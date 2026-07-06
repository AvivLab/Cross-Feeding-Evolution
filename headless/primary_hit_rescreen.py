#!/usr/bin/env python3
"""
Re-screen primary-batch offload rows with new random seeds.

After a primary campaign finishes, this tool:

1. Reads offload data and finds rows that were *hits* (default) or *non-hits* (``--non-hits``).
2. Re-runs each unique parameter vector N times with fresh random seeds.
3. Reports hit_rate = fraction of those re-runs that pass metric filters again.

Usage (from this folder)::

    python headless/primary_hit_rescreen.py SESSION_FOLDER --n-seeds 20
    python headless/primary_hit_rescreen.py SESSION_FOLDER --non-hits --max-hits 500

Outputs under ``<session>/Re-Runs/`` (hits) or ``<session>/Re-Runs-NonHits/`` (non-hits).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gui.apps.neutral_comparison.batch import (
    _merge_params,
    _metric_checks_ordered_cheapest_first_many,
    _metric_passes,
    simulation_light_tracking_plan,
)
from gui.apps.neutral_comparison.offload import param_dict_from_offload_vector
from gui.apps.batch_runner.csv_output import update_session_hit_counts_csv_from_rescreen
from gui.apps.neutral_comparison.parameter_heatmap import load_primary_offload_context
from gui.apps.neutral_comparison.primary_event_chart import _primary_hit_from_record
from gui.common.simulation_settings import RANDOM_SEED_OPTIONAL
from gui.models.registry import get_model_by_key
from gui.persistence.full_save import full_save_settings_path_json
from gui.persistence.json_io import make_read_json_maybe_gz_fn, write_json_maybe_gz_atomic
from headless.hpc_common import (
    discover_primary_session_dirs,
    effective_process_pool_workers,
    is_primary_batch_session_folder,
    prune_irrelevant_bounds_for_toggles,
)
from headless.neutral_comparison import _bounds_from_save_json, _metric_checks_from_filters
from headless.primary_batch_session_meta import (
    _plot_label_config_key,
    _resolve_session_meta,
    load_campaign_summary,
)
from headless.primary_events_chart import infer_run_headline, infer_session_id
from simulation.change_history import event_totals_from_metric_input
from tools.hpc_suite_manifest import COMPARE_CONFIG_ORDER

RE_RUNNER_VERSION = "1.6.0"
SCREEN_MODE_HITS = "hits"
SCREEN_MODE_NON_HITS = "non_hits"
DEFAULT_OUT_SUBDIR = "Re-Runs"
NON_HIT_OUT_SUBDIR = "Re-Runs-NonHits"
_ARTIFACT_PREFIX_BY_MODE = {
    SCREEN_MODE_HITS: "primary_hit_rescreen",
    SCREEN_MODE_NON_HITS: "primary_non_hit_rescreen",
}
_KIND_BY_MODE = {
    SCREEN_MODE_HITS: "primary_hit_rescreen",
    SCREEN_MODE_NON_HITS: "primary_non_hit_rescreen",
}
_COMPARE_KIND_BY_MODE = {
    SCREEN_MODE_HITS: "primary_hit_rescreen_compare",
    SCREEN_MODE_NON_HITS: "primary_non_hit_rescreen_compare",
}
_SUITE_TAG_RE = re.compile(r"^(Variable_Ratio|Fixed_\d+(?:\.\d+)?_ratio)$")

RescreenProgressCallback = Callable[[int, int, str], None]


def _emit_rescreen_progress(
    callback: Optional[RescreenProgressCallback],
    done: int,
    total: int,
    message: str,
) -> None:
    if callback is None:
        return
    try:
        t = max(0, int(total))
        d = max(0, int(done))
        if t == 0:
            callback(0, 1, str(message))
            return
        callback(min(d, t), t, str(message))
    except Exception:
        pass


def _rescreen_trial_total(n_points: int, n_seeds: int) -> int:
    return max(0, int(n_points)) * max(1, int(n_seeds))


def _trial_progress_message(
    trials_done: int,
    total_trials: int,
    *,
    point_number: int,
    n_points: int,
    source_run_index: int,
) -> str:
    return (
        f"Re-run {trials_done}/{total_trials} "
        f"(point {point_number}/{n_points}, source run {source_run_index})"
    )


def normalize_screen_mode(mode: str) -> str:
    """Return ``hits`` or ``non_hits`` from CLI/env synonyms."""
    raw = str(mode or SCREEN_MODE_HITS).strip().lower().replace("-", "_")
    if raw in ("non_hits", "nonhit", "nonhits", "miss", "misses", "inverse"):
        return SCREEN_MODE_NON_HITS
    return SCREEN_MODE_HITS


def screen_mode_from_env() -> str:
    return normalize_screen_mode(os.environ.get("RE_RUNNER_SCREEN_MODE", SCREEN_MODE_HITS))


def out_subdir_for_screen_mode(screen_mode: str) -> str:
    return NON_HIT_OUT_SUBDIR if normalize_screen_mode(screen_mode) == SCREEN_MODE_NON_HITS else DEFAULT_OUT_SUBDIR


def artifact_prefix_for_screen_mode(screen_mode: str) -> str:
    return _ARTIFACT_PREFIX_BY_MODE[normalize_screen_mode(screen_mode)]


def rescreen_kind_for_screen_mode(screen_mode: str) -> str:
    return _KIND_BY_MODE[normalize_screen_mode(screen_mode)]


def rescreen_artifact_basename(session_id: str, screen_mode: str) -> str:
    return f"{artifact_prefix_for_screen_mode(screen_mode)}_{session_id}"


def _record_matches_screen_mode(rec: Dict[str, Any], screen_mode: str) -> bool:
    hit_flag = _primary_hit_from_record(rec)
    if hit_flag is None:
        return False
    if normalize_screen_mode(screen_mode) == SCREEN_MODE_NON_HITS:
        return hit_flag is False
    return hit_flag is True


def _infer_suite_tag_from_session_dir(session_dir: str) -> str:
    """Infer suite folder name from ``Output/<suite>/<session>/`` layout."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(session_dir)))
    if _SUITE_TAG_RE.match(parent):
        return parent
    return ""
DEFAULT_N_SEEDS = 20
DEFAULT_RESCREEN_BASE_SEED = 900_000_001
BATCH_SESSIONS_SUBDIR = "sessions"


def _default_rescreen_workers() -> int:
    """Slurm CPUs per task when set; else local CPU count (minimum 1)."""
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if cpus > 0:
            return cpus
    except Exception:
        pass
    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1

_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)


@dataclass(frozen=True)
class HitSpec:
    """One original hit simulation to re-screen."""

    hit_index: int
    source_run_index: int
    param_vector: Tuple[float, ...]
    param_key: str
    original_seed: Optional[int]
    total_deaths: Optional[int] = None
    total_duplications: Optional[int] = None
    total_mutations: Optional[int] = None
    total_outflows: Optional[int] = None


@dataclass(frozen=True)
class SessionRescreenContext:
    session_id: str
    session_dir: str
    param_names_list: List[str]
    numeric_base: Dict[str, Any]
    toggles: Dict[str, Any]
    metric_checks: List[Tuple[str, str, float]]
    sim_light_used: bool
    sim_light_canon: Tuple[str, ...]
    settings_path: str
    plot_label: str = ""
    config_key: str = ""
    suite_tag: str = ""


def _load_settings_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        data = _read_json_maybe_gz(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve_settings_path(session_dir: str, session_id: str) -> Optional[str]:
    session_dir = os.path.abspath(session_dir)
    sid = str(session_id or "").strip()
    for path in (
        full_save_settings_path_json(session_dir, sid),
        full_save_settings_path_json(session_dir, sid) + ".gz",
    ):
        if os.path.isfile(path):
            return path
    return None


def _config_key_from_job_stem(stem: str) -> str:
    from tools.hpc_suite_manifest import primary_job_key

    key = primary_job_key(str(stem or "").strip())
    return key or str(stem or "").strip()


def _campaign_summary_fallback(session_dir: str) -> Optional[Dict[str, Any]]:
    data = load_campaign_summary(session_dir)
    if not isinstance(data, dict):
        return None
    if str(data.get("kind") or "") not in ("primary_batch_campaign", "neutral_set_comparison"):
        return None
    return data


def load_session_rescreen_context(
    session_dir: str,
    session_id: Optional[str] = None,
    *,
    suite_tag: str = "",
) -> SessionRescreenContext:
    """Build rerun context from ``full_save_settings_<sid>.json`` + offload param names."""
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    sid = (session_id or "").strip() or infer_session_id(session_dir) or ""
    if not sid:
        raise ValueError(f"Could not infer session_id under {session_dir!r}")

    settings_path = _resolve_settings_path(session_dir, sid)
    settings = _load_settings_json(settings_path or "")
    campaign = _campaign_summary_fallback(session_dir)
    if not settings and campaign:
        settings = dict(campaign)
        settings["kind"] = "primary_batch_campaign_session"
        settings_path = settings_path or ""
    if not settings:
        raise FileNotFoundError(
            f"Missing full_save_settings for session {sid!r} in {session_dir!r}"
        )
    if str(settings.get("kind") or "") not in (
        "primary_batch_campaign_session",
        "primary_batch_campaign",
    ):
        raise ValueError(
            f"Unexpected settings kind {settings.get('kind')!r} in {settings_path!r}"
        )

    num_p = settings.get("primary_numeric_parameters")
    tog_p = settings.get("primary_toggles")
    if (not isinstance(num_p, dict) or not isinstance(tog_p, dict)) and campaign:
        if not isinstance(num_p, dict):
            num_p = campaign.get("primary_numeric_parameters")
        if not isinstance(tog_p, dict):
            tog_p = campaign.get("primary_toggles")
    if not isinstance(num_p, dict) or not isinstance(tog_p, dict):
        raise ValueError("primary_numeric_parameters / primary_toggles missing in settings")

    metric_filters = settings.get("metric_filters")
    if metric_filters is None and campaign:
        metric_filters = campaign.get("metric_filters")
    bounds_raw = settings.get("primary_bounds")
    if bounds_raw is None and campaign:
        bounds_raw = campaign.get("primary_bounds")
    primary_bounds: Dict[str, Tuple[float, float]] = {}
    if bounds_raw is not None:
        primary_bounds = _bounds_from_save_json(bounds_raw, label="primary_bounds")
        primary_bounds = prune_irrelevant_bounds_for_toggles(primary_bounds, tog_p)
    metric_checks, _snapshot = _metric_checks_from_filters(metric_filters)

    param_names = settings.get("primary_offload_param_names")
    if not isinstance(param_names, list) or not param_names:
        _records, param_names_loaded = load_primary_offload_context(session_dir, sid, _read_json_maybe_gz)
        param_names = param_names_loaded
        if not param_names:
            raise ValueError(f"No param_names_list for session {sid!r}")

    model_spec = get_model_by_key(str((settings.get("model") or {}).get("key") or "simulation"))
    sim_light_used = bool(settings.get("simulation_light_tracking"))
    sim_light_canon = tuple(settings.get("simulation_light_tracking_metrics") or ())
    if not sim_light_canon:
        sim_light_used, sim_light_canon = simulation_light_tracking_plan(model_spec, metric_checks)

    tag = str(suite_tag or "").strip()
    if not tag:
        tag = _infer_suite_tag_from_session_dir(session_dir)
    resolved = _resolve_session_meta(session_dir, tag)
    meta_label = ""
    if resolved:
        meta_sid, meta_label, _ = resolved
        if meta_sid:
            sid = str(meta_sid).strip() or sid

    stem = str(settings.get("headless_primary_job_file_stem") or "").strip()
    if not stem:
        stem = (infer_run_headline(session_dir, sid) or "").strip()
    if meta_label:
        plot_label = str(meta_label).strip()
        config_key = _plot_label_config_key(plot_label)
    else:
        plot_label = f"{stem} [{tag}]" if tag and stem else (stem or sid)
        config_key = _config_key_from_job_stem(stem) if stem else ""

    return SessionRescreenContext(
        session_id=sid,
        session_dir=session_dir,
        param_names_list=[str(x) for x in param_names],
        numeric_base=dict(num_p),
        toggles=dict(tog_p),
        metric_checks=list(metric_checks),
        sim_light_used=bool(sim_light_used),
        sim_light_canon=tuple(str(x) for x in sim_light_canon),
        settings_path=str(settings_path),
        plot_label=plot_label,
        config_key=config_key,
        suite_tag=tag,
    )


def _param_vector_key(vec: Sequence[Any], *, decimals: int = 8) -> str:
    parts: List[str] = []
    for x in vec:
        try:
            v = float(x)
            parts.append(f"{round(v, decimals):.{decimals}f}" if np.isfinite(v) else "nan")
        except (TypeError, ValueError):
            parts.append("nan")
    return "|".join(parts)


def _original_seed_from_record(rec: Dict[str, Any]) -> Optional[int]:
    mi = rec.get("metric_input")
    if not isinstance(mi, dict):
        return None
    raw = mi.get("random_seed_used")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def count_hit_specs(
    session_dir: str,
    session_id: str,
    *,
    screen_mode: str = SCREEN_MODE_HITS,
) -> Tuple[int, int]:
    """Return ``(total_rows, unique_param_vectors)`` for hits or non-hits."""
    records, _ = load_primary_offload_context(session_dir, session_id, _read_json_maybe_gz)
    total = 0
    seen: set = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if not _record_matches_screen_mode(rec, screen_mode):
            continue
        vec = rec.get("param_vector")
        if not isinstance(vec, list) or not vec:
            continue
        total += 1
        seen.add(_param_vector_key(vec))
    return total, len(seen)


def collect_hit_specs(
    session_dir: str,
    session_id: str,
    *,
    max_hits: int = 0,
    dedupe_params: bool = True,
    screen_mode: str = SCREEN_MODE_HITS,
) -> List[HitSpec]:
    """Scan offload rows and return parameter sets to re-screen (hits or non-hits)."""
    records, param_names = load_primary_offload_context(session_dir, session_id, _read_json_maybe_gz)
    if not param_names:
        ctx = load_session_rescreen_context(session_dir, session_id)
        param_names = ctx.param_names_list

    hits: List[HitSpec] = []
    seen: set = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if not _record_matches_screen_mode(rec, screen_mode):
            continue
        vec = rec.get("param_vector")
        if not isinstance(vec, list) or not vec:
            continue
        key = _param_vector_key(vec) if dedupe_params else f"run_{rec.get('run_index', len(hits))}"
        if dedupe_params and key in seen:
            continue
        seen.add(key)
        try:
            run_index = int(rec.get("run_index") if rec.get("run_index") is not None else rec.get("run_id") or 0)
        except (TypeError, ValueError):
            run_index = len(hits)
        totals = event_totals_from_metric_input(rec.get("metric_input"))
        total_kwargs: Dict[str, Optional[int]] = {
            "total_deaths": None,
            "total_duplications": None,
            "total_mutations": None,
            "total_outflows": None,
        }
        if totals is not None:
            td, tdup, tmut, tout = totals
            total_kwargs = {
                "total_deaths": int(td),
                "total_duplications": int(tdup),
                "total_mutations": int(tmut),
                "total_outflows": int(tout),
            }
        try:
            param_vector = tuple(float(x) for x in vec)
        except (TypeError, ValueError):
            continue
        # hit_index order matches build_primary_hit_csv_rows when dedupe_params matches.
        hits.append(
            HitSpec(
                hit_index=len(hits),
                source_run_index=run_index,
                param_vector=param_vector,
                param_key=key,
                original_seed=_original_seed_from_record(rec),
                **total_kwargs,
            )
        )
        if max_hits > 0 and len(hits) >= max_hits:
            break
    return hits


def decode_hit_params(
    ctx: SessionRescreenContext,
    hit: HitSpec,
) -> Dict[str, float]:
    """Parameter values stored in the offload ``param_vector`` for one hit."""
    return param_dict_from_offload_vector(hit.param_vector, ctx.param_names_list, strict=False)


def _merged_params_for_hit(
    ctx: SessionRescreenContext,
    hit: HitSpec,
    *,
    seed: int,
) -> Dict[str, Any]:
    """Rebuild full simulation params for one hit: session toggles + stored draw + new seed."""
    draw = decode_hit_params(ctx, hit)
    params = _merge_params(
        ctx.numeric_base,
        ctx.toggles,
        draw,
        simulation_light_batch=ctx.sim_light_used,
        tracking_metric_names_union=ctx.sim_light_canon if ctx.sim_light_used else None,
        keep_optional_final_arrays=False,
    )
    params[RANDOM_SEED_OPTIONAL] = int(seed)
    return params


def assert_rescreen_params_match_hit(
    ctx: SessionRescreenContext,
    hit: HitSpec,
    rebuilt: Dict[str, Any],
    *,
    expected_seed: int,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """
    Verify re-screen simulation inputs match the stored hit (same conditions), except seed.

    Raises ``AssertionError`` when decoded offload parameters or session toggles do not match.
    """
    decoded = decode_hit_params(ctx, hit)
    for name in ctx.param_names_list:
        if name not in decoded:
            continue
        if name not in rebuilt:
            raise AssertionError(f"rebuilt params missing stored hit field {name!r}")
        want = float(decoded[name])
        got = float(rebuilt[name])
        if not np.isclose(got, want, rtol=rtol, atol=atol):
            raise AssertionError(
                f"param {name!r} mismatch: stored hit {want!r} vs rebuilt {got!r}"
            )
    for key, want in ctx.toggles.items():
        if key not in rebuilt:
            raise AssertionError(f"rebuilt params missing toggle {key!r}")
        if bool(rebuilt[key]) != bool(want):
            raise AssertionError(
                f"toggle {key!r} mismatch: settings {want!r} vs rebuilt {rebuilt[key]!r}"
            )
    if int(rebuilt.get(RANDOM_SEED_OPTIONAL, -1)) != int(expected_seed):
        raise AssertionError(
            f"seed mismatch: expected {expected_seed}, got {rebuilt.get(RANDOM_SEED_OPTIONAL)!r}"
        )


def evaluate_hit_for_params(
    params: Dict[str, Any],
    metric_checks: Sequence[Tuple[str, str, float]],
) -> Tuple[bool, Dict[str, float]]:
    """Run one simulation and evaluate metric filters (same logic as primary batch)."""
    model_spec = get_model_by_key("simulation")
    ordered = _metric_checks_ordered_cheapest_first_many(list(metric_checks))
    try:
        res = model_spec.run_simulation(params)
    except Exception as exc:
        print(
            f"[rescreen] simulation exception: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        res = None
    if res is None or model_spec.is_failed(res):
        return False, {}
    evaluated: Dict[str, float] = {}
    ok = True
    for mname, op, thr in ordered:
        try:
            v = float(model_spec.compute_metric(res, mname))
        except Exception:
            v = float("nan")
        evaluated[str(mname)] = v
        if not _metric_passes(v, op, thr):
            ok = False
            break
    return ok, evaluated


def _rescreen_worker(payload: Dict[str, Any]) -> Tuple[int, int, int, bool]:
    """Process-pool worker: (hit_index, trial, seed, hit_again)."""
    params = payload["params"]
    metric_checks = [(str(t[0]), str(t[1]), float(t[2])) for t in payload["metric_checks"]]
    hit_again, _ = evaluate_hit_for_params(params, metric_checks)
    return (
        int(payload["hit_index"]),
        int(payload["trial"]),
        int(payload["seed"]),
        bool(hit_again),
    )


def _ctx_to_payload(ctx: SessionRescreenContext) -> Dict[str, Any]:
    return {
        "session_id": ctx.session_id,
        "session_dir": ctx.session_dir,
        "param_names_list": list(ctx.param_names_list),
        "numeric_base": dict(ctx.numeric_base),
        "toggles": dict(ctx.toggles),
        "metric_checks": list(ctx.metric_checks),
        "sim_light_used": bool(ctx.sim_light_used),
        "sim_light_canon": list(ctx.sim_light_canon),
        "settings_path": ctx.settings_path,
        "plot_label": ctx.plot_label,
        "config_key": ctx.config_key,
        "suite_tag": ctx.suite_tag,
    }


def _ctx_from_payload(data: Dict[str, Any]) -> SessionRescreenContext:
    return SessionRescreenContext(
        session_id=str(data["session_id"]),
        session_dir=str(data["session_dir"]),
        param_names_list=list(data["param_names_list"]),
        numeric_base=dict(data["numeric_base"]),
        toggles=dict(data["toggles"]),
        metric_checks=[
            (str(t[0]), str(t[1]), float(t[2])) for t in data["metric_checks"]
        ],
        sim_light_used=bool(data["sim_light_used"]),
        sim_light_canon=tuple(str(x) for x in (data.get("sim_light_canon") or ())),
        settings_path=str(data.get("settings_path") or ""),
        plot_label=str(data.get("plot_label") or ""),
        config_key=str(data.get("config_key") or ""),
        suite_tag=str(data.get("suite_tag") or ""),
    )


def _event_total_row_columns(hit: HitSpec) -> Dict[str, Any]:
    """Primary-run event totals from offload change_history (blank when unavailable)."""
    if hit.total_deaths is None:
        return _event_total_row_columns_from_totals(None)
    return _event_total_row_columns_from_totals(
        (hit.total_deaths, hit.total_duplications, hit.total_mutations, hit.total_outflows)
    )


def _event_total_row_columns_from_totals(
    totals: Optional[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]],
) -> Dict[str, Any]:
    keys = (
        "total_deaths",
        "total_duplications",
        "total_mutations",
        "total_outflows",
    )
    if totals is None or totals[0] is None:
        return {k: "" for k in keys}
    td, tdup, tmut, tout = totals
    return {
        "total_deaths": int(td),
        "total_duplications": int(tdup),
        "total_mutations": int(tmut),
        "total_outflows": int(tout),
    }


def _event_totals_by_source_run_index(
    session_dir: str,
    session_id: str,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Map primary offload ``run_index`` to summed event totals."""
    records, _ = load_primary_offload_context(session_dir, session_id, _read_json_maybe_gz)
    out: Dict[int, Tuple[int, int, int, int]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        try:
            run_index = int(
                rec.get("run_index")
                if rec.get("run_index") is not None
                else rec.get("run_id") or -1
            )
        except (TypeError, ValueError):
            continue
        if run_index < 0:
            continue
        totals = event_totals_from_metric_input(rec.get("metric_input"))
        if totals is not None:
            out[run_index] = totals
    return out


def backfill_rescreen_event_totals_in_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    session_dir: str,
    session_id: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Add/update event-total columns on existing rescreen rows (no re-simulation)."""
    lookup = _event_totals_by_source_run_index(session_dir, session_id)
    updated_rows: List[Dict[str, Any]] = []
    n_filled = 0
    n_missing = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        patched = dict(row)
        try:
            src = int(row.get("source_run_index"))
        except (TypeError, ValueError):
            n_missing += 1
            patched.update(_event_total_row_columns_from_totals(None))
            updated_rows.append(patched)
            continue
        totals = lookup.get(src)
        patched.update(_event_total_row_columns_from_totals(totals))
        if totals is None:
            n_missing += 1
        else:
            n_filled += 1
        updated_rows.append(patched)
    return updated_rows, n_filled, n_missing


def _is_rescreen_session_json_path(json_path: str, *, screen_mode: str) -> bool:
    base = os.path.basename(json_path)
    prefix = artifact_prefix_for_screen_mode(screen_mode)
    if not base.startswith(f"{prefix}_") or not base.endswith(".json"):
        return False
    if ".rank" in base or ".partial" in base:
        return False
    sid = base[len(prefix) + 1 : -5]
    return bool(sid) and sid != "compare"


def discover_rescreen_session_json_paths(path: str, *, screen_mode: str = SCREEN_MODE_HITS) -> List[str]:
    """Find completed per-session rescreen JSON(s) for backfill."""
    root = Path(os.path.abspath(os.path.expanduser(path)))
    prefix = artifact_prefix_for_screen_mode(screen_mode)

    def _filter_json_paths(candidates: Iterable[Path]) -> List[str]:
        found = sorted(
            str(p)
            for p in candidates
            if p.is_file() and _is_rescreen_session_json_path(str(p), screen_mode=screen_mode)
        )
        return found

    if root.is_file():
        if root.suffix == ".json":
            if not _is_rescreen_session_json_path(str(root), screen_mode=screen_mode):
                raise FileNotFoundError(f"Not a per-session rescreen JSON: {str(root)!r}")
            return [str(root)]
        if root.suffix == ".csv":
            json_path = root.with_suffix(".json")
            if json_path.is_file():
                return [str(json_path)]
            raise FileNotFoundError(f"No sibling rescreen JSON for CSV: {str(root)!r}")
    if root.is_dir():
        if root.name in (DEFAULT_OUT_SUBDIR, NON_HIT_OUT_SUBDIR):
            found = _filter_json_paths(
                root.joinpath(BATCH_SESSIONS_SUBDIR).glob(f"*/{prefix}_*.json")
            )
            if found:
                return found
        if root.name == BATCH_SESSIONS_SUBDIR:
            found = _filter_json_paths(root.glob(f"*/{prefix}_*.json"))
            if found:
                return found
        found = _filter_json_paths(root.glob(f"{prefix}_*.json"))
        if found:
            return found
    raise FileNotFoundError(f"No rescreen session JSON found under {str(root)!r}")


def backfill_rescreen_event_totals_for_json(
    json_path: str,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Rewrite one rescreen session CSV/JSON with primary event-total columns."""
    json_path = os.path.abspath(os.path.expanduser(json_path))
    meta = _read_json_maybe_gz(json_path)
    if not isinstance(meta, dict):
        raise ValueError(f"invalid rescreen JSON: {json_path!r}")
    session_dir = str(meta.get("session_dir") or "").strip()
    session_id = str(meta.get("session_id") or "").strip()
    if not session_dir or not os.path.isdir(session_dir):
        raise FileNotFoundError(
            f"rescreen JSON {json_path!r} lacks usable session_dir {session_dir!r}"
        )
    if not session_id:
        session_id = infer_session_id(session_dir) or ""
    rows = meta.get("rows")
    if not isinstance(rows, list) or not rows:
        csv_path = str(meta.get("csv_path") or "").strip()
        if csv_path and os.path.isfile(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            sibling_csv = os.path.splitext(json_path)[0] + ".csv"
            if os.path.isfile(sibling_csv):
                with open(sibling_csv, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            else:
                raise ValueError(f"no rows in JSON and no CSV beside {json_path!r}")

    csv_path = str(meta.get("csv_path") or os.path.splitext(json_path)[0] + ".csv")
    csv_path = os.path.abspath(csv_path)
    if dry_run:
        return {
            "session_id": session_id,
            "json_path": json_path,
            "csv_path": csv_path,
            "n_rows": len(rows),
            "n_filled": None,
            "n_missing": None,
            "dry_run": True,
        }

    print(f"  loading primary offload: {session_dir}", flush=True)
    patched_rows, n_filled, n_missing = backfill_rescreen_event_totals_in_rows(
        rows,
        session_dir=session_dir,
        session_id=session_id,
    )
    result = {
        "session_id": session_id,
        "json_path": json_path,
        "csv_path": csv_path,
        "n_rows": len(patched_rows),
        "n_filled": int(n_filled),
        "n_missing": int(n_missing),
        "dry_run": False,
    }

    write_rescreen_csv(csv_path, patched_rows)
    meta["rows"] = patched_rows
    meta["csv_path"] = csv_path
    _write_rescreen_json_atomic(json_path, meta)
    return result


def backfill_rescreen_event_totals_batch(
    path: str,
    *,
    screen_mode: str = SCREEN_MODE_HITS,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Backfill all rescreen session artifacts under a batch root or one session folder."""
    json_paths = discover_rescreen_session_json_paths(path, screen_mode=screen_mode)
    ok = 0
    fail = 0
    results: List[Dict[str, Any]] = []
    for json_path in json_paths:
        base = os.path.basename(json_path)
        prefix = artifact_prefix_for_screen_mode(screen_mode)
        sid = base[len(prefix) + 1 : -5] if base.startswith(f"{prefix}_") else base
        print(f"[backfill_event_totals] {sid}", flush=True)
        try:
            summary = backfill_rescreen_event_totals_for_json(json_path, dry_run=dry_run)
        except Exception as exc:
            fail += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        ok += 1
        results.append(summary)
        if dry_run:
            print(
                f"  rows {summary.get('n_rows')} | (dry-run; primary offload not loaded)",
                flush=True,
            )
        else:
            print(
                f"  rows {summary.get('n_rows')} | filled {summary.get('n_filled')} | "
                f"missing {summary.get('n_missing')} | {summary.get('csv_path')}",
                flush=True,
            )
    return {"ok": ok, "fail": fail, "sessions": results, "dry_run": bool(dry_run)}


def _hit_to_payload(hit: HitSpec) -> Dict[str, Any]:
    return {
        "hit_index": int(hit.hit_index),
        "source_run_index": int(hit.source_run_index),
        "param_vector": list(hit.param_vector),
        "param_key": str(hit.param_key),
        "original_seed": hit.original_seed,
        "total_deaths": hit.total_deaths,
        "total_duplications": hit.total_duplications,
        "total_mutations": hit.total_mutations,
        "total_outflows": hit.total_outflows,
    }


def _hit_from_payload(data: Dict[str, Any]) -> HitSpec:
    original_seed = data.get("original_seed")
    if original_seed is not None and original_seed != "":
        original_seed = int(original_seed)
    else:
        original_seed = None

    def _optional_int(key: str) -> Optional[int]:
        raw = data.get(key)
        if raw is None or raw == "":
            return None
        return int(raw)

    return HitSpec(
        hit_index=int(data["hit_index"]),
        source_run_index=int(data["source_run_index"]),
        param_vector=tuple(float(x) for x in data["param_vector"]),
        param_key=str(data["param_key"]),
        original_seed=original_seed,
        total_deaths=_optional_int("total_deaths"),
        total_duplications=_optional_int("total_duplications"),
        total_mutations=_optional_int("total_mutations"),
        total_outflows=_optional_int("total_outflows"),
    )


def _seed_for_trial(
    rescreen_base_seed: int,
    hit_index: int,
    trial: int,
) -> int:
    """Deterministic re-screen seed (separate formula from primary-batch simulation_seed_for_run)."""
    return int(rescreen_base_seed) + int(hit_index) * 10_007 + int(trial) * 7919


def rescreen_hit(
    ctx: SessionRescreenContext,
    hit: HitSpec,
    *,
    n_seeds: int,
    rescreen_base_seed: int,
    workers: int = 1,
    on_trial_complete: Optional[Callable[[HitSpec, int], None]] = None,
) -> Tuple[int, List[Tuple[int, int, bool]]]:
    """Re-run one hit ``n_seeds`` times; return (n_hit_again, per-trial details)."""
    n_seeds = max(1, int(n_seeds))
    trials: List[Tuple[int, int, bool]] = []
    if workers <= 1:
        n_hit_again = 0
        for trial in range(n_seeds):
            seed = _seed_for_trial(rescreen_base_seed, hit.hit_index, trial)
            params = _merged_params_for_hit(ctx, hit, seed=seed)
            assert_rescreen_params_match_hit(ctx, hit, params, expected_seed=seed)
            hit_again, _ = evaluate_hit_for_params(params, ctx.metric_checks)
            trials.append((trial, seed, hit_again))
            if hit_again:
                n_hit_again += 1
            if on_trial_complete is not None:
                on_trial_complete(hit, trial)
        return n_hit_again, trials

    payloads: List[Dict[str, Any]] = []
    for trial in range(n_seeds):
        seed = _seed_for_trial(rescreen_base_seed, hit.hit_index, trial)
        params = _merged_params_for_hit(ctx, hit, seed=seed)
        assert_rescreen_params_match_hit(ctx, hit, params, expected_seed=seed)
        payloads.append(
            {
                "hit_index": hit.hit_index,
                "trial": trial,
                "seed": seed,
                "params": params,
                "metric_checks": list(ctx.metric_checks),
            }
        )
    n_hit_again = 0
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futs = [pool.submit(_rescreen_worker, p) for p in payloads]
        for fut in as_completed(futs):
            _hi, trial, seed, hit_again = fut.result()
            trials.append((trial, seed, hit_again))
            if hit_again:
                n_hit_again += 1
            if on_trial_complete is not None:
                on_trial_complete(hit, trial)
    trials.sort(key=lambda t: t[0])
    return n_hit_again, trials


def _row_from_rescreen_trials(
    ctx: SessionRescreenContext,
    hit: HitSpec,
    *,
    sid: str,
    n_seeds: int,
    trials: Sequence[Tuple[int, int, bool]],
    rescreen_base_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one rescreen CSV row from completed per-trial results."""
    ordered = sorted(trials, key=lambda t: int(t[0]))
    if rescreen_base_seed is not None:
        _validate_trial_list_complete(
            ordered,
            hit_index=int(hit.hit_index),
            n_seeds=int(n_seeds),
            rescreen_base_seed=int(rescreen_base_seed),
        )
    n_hit_again = sum(1 for _trial, _seed, hit_again in ordered if hit_again)
    rate = float(n_hit_again) / float(n_seeds) if n_seeds else 0.0
    params_json = _params_json_for_row(ctx, hit)
    _validate_params_json_matches_hit(ctx, hit, params_json)
    row: Dict[str, Any] = {
        "session_id": sid,
        "hit_index": hit.hit_index,
        "source_run_index": hit.source_run_index,
        "original_seed": hit.original_seed if hit.original_seed is not None else "",
        "param_key": hit.param_key,
        "params_json": params_json,
        "n_seeds": int(n_seeds),
        "n_hit_again": int(n_hit_again),
        "hit_rate": round(rate, 4),
        **_event_total_row_columns(hit),
        "seeds_tested": ";".join(str(t[1]) for t in ordered),
        "trial_hits": ";".join("1" if t[2] else "0" for t in ordered),
    }
    _validate_rescreen_row(
        row,
        n_seeds=n_seeds,
        hit_index=int(hit.hit_index),
        rescreen_base_seed=rescreen_base_seed,
    )
    return row


def _build_rescreen_row(
    ctx: SessionRescreenContext,
    hit: HitSpec,
    *,
    sid: str,
    n_seeds: int,
    rescreen_base_seed: int,
    seed_workers: int = 1,
    on_trial_complete: Optional[Callable[[HitSpec, int], None]] = None,
) -> Dict[str, Any]:
    """Build one CSV/JSON row for a hit (sequential or inside a process-pool worker)."""
    n_hit_again, trials = rescreen_hit(
        ctx,
        hit,
        n_seeds=n_seeds,
        rescreen_base_seed=rescreen_base_seed,
        workers=max(1, int(seed_workers)),
        on_trial_complete=on_trial_complete,
    )
    return _row_from_rescreen_trials(
        ctx,
        hit,
        sid=sid,
        n_seeds=n_seeds,
        trials=trials,
        rescreen_base_seed=rescreen_base_seed,
    )


def _rescreen_trial_worker(payload: Dict[str, Any]) -> Tuple[int, int, int, bool]:
    """Process-pool worker: evaluate one (hit, seed trial) re-screen."""
    ctx = _ctx_from_payload(payload["ctx"])
    hit = _hit_from_payload(payload["hit"])
    trial = int(payload["trial"])
    base = int(payload["rescreen_base_seed"])
    seed = _seed_for_trial(base, hit.hit_index, trial)
    params = _merged_params_for_hit(ctx, hit, seed=seed)
    assert_rescreen_params_match_hit(ctx, hit, params, expected_seed=seed)
    hit_again, _ = evaluate_hit_for_params(params, ctx.metric_checks)
    return hit.hit_index, trial, seed, bool(hit_again)


def _rescreen_hits_flat_parallel(
    *,
    ctx: SessionRescreenContext,
    hits: Sequence[HitSpec],
    all_hits: Sequence[HitSpec],
    sid: str,
    n_seeds: int,
    rescreen_base_seed: int,
    workers: int,
    rank: int,
    n_nodes: int,
    progress: bool,
    progress_callback: Optional[RescreenProgressCallback] = None,
) -> List[Dict[str, Any]]:
    n_tasks = len(hits) * int(n_seeds)
    pool_workers = effective_process_pool_workers(workers, n_tasks)
    if progress:
        print(
            f"[rescreen] rank {rank}/{n_nodes} flat trial pool max_workers={pool_workers} "
            f"(workers={workers}, hits={len(hits)}, n_seeds={n_seeds}, tasks={n_tasks}, "
            f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', '(unset)')})",
            flush=True,
        )
    ctx_payload = _ctx_to_payload(ctx)
    payloads = [
        {
            "ctx": ctx_payload,
            "hit": _hit_to_payload(hit),
            "trial": trial,
            "rescreen_base_seed": int(rescreen_base_seed),
        }
        for hit in hits
        for trial in range(int(n_seeds))
    ]
    trials_by_hit: Dict[int, List[Tuple[int, int, bool]]] = defaultdict(list)
    trials_done = 0
    total_trials = _rescreen_trial_total(len(all_hits), n_seeds)
    with ProcessPoolExecutor(max_workers=pool_workers) as pool:
        futs = [pool.submit(_rescreen_trial_worker, p) for p in payloads]
        for fut in as_completed(futs):
            hit_index, trial, seed, hit_again = fut.result()
            trials_by_hit[int(hit_index)].append((int(trial), int(seed), bool(hit_again)))
            trials_done += 1
            hit = next(h for h in hits if int(h.hit_index) == int(hit_index))
            _emit_rescreen_progress(
                progress_callback,
                trials_done,
                total_trials,
                _trial_progress_message(
                    trials_done,
                    total_trials,
                    point_number=int(hit.hit_index) + 1,
                    n_points=len(all_hits),
                    source_run_index=hit.source_run_index,
                ),
            )
    rows: List[Dict[str, Any]] = []
    for hit in hits:
        row = _row_from_rescreen_trials(
            ctx,
            hit,
            sid=sid,
            n_seeds=n_seeds,
            trials=trials_by_hit[int(hit.hit_index)],
            rescreen_base_seed=rescreen_base_seed,
        )
        rows.append(row)
        if progress:
            print(
                f"  hit {hit.hit_index + 1}/{len(all_hits)} "
                f"(source run {hit.source_run_index}, rank {rank}/{n_nodes}): "
                f"hit_rate={row.get('hit_rate')} …",
                flush=True,
            )
    rows.sort(key=lambda r: int(r["hit_index"]))
    _validate_rescreen_rows_integrity(
        rows,
        hits,
        ctx,
        sid=sid,
        n_seeds=n_seeds,
        rescreen_base_seed=rescreen_base_seed,
    )
    return rows


def _rescreen_hit_row_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool worker: re-screen one hit and return its CSV row."""
    track_file = os.environ.get("RESCREEN_PARALLEL_TRACK_FILE", "").strip()
    if track_file:
        track_sleep = float(os.environ.get("RESCREEN_PARALLEL_TRACK_SLEEP", "0.1"))
        with open(track_file, "a", encoding="utf-8") as tf:
            tf.write(f"start\t{os.getpid()}\t{time.time()}\n")
            tf.flush()
        if track_sleep > 0:
            time.sleep(track_sleep)
    ctx = _ctx_from_payload(payload["ctx"])
    hit = _hit_from_payload(payload["hit"])
    progress_label = str(payload.get("progress_label") or "").strip()
    if progress_label:
        print(
            f"[{progress_label}] hit {hit.hit_index + 1} "
            f"worker_pid={os.getpid()} n_seeds={int(payload['n_seeds'])}",
            flush=True,
        )
    row = _build_rescreen_row(
        ctx,
        hit,
        sid=str(payload["session_id"]),
        n_seeds=int(payload["n_seeds"]),
        rescreen_base_seed=int(payload["rescreen_base_seed"]),
        seed_workers=int(payload.get("seed_workers") or 1),
    )
    if track_file:
        with open(track_file, "a", encoding="utf-8") as tf:
            tf.write(f"end\t{os.getpid()}\t{time.time()}\n")
    return row


def _rescreen_hits_parallel(
    *,
    ctx: SessionRescreenContext,
    hits: Sequence[HitSpec],
    all_hits: Sequence[HitSpec],
    sid: str,
    n_seeds: int,
    rescreen_base_seed: int,
    workers: int,
    rank: int,
    n_nodes: int,
    progress: bool,
    progress_callback: Optional[RescreenProgressCallback] = None,
) -> List[Dict[str, Any]]:
    """Re-screen hits with a process pool (same pattern as primary batch workers)."""
    hit_pool = effective_process_pool_workers(workers, len(hits))
    if len(hits) > 1 and int(n_seeds) > 1 and int(workers) > len(hits):
        return _rescreen_hits_flat_parallel(
            ctx=ctx,
            hits=hits,
            all_hits=all_hits,
            sid=sid,
            n_seeds=n_seeds,
            rescreen_base_seed=rescreen_base_seed,
            workers=workers,
            rank=rank,
            n_nodes=n_nodes,
            progress=progress,
            progress_callback=progress_callback,
        )
    pool_workers = hit_pool
    seed_workers = 1 if pool_workers > 1 else max(1, int(workers))
    if progress:
        print(
            f"[rescreen] rank {rank}/{n_nodes} process pool max_workers={pool_workers} "
            f"(workers={workers}, hits={len(hits)}, seed_workers={seed_workers}, "
            f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', '(unset)')})",
            flush=True,
        )
    payloads = [
        {
            "session_id": sid,
            "ctx": _ctx_to_payload(ctx),
            "hit": _hit_to_payload(hit),
            "n_seeds": int(n_seeds),
            "rescreen_base_seed": int(rescreen_base_seed),
            "seed_workers": seed_workers,
            "progress_label": f"rescreen_hit_{hit.hit_index}" if progress else "",
        }
        for hit in hits
    ]
    rows: List[Dict[str, Any]] = []
    trials_done = 0
    total_trials = _rescreen_trial_total(len(all_hits), n_seeds)
    with ProcessPoolExecutor(max_workers=pool_workers) as pool:
        futs = {pool.submit(_rescreen_hit_row_worker, p): p for p in payloads}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            hit = _hit_from_payload(futs[fut]["hit"])
            trials_done += int(n_seeds)
            if progress:
                print(
                    f"  hit {hit.hit_index + 1}/{len(all_hits)} "
                    f"(source run {hit.source_run_index}, rank {rank}/{n_nodes}): "
                    f"hit_rate={row.get('hit_rate')} …",
                    flush=True,
                )
            rate = row.get("hit_rate")
            rate_txt = f", re-hit rate {float(rate):.3f}" if rate is not None else ""
            _emit_rescreen_progress(
                progress_callback,
                trials_done,
                total_trials,
                _trial_progress_message(
                    trials_done,
                    total_trials,
                    point_number=int(hit.hit_index) + 1,
                    n_points=len(all_hits),
                    source_run_index=hit.source_run_index,
                )
                + rate_txt,
            )
    rows.sort(key=lambda r: int(r["hit_index"]))
    _validate_rescreen_rows_integrity(
        rows,
        hits,
        ctx,
        sid=sid,
        n_seeds=n_seeds,
        rescreen_base_seed=rescreen_base_seed,
    )
    return rows


def _default_output_dir(session_dir: str, *, screen_mode: str = SCREEN_MODE_HITS) -> str:
    return os.path.join(os.path.abspath(session_dir), out_subdir_for_screen_mode(screen_mode))


def _params_json_for_row(ctx: SessionRescreenContext, hit: HitSpec) -> str:
    decoded = decode_hit_params(ctx, hit)
    payload = {str(k): float(v) for k, v in sorted(decoded.items())}
    return json.dumps(payload, sort_keys=True)


def _validate_params_json_matches_hit(
    ctx: SessionRescreenContext,
    hit: HitSpec,
    params_json: str,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """Ensure stored ``params_json`` matches offload-decoded hit parameters."""
    decoded = decode_hit_params(ctx, hit)
    try:
        stored = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid params_json for hit {hit.hit_index}") from exc
    if not isinstance(stored, dict):
        raise ValueError(f"params_json for hit {hit.hit_index} must be a JSON object")
    for name, want in decoded.items():
        if name not in stored:
            raise ValueError(f"params_json missing {name!r} for hit {hit.hit_index}")
        got = float(stored[name])
        if not np.isclose(float(want), got, rtol=rtol, atol=atol):
            raise ValueError(
                f"params_json drift for hit {hit.hit_index} field {name!r}: "
                f"offload {float(want)!r} vs stored {got!r}"
            )


def _validate_trial_list_complete(
    trials: Sequence[Tuple[int, int, bool]],
    *,
    hit_index: int,
    n_seeds: int,
    rescreen_base_seed: int,
) -> None:
    """Every trial index 0..n_seeds-1 present once with expected deterministic seed."""
    n = max(1, int(n_seeds))
    if len(trials) != n:
        raise ValueError(
            f"hit {hit_index}: expected {n} trials, got {len(trials)}"
        )
    trial_nums = sorted(int(t[0]) for t in trials)
    if trial_nums != list(range(n)):
        raise ValueError(
            f"hit {hit_index}: trial indices {trial_nums} != expected {list(range(n))}"
        )
    ordered = sorted(trials, key=lambda t: int(t[0]))
    seeds = [int(t[1]) for t in ordered]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"hit {hit_index}: duplicate seeds in trial list {seeds}")
    for trial, seed in enumerate(seeds):
        expect = _seed_for_trial(int(rescreen_base_seed), int(hit_index), trial)
        if seed != expect:
            raise ValueError(
                f"hit {hit_index} trial {trial}: seed {seed} != expected {expect}"
            )


def _validate_rescreen_row(
    row: Dict[str, Any],
    *,
    n_seeds: int,
    hit_index: Optional[int] = None,
    rescreen_base_seed: Optional[int] = None,
) -> None:
    """CSV/JSON row internal consistency (trial strings vs counts)."""
    if row.get("export_hits_only"):
        return
    hi = int(hit_index if hit_index is not None else row.get("hit_index") or -1)
    n_again = int(row["n_hit_again"])
    rate = float(row["hit_rate"])
    trials = str(row.get("trial_hits") or "").split(";")
    seeds_raw = str(row.get("seeds_tested") or "").split(";")
    if len(trials) != int(n_seeds) or len(seeds_raw) != int(n_seeds):
        raise ValueError(
            f"trial/seeds length mismatch for hit {hi}: "
            f"{len(trials)} trials, {len(seeds_raw)} seeds, n_seeds={n_seeds}"
        )
    if sum(1 for t in trials if t == "1") != n_again:
        raise ValueError(f"n_hit_again {n_again} != count of trial_hits for hit {hi}")
    expect_rate = round(float(n_again) / float(n_seeds), 4) if n_seeds else 0.0
    if abs(rate - expect_rate) > 1e-4:
        raise ValueError(
            f"hit_rate {rate} != n_hit_again/n_seeds ({expect_rate}) for hit {hi}"
        )
    try:
        seeds = [int(s) for s in seeds_raw]
    except ValueError as exc:
        raise ValueError(f"non-integer seeds_tested for hit {hi}: {seeds_raw!r}") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds_tested for hit {hi}: {seeds_raw}")
    if rescreen_base_seed is not None:
        for trial, seed in enumerate(seeds):
            expect = _seed_for_trial(int(rescreen_base_seed), hi, trial)
            if seed != expect:
                raise ValueError(
                    f"stored row seed drift for hit {hi} trial {trial}: "
                    f"{seed} != expected {expect}"
                )


def _validate_rescreen_rows_integrity(
    rows: Sequence[Dict[str, Any]],
    hits: Sequence[HitSpec],
    ctx: SessionRescreenContext,
    *,
    sid: str,
    n_seeds: int,
    rescreen_base_seed: int,
) -> None:
    """Batch check: no missing/duplicate hits, params_json fidelity, row tallies."""
    if len(rows) != len(hits):
        raise ValueError(
            f"rescreen row count {len(rows)} != hit count {len(hits)} on this rank"
        )
    hit_by_index = {int(h.hit_index): h for h in hits}
    seen: set[int] = set()
    for row in rows:
        hi = int(row["hit_index"])
        if hi in seen:
            raise ValueError(f"duplicate hit_index {hi} in rescreen output")
        seen.add(hi)
        if hi not in hit_by_index:
            raise ValueError(f"unexpected hit_index {hi} in rescreen output")
        if str(row.get("session_id") or "") != str(sid):
            raise ValueError(
                f"session_id mismatch for hit {hi}: {row.get('session_id')!r} != {sid!r}"
            )
        if int(row.get("n_seeds") or 0) != int(n_seeds):
            raise ValueError(
                f"n_seeds field mismatch for hit {hi}: {row.get('n_seeds')!r} != {n_seeds}"
            )
        if str(row.get("param_key") or "") != str(hit_by_index[hi].param_key):
            raise ValueError(f"param_key drift for hit {hi}")
        _validate_params_json_matches_hit(
            ctx, hit_by_index[hi], str(row.get("params_json") or "")
        )
        _validate_rescreen_row(
            row,
            n_seeds=n_seeds,
            hit_index=hi,
            rescreen_base_seed=rescreen_base_seed,
        )
    expected = sorted(int(h.hit_index) for h in hits)
    if sorted(seen) != expected:
        raise ValueError(
            f"hit_index set mismatch: got {sorted(seen)}, expected {expected}"
        )


def _write_rescreen_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Atomically write rescreen JSON (plain ``.json`` only)."""
    write_json_maybe_gz_atomic(path, payload, indent=2, ensure_ascii=False)


def _write_csv_atomic(
    out_path: str,
    fieldnames: Sequence[str],
    rows: Sequence[Dict[str, Any]],
) -> str:
    """Write CSV via temp file + ``os.replace``."""
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    target_dir = os.path.dirname(out_path) or "."
    base = os.path.basename(out_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{base}.tmp.", dir=target_dir, text=True)
    os.close(fd)
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return out_path


def _validate_multinode_merge_completeness(
    all_rows: Sequence[Dict[str, Any]],
    template: Dict[str, Any],
    partials: Sequence[Dict[str, Any]],
) -> None:
    """Ensure every hit appears exactly once across rank partials."""
    n_hits_total = int(template.get("n_hits_total") or template.get("n_hits_screened") or 0)
    if n_hits_total <= 0:
        n_hits_total = len(all_rows)
    if len(all_rows) != n_hits_total:
        raise ValueError(
            f"multinode merge row count {len(all_rows)} != n_hits_total {n_hits_total}"
        )
    hit_indices = [int(r.get("hit_index")) for r in all_rows]
    if len(set(hit_indices)) != len(hit_indices):
        raise ValueError(
            f"multinode merge has duplicate hit_index values among {len(all_rows)} rows"
        )
    expected = list(range(n_hits_total))
    if sorted(hit_indices) != expected:
        raise ValueError(
            f"multinode merge hit_index set {sorted(hit_indices)} != expected {expected}"
        )
    rank_rows = sum(int(p.get("n_hits_this_rank") or len(p.get("rows") or [])) for p in partials)
    if rank_rows != n_hits_total:
        raise ValueError(
            f"multinode merge rank row sum {rank_rows} != n_hits_total {n_hits_total}"
        )


def write_rescreen_csv(
    out_path: str,
    rows: Sequence[Dict[str, Any]],
) -> str:
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not rows:
        fieldnames = [
            "session_id",
            "hit_index",
            "source_run_index",
            "n_seeds",
            "n_hit_again",
            "hit_rate",
            "total_deaths",
            "total_duplications",
            "total_mutations",
            "total_outflows",
            "original_seed",
            "param_key",
            "params_json",
        ]
    else:
        fieldnames = list(rows[0].keys())
    return _write_csv_atomic(out_path, fieldnames, rows)


def discover_rescreen_targets(
    path: str,
    *,
    scan_suite_subdirs: bool = True,
) -> List[Tuple[str, str]]:
    """Return sorted ``(session_dir, suite_tag)`` pairs under an output folder."""
    path = os.path.abspath(os.path.expanduser(path))
    if is_primary_batch_session_folder(path):
        return [(path, "")]
    pairs = discover_primary_session_dirs(path, scan_suite_subdirs=bool(scan_suite_subdirs))
    if not pairs:
        pairs = discover_primary_session_dirs(path, scan_suite_subdirs=False)
    return pairs


def batch_output_root(output_folder: str, *, screen_mode: str = SCREEN_MODE_HITS) -> str:
    return os.path.join(
        os.path.abspath(os.path.expanduser(output_folder)),
        out_subdir_for_screen_mode(screen_mode),
    )


def _partition_hits(
    hits: Sequence[HitSpec],
    *,
    multi_node_index: int,
    multi_node_count: int,
) -> List[HitSpec]:
    """Assign hits to one multi-node rank (``index``, ``index+count``, …)."""
    n_nodes = max(1, int(multi_node_count))
    rank = int(multi_node_index)
    if n_nodes <= 1:
        return list(hits)
    if rank < 0 or rank >= n_nodes:
        raise ValueError(f"multi_node_index {rank} out of range for count {n_nodes}")
    return [hit for i, hit in enumerate(hits) if i % n_nodes == rank]


def _multinode_partial_json_path(
    out_dir: str,
    session_id: str,
    rank: int,
    *,
    screen_mode: str = SCREEN_MODE_HITS,
) -> str:
    basename = rescreen_artifact_basename(session_id, screen_mode)
    return os.path.join(
        os.path.abspath(out_dir),
        f"{basename}.rank{int(rank)}.partial.json",
    )


def _maybe_update_batch_hit_counts_csv(
    summary: Dict[str, Any],
    *,
    screen_mode: str,
    progress: bool = False,
) -> None:
    # Skip export-only runs, multinode partial writes, and empty result sets.
    rows = summary.get("rows") or []
    if summary.get("export_hits_only") or not rows or summary.get("partial_json_path"):
        return
    session_dir = str(summary.get("session_dir") or "").strip()
    session_id = str(summary.get("session_id") or "").strip()
    if not session_dir:
        return
    try:
        hit_counts_csv = update_session_hit_counts_csv_from_rescreen(
            session_dir=session_dir,
            session_id=session_id,
            rescreen_rows=rows,
            screen_mode=screen_mode,
        )
        if hit_counts_csv:
            summary["hit_counts_csv"] = hit_counts_csv
            if progress:
                print(f"Updated batch hit-counts CSV: {hit_counts_csv}", flush=True)
    except Exception as exc:
        if progress:
            print(
                f"Note: could not update batch hit-counts CSV for {session_id}: {exc}",
                flush=True,
            )


def merge_multinode_rescreen_partials(
    out_dir: str,
    session_id: str,
    *,
    multi_node_count: int,
    write_rehit_heatmap: bool = True,
    screen_mode: str = SCREEN_MODE_HITS,
) -> Dict[str, Any]:
    """Merge per-rank partial JSON from a two-node rescreen into final CSV/JSON."""
    out_dir = os.path.abspath(out_dir)
    sid = str(session_id or "").strip()
    mode = normalize_screen_mode(screen_mode)
    n_nodes = max(1, int(multi_node_count))
    partials: List[Dict[str, Any]] = []
    for rank in range(n_nodes):
        path = _multinode_partial_json_path(out_dir, sid, rank, screen_mode=mode)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing multinode partial: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"invalid partial JSON: {path}")
        partials.append(data)

    template = dict(partials[0])
    screen_mode = str(template.get("screen_mode") or mode)
    sid = str(session_id or template.get("session_id") or "").strip()
    all_rows: List[Dict[str, Any]] = []
    for part in partials:
        rows = part.get("rows")
        if isinstance(rows, list):
            all_rows.extend(rows)
    all_rows.sort(key=lambda r: int(r.get("hit_index") or 0))

    n_seeds = int(template.get("n_seeds_per_hit") or DEFAULT_N_SEEDS)
    for row in all_rows:
        _validate_rescreen_row(row, n_seeds=n_seeds)
    _validate_multinode_merge_completeness(all_rows, template, partials)

    summary = {k: v for k, v in template.items() if k != "rows"}
    summary["n_hits_screened"] = int(template.get("n_hits_total") or len(all_rows))
    summary["multi_node_count"] = n_nodes
    summary.pop("multi_node_index", None)
    summary["workers"] = int(template.get("workers") or 1)
    summary["saved_at_epoch"] = int(time.time())

    if all_rows and not summary.get("export_hits_only"):
        rates = [float(r["hit_rate"]) for r in all_rows if r.get("hit_rate") is not None]
        summary["mean_hit_rate"] = round(float(np.mean(rates)), 4) if rates else None
        summary["median_hit_rate"] = round(float(np.median(rates)), 4) if rates else None
        summary["hits_always"] = sum(1 for r in rates if r >= 1.0)
        summary["hits_never"] = sum(1 for r in rates if r <= 0.0)

    artifact_base = rescreen_artifact_basename(sid, screen_mode)
    csv_path = os.path.join(out_dir, f"{artifact_base}.csv")
    json_path = os.path.join(out_dir, f"{artifact_base}.json")
    write_rescreen_csv(csv_path, all_rows)
    summary["csv_path"] = csv_path
    summary["rows"] = all_rows
    _write_rescreen_json_atomic(json_path, summary)
    summary["json_path"] = json_path

    _maybe_update_batch_hit_counts_csv(summary, screen_mode=screen_mode, progress=True)

    for rank in range(n_nodes):
        try:
            os.remove(_multinode_partial_json_path(out_dir, sid, rank, screen_mode=screen_mode))
        except OSError:
            pass
    hm = maybe_write_rescreen_rehit_heatmap_for_summary(
        summary,
        skip=not bool(write_rehit_heatmap),
    )
    if hm and hm.get("png_path"):
        print(f"Wrote: {hm.get('png_path')}", flush=True)
    elif hm and not hm.get("png_path"):
        print("Note: re-hit parameter heatmap not written (matplotlib unavailable).", flush=True)
    gt90_png = summary.get("rehit_rate_gt90_heatmap_png")
    if gt90_png:
        print(f"Wrote: {gt90_png}", flush=True)
    return summary


def load_session_summaries_from_batch_root(
    batch_root: str,
    *,
    screen_mode: str = SCREEN_MODE_HITS,
) -> List[Dict[str, Any]]:
    """Load completed per-session rescreen JSON under ``<batch_root>/sessions/``."""
    sessions_root = os.path.join(os.path.abspath(batch_root), BATCH_SESSIONS_SUBDIR)
    if not os.path.isdir(sessions_root):
        return []
    prefix = artifact_prefix_for_screen_mode(screen_mode)
    order = {k: i for i, k in enumerate(COMPARE_CONFIG_ORDER)}
    out: List[Dict[str, Any]] = []
    for sid_dir in sorted(os.listdir(sessions_root)):
        sess_path = os.path.join(sessions_root, sid_dir)
        if not os.path.isdir(sess_path):
            continue
        json_hits = sorted(
            f
            for f in os.listdir(sess_path)
            if f.startswith(f"{prefix}_")
            and f.endswith(".json")
            and ".rank" not in f
            and ".partial" not in f
        )
        if not json_hits:
            continue
        with open(os.path.join(sess_path, json_hits[0]), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and not data.get("export_hits_only"):
            out.append(data)
    out.sort(
        key=lambda s: (
            order.get(str(s.get("config_key") or ""), 999),
            str(s.get("suite_tag") or ""),
            str(s.get("session_id") or ""),
        )
    )
    return out


def write_batch_compare_from_disk(
    output_folder: str,
    *,
    batch_root: Optional[str] = None,
    n_seeds: int = DEFAULT_N_SEEDS,
    screen_mode: str = SCREEN_MODE_HITS,
) -> Dict[str, Any]:
    """Build compare CSV/JSON/category PNG from per-session rescreen outputs on disk."""
    folder = os.path.abspath(os.path.expanduser(output_folder))
    mode = normalize_screen_mode(screen_mode)
    root = os.path.abspath(batch_root or batch_output_root(folder, screen_mode=mode))
    summaries = load_session_summaries_from_batch_root(root, screen_mode=mode)
    if not summaries:
        raise RuntimeError(f"no completed session rescreen JSON under {root}/{BATCH_SESSIONS_SUBDIR}/")
    return write_batch_compare_artifacts(
        root,
        summaries,
        output_folder=folder,
        n_seeds=max(1, int(n_seeds)),
        screen_mode=mode,
    )


def aggregate_category_stats(
    session_summaries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pool re-hit trials across sessions, grouped by config category."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for summary in session_summaries:
        if summary.get("export_hits_only"):
            continue
        cfg = str(summary.get("config_key") or "unknown").strip() or "unknown"
        bucket = buckets.setdefault(
            cfg,
            {
                "config_key": cfg,
                "n_hit_again": 0,
                "n_trials": 0,
                "n_hits": 0,
                "n_sessions": 0,
            },
        )
        for row in summary.get("rows") or []:
            bucket["n_hit_again"] += int(row.get("n_hit_again") or 0)
            bucket["n_trials"] += int(row.get("n_seeds") or 0)
            bucket["n_hits"] += 1
        bucket["n_sessions"] += 1

    order = {k: i for i, k in enumerate(COMPARE_CONFIG_ORDER)}
    out: List[Dict[str, Any]] = []
    for cfg in sorted(buckets.keys(), key=lambda k: (order.get(k, 999), k)):
        bucket = buckets[cfg]
        n_trials = int(bucket["n_trials"])
        n_again = int(bucket["n_hit_again"])
        pct = 100.0 * float(n_again) / float(n_trials) if n_trials else 0.0
        out.append(
            {
                **bucket,
                "pct_rehit": round(pct, 2),
                "mean_hit_rate": round(float(n_again) / float(n_trials), 4) if n_trials else None,
            }
        )
    return out


def write_compare_csv(out_path: str, session_summaries: Sequence[Dict[str, Any]]) -> str:
    rows: List[Dict[str, Any]] = []
    for summary in session_summaries:
        rows.append(
            {
                "session_id": summary.get("session_id"),
                "plot_label": summary.get("plot_label"),
                "config_key": summary.get("config_key"),
                "suite_tag": summary.get("suite_tag"),
                "n_hits_screened": summary.get("n_hits_screened"),
                "n_seeds_per_hit": summary.get("n_seeds_per_hit"),
                "mean_hit_rate": summary.get("mean_hit_rate"),
                "median_hit_rate": summary.get("median_hit_rate"),
                "hits_always": summary.get("hits_always"),
                "hits_never": summary.get("hits_never"),
                "csv_path": summary.get("csv_path"),
                "json_path": summary.get("json_path"),
            }
        )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "session_id",
        "plot_label",
        "config_key",
        "suite_tag",
        "n_hits_screened",
        "n_seeds_per_hit",
        "mean_hit_rate",
        "median_hit_rate",
        "hits_always",
        "hits_never",
        "csv_path",
        "json_path",
    ]
    return _write_csv_atomic(out_path, fieldnames, rows)


def write_category_summary_plot(
    out_path: str,
    category_stats: Sequence[Dict[str, Any]],
    *,
    title: str = "Primary hit re-screen rate by category",
) -> Optional[str]:
    if not category_stats:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [str(s["config_key"]) for s in category_stats]
    pcts = [float(s["pct_rehit"]) for s in category_stats]
    n_again = [int(s["n_hit_again"]) for s in category_stats]
    trials = [int(s["n_trials"]) for s in category_stats]

    fig_w = max(6.0, 0.9 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, pcts, color="#4C78A8", edgecolor="#2F4B7C", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("% re-hits (trials that hit again)")
    ax.set_ylim(0, min(100.0, max(pcts) * 1.15 + 5.0) if pcts else 100.0)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, pct, again, n_trial in zip(bars, pcts, n_again, trials):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{pct:.1f}%\n({again}/{n_trial})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_batch_compare_artifacts(
    batch_root: str,
    session_summaries: Sequence[Dict[str, Any]],
    *,
    output_folder: str,
    n_seeds: int,
    screen_mode: str = SCREEN_MODE_HITS,
) -> Dict[str, Any]:
    batch_root = os.path.abspath(batch_root)
    os.makedirs(batch_root, exist_ok=True)
    mode = normalize_screen_mode(screen_mode)
    prefix = artifact_prefix_for_screen_mode(mode)
    category_stats = aggregate_category_stats(session_summaries)
    compare_csv = os.path.join(batch_root, f"{prefix}_compare.csv")
    compare_json = os.path.join(batch_root, f"{prefix}_compare.json")
    plot_path = os.path.join(batch_root, f"{prefix}_by_category.png")
    plot_title = (
        "Primary non-hit re-screen: % hit rate by category"
        if mode == SCREEN_MODE_NON_HITS
        else "Primary hit re-screen: % re-hits by category"
    )

    write_compare_csv(compare_csv, session_summaries)
    payload: Dict[str, Any] = {
        "kind": _COMPARE_KIND_BY_MODE[mode],
        "version": 1,
        "re_runner_version": RE_RUNNER_VERSION,
        "screen_mode": mode,
        "output_folder": os.path.abspath(output_folder),
        "batch_root": batch_root,
        "n_seeds_per_hit": int(n_seeds),
        "n_sessions": len(session_summaries),
        "sessions": [
            {
                "session_id": s.get("session_id"),
                "plot_label": s.get("plot_label"),
                "config_key": s.get("config_key"),
                "suite_tag": s.get("suite_tag"),
                "mean_hit_rate": s.get("mean_hit_rate"),
                "n_hits_screened": s.get("n_hits_screened"),
                "csv_path": s.get("csv_path"),
            }
            for s in session_summaries
        ],
        "by_category": category_stats,
        "saved_at_epoch": int(time.time()),
    }
    _write_rescreen_json_atomic(compare_json, payload)

    plot_written = write_category_summary_plot(
        plot_path,
        category_stats,
        title=plot_title,
    )
    return {
        "compare_csv": compare_csv,
        "compare_json": compare_json,
        "category_plot": plot_written,
        "by_category": category_stats,
    }


def maybe_write_rescreen_rehit_heatmap_for_summary(
    summary: Dict[str, Any],
    *,
    skip: bool = False,
    log_errors: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build re-hit parameter heatmap PNG/JSON when a rescreen session finishes."""
    if skip:
        return None
    if os.environ.get("SKIP_RESCREEN_REHIT_HEATMAP", "").strip().lower() in ("1", "true", "yes"):
        return None
    if summary.get("export_hits_only") or summary.get("partial_json_path"):
        return None
    csv_path = str(summary.get("csv_path") or "").strip()
    if not csv_path or not os.path.isfile(csv_path):
        return None
    try:
        from rescreen_rehit_parameter_heatmap import build_rescreen_rehit_heatmaps

        heatmaps = build_rescreen_rehit_heatmaps(
            csv_path,
            settings_path=(str(summary.get("settings_path") or "").strip() or None),
            primary_session_dir=(str(summary.get("session_dir") or "").strip() or None),
            run_headline=(
                str(summary.get("plot_label") or summary.get("session_id") or "").strip() or None
            ),
            log_errors=log_errors,
        )
    except Exception as exc:
        if log_errors:
            sid = summary.get("session_id") or "?"
            print(
                f"Note: re-hit parameter heatmap not written for {sid}: {exc}",
                file=sys.stderr,
            )
        return None
    hm = heatmaps.get("default")
    hm90 = heatmaps.get("high_rehit_rate")
    if hm:
        summary["rehit_heatmap_png"] = hm.get("png_path")
        summary["rehit_heatmap_json"] = hm.get("json_path")
    if hm90:
        summary["rehit_rate_gt90_heatmap_png"] = hm90.get("png_path")
        summary["rehit_rate_gt90_heatmap_json"] = hm90.get("json_path")
    json_path = str(summary.get("json_path") or "").strip()
    if json_path and (hm or hm90):
        _write_rescreen_json_atomic(json_path, summary)
    return hm


def rescreen_session(
    session_dir: str,
    *,
    session_id: Optional[str] = None,
    suite_tag: str = "",
    n_seeds: int = DEFAULT_N_SEEDS,
    rescreen_base_seed: int = DEFAULT_RESCREEN_BASE_SEED,
    max_hits: int = 0,
    dedupe_params: bool = True,
    screen_mode: str = SCREEN_MODE_HITS,
    workers: Optional[int] = None,
    output_dir: Optional[str] = None,
    batch_root: Optional[str] = None,
    export_hits_only: bool = False,
    progress: bool = True,
    progress_callback: Optional[RescreenProgressCallback] = None,
    multi_node_index: int = 0,
    multi_node_count: int = 1,
    write_rehit_heatmap: bool = True,
) -> Dict[str, Any]:
    mode = normalize_screen_mode(screen_mode)
    ctx = load_session_rescreen_context(session_dir, session_id, suite_tag=suite_tag)
    worker_count = max(1, int(workers if workers is not None else _default_rescreen_workers()))
    if mode == SCREEN_MODE_NON_HITS and max_hits <= 0 and progress:
        print(
            "[rescreen] WARNING: non-hits mode with --max-hits 0 will screen ALL unique "
            "primary misses (often 10³–10⁵+ points × n_seeds). Set RE_RUNNER_MAX_HITS "
            "or --max-hits to cap.",
            flush=True,
        )
    all_hits = collect_hit_specs(
        ctx.session_dir,
        ctx.session_id,
        max_hits=max_hits,
        dedupe_params=dedupe_params,
        screen_mode=mode,
    )
    n_nodes = max(1, int(multi_node_count))
    rank = int(multi_node_index)
    hits = _partition_hits(
        all_hits,
        multi_node_index=rank,
        multi_node_count=n_nodes,
    )
    mode_label = "hits" if mode == SCREEN_MODE_HITS else "non-hits"
    total_trials = _rescreen_trial_total(len(all_hits), n_seeds)
    _emit_rescreen_progress(
        progress_callback,
        0,
        total_trials,
        (
            f"Re-screening {len(all_hits)} {mode_label} "
            f"({int(n_seeds)} seeds per point, {total_trials} re-runs)…"
        ),
    )
    if batch_root:
        out_dir = os.path.join(
            os.path.abspath(batch_root),
            BATCH_SESSIONS_SUBDIR,
            ctx.session_id,
        )
    else:
        out_dir = os.path.abspath(output_dir or _default_output_dir(ctx.session_dir, screen_mode=mode))
    os.makedirs(out_dir, exist_ok=True)
    sid = ctx.session_id
    artifact_base = rescreen_artifact_basename(sid, mode)

    summary: Dict[str, Any] = {
        "kind": rescreen_kind_for_screen_mode(mode),
        "version": 1,
        "re_runner_version": RE_RUNNER_VERSION,
        "screen_mode": mode,
        "session_id": sid,
        "session_dir": ctx.session_dir,
        "plot_label": ctx.plot_label,
        "config_key": ctx.config_key,
        "suite_tag": ctx.suite_tag,
        "settings_path": ctx.settings_path,
        "param_names_list": list(ctx.param_names_list),
        "metric_checks": [
            {"metric": m, "operator": op, "threshold": thr} for m, op, thr in ctx.metric_checks
        ],
        "primary_toggles": dict(ctx.toggles),
        "simulation_light_tracking": bool(ctx.sim_light_used),
        "n_seeds_per_hit": int(n_seeds),
        "rescreen_base_seed": int(rescreen_base_seed),
        "dedupe_params": bool(dedupe_params),
        "max_hits": int(max_hits),
        "n_hits_screened": len(all_hits),
        "n_hits_total": len(all_hits),
        "n_hits_this_rank": len(hits),
        "multi_node_index": rank if n_nodes > 1 else None,
        "multi_node_count": n_nodes if n_nodes > 1 else None,
        "export_hits_only": bool(export_hits_only),
        "workers": int(worker_count),
        "saved_at_epoch": int(time.time()),
    }

    csv_rows: List[Dict[str, Any]] = []
    if export_hits_only:
        for hit in hits:
            row: Dict[str, Any] = {
                "session_id": sid,
                "hit_index": hit.hit_index,
                "source_run_index": hit.source_run_index,
                "original_seed": hit.original_seed if hit.original_seed is not None else "",
                "param_key": hit.param_key,
                "params_json": _params_json_for_row(ctx, hit),
                "n_seeds": int(n_seeds),
                **_event_total_row_columns(hit),
            }
            decoded = decode_hit_params(ctx, hit)
            for name in ctx.param_names_list:
                if name in decoded:
                    row[f"param_{name}"] = decoded[name]
            csv_rows.append(row)
    elif int(worker_count) > 1 and len(hits) > 1:
        csv_rows = _rescreen_hits_parallel(
            ctx=ctx,
            hits=hits,
            all_hits=all_hits,
            sid=sid,
            n_seeds=n_seeds,
            rescreen_base_seed=rescreen_base_seed,
            workers=worker_count,
            rank=rank,
            n_nodes=n_nodes,
            progress=progress,
            progress_callback=progress_callback,
        )
    else:
        seed_workers = max(1, int(worker_count))
        trials_done = 0

        def _on_trial_complete(trial_hit: HitSpec, _trial: int) -> None:
            nonlocal trials_done
            trials_done += 1
            _emit_rescreen_progress(
                progress_callback,
                trials_done,
                total_trials,
                _trial_progress_message(
                    trials_done,
                    total_trials,
                    point_number=int(trial_hit.hit_index) + 1,
                    n_points=len(all_hits),
                    source_run_index=trial_hit.source_run_index,
                ),
            )

        for hit in hits:
            if progress:
                print(
                    f"  hit {hit.hit_index + 1}/{len(all_hits)} "
                    f"(source run {hit.source_run_index}, rank {rank}/{n_nodes}): {n_seeds} seeds …",
                    flush=True,
                )
            row = _build_rescreen_row(
                ctx,
                hit,
                sid=sid,
                n_seeds=n_seeds,
                rescreen_base_seed=rescreen_base_seed,
                seed_workers=seed_workers,
                on_trial_complete=_on_trial_complete,
            )
            csv_rows.append(row)
            rate = row.get("hit_rate")
            if rate is not None:
                _emit_rescreen_progress(
                    progress_callback,
                    trials_done,
                    total_trials,
                    _trial_progress_message(
                        trials_done,
                        total_trials,
                        point_number=int(hit.hit_index) + 1,
                        n_points=len(all_hits),
                        source_run_index=hit.source_run_index,
                    )
                    + f", re-hit rate {float(rate):.3f}",
                )

    if not export_hits_only and csv_rows:
        _validate_rescreen_rows_integrity(
            csv_rows,
            hits,
            ctx,
            sid=sid,
            n_seeds=n_seeds,
            rescreen_base_seed=rescreen_base_seed,
        )

    if n_nodes > 1 and not export_hits_only:
        partial_path = _multinode_partial_json_path(out_dir, sid, rank, screen_mode=mode)
        summary["rows"] = csv_rows
        _write_rescreen_json_atomic(partial_path, summary)
        summary["partial_json_path"] = partial_path
        return summary

    csv_path = os.path.join(out_dir, f"{artifact_base}.csv")
    json_path = os.path.join(out_dir, f"{artifact_base}.json")
    _emit_rescreen_progress(
        progress_callback,
        total_trials,
        total_trials,
        f"Writing {mode_label} results…",
    )
    write_rescreen_csv(csv_path, csv_rows)

    if not export_hits_only and csv_rows:
        rates = [float(r["hit_rate"]) for r in csv_rows if r.get("hit_rate") is not None]
        summary["mean_hit_rate"] = round(float(np.mean(rates)), 4) if rates else None
        summary["median_hit_rate"] = round(float(np.median(rates)), 4) if rates else None
        summary["hits_always"] = sum(1 for r in rates if r >= 1.0)
        summary["hits_never"] = sum(1 for r in rates if r <= 0.0)

    summary["csv_path"] = csv_path
    summary["rows"] = csv_rows
    _write_rescreen_json_atomic(json_path, summary)
    summary["json_path"] = json_path
    _maybe_update_batch_hit_counts_csv(
        summary,
        screen_mode=mode,
        progress=bool(progress),
    )
    hm = maybe_write_rescreen_rehit_heatmap_for_summary(
        summary,
        skip=not bool(write_rehit_heatmap),
    )
    if progress and hm and hm.get("png_path"):
        print(f"Wrote: {hm.get('png_path')}", flush=True)
    elif progress and hm and not hm.get("png_path"):
        print("Note: re-hit parameter heatmap not written (matplotlib unavailable).", flush=True)
    gt90_png = summary.get("rehit_rate_gt90_heatmap_png")
    if progress and gt90_png:
        print(f"Wrote: {gt90_png}", flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Re-screen primary batch hits with new random seeds (Re-Runner)."
    )
    p.add_argument(
        "path",
        help="Session folder, or output/suite root (discovers all campaign sessions).",
    )
    p.add_argument(
        "--all-campaigns",
        action="store_true",
        help="Scan output/<suite>/<session>/ (default when PATH is not a single session).",
    )
    p.add_argument(
        "--output-root",
        action="store_true",
        help="Only immediate children of PATH (no suite subfolders).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Re-runs per hit with distinct seeds (default {DEFAULT_N_SEEDS}).",
    )
    p.add_argument(
        "--rescreen-base-seed",
        type=int,
        default=DEFAULT_RESCREEN_BASE_SEED,
        help="Base for deterministic per-hit/trial seeds.",
    )
    p.add_argument(
        "--max-hits",
        type=int,
        default=0,
        help="Cap screened points per session (0 = all unique hits or non-hits).",
    )
    p.add_argument(
        "--non-hits",
        action="store_true",
        help=(
            "Re-screen primary *misses* instead of hits. Writes under Re-Runs-NonHits/. "
            "Without --max-hits this can be very large."
        ),
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate parameter vectors as separate hits.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=_default_rescreen_workers(),
        help=(
            "Parallel worker processes per rank (default: SLURM_CPUS_PER_TASK or local CPU count). "
            "Uses one pool over hits when many hits remain; otherwise one pool over all "
            "(hit, seed) trials to keep CPUs busy."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-point progress and worker logs on the terminal.",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help=f"Override output directory (default <session>/{DEFAULT_OUT_SUBDIR}/).",
    )
    p.add_argument(
        "--batch-root",
        default="",
        help=f"Central batch output root (default <output-folder>/{DEFAULT_OUT_SUBDIR}/).",
    )
    p.add_argument(
        "--export-hits-only",
        action="store_true",
        help="Write hit parameter manifest CSV/JSON only; do not re-simulate.",
    )
    p.add_argument(
        "--skip-batch-compare",
        action="store_true",
        help="Do not write batch compare CSV/JSON/category PNG after sessions.",
    )
    p.add_argument(
        "--skip-rehit-heatmap",
        action="store_true",
        help="Do not write re-hit parameter heatmap PNG/JSON after each session finishes.",
    )
    p.add_argument(
        "--batch-compare-only",
        action="store_true",
        help="Load completed session JSON under Re-Runs/sessions/ and write compare artifacts only.",
    )
    p.add_argument(
        "--backfill-event-totals",
        action="store_true",
        help=(
            "Add primary-run event-total columns to existing rescreen CSV/JSON from offload "
            "(no re-simulation). PATH may be Re-Runs root, sessions/<id>/, or one rescreen JSON."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "With --backfill-event-totals, list rescreen row counts only; do not load "
            "primary offload or rewrite files. For filled/missing totals on one session, "
            "run without --dry-run on that session only."
        ),
    )
    p.add_argument(
        "--merge-multinode",
        action="store_true",
        help="Merge rank partial JSON into final per-session CSV/JSON (after srun).",
    )
    p.add_argument(
        "--multi-node-index",
        type=int,
        default=0,
        help="Multi-node rank index for hit partitioning (default 0).",
    )
    p.add_argument(
        "--multi-node-count",
        type=int,
        default=1,
        help="Number of ranks that split hits (default 1 = single node).",
    )
    p.add_argument(
        "--session-id",
        default="",
        help="Session id for --merge-multinode (default: infer from PATH).",
    )
    p.add_argument(
        "--suite-tag",
        default="",
        help="Suite folder tag for plot labels (e.g. Fixed_10_ratio).",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    path = os.path.abspath(os.path.expanduser(args.path))
    n_seeds = max(1, int(args.n_seeds))
    n_nodes = max(1, int(args.multi_node_count))
    suite_tag_arg = str(args.suite_tag or "").strip()
    screen_mode = SCREEN_MODE_NON_HITS if bool(args.non_hits) else screen_mode_from_env()

    if args.batch_compare_only:
        batch_root_arg = (args.batch_root or "").strip() or None
        try:
            artifacts = write_batch_compare_from_disk(
                path,
                batch_root=batch_root_arg,
                n_seeds=n_seeds,
                screen_mode=screen_mode,
            )
        except Exception as exc:
            print(f"FAILED batch compare: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote: {artifacts.get('compare_csv')}")
        print(f"Wrote: {artifacts.get('compare_json')}")
        if artifacts.get("category_plot"):
            print(f"Wrote: {artifacts.get('category_plot')}")
        return 0

    if args.backfill_event_totals:
        try:
            summary = backfill_rescreen_event_totals_batch(
                path,
                screen_mode=screen_mode,
                dry_run=bool(args.dry_run),
            )
        except Exception as exc:
            print(f"FAILED backfill event totals: {exc}", file=sys.stderr)
            return 1
        print(
            f"[backfill_event_totals] done: {summary.get('ok')} ok, {summary.get('fail')} failed"
            + (" (dry-run)" if summary.get("dry_run") else "")
        )
        return 0 if int(summary.get("fail") or 0) == 0 else 1

    single_session = is_primary_batch_session_folder(path)
    if args.merge_multinode:
        if not single_session:
            print("--merge-multinode requires a single session folder PATH", file=sys.stderr)
            return 1
        sid = (args.session_id or "").strip() or infer_session_id(path) or os.path.basename(path)
        batch_root_arg = (args.batch_root or "").strip()
        if batch_root_arg:
            root = os.path.abspath(batch_root_arg)
        else:
            root = batch_output_root(os.path.dirname(path), screen_mode=screen_mode)
        out_dir = os.path.join(root, BATCH_SESSIONS_SUBDIR, sid)
        try:
            summary = merge_multinode_rescreen_partials(
                out_dir,
                sid,
                multi_node_count=n_nodes,
                write_rehit_heatmap=not bool(args.skip_rehit_heatmap),
                screen_mode=screen_mode,
            )
        except Exception as exc:
            print(f"FAILED merge multinode partials for {sid}: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote: {summary.get('csv_path')}")
        print(f"Wrote: {summary.get('json_path')}")
        print(
            f"  hits screened: {summary.get('n_hits_screened')} | "
            f"mean hit rate: {summary.get('mean_hit_rate')} | "
            f"category: {summary.get('config_key')}"
        )
        return 0

    scan_subdirs = bool(args.all_campaigns) or (not single_session and not args.output_root)
    targets = discover_rescreen_targets(path, scan_suite_subdirs=scan_subdirs)
    if not targets:
        print(f"No primary batch session folders found under {args.path!r}", file=sys.stderr)
        return 1

    batch_mode = not single_session
    batch_root = (args.batch_root or "").strip() or None
    if batch_root:
        batch_root = os.path.abspath(batch_root)
    elif batch_mode:
        batch_root = batch_output_root(path, screen_mode=screen_mode)
    override_out = (args.output_dir or "").strip() or None
    if override_out and batch_mode and not batch_root:
        batch_root = os.path.abspath(override_out)

    rc = 0
    session_summaries: List[Dict[str, Any]] = []
    failed_sessions: List[str] = []
    for session_dir, suite_tag in targets:
        sid = infer_session_id(session_dir) or os.path.basename(session_dir)
        tag = str(suite_tag or suite_tag_arg or "").strip()
        if not tag:
            tag = _infer_suite_tag_from_session_dir(session_dir)
        label = f"{sid} ({session_dir})"
        if tag:
            label = f"{sid} [{tag}] ({session_dir})"
        print(f"=== Re-Runner ({screen_mode}): {label} ===", flush=True)
        try:
            summary = rescreen_session(
                session_dir,
                suite_tag=tag,
                n_seeds=n_seeds,
                rescreen_base_seed=int(args.rescreen_base_seed),
                max_hits=max(0, int(args.max_hits)),
                dedupe_params=not bool(args.no_dedupe),
                screen_mode=screen_mode,
                workers=max(1, int(args.workers)),
                output_dir=override_out if not batch_mode else None,
                batch_root=batch_root,
                export_hits_only=bool(args.export_hits_only),
                progress=not bool(args.quiet),
                multi_node_index=int(args.multi_node_index),
                multi_node_count=n_nodes,
                write_rehit_heatmap=not bool(args.skip_rehit_heatmap),
            )
        except Exception as exc:
            print(f"FAILED {sid}: {exc}", file=sys.stderr)
            failed_sessions.append(str(sid))
            rc = 1
            continue
        session_summaries.append(summary)
        if summary.get("partial_json_path"):
            print(f"Wrote: {summary.get('partial_json_path')}")
        else:
            print(f"Wrote: {summary.get('csv_path')}")
            print(f"Wrote: {summary.get('json_path')}")
        if not args.export_hits_only and not summary.get("partial_json_path"):
            print(
                f"  hits screened: {summary.get('n_hits_screened')} | "
                f"mean hit rate: {summary.get('mean_hit_rate')} | "
                f"category: {summary.get('config_key')}"
            )

    if (
        batch_mode
        and session_summaries
        and not failed_sessions
        and len(session_summaries) == len(targets)
        and not args.export_hits_only
        and not args.skip_batch_compare
        and n_nodes <= 1
    ):
        try:
            artifacts = write_batch_compare_artifacts(
                batch_root or batch_output_root(path, screen_mode=screen_mode),
                session_summaries,
                output_folder=path,
                n_seeds=n_seeds,
                screen_mode=screen_mode,
            )
            print(f"Wrote: {artifacts.get('compare_csv')}")
            print(f"Wrote: {artifacts.get('compare_json')}")
            if artifacts.get("category_plot"):
                print(f"Wrote: {artifacts.get('category_plot')}")
            else:
                print("Note: category summary PNG not written (matplotlib unavailable).")
            for row in artifacts.get("by_category") or []:
                print(
                    f"  {row.get('config_key')}: {row.get('pct_rehit')}% re-hit "
                    f"({row.get('n_hit_again')}/{row.get('n_trials')} trials, "
                    f"{row.get('n_sessions')} session(s))"
                )
        except Exception as exc:
            print(f"FAILED batch compare artifacts: {exc}", file=sys.stderr)
            rc = 1
    elif (
        batch_mode
        and not args.skip_batch_compare
        and not args.export_hits_only
        and n_nodes <= 1
        and (failed_sessions or len(session_summaries) < len(targets))
    ):
        print(
            "Skipping batch compare: "
            f"{len(failed_sessions)} session(s) failed, "
            f"{len(session_summaries)}/{len(targets)} succeeded "
            f"({', '.join(failed_sessions) if failed_sessions else 'partial success'})",
            file=sys.stderr,
        )
        rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
