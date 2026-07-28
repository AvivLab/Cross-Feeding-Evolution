#!/usr/bin/env python3
"""
Headless Batch Runner — Monte Carlo primary-batch campaigns from saved job JSON.

Load a job description with ``kind: primary_batch_campaign`` (the same JSON shape
written by **Save JSON Settings…** in the Batch Runner GUI), then write a session
folder with offload batches, hit-count CSV, and ``primary_batch_campaign_<session>.json``.

Typical workflow: configure once in the GUI, save settings JSON, then run from a
terminal in this folder::

    python headless/primary_batch_campaign.py batch_settings.json --output-dir OUTPUT_FOLDER

Or::

    python -m headless.primary_batch_campaign batch_settings.json --output-dir OUTPUT_FOLDER
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Allow ``python headless/primary_batch_campaign.py …`` from the bundle root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gui.apps.batch_runner.csv_output import write_session_hit_counts_csv
from gui.apps.neutral_comparison.batch import run_hit_count_batch, simulation_light_tracking_plan
from gui.apps.neutral_comparison.offload import NeutralComparisonOffloadWriter
from gui.apps.neutral_comparison.primary_event_chart import load_primary_offload_records
from gui.common.simulation_settings import normalize_simulation_params
from gui.metrics import filter_metric_options_for_simulation_settings
from gui.models.registry import get_model_by_key
from gui.persistence.full_save import full_save_settings_path_json
from gui.persistence.json_io import make_read_json_maybe_gz_fn, make_write_json_maybe_gz_atomic_fn
from headless.neutral_comparison import (
    _bounds_from_save_json,
    _bounds_jsonable,
    _metric_checks_from_filters,
)

PRIMARY_BATCH_HEADLESS_VERSION = "1.0.0"

_write_json = make_write_json_maybe_gz_atomic_fn(indent=2)
_read_json = make_read_json_maybe_gz_fn(plain_twin_fallback=False)


def _primary_batch_seed(base_seed: Optional[int], batch_index: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + 19 + int(batch_index) * 10_007


def validate_primary_metrics(
    metric_checks: List[Tuple[str, str, float]],
    num_p: Dict[str, Any],
    tog_p: Dict[str, Any],
) -> None:
    spec = get_model_by_key("simulation")
    merged = normalize_simulation_params({**num_p, **tog_p})
    m1_any = bool(
        merged.get("Enable M1 Diffusion")
        or merged.get("Enable M1 Facilitated Diffusion")
        or merged.get("Enable M1 Porin Diffusion")
    )
    allowed = set(
        filter_metric_options_for_simulation_settings(
            list(spec.metric_names),
            enable_m1_diffusion=m1_any,
            enable_m2_diffusion=bool(merged.get("Enable M2 Diffusion")),
            enable_intermediate_costs=bool(merged.get("Enable Intermediate Costs")),
        )
    )
    if bool(merged.get("Independent Traits")):
        allowed -= {
            "Trait Std Dev (Coupled)",
            "Trait Std Dev (Neutral Perc.)",
            "Trait Entropy (Neutral Perc.)",
        }
    for mname, _op, _thr in metric_checks:
        if mname not in allowed:
            raise ValueError(f"Metric {mname!r} is not valid for the current simulation settings.")


def run_from_primary_batch_campaign_json(
    data: Dict[str, Any],
    *,
    output_dir: str,
    session_id: Optional[str] = None,
    progress_every: int = 0,
    job_settings_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a primary batch campaign like the Batch Runner GUI worker."""
    kind = str(data.get("kind", "") or "")
    if kind != "primary_batch_campaign":
        raise ValueError(
            f"Expected kind 'primary_batch_campaign' (Batch Runner job JSON), got {kind!r}."
        )

    folder = os.path.abspath(os.path.expanduser(str(output_dir)))
    os.makedirs(folder, exist_ok=True)
    if session_id is None:
        sid = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        sid = str(session_id).strip()
        if not sid:
            raise ValueError("session_id is required")

    n_runs = int(data["n_runs"])
    n_primary = int(data["n_primary_batches"])
    if n_runs < 1 or n_primary < 1:
        raise ValueError("n_runs and n_primary_batches must be >= 1")

    base_seed = data.get("base_seed")
    if base_seed is not None:
        base_seed = int(base_seed)

    spec = get_model_by_key("simulation")
    primary_bounds = _bounds_from_save_json(data.get("primary_bounds"), label="primary_bounds")
    num_p = dict(data.get("primary_numeric_parameters") or {})
    tog_p = dict(data.get("primary_toggles") or {})
    metric_checks, metric_filters_snapshot = _metric_checks_from_filters(data.get("metric_filters"))
    validate_primary_metrics(metric_checks, num_p, tog_p)

    sim_light_used, sim_light_canon = simulation_light_tracking_plan(spec, metric_checks)
    param_names_list = list(spec.default_params.keys())

    settings_snapshot = {
        "kind": "primary_batch_campaign_session",
        "version": 1,
        "session_id": sid,
        "model": data.get("model") or {"key": "simulation"},
        "n_runs": n_runs,
        "n_primary_batches": n_primary,
        "base_seed": base_seed,
        "configuration_name": str(data.get("configuration_name") or ""),
        "metric_filters": list(metric_filters_snapshot),
        "primary_bounds": _bounds_jsonable(primary_bounds),
        "primary_numeric_parameters": num_p,
        "primary_toggles": tog_p,
        "primary_offload_param_names": list(param_names_list),
        "simulation_light_tracking": bool(sim_light_used),
        "simulation_light_tracking_metrics": list(sim_light_canon),
        "full_save_folder": folder,
        "saved_at_epoch": int(time.time()),
    }
    _write_json(full_save_settings_path_json(folder, sid), settings_snapshot)

    writer = NeutralComparisonOffloadWriter(
        folder,
        sid,
        model_key=str(spec.key),
        param_names_list=param_names_list,
        metric_name_at_offload=str(metric_checks[0][0]),
        job_settings_json_path=job_settings_json_path,
    )

    batch_hit_counts: List[int] = []
    batch_seeds: List[Optional[int]] = []
    pe = max(0, int(progress_every))

    def progress_cb(batch_index: int, sim_done: int, hits: int, phase: str) -> None:
        if pe <= 0 or sim_done % pe != 0:
            return
        print(
            f"[batch {batch_index + 1}/{n_primary}] {phase}: simulation {sim_done}/{n_runs} "
            f"(hits so far: {hits})",
            flush=True,
        )

    try:
        for r in range(n_primary):
            local_seed = _primary_batch_seed(base_seed, r)
            batch_seeds.append(local_seed)
            if pe > 0:
                print(
                    f"Starting primary batch {r + 1}/{n_primary} ({n_runs} runs)…",
                    flush=True,
                )

            def _cb(done: int, total: int, hits: int, *, _r: int = r) -> None:
                progress_cb(_r, done, hits, "primary")

            hc = run_hit_count_batch(
                model_spec=spec,
                n_runs=n_runs,
                bounds=primary_bounds,
                numeric_base=num_p,
                toggles=tog_p,
                metric_checks=metric_checks,
                base_seed=local_seed,
                progress_callback=_cb if pe > 0 else None,
                offload_writer=writer,
                offload_stage="primary",
            )
            batch_hit_counts.append(int(hc))
            if pe > 0:
                print(
                    f"Finished primary batch {r + 1}/{n_primary}: {int(hc)} hits.",
                    flush=True,
                )
    finally:
        writer.finalize()

    records = load_primary_offload_records(folder, sid, _read_json)
    csv_path = write_session_hit_counts_csv(
        folder=folder,
        session_id=sid,
        hit_counts=batch_hit_counts,
        primary_numeric=num_p,
        primary_toggles=tog_p,
        primary_bounds=_bounds_jsonable(primary_bounds),
        configuration=str(data.get("configuration_name") or ""),
        session_dir=folder,
        param_names_list=param_names_list,
        offload_records=records,
        n_runs=n_runs,
    )
    summary_path = os.path.join(folder, f"primary_batch_campaign_{sid}.json")
    summary: Dict[str, Any] = {
        "kind": "primary_batch_campaign",
        "version": 1,
        "full_save_session_id": sid,
        "save_folder": folder,
        "n_runs": n_runs,
        "n_primary_batches": n_primary,
        "base_seed": base_seed,
        "configuration_name": str(data.get("configuration_name") or ""),
        "metric_filters": list(metric_filters_snapshot),
        "metric_filters_logical": "AND",
        "simulation_light_tracking": bool(sim_light_used),
        "simulation_light_tracking_metrics": list(sim_light_canon),
        "primary_hit_counts": [int(x) for x in batch_hit_counts],
        "primary_bounds": _bounds_jsonable(primary_bounds),
        "primary_numeric_parameters": num_p,
        "primary_toggles": tog_p,
        "offload_batches_written": len(writer.manifest.get("batches", [])),
    }
    if csv_path:
        summary["hit_counts_csv"] = os.path.basename(csv_path)
    _write_json(summary_path, summary)

    return {
        "session_id": sid,
        "folder": folder,
        "summary_path": summary_path,
        "csv_path": csv_path,
        "primary_hit_counts": batch_hit_counts,
        "batch_seeds": batch_seeds,
        "n_runs": n_runs,
        "n_primary_batches": n_primary,
        "offload_batches_written": len(writer.manifest.get("batches", [])),
        "summary": summary,
    }


def main(argv: Optional[List[str]] = None) -> int:
    print(
        f"[headless batch runner] version={PRIMARY_BATCH_HEADLESS_VERSION} "
        f"runner={os.path.abspath(__file__)}",
        flush=True,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PRIMARY_BATCH_HEADLESS_VERSION}",
    )
    p.add_argument(
        "config",
        help="Path to primary_batch_campaign JSON (from Batch Runner → Save JSON Settings…).",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Folder for the session (default: save_folder from the JSON, or the current directory).",
    )
    p.add_argument(
        "--session-id",
        default="",
        help="Session name for output files (default: timestamp YYYYMMDD-HHMMSS).",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        metavar="N",
        help="Print progress every N finished simulations per batch (0 disables).",
    )
    args = p.parse_args(argv)

    config_path = (args.config or "").replace("\r", "").replace("\n", "").strip()
    if not config_path:
        print("Error: config path is empty.", file=sys.stderr)
        return 1

    try:
        raw = _read_json(config_path)
    except Exception as exc:
        print(f"Error reading config: {exc}", file=sys.stderr)
        return 1
    if not isinstance(raw, dict):
        print("Config must be a JSON object.", file=sys.stderr)
        return 1

    explicit_dir = (args.output_dir or "").strip()
    if explicit_dir:
        out_dir = os.path.abspath(os.path.expanduser(explicit_dir))
    else:
        out_dir = str(raw.get("save_folder") or "").strip() or os.getcwd()
        out_dir = os.path.abspath(os.path.expanduser(out_dir))

    sid = (args.session_id or "").strip() or None
    print(
        f"[headless batch runner] starting: config={config_path!r} output_dir={out_dir!r} "
        f"session_id={sid or '(timestamp)'}",
        flush=True,
    )

    try:
        result = run_from_primary_batch_campaign_json(
            raw,
            output_dir=out_dir,
            session_id=sid,
            progress_every=max(0, int(args.progress_every)),
            job_settings_json_path=os.path.abspath(config_path),
        )
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Finished session {result['session_id']!r} in {result['folder']!r} "
        f"({result['n_primary_batches']} batches × {result['n_runs']} runs; "
        f"hit counts: {result['primary_hit_counts']}).",
        flush=True,
    )
    print(f"Summary: {result['summary_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
