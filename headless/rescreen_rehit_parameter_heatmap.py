#!/usr/bin/env python3
"""
Parameter heatmaps for Re-Runner re-hits (same 1D bin layout as primary batch heatmaps).

Reads ``primary_hit_rescreen_<session>.csv`` (+ optional JSON metadata), filters rows
that re-met metric filters on re-screen, and writes:

  ``rescreen_rehit_param_heatmap_<session_id>.png``
  ``rescreen_rehit_param_heatmap_<session_id>.json``
  ``rescreen_rehit_rate_gt90_param_heatmap_<session_id>.png``  (hit_rate > 0.9 only)
  ``rescreen_rehit_rate_gt90_param_heatmap_<session_id>.json``

Examples:
  python Re-Runner/rescreen_rehit_parameter_heatmap.py \\
    /path/to/Output/Re-Runs/sessions/d_justDup_Fixed_0.1

  python Re-Runner/rescreen_rehit_parameter_heatmap.py \\
    /path/to/Output/Re-Runs

  python Re-Runner/rescreen_rehit_parameter_heatmap.py \\
    /path/to/Output/Re-Runs/sessions/d_justDup_Fixed_0.1/primary_hit_rescreen_d_justDup_Fixed_0.1.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as ``python headless/rescreen_rehit_parameter_heatmap.py …``.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gui.apps.neutral_comparison.parameter_heatmap import (
    draw_parameter_heatmaps_on_figure,
    pop_last_parameter_heatmap_png_error,
    resolve_heatmap_param_names,
)
from gui.persistence.json_io import make_read_json_maybe_gz_fn

RESCREEN_REHIT_HEATMAP_VERSION = "1.1.0"
HIGH_REHIT_RATE_THRESHOLD = 0.9
DEFAULT_BATCH_SUBDIR = "Re-Runs"
DEFAULT_SESSIONS_SUBDIR = "sessions"
CSV_GLOB = "primary_hit_rescreen_*.csv"
PNG_BASENAME = "rescreen_rehit_param_heatmap_{session_id}.png"
JSON_BASENAME = "rescreen_rehit_param_heatmap_{session_id}.json"
HIGH_REHIT_PNG_BASENAME = "rescreen_rehit_rate_gt90_param_heatmap_{session_id}.png"
HIGH_REHIT_JSON_BASENAME = "rescreen_rehit_rate_gt90_param_heatmap_{session_id}.json"

HEATMAP_VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {
        "png_basename": PNG_BASENAME,
        "json_basename": JSON_BASENAME,
        "kind": "rescreen_rehit_parameter_heatmap",
        "title": "Re-screen re-hits: parameter heatmaps",
        "count_note_tpl": "({n_rehit} / {n_screened} primary hits re-met filters on re-screen)",
        "colorbar_label": "Count of re-screen re-hits",
        "empty_note": "No re-screen re-hits — empty heatmaps",
        "min_hit_rate": 0.0,
        "min_n_hit_again": 1,
        "strict_hit_rate": False,
        "note": (
            "1D bin counts of primary hits that re-met all active metric filters on "
            "Re-Runner re-screen (same layout as primary_param_heatmap)."
        ),
    },
    "high_rehit_rate": {
        "png_basename": HIGH_REHIT_PNG_BASENAME,
        "json_basename": HIGH_REHIT_JSON_BASENAME,
        "kind": "rescreen_rehit_rate_gt90_parameter_heatmap",
        "title": "Re-screen re-hits (>90% seeds): parameter heatmaps",
        "count_note_tpl": (
            "({n_rehit} / {n_screened} primary hits with >90% re-hit rate)"
        ),
        "colorbar_label": "Count of high-confidence re-hits (>90%)",
        "empty_note": "No primary hits with >90% re-hit rate — empty heatmaps",
        "min_hit_rate": HIGH_REHIT_RATE_THRESHOLD,
        "min_n_hit_again": 0,
        "strict_hit_rate": True,
        "note": (
            "1D bin counts of primary hits where strictly more than 90% of re-screen "
            "seeds re-met all active metric filters (same layout as primary_param_heatmap)."
        ),
    },
}

_read_json_maybe_gz = make_read_json_maybe_gz_fn()


def _session_id_from_csv_path(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    prefix = "primary_hit_rescreen_"
    if base.startswith(prefix) and base.endswith(".csv"):
        return base[len(prefix) : -4]
    return Path(csv_path).parent.name


def _rescreen_json_path_for_csv(csv_path: str, session_id: str) -> str:
    folder = os.path.dirname(os.path.abspath(csv_path))
    return os.path.join(folder, f"primary_hit_rescreen_{session_id}.json")


def _load_rescreen_meta(csv_path: str, session_id: str) -> Dict[str, Any]:
    json_path = _rescreen_json_path_for_csv(csv_path, session_id)
    if os.path.isfile(json_path):
        data = _read_json_maybe_gz(json_path)
        if isinstance(data, dict):
            return data
    return {}


def _load_settings(path: str) -> Dict[str, Any]:
    data = _read_json_maybe_gz(path)
    if not isinstance(data, dict):
        raise ValueError(f"invalid settings JSON: {path!r}")
    return data


def _load_rescreen_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_is_rehit(
    row: Dict[str, Any],
    *,
    min_hit_rate: float,
    min_n_hit_again: int,
    strict_hit_rate: bool = False,
) -> bool:
    try:
        n_again = int(row.get("n_hit_again") or 0)
        rate = float(row.get("hit_rate") or 0.0)
    except (TypeError, ValueError):
        return False
    if strict_hit_rate:
        return rate > float(min_hit_rate)
    if n_again >= max(1, int(min_n_hit_again)):
        return True
    return rate > float(min_hit_rate)


def collect_rescreen_rehit_param_values(
    rows: Sequence[Dict[str, Any]],
    heatmap_param_names: Sequence[str],
    *,
    min_hit_rate: float = 0.0,
    min_n_hit_again: int = 1,
    strict_hit_rate: bool = False,
) -> Tuple[Dict[str, List[float]], int, int]:
    """Return (values_by_param, n_rehit, n_screened) from rescreen CSV rows."""
    values_by_param: Dict[str, List[float]] = {str(p): [] for p in heatmap_param_names}
    n_screened = 0
    n_rehit = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        n_screened += 1
        if not _row_is_rehit(
            row,
            min_hit_rate=min_hit_rate,
            min_n_hit_again=min_n_hit_again,
            strict_hit_rate=strict_hit_rate,
        ):
            continue
        raw = row.get("params_json")
        if not raw:
            continue
        try:
            params = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if not isinstance(params, dict):
            continue
        row_ok = True
        row_vals: Dict[str, float] = {}
        for pname in heatmap_param_names:
            if pname not in params:
                row_ok = False
                break
            try:
                v = float(params[pname])
            except (TypeError, ValueError):
                row_ok = False
                break
            if not np.isfinite(v):
                row_ok = False
                break
            row_vals[str(pname)] = v
        if not row_ok:
            continue
        n_rehit += 1
        for pname, v in row_vals.items():
            values_by_param[pname].append(v)
    return values_by_param, n_rehit, n_screened


def _resolve_bounds_and_param_names(
    meta: Dict[str, Any],
    *,
    settings_path_override: Optional[str] = None,
    job_settings_json: Optional[str] = None,
    primary_session_dir: Optional[str] = None,
    sample_params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Any], Optional[str]]:
    settings_path = str(settings_path_override or meta.get("settings_path") or "").strip()
    if not settings_path or not os.path.isfile(settings_path):
        session_dir = str(primary_session_dir or meta.get("session_dir") or "").strip()
        sid = str(meta.get("session_id") or "").strip()
        if session_dir and sid:
            from gui.persistence.full_save import full_save_settings_path_json

            candidate = full_save_settings_path_json(session_dir, sid)
            if os.path.isfile(candidate):
                settings_path = candidate
    session_dir = str(primary_session_dir or meta.get("session_dir") or "").strip()

    def _campaign_summary_bounds() -> Optional[Dict[str, Any]]:
        if not session_dir or not os.path.isdir(session_dir):
            return None
        for name in sorted(os.listdir(session_dir)):
            if not (
                name.startswith("primary_batch_campaign_")
                and name.endswith(".json")
                and not name.endswith(".gz")
            ):
                continue
            try:
                camp = _load_settings(os.path.join(session_dir, name))
            except Exception:
                continue
            if str(camp.get("kind") or "") != "primary_batch_campaign":
                continue
            bounds = camp.get("primary_bounds")
            if isinstance(bounds, dict):
                return bounds
        return None

    if settings_path and os.path.isfile(settings_path):
        settings = _load_settings(settings_path)
        param_names = list(
            meta.get("param_names_list") or settings.get("primary_offload_param_names") or []
        )
        if not param_names and isinstance(sample_params, dict):
            param_names = sorted(str(k) for k in sample_params.keys())
        if not param_names:
            raise ValueError(f"no param_names_list in rescreen meta or {settings_path!r}")
        primary_bounds = settings.get("primary_bounds")
        if not isinstance(primary_bounds, dict):
            primary_bounds = _campaign_summary_bounds()
        if not isinstance(primary_bounds, dict):
            raise ValueError(
                f"primary_bounds missing in {settings_path!r} "
                f"(and no primary_batch_campaign_*.json fallback under {session_dir!r})"
            )
        return param_names, primary_bounds, settings_path

    if job_settings_json:
        job_path = os.path.abspath(os.path.expanduser(job_settings_json))
        job = _load_settings(job_path)
        if str(job.get("kind") or "") != "primary_batch_campaign":
            raise ValueError(f"expected primary_batch_campaign job JSON: {job_path!r}")
        primary_bounds = job.get("primary_bounds")
        if not isinstance(primary_bounds, dict):
            raise ValueError(f"primary_bounds missing in {job_path!r}")
        param_names = list(meta.get("param_names_list") or [])
        if not param_names and isinstance(sample_params, dict):
            param_names = sorted(str(k) for k in sample_params.keys())
        if not param_names:
            from headless.hpc_common import prune_irrelevant_bounds_for_toggles
            from headless.neutral_comparison import _bounds_from_save_json

            bounds = prune_irrelevant_bounds_for_toggles(
                _bounds_from_save_json(primary_bounds, label="primary_bounds"),
                job.get("primary_toggles") or {},
            )
            param_names = sorted(str(k) for k in bounds.keys())
        return param_names, primary_bounds, job_path

    raise FileNotFoundError(
        "Could not locate full_save_settings. Pass --settings-path, --primary-session-dir, "
        "or --job-settings-json (campaign job JSON with primary_bounds)."
    )


def _write_rehit_heatmap_png(
    *,
    out_dir: str,
    session_id: str,
    values_by_param: Dict[str, List[float]],
    primary_bounds: Dict[str, Any],
    n_rehit: int,
    n_screened: int,
    run_headline: Optional[str],
    num_bins: int,
    use_log: bool,
    log_errors: bool,
    png_basename: str = PNG_BASENAME,
    title: str = "Re-screen re-hits: parameter heatmaps",
    count_note: Optional[str] = None,
    colorbar_label: str = "Count of re-screen re-hits",
    empty_note: str = "No re-screen re-hits — empty heatmaps",
) -> Optional[str]:
    heatmap_params = resolve_heatmap_param_names(list(values_by_param.keys()), primary_bounds)
    if not heatmap_params:
        if log_errors:
            print("[rescreen_rehit_heatmap] no plottable parameters", file=sys.stderr)
        return None
    filtered_values = {p: list(values_by_param.get(p) or []) for p in heatmap_params}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        if log_errors:
            print(f"[rescreen_rehit_heatmap] matplotlib unavailable: {exc}", file=sys.stderr)
        return None

    fig = None
    try:
        n_params = len(heatmap_params)
        fig_h = max(4.0, 1.15 * n_params)
        fig = plt.figure(figsize=(12, fig_h), dpi=100)
        resolved_count_note = count_note or (
            f"({n_rehit} / {n_screened} primary hits re-met filters on re-screen)"
        )
        if not draw_parameter_heatmaps_on_figure(
            fig,
            values_by_param=filtered_values,
            primary_bounds=primary_bounds,
            num_bins=num_bins,
            use_log=use_log,
            title=title,
            n_hit=int(n_rehit),
            n_primary=int(n_screened),
        ):
            err = pop_last_parameter_heatmap_png_error()
            if log_errors and err:
                print(f"[rescreen_rehit_heatmap] {err}", file=sys.stderr)
            return None
        if run_headline:
            fig.suptitle(
                f"{run_headline.strip()}\n{title}\n{resolved_count_note}",
                fontsize=11,
            )
        out_abs = os.path.join(
            os.path.abspath(out_dir),
            png_basename.format(session_id=session_id),
        )
        fig.savefig(out_abs, dpi=100, bbox_inches="tight")
        return out_abs
    except Exception as exc:
        if log_errors:
            print(f"[rescreen_rehit_heatmap] PNG save failed: {exc}", file=sys.stderr)
        return None
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass


def _variant_config(variant: str) -> Dict[str, Any]:
    cfg = HEATMAP_VARIANTS.get(str(variant))
    if cfg is None:
        known = ", ".join(sorted(HEATMAP_VARIANTS))
        raise ValueError(f"unknown heatmap variant {variant!r} (expected one of: {known})")
    return dict(cfg)


def build_rescreen_rehit_heatmap(
    csv_path: str,
    *,
    out_dir: Optional[str] = None,
    settings_path: Optional[str] = None,
    job_settings_json: Optional[str] = None,
    primary_session_dir: Optional[str] = None,
    min_hit_rate: Optional[float] = None,
    min_n_hit_again: Optional[int] = None,
    strict_hit_rate: Optional[bool] = None,
    variant: str = "default",
    num_bins: int = 60,
    use_log: bool = False,
    run_headline: Optional[str] = None,
    log_errors: bool = True,
) -> Dict[str, Any]:
    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    session_id = _session_id_from_csv_path(csv_path)
    out_dir = os.path.abspath(out_dir or os.path.dirname(csv_path))
    os.makedirs(out_dir, exist_ok=True)
    variant_cfg = _variant_config(variant)
    resolved_min_hit_rate = (
        float(min_hit_rate) if min_hit_rate is not None else float(variant_cfg["min_hit_rate"])
    )
    resolved_min_n_hit_again = (
        int(min_n_hit_again)
        if min_n_hit_again is not None
        else int(variant_cfg["min_n_hit_again"])
    )
    resolved_strict_hit_rate = (
        bool(strict_hit_rate)
        if strict_hit_rate is not None
        else bool(variant_cfg["strict_hit_rate"])
    )

    meta = _load_rescreen_meta(csv_path, session_id)
    rows = _load_rescreen_rows(csv_path)
    sample_params: Optional[Dict[str, Any]] = None
    for row in rows:
        raw = row.get("params_json")
        if not raw:
            continue
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            sample_params = parsed
            break

    param_names, primary_bounds, resolved_settings = _resolve_bounds_and_param_names(
        meta,
        settings_path_override=(settings_path or None),
        job_settings_json=(job_settings_json or None),
        primary_session_dir=(primary_session_dir or None),
        sample_params=sample_params,
    )
    heatmap_params = resolve_heatmap_param_names(param_names, primary_bounds)

    values_by_param, n_rehit, n_screened = collect_rescreen_rehit_param_values(
        rows,
        heatmap_params,
        min_hit_rate=resolved_min_hit_rate,
        min_n_hit_again=resolved_min_n_hit_again,
        strict_hit_rate=resolved_strict_hit_rate,
    )
    # Ensure every heatmap param key exists even if empty
    for pname in heatmap_params:
        values_by_param.setdefault(str(pname), [])

    headline = run_headline
    if not headline:
        headline = str(meta.get("plot_label") or session_id).strip() or session_id

    count_note = str(variant_cfg["count_note_tpl"]).format(
        n_rehit=n_rehit,
        n_screened=n_screened,
    )
    png_path = _write_rehit_heatmap_png(
        out_dir=out_dir,
        session_id=session_id,
        values_by_param=values_by_param,
        primary_bounds=primary_bounds,
        n_rehit=n_rehit,
        n_screened=n_screened,
        run_headline=headline,
        num_bins=num_bins,
        use_log=use_log,
        log_errors=log_errors,
        png_basename=str(variant_cfg["png_basename"]),
        title=str(variant_cfg["title"]),
        count_note=count_note,
        colorbar_label=str(variant_cfg["colorbar_label"]),
        empty_note=str(variant_cfg["empty_note"]),
    )

    summary: Dict[str, Any] = {
        "kind": str(variant_cfg["kind"]),
        "variant": str(variant),
        "version": 1,
        "chart_tool_version": RESCREEN_REHIT_HEATMAP_VERSION,
        "session_id": session_id,
        "rescreen_csv": csv_path,
        "settings_path": resolved_settings,
        "heatmap_param_names": list(heatmap_params),
        "n_hits_screened": int(n_screened),
        "n_rehit": int(n_rehit),
        "min_hit_rate": float(resolved_min_hit_rate),
        "min_n_hit_again": int(resolved_min_n_hit_again),
        "strict_hit_rate": bool(resolved_strict_hit_rate),
        "values_by_param": {k: list(v) for k, v in values_by_param.items()},
        "note": str(variant_cfg["note"]),
    }
    if png_path:
        summary["png_path"] = png_path
    json_path = os.path.join(
        out_dir,
        str(variant_cfg["json_basename"]).format(session_id=session_id),
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["json_path"] = json_path
    return summary


def build_rescreen_rehit_heatmaps(
    csv_path: str,
    *,
    variants: Optional[Sequence[str]] = None,
    min_hit_rate: Optional[float] = None,
    min_n_hit_again: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """Build one or more rescreen re-hit heatmap variants for a session CSV."""
    selected = list(variants) if variants is not None else list(HEATMAP_VARIANTS)
    out: Dict[str, Dict[str, Any]] = {}
    for variant in selected:
        variant_kwargs = dict(kwargs)
        if str(variant) == "default":
            if min_hit_rate is not None:
                variant_kwargs["min_hit_rate"] = min_hit_rate
            if min_n_hit_again is not None:
                variant_kwargs["min_n_hit_again"] = min_n_hit_again
        out[str(variant)] = build_rescreen_rehit_heatmap(
            csv_path,
            variant=str(variant),
            **variant_kwargs,
        )
    return out


COMPARE_CSV_BASENAME = "primary_hit_rescreen_compare.csv"


def _is_session_rescreen_csv(csv_path: str) -> bool:
    """Per-session rescreen CSV only (exclude batch compare summary at Re-Runs root)."""
    base = os.path.basename(csv_path)
    if base == COMPARE_CSV_BASENAME:
        return False
    if not base.startswith("primary_hit_rescreen_") or not base.endswith(".csv"):
        return False
    sid = _session_id_from_csv_path(csv_path)
    return bool(sid) and sid != "compare"


def discover_rescreen_csv_paths(path: str) -> List[str]:
    """Find rescreen CSV(s): one file, one session folder, or Re-Runs batch root."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path) and path.endswith(".csv"):
        if not _is_session_rescreen_csv(path):
            raise FileNotFoundError(f"Not a per-session rescreen CSV: {path!r}")
        return [path]
    if os.path.isdir(path):
        base = os.path.basename(path.rstrip("/"))
        if base == DEFAULT_BATCH_SUBDIR:
            found = sorted(
                glob.glob(os.path.join(path, DEFAULT_SESSIONS_SUBDIR, "*", CSV_GLOB))
            )
            found = [p for p in found if _is_session_rescreen_csv(p)]
            if found:
                return found
        if base == DEFAULT_SESSIONS_SUBDIR:
            found = sorted(glob.glob(os.path.join(path, "*", CSV_GLOB)))
            found = [p for p in found if _is_session_rescreen_csv(p)]
            if found:
                return found
        direct = sorted(glob.glob(os.path.join(path, CSV_GLOB)))
        direct = [p for p in direct if _is_session_rescreen_csv(p)]
        if direct:
            return direct
        parent = os.path.basename(os.path.dirname(path.rstrip("/")))
        if parent == DEFAULT_SESSIONS_SUBDIR:
            one = [p for p in sorted(glob.glob(os.path.join(path, CSV_GLOB))) if _is_session_rescreen_csv(p)]
            if one:
                return one
    raise FileNotFoundError(f"No rescreen CSV found under {path!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="Rescreen CSV, session folder under Re-Runs/sessions/, or Re-Runs root",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output folder (default: same directory as the rescreen CSV)",
    )
    parser.add_argument(
        "--settings-path",
        default="",
        help="Override full_save_settings JSON (if rescreen JSON lacks session_dir)",
    )
    parser.add_argument(
        "--primary-session-dir",
        default="",
        help="Primary batch session folder (find full_save_settings_<sid>.json)",
    )
    parser.add_argument(
        "--job-settings-json",
        default="",
        help="Campaign job JSON with primary_bounds (fallback when settings JSON unavailable)",
    )
    parser.add_argument(
        "--variant",
        choices=("default", "high_rehit_rate", "all"),
        default="all",
        help=(
            "Heatmap filter/output variant: all re-hits (default), >90%% re-hit rate only, "
            "or both (default: all)"
        ),
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help=(
            "Override min hit_rate for the default variant only "
            "(strictly greater than; default 0)"
        ),
    )
    parser.add_argument(
        "--min-n-hit-again",
        type=int,
        default=None,
        help="Override min n_hit_again for the default variant only (default 1)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=60,
        help="Histogram bins per parameter strip (default 60)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Log10 x-axis bins where values are positive",
    )
    parser.add_argument(
        "--headline",
        default="",
        help="Optional plot headline (default: rescreen plot_label or session_id)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        csv_paths = discover_rescreen_csv_paths(args.path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    variants = (
        list(HEATMAP_VARIANTS)
        if args.variant == "all"
        else [str(args.variant)]
    )
    ok = 0
    fail = 0
    for csv_path in csv_paths:
        sid = _session_id_from_csv_path(csv_path)
        print(f"[rescreen_rehit_heatmap] {sid}")
        try:
            summaries = build_rescreen_rehit_heatmaps(
                csv_path,
                variants=variants,
                out_dir=(args.out_dir or None),
                settings_path=(args.settings_path or None),
                job_settings_json=(args.job_settings_json or None),
                primary_session_dir=(args.primary_session_dir or None),
                min_hit_rate=(None if args.min_hit_rate is None else float(args.min_hit_rate)),
                min_n_hit_again=(
                    None if args.min_n_hit_again is None else int(args.min_n_hit_again)
                ),
                num_bins=int(args.num_bins),
                use_log=bool(args.log_scale),
                run_headline=(args.headline or None),
            )
        except Exception as exc:
            fail += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        ok += 1
        for variant, summary in summaries.items():
            print(
                f"  [{variant}] re-hits {summary.get('n_rehit')} / "
                f"{summary.get('n_hits_screened')} "
                f"→ {summary.get('png_path') or '(no PNG)'}"
            )

    print(f"[rescreen_rehit_heatmap] done: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
