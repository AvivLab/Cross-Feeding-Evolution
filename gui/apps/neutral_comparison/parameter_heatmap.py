"""
1D binned parameter heatmaps for primary-batch runs that met the metric filter (hit=True).

Mirrors the Gradient Descent GUI parameter heatmap layout: one horizontal strip per sampled
parameter, counting threshold-crossing primary simulations per bin.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.apps.neutral_comparison.offload import NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR
from gui.apps.neutral_comparison.primary_event_chart import (
    PRIMARY_OFFLOAD_STAGE,
    _primary_hit_from_record,
    load_primary_offload_records,
)
from gui.common.colors import DEFAULT_HEATMAP_CMAP
from gui.persistence.full_save import load_full_save_manifest

_LAST_PNG_ERROR: Optional[str] = None


def pop_last_parameter_heatmap_png_error() -> Optional[str]:
    """Last error from ``write_parameter_heatmap_png`` (cleared after read)."""
    global _LAST_PNG_ERROR
    err = _LAST_PNG_ERROR
    _LAST_PNG_ERROR = None
    return err


def _fail(msg: str, *, log_errors: bool = False) -> None:
    global _LAST_PNG_ERROR
    _LAST_PNG_ERROR = msg
    if log_errors:
        print(f"  [parameter_heatmap] {msg}", file=sys.stderr)


def _parse_bounds_pair(raw: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        lo = float(raw[0]) if raw[0] not in ("-inf", "-Infinity", None) else float("-inf")
        hi = float(raw[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(hi):
        return None
    if lo >= hi:
        return None
    return lo, hi


def load_param_names_from_offload(
    folder: str,
    session_id: str,
    read_json: Callable[[str], Any],
) -> List[str]:
    """Read ``param_names_list`` from the first offload batch under ``folder``."""
    folder = os.path.abspath(os.path.expanduser(str(folder or "").strip()))
    sid = str(session_id or "").strip()
    if not folder or not sid:
        return []
    manifest = load_full_save_manifest(folder, sid, read_json=read_json)
    if not isinstance(manifest, dict):
        return []
    for batch in manifest.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        rel = str(batch.get("path") or "").strip()
        if not rel:
            continue
        path = rel if os.path.isabs(rel) else os.path.join(folder, rel)
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        pnl = payload.get("param_names_list")
        if isinstance(pnl, list) and pnl:
            return [str(x) for x in pnl]
    return []


def resolve_heatmap_param_names(
    param_names_list: Sequence[str],
    primary_bounds: Optional[Dict[str, Any]],
) -> List[str]:
    """Parameters to plot: primary_bounds keys in offload vector order."""
    bounds = primary_bounds if isinstance(primary_bounds, dict) else {}
    bounds_keys = set(bounds.keys())
    ordered = [str(p) for p in param_names_list if str(p) in bounds_keys]
    if ordered:
        return ordered
    if bounds_keys:
        return sorted(str(k) for k in bounds_keys)
    return []


def collect_primary_hit_param_values(
    records: Sequence[Dict[str, Any]],
    param_names_list: Sequence[str],
    heatmap_param_names: Sequence[str],
) -> Tuple[Dict[str, List[float]], int, int]:
    """
    Return (values_by_param, n_hit, n_primary) for primary-stage offload rows.

    Only runs with ``neutral_comparison.hit`` True are included in ``values_by_param``.
    """
    name_to_idx = {str(n): i for i, n in enumerate(param_names_list)}
    values_by_param: Dict[str, List[float]] = {str(p): [] for p in heatmap_param_names}
    n_primary = 0
    n_hit = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        hit_flag = _primary_hit_from_record(rec)
        if hit_flag is None:
            continue
        n_primary += 1
        if not hit_flag:
            continue
        vec = rec.get("param_vector")
        if not isinstance(vec, list):
            continue
        row_ok = True
        row_vals: Dict[str, float] = {}
        for pname in heatmap_param_names:
            idx = name_to_idx.get(str(pname))
            if idx is None or idx >= len(vec):
                row_ok = False
                break
            try:
                v = float(vec[idx])
            except (TypeError, ValueError):
                row_ok = False
                break
            if not np.isfinite(v):
                row_ok = False
                break
            row_vals[str(pname)] = v
        if not row_ok:
            continue
        n_hit += 1
        for pname, v in row_vals.items():
            values_by_param[pname].append(v)
    return values_by_param, n_hit, n_primary


def summarize_parameter_heatmap(
    records: Sequence[Dict[str, Any]],
    *,
    param_names_list: Sequence[str],
    primary_bounds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summary dict for JSON export / regen tools."""
    heatmap_params = resolve_heatmap_param_names(param_names_list, primary_bounds)
    values_by_param, n_hit, n_primary = collect_primary_hit_param_values(
        records, param_names_list, heatmap_params
    )
    return {
        "heatmap_param_names": list(heatmap_params),
        "n_primary_runs": int(n_primary),
        "n_threshold_hits": int(n_hit),
        "values_by_param": {k: list(v) for k, v in values_by_param.items()},
        "note": (
            "1D bin counts of primary-batch runs that met all active metric filters (AND), "
            f"stage={PRIMARY_OFFLOAD_STAGE!r}."
        ),
    }


def _bounds_for_param(
    pname: str,
    vals: np.ndarray,
    primary_bounds: Optional[Dict[str, Any]],
) -> Tuple[float, float]:
    bounds = primary_bounds if isinstance(primary_bounds, dict) else {}
    pair = _parse_bounds_pair(bounds.get(pname))
    if pair is not None:
        lo, hi = pair
        if lo == float("-inf") and vals.size > 0:
            lo = float(np.min(vals))
        return lo, hi
    if vals.size > 0:
        return float(np.min(vals)), float(np.max(vals))
    return 0.0, 1.0


def _bin_counts_1d(
    vals: np.ndarray,
    *,
    pmin: float,
    pmax: float,
    num_bins: int,
    use_log: bool,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    if use_log:
        if pmin <= 0.0 or pmax <= 0.0:
            pos = vals[vals > 0] if vals.size > 0 else vals
            if pos.size == 0:
                raise ValueError("no positive values for log-scale bins")
            pmin = float(np.min(pos))
            pmax = float(np.max(pos))
        plot_min = float(np.log10(pmin))
        plot_max = float(np.log10(pmax))
        vals_plot = np.log10(vals) if vals.size > 0 else vals
    else:
        plot_min = float(pmin)
        plot_max = float(pmax)
        vals_plot = vals
    bins = np.linspace(plot_min, plot_max, int(num_bins) + 1)
    if vals_plot.size > 0:
        vals_plot = np.clip(vals_plot, plot_min, plot_max)
        counts, _ = np.histogram(vals_plot, bins=bins)
    else:
        counts = np.zeros(int(num_bins), dtype=float)
    return counts, plot_min, plot_max, bins


def draw_parameter_heatmaps_on_figure(
    fig: Any,
    *,
    values_by_param: Dict[str, List[float]],
    primary_bounds: Optional[Dict[str, Any]] = None,
    num_bins: int = 60,
    use_log: bool = False,
    title: str = "Primary batch: parameter heatmaps (threshold-crossing runs)",
    n_hit: Optional[int] = None,
    n_primary: Optional[int] = None,
) -> bool:
    """
    Draw one 1D heatmap strip per parameter on ``fig`` (creates its own subplots).

    Returns False if there are no parameters to plot.
    """
    param_names = [p for p in values_by_param.keys()]
    if not param_names:
        return False

    n_params = len(param_names)
    fig.clf()
    try:
        fig.set_constrained_layout(True)
    except Exception:
        pass
    axes = fig.subplots(n_params, 1)
    if n_params == 1:
        axes = [axes]

    cmap = DEFAULT_HEATMAP_CMAP
    im = None
    no_matches = int(n_hit or 0) == 0

    for ax_i, pname in zip(axes, param_names):
        vals = np.asarray(values_by_param.get(pname) or [], dtype=float)
        vals = vals[np.isfinite(vals)]
        try:
            pmin, pmax = _bounds_for_param(pname, vals, primary_bounds)
        except ValueError:
            return False
        counts, plot_min, plot_max, _bins = _bin_counts_1d(
            vals, pmin=pmin, pmax=pmax, num_bins=num_bins, use_log=use_log
        )
        im = ax_i.imshow(
            counts[np.newaxis, :],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=(plot_min, plot_max, 0, 1),
        )
        ax_i.set_yticks([])
        ax_i.set_ylabel(pname, rotation=0, ha="right", va="center", fontsize=9)
        ax_i.set_xlim(plot_min, plot_max)
        mid_tick = (plot_min + plot_max) / 2.0
        ax_i.set_xticks([plot_min, mid_tick, plot_max])
        if use_log:
            ax_i.set_xticklabels([f"{pmin:g}", f"{(10 ** mid_tick):g}", f"{pmax:g}"], fontsize=8)
        else:
            ax_i.set_xticklabels([f"{plot_min:g}", f"{mid_tick:g}", f"{pmax:g}"], fontsize=8)
        if vals.size > 0:
            obs_min = float(np.min(vals))
            obs_max = float(np.max(vals))
            if use_log and obs_min > 0 and obs_max > 0:
                ax_i.axvline(np.log10(obs_min), color="white", lw=1, alpha=0.9)
                ax_i.axvline(np.log10(obs_max), color="white", lw=1, alpha=0.9)
            elif not use_log:
                ax_i.axvline(obs_min, color="white", lw=1, alpha=0.9)
                ax_i.axvline(obs_max, color="white", lw=1, alpha=0.9)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
        cbar.set_label("Count of threshold-crossing primary runs")

    subtitle_parts = [title]
    if n_hit is not None and n_primary is not None:
        subtitle_parts.append(f"({n_hit} / {n_primary} primary runs met filters)")
    fig.suptitle("\n".join(subtitle_parts), fontsize=11)
    if no_matches:
        fig.text(
            0.5,
            0.01,
            "No primary runs met the metric filters — empty heatmaps",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#666666",
        )
    return True


def write_parameter_heatmap_png(
    *,
    folder: str,
    session_id: str,
    records: Sequence[Dict[str, Any]],
    param_names_list: Sequence[str],
    primary_bounds: Optional[Dict[str, Any]] = None,
    run_headline: Optional[str] = None,
    num_bins: int = 60,
    use_log: bool = False,
    log_errors: bool = False,
) -> Optional[str]:
    """
    Write ``neutral_param_heatmap_<session_id>.png`` under ``folder``.

    Returns absolute path written, or None if matplotlib unavailable or no plottable parameters.
    """
    global _LAST_PNG_ERROR
    _LAST_PNG_ERROR = None

    heatmap_params = resolve_heatmap_param_names(param_names_list, primary_bounds)
    if not heatmap_params:
        _fail("no heatmap parameters (primary_bounds empty or missing from offload)", log_errors=log_errors)
        return None

    values_by_param, n_hit, n_primary = collect_primary_hit_param_values(
        records, param_names_list, heatmap_params
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        _fail(f"matplotlib not installed ({exc})", log_errors=log_errors)
        return None

    fig = None
    try:
        n_params = len(heatmap_params)
        fig_h = max(4.0, 1.15 * n_params)
        fig = plt.figure(figsize=(12, fig_h), dpi=100)
        title = "Primary batch: parameter heatmaps (threshold-crossing runs)"
        if not draw_parameter_heatmaps_on_figure(
            fig,
            values_by_param=values_by_param,
            primary_bounds=primary_bounds,
            num_bins=num_bins,
            use_log=use_log,
            title=title,
            n_hit=n_hit,
            n_primary=n_primary,
        ):
            _fail("draw_parameter_heatmaps_on_figure returned no data", log_errors=log_errors)
            return None
        h = (run_headline or "").strip()
        if h:
            fig.suptitle(
                f"{h}\n{title}\n({n_hit} / {n_primary} primary runs met filters)",
                fontsize=11,
            )
        fname = f"neutral_param_heatmap_{session_id}.png"
        out_abs = os.path.abspath(os.path.join(folder, fname))
        fig.savefig(out_abs, dpi=100, bbox_inches="tight")
        return out_abs
    except Exception as exc:
        _fail(f"PNG save failed: {exc}", log_errors=log_errors)
        return None
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass


def load_primary_offload_context(
    folder: str,
    session_id: str,
    read_json: Callable[[str], Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Primary offload records plus ``param_names_list`` (session root, then shard fallback).
    """
    folder = os.path.abspath(os.path.expanduser(str(folder or "").strip()))
    sid = str(session_id or "").strip()
    records = load_primary_offload_records(folder, sid, read_json)
    param_names = load_param_names_from_offload(folder, sid, read_json)
    if not param_names and records:
        shard_primary = os.path.join(folder, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, "primary")
        if os.path.isdir(shard_primary):
            param_names = load_param_names_from_offload(shard_primary, sid, read_json)
    return records, param_names
