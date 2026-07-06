"""
Primary-batch event rates (deaths, duplications, mutations, flow removal) vs metric-filter hits.

Reads Full Save offload rows for stage ``primary``, computes per-generation means within each
simulation from change_history, then averages those rates across runs that met vs did not meet
the active metric AND-chain.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gui.apps.neutral_comparison.offload import NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR
from gui.persistence.full_save import load_full_save_manifest
from simulation.change_history import parse_change_history_row

PRIMARY_OFFLOAD_STAGE = "primary"

# Chart / summary JSON field order (matches grouped bar x-axis).
_EVENT_METRIC_LABELS: Tuple[str, ...] = (
    "Deaths",
    "Duplications",
    "Mutations",
    "Flow removal",
)
_EVENT_SUMMARY_KEYS: Tuple[str, ...] = (
    "deaths_mean",
    "duplications_mean",
    "mutations_mean",
    "flow_removed_mean",
)
_EVENT_SUMMARY_CI95_KEYS: Tuple[str, ...] = tuple(
    k.replace("_mean", "_ci95") for k in _EVENT_SUMMARY_KEYS
)

# Two-sided 95% Student-t critical values for df = 1 … 120 (df > 120 → 1.96).
_T_CRIT_95_LUT: Tuple[float, ...] = (
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.306, 2.2622, 2.2281,
    2.201, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.093, 2.086,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
    2.0395, 2.0369, 2.0345, 2.0322, 2.0301, 2.0281, 2.0262, 2.0244, 2.0227, 2.0211,
    2.0195, 2.0181, 2.0167, 2.0154, 2.0141, 2.0129, 2.0117, 2.0106, 2.0096, 2.0086,
    2.0076, 2.0066, 2.0057, 2.0049, 2.004, 2.0032, 2.0025, 2.0017, 2.001, 2.0003,
    1.9996, 1.999, 1.9983, 1.9977, 1.9971, 1.9966, 1.996, 1.9955, 1.9949, 1.9944,
    1.9939, 1.9935, 1.993, 1.9925, 1.9921, 1.9917, 1.9913, 1.9908, 1.9905, 1.9901,
    1.9897, 1.9893, 1.989, 1.9886, 1.9883, 1.9879, 1.9876, 1.9873, 1.987, 1.9867,
    1.9864, 1.9861, 1.9858, 1.9855, 1.9853, 1.985, 1.9847, 1.9845, 1.9842, 1.984,
    1.9837, 1.9835, 1.9833, 1.983, 1.9828, 1.9826, 1.9824, 1.9822, 1.982, 1.9818,
    1.9816, 1.9814, 1.9812, 1.981, 1.9808, 1.9806, 1.9804, 1.9803, 1.9801, 1.9799,
)


def _t_critical_975(df: int) -> float:
    """Two-sided 95% t critical value (``t_{0.975, df}``) for sample size ``n = df + 1``."""
    if df <= 0:
        return 0.0
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, df))
    except ImportError:
        pass
    if df <= len(_T_CRIT_95_LUT):
        return _T_CRIT_95_LUT[df - 1]
    return 1.96


def _ci95_half_width(vals: Sequence[float]) -> float:
    """Half-width of two-sided 95% CI for the mean (Student-t, ddof=1)."""
    arr = np.asarray(vals, dtype=float)
    n = int(arr.size)
    if n <= 1:
        return 0.0
    sem = float(np.std(arr, ddof=1)) / np.sqrt(n)
    return _t_critical_975(n - 1) * sem


def per_run_event_rates_per_generation_from_metric_input(
    metric_input: Any,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Return (deaths, duplications, mutations, flow_removed) **per generation** for one simulation.

    Each value is the arithmetic mean over ``change_history`` rows (one row per generation).
    mutations = accepted trait mutations + accepted auxiliary (T/D) mutations per generation.
    Returns None if change_history is missing (cannot summarize).
    """
    if not isinstance(metric_input, dict) or metric_input.get("collapsed"):
        return None
    rows = metric_input.get("change_history")
    if not isinstance(rows, list) or not rows:
        return (0.0, 0.0, 0.0, 0.0)
    deaths = dups = mutations = flow_removed = 0
    n_gen = 0
    for row in rows:
        try:
            d, dup, acc, flow, acc_aux = parse_change_history_row(row)
        except Exception:
            continue
        n_gen += 1
        deaths += int(d)
        dups += int(dup)
        mutations += int(acc) + int(acc_aux)
        flow_removed += int(flow)
    if n_gen <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    inv = 1.0 / float(n_gen)
    return (
        deaths * inv,
        dups * inv,
        mutations * inv,
        flow_removed * inv,
    )


def _primary_hit_from_record(rec: Dict[str, Any]) -> Optional[bool]:
    nc = rec.get("neutral_comparison")
    if not isinstance(nc, dict):
        return None
    if str(nc.get("stage") or "").strip() != PRIMARY_OFFLOAD_STAGE:
        return None
    return bool(nc.get("hit"))


def summarize_primary_events_by_hit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate primary offload records into mean per-generation event rates for hit vs non-hit runs.
    """
    hit_buckets: List[List[float]] = [[] for _ in range(4)]
    miss_buckets: List[List[float]] = [[] for _ in range(4)]
    skipped_no_history = 0

    for rec in records:
        if not isinstance(rec, dict):
            continue
        hit_flag = _primary_hit_from_record(rec)
        if hit_flag is None:
            continue
        rates = per_run_event_rates_per_generation_from_metric_input(rec.get("metric_input"))
        if rates is None:
            skipped_no_history += 1
            continue
        bucket = hit_buckets if hit_flag else miss_buckets
        for i, val in enumerate(rates):
            bucket[i].append(float(val))

    def _pack(buckets: List[List[float]]) -> Dict[str, Any]:
        n = len(buckets[0]) if buckets and buckets[0] else 0
        if n == 0:
            empty = {k: None for k in _EVENT_SUMMARY_KEYS}
            empty.update({k: None for k in _EVENT_SUMMARY_CI95_KEYS})
            empty["n"] = 0
            return empty
        out: Dict[str, Any] = {"n": n}
        for mean_key, ci_key, vals in zip(_EVENT_SUMMARY_KEYS, _EVENT_SUMMARY_CI95_KEYS, buckets):
            arr = np.asarray(vals, dtype=float)
            out[mean_key] = float(np.mean(arr))
            out[ci_key] = _ci95_half_width(arr)
        return out

    return {
        "meets_threshold": _pack(hit_buckets),
        "below_threshold": _pack(miss_buckets),
        "skipped_no_change_history": int(skipped_no_history),
        "definition": (
            "Per primary simulation: for each generation in change_history, record deaths, "
            "duplications, accepted mutations (A/B + auxiliary), and chemostat flow removals; "
            "take the arithmetic mean across generations in that run. Bars show the mean of those "
            "per-run rates within each hit group; error bars are half the two-sided 95% CI of "
            "the mean across runs (Student-t, ddof=1)."
        ),
    }


def _collect_primary_offload_records_in_folder(
    search_folder: str,
    session_id: str,
    read_json: Callable[[str], Any],
) -> List[Dict[str, Any]]:
    """Read primary-stage records from offload batches under ``search_folder`` (manifest layout)."""
    search_folder = os.path.abspath(os.path.expanduser(str(search_folder or "").strip()))
    sid = str(session_id or "").strip()
    if not search_folder or not sid:
        return []
    manifest = load_full_save_manifest(search_folder, sid, read_json=read_json)
    if not isinstance(manifest, dict):
        return []
    out: List[Dict[str, Any]] = []
    for batch in manifest.get("batches") or []:
        if not isinstance(batch, dict):
            continue
        rel = str(batch.get("path") or "").strip()
        if not rel:
            continue
        path = rel if os.path.isabs(rel) else os.path.join(search_folder, rel)
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        records = payload.get("records")
        if not isinstance(records, list):
            continue
        for rec in records:
            if isinstance(rec, dict) and _primary_hit_from_record(rec) is not None:
                out.append(rec)
    return out


def load_primary_offload_records(
    folder: str,
    session_id: str,
    read_json: Callable[[str], Any],
) -> List[Dict[str, Any]]:
    """
    Load all offload records for the primary batch.

    Tries session-root offload batches first. If empty (common when multi-node or parallel
    runs use ``--skip-offload-merge``), reads ``_parallel_shards/primary/`` under the session folder.
    """
    folder = os.path.abspath(os.path.expanduser(str(folder or "").strip()))
    sid = str(session_id or "").strip()
    if not folder or not sid:
        return []
    out = _collect_primary_offload_records_in_folder(folder, sid, read_json)
    if out:
        return out
    shard_primary = os.path.join(folder, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, "primary")
    if os.path.isdir(shard_primary):
        out = _collect_primary_offload_records_in_folder(shard_primary, sid, read_json)
    return out


def _summary_side_values(side: Dict[str, Any], *, keys: Tuple[str, ...] = _EVENT_SUMMARY_KEYS) -> List[float]:
    out: List[float] = []
    for k in keys:
        v = side.get(k)
        out.append(float(v) if v is not None and np.isfinite(float(v)) else 0.0)
    return out


def _error_bar_half_widths(side: Dict[str, Any]) -> List[float]:
    """95% CI half-width for matplotlib ``yerr``."""
    return _summary_side_values(side, keys=_EVENT_SUMMARY_CI95_KEYS)


def _draw_primary_event_bars(
    ax: Any,
    *,
    meets: Dict[str, Any],
    below: Dict[str, Any],
    summary: Dict[str, Any],
    title: str,
) -> bool:
    n_hit = int(meets.get("n") or 0)
    n_miss = int(below.get("n") or 0)
    if n_hit + n_miss == 0:
        return False
    hit_vals = _summary_side_values(meets)
    miss_vals = _summary_side_values(below)
    hit_err = _error_bar_half_widths(meets)
    miss_err = _error_bar_half_widths(below)
    x = np.arange(len(_EVENT_METRIC_LABELS), dtype=float)
    width = 0.36
    _err_kw = {"elinewidth": 1.0, "capthick": 1.0, "capsize": 4}
    ax.bar(
        x - width / 2,
        hit_vals,
        width,
        yerr=hit_err,
        label=f"Meets threshold (n={n_hit})",
        color="#009E73",
        alpha=0.9,
        error_kw=_err_kw,
    )
    ax.bar(
        x + width / 2,
        miss_vals,
        width,
        yerr=miss_err,
        label=f"Below threshold (n={n_miss})",
        color="#CC79A7",
        alpha=0.9,
        error_kw=_err_kw,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(_EVENT_METRIC_LABELS)
    ax.set_ylabel("Mean events per generation ± 95% CI\n(across runs in group)")
    ax.legend(loc="upper right")
    ax.set_title(title)
    skipped = int(summary.get("skipped_no_change_history") or 0)
    if skipped:
        ax.text(
            0.02,
            0.02,
            f"{skipped} run(s) omitted (no change_history in offload).",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
    return True


_LAST_PNG_ERROR: Optional[str] = None


def pop_last_primary_event_png_error() -> Optional[str]:
    """Last error from ``write_primary_event_rates_png`` (cleared after read)."""
    global _LAST_PNG_ERROR
    err = _LAST_PNG_ERROR
    _LAST_PNG_ERROR = None
    return err


def write_primary_event_rates_png(
    *,
    folder: str,
    session_id: str,
    summary: Dict[str, Any],
    run_headline: Optional[str] = None,
    log_errors: bool = False,
) -> Optional[str]:
    """
    Grouped bar chart: mean per-generation deaths / duplications / mutations / flow removal.

    Returns absolute path written, or None if matplotlib unavailable or no data.
    On failure, ``pop_last_primary_event_png_error()`` may return a short reason.
    """
    global _LAST_PNG_ERROR
    _LAST_PNG_ERROR = None

    def _fail(msg: str) -> None:
        global _LAST_PNG_ERROR
        _LAST_PNG_ERROR = msg
        if log_errors:
            print(f"  [primary_event_chart] {msg}", file=sys.stderr)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        _fail(f"matplotlib not installed ({exc})")
        return None

    meets = summary.get("meets_threshold") if isinstance(summary.get("meets_threshold"), dict) else {}
    below = summary.get("below_threshold") if isinstance(summary.get("below_threshold"), dict) else {}
    n_hit = int(meets.get("n") or 0)
    n_miss = int(below.get("n") or 0)
    if n_hit + n_miss == 0:
        return None

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=100)
        if not _draw_primary_event_bars(
            ax,
            meets=meets,
            below=below,
            summary=summary,
            title="Primary batch: per-generation events by filter outcome",
        ):
            return None
        # With yerr, ax.containers includes error-bar artists; only annotate bar patches.
        for container in ax.containers:
            patches = getattr(container, "patches", ())
            for bar in patches:
                if bar is None or not hasattr(bar, "get_height"):
                    continue
                h = float(bar.get_height())
                if h > 0:
                    ax.annotate(
                        f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        h = (run_headline or "").strip()
        if h:
            fig.suptitle(h, fontsize=12, fontweight="semibold", y=1.03)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
        else:
            fig.tight_layout()
        fname = f"neutral_primary_events_{session_id}.png"
        out_abs = os.path.abspath(os.path.join(folder, fname))
        fig.savefig(out_abs, dpi=100, bbox_inches="tight")
        return out_abs
    except Exception as exc:
        _fail(f"PNG save failed: {exc}")
        return None
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass


def draw_primary_event_rates_on_axes(
    ax: Any,
    summary: Dict[str, Any],
    *,
    title: Optional[str] = None,
) -> bool:
    """Draw the same chart on an existing matplotlib Axes (GUI). Returns False if no data."""
    meets = summary.get("meets_threshold") if isinstance(summary.get("meets_threshold"), dict) else {}
    below = summary.get("below_threshold") if isinstance(summary.get("below_threshold"), dict) else {}
    n_hit = int(meets.get("n") or 0)
    n_miss = int(below.get("n") or 0)
    if n_hit + n_miss == 0:
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "No primary offload rows with change_history.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        return False
    return _draw_primary_event_bars(
        ax,
        meets=meets,
        below=below,
        summary=summary,
        title=title or "Primary: per-generation events by filter outcome",
    )
