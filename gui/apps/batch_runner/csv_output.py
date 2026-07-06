"""CSV output for Batch Runner batch campaigns.

Session hit-counts CSVs use a hit-only schema: one row per primary hit with
``(hit_index, source_run_index)`` identity, parameter columns, and
``n_seeds`` / ``n_hit_again`` (``nan`` until Batch Re-Runner patches them in place).
"""

from __future__ import annotations

import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Row-type label for per-batch summary rows in wide CSV schemas.
ROW_BATCH = "batch"
ROW_PRIMARY_BATCH = ROW_BATCH
ROW_HIT = "hit"
ROW_RESCREEN_HIT = "rescreen_hit"
ROW_RESCREEN_NON_HIT = "rescreen_non_hit"
NAN = "nan"

# row_type values still read from older wide CSVs (batch summary rows, rescreen sidecars).
_LEGACY_BATCH_ROW_TYPES = frozenset({ROW_BATCH, "primary_batch"})
_HIT_ROW_TYPES = frozenset({ROW_HIT})
_RESCREEN_EXTRA_ROW_TYPES = frozenset({ROW_RESCREEN_HIT, ROW_RESCREEN_NON_HIT})

PARAM_COLUMNS: Tuple[str, ...] = (
    "Acetate Ratio",
    "Average In_Flow",
    "Cost of Life",
    "Flow Percentage",
    "Initial A",
    "Initial Energy",
    "Initial Organism Count",
    "Investment Modifier",
    "Mutation Rate",
    "Mutation Scale",
    "Number of Generations",
    "Constant Probability",
    "Death Decay Rate",
    "Duplication Sigmoid Intensity",
    "Duplication Sigmoid Midpoint",
)

PAPER_HIT_COUNT_FIELDS: Tuple[str, ...] = (
    "plot_label",
    "configuration",
    "suite",
    "session_dir",
    "Y",
    "batch_index",
    "hit_index",
    "source_run_index",
    "n_seeds",
    "n_hit_again",
    *PARAM_COLUMNS,
)

PAPER_BATCH_HIT_COUNT_FIELDS = PAPER_HIT_COUNT_FIELDS

COMPARE_HIT_COUNT_FIELDS: Tuple[str, ...] = (
    "plot_label",
    "configuration",
    "suite",
    "session_dir",
    "batch_index",
    "hit_count",
)


def normalize_configuration_name(name: Any) -> str:
    """User-facing configuration label for CSV/metadata (single line, no commas)."""
    text = str(name or "").strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    if "," in text:
        text = text.replace(",", " ")
    return text[:80].strip()


def sanitize_configuration_slug(name: str) -> str:
    """Filesystem-safe slug derived from a configuration name."""
    text = normalize_configuration_name(name)
    if not text:
        return ""
    slug = re.sub(r"[^\w.\-+]+", "_", text.replace(" ", "_"))
    slug = slug.strip("._-")
    return slug[:48]


def session_hit_counts_csv_basename(session_id: str, *, configuration: str = "") -> str:
    sid = str(session_id or "").strip()
    slug = sanitize_configuration_slug(configuration)
    if slug:
        return f"batch_hit_counts_{slug}_{sid}.csv"
    return f"batch_hit_counts_{sid}.csv"


def session_compare_hit_counts_csv_basename(session_id: str) -> str:
    """Basename for optional per-session compare sidecar CSV."""
    return f"batch_compare_hit_counts_{session_id}.csv"


def compare_rows_from_batch_hit_counts(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """Extract compare-CSV columns (per-batch hit totals) from session CSV rows."""
    legacy_batch_rows = [row for row in rows if _is_batch_summary_row(row)]
    if legacy_batch_rows:
        out: List[Dict[str, str]] = []
        for row in legacy_batch_rows:
            out.append({name: str(row.get(name, "")) for name in COMPARE_HIT_COUNT_FIELDS})
        return out

    # Hit-only CSV: tally one hit per row grouped by campaign metadata + batch_index.
    tallies: Dict[Tuple[str, str, str, str, int], Dict[str, str]] = {}
    for row in rows:
        if not _is_data_hit_row(row):
            continue
        try:
            batch_idx = int(row.get("batch_index") or 0)
        except (TypeError, ValueError):
            continue
        key = (
            str(row.get("plot_label") or ""),
            str(row.get("configuration") or ""),
            str(row.get("suite") or ""),
            str(row.get("session_dir") or ""),
            batch_idx,
        )
        bucket = tallies.setdefault(
            key,
            {
                "plot_label": key[0],
                "configuration": key[1],
                "suite": key[2],
                "session_dir": key[3],
                "batch_index": str(batch_idx),
                "hit_count": "0",
            },
        )
        bucket["hit_count"] = str(int(bucket["hit_count"]) + 1)
    return [tallies[k] for k in sorted(tallies.keys(), key=lambda item: item[4])]


def hit_counts_by_batch_from_csv_rows(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    """Return ``[(batch_index, hit_count), ...]`` from batch-summary or aggregated hit rows."""
    compare = compare_rows_from_batch_hit_counts(rows)
    out: List[Tuple[int, int]] = []
    for row in compare:
        out.append((int(row["batch_index"]), int(row["hit_count"])))
    return out


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def infer_configuration(
    primary_toggles: Optional[Mapping[str, Any]],
    primary_bounds: Optional[Mapping[str, Any]] = None,
) -> str:
    """Map simulation toggles and sampled bounds to configuration keys."""
    toggles = primary_toggles or {}
    bound_keys = set((primary_bounds or {}).keys())
    no_death = _coerce_bool(toggles.get("No Death", False))
    const_death = _coerce_bool(toggles.get("Constant Death Probability", False))
    const_dup = _coerce_bool(toggles.get("Constant Duplication Probability", False))
    binary = _coerce_bool(toggles.get("Binary Death at Zero Energy", False))
    has_death_decay = "Death Decay Rate" in bound_keys
    has_dup_sigmoid = (
        "Duplication Sigmoid Intensity" in bound_keys
        or "Duplication Sigmoid Midpoint" in bound_keys
    )
    # Prefer sampled-bound signatures over toggle-only labels when both are present.
    if no_death and const_dup and not has_death_decay:
        return "Neutral"
    if has_death_decay and has_dup_sigmoid:
        return "Death+Dup"
    if has_death_decay:
        return "justDeath"
    if no_death and has_dup_sigmoid:
        return "justDup"
    if const_death and const_dup:
        return "Death+Dup"
    if const_death:
        return "justDeath"
    if const_dup and not no_death:
        return "justDup"
    if binary:
        return "binary_death"
    return ""


def _format_y_value(acetate_ratio: float) -> str:
    if acetate_ratio == int(acetate_ratio):
        return str(int(acetate_ratio))
    text = f"{acetate_ratio:g}"
    return text


def infer_suite_and_y(primary_numeric: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
    """Derive paper suite tag (e.g. Fixed_1_ratio) and Y from fixed Acetate Ratio when present."""
    numeric = primary_numeric or {}
    if "Acetate Ratio" not in numeric:
        return "", ""
    try:
        val = float(numeric["Acetate Ratio"])
    except (TypeError, ValueError):
        return "", ""
    y = _format_y_value(val)
    return f"Fixed_{y}_ratio", y


def resolve_paper_campaign_metadata(
    *,
    session_id: str,
    folder: str,
    primary_numeric: Optional[Mapping[str, Any]] = None,
    primary_toggles: Optional[Mapping[str, Any]] = None,
    primary_bounds: Optional[Mapping[str, Any]] = None,
    configuration: str = "",
    suite: str = "",
    y_value: str = "",
    plot_label: str = "",
    session_dir: str = "",
) -> Dict[str, str]:
    """Fill CSV metadata with explicit values or inference from the run settings."""
    explicit = normalize_configuration_name(configuration)
    configuration_out = explicit or infer_configuration(primary_toggles, primary_bounds)
    suite = str(suite or "").strip()
    y_value = str(y_value or "").strip()
    if not suite or not y_value:
        inferred_suite, inferred_y = infer_suite_and_y(primary_numeric)
        suite = suite or inferred_suite
        y_value = y_value or inferred_y
    if not y_value and suite:
        m = re.match(r"^Fixed_(?P<y>.+)_ratio$", suite)
        if m:
            y_value = m.group("y")
    session_dir = str(session_dir or "").strip() or os.path.abspath(os.path.expanduser(folder))
    plot_label = str(plot_label or "").strip()
    if not plot_label:
        base = configuration_out or "campaign"
        plot_label = f"{base}_{session_id}"
        if suite:
            plot_label = f"{plot_label} [{suite}]"
    return {
        "configuration": configuration_out,
        "suite": suite,
        "y_value": y_value,
        "plot_label": plot_label,
        "session_dir": session_dir,
    }


def _blank_param_cells() -> Dict[str, str]:
    return {name: NAN for name in PARAM_COLUMNS}


def format_csv_param_value(value: Any) -> str:
    """Format a scalar for a PARAM_COLUMNS CSV cell."""
    if value is None:
        return NAN
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in ("nan", "none"):
            return NAN
        try:
            value = float(text)
        except ValueError:
            return NAN
    try:
        num = float(value)
    except (TypeError, ValueError):
        return NAN
    if not math.isfinite(num):
        return NAN
    if num == int(num):
        return str(int(num))
    return f"{num:g}"


def _bound_midpoint(raw: Any) -> Optional[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    lo_raw, hi_raw = raw[0], raw[1]
    try:
        lo = float("-inf") if lo_raw in ("-inf", "-Infinity", None) else float(lo_raw)
        hi = float(hi_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(hi):
        return None
    if math.isfinite(lo):
        return 0.5 * (lo + hi)
    return 0.5 * hi


def param_cells_for_csv_row(
    *,
    primary_numeric: Optional[Mapping[str, Any]] = None,
    primary_bounds: Optional[Mapping[str, Any]] = None,
    sampled_means: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """
    Build PARAM_COLUMNS cells for one batch row.

    Fixed scalars come from ``primary_numeric``. Sampled parameters use per-batch
    simulation means when provided; otherwise the midpoint of ``primary_bounds``.
    """
    cells = _blank_param_cells()
    numeric = dict(primary_numeric or {})
    bounds = dict(primary_bounds or {})
    means = dict(sampled_means or {})

    for name in PARAM_COLUMNS:
        if name in numeric:
            cells[name] = format_csv_param_value(numeric[name])
    for name in PARAM_COLUMNS:
        if name in means:
            cells[name] = format_csv_param_value(means[name])
    for name in PARAM_COLUMNS:
        if cells[name] != NAN:
            continue
        if name not in bounds:
            continue
        mid = _bound_midpoint(bounds[name])
        if mid is not None:
            cells[name] = format_csv_param_value(mid)
    return cells


def mean_param_values_for_campaign_batches(
    records: Sequence[Mapping[str, Any]],
    param_names_list: Sequence[str],
    *,
    n_runs: int,
    n_batches: int,
) -> List[Dict[str, float]]:
    """Mean parameter values per campaign batch from primary offload simulation records."""
    from gui.apps.neutral_comparison.offload import param_dict_from_offload_vector

    n_runs_i = int(n_runs)
    n_batches_i = int(n_batches)
    if n_runs_i < 1 or n_batches_i < 1:
        return []

    ordered = sorted(
        records,
        key=lambda rec: int(rec.get("run_index", rec.get("run_id", 0)) or 0),
    )
    out: List[Dict[str, float]] = []
    for batch_idx in range(n_batches_i):
        # Offload records are stored in run_index order; each batch is a contiguous n_runs slice.
        chunk = ordered[batch_idx * n_runs_i : (batch_idx + 1) * n_runs_i]
        acc: Dict[str, List[float]] = {}
        for rec in chunk:
            pv = rec.get("param_vector")
            if not isinstance(pv, list):
                continue
            decoded = param_dict_from_offload_vector(pv, param_names_list, strict=False)
            for key, val in decoded.items():
                if key not in PARAM_COLUMNS:
                    continue
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fv):
                    acc.setdefault(str(key), []).append(fv)
        out.append({k: float(np.mean(vs)) for k, vs in acc.items() if vs})
    return out


def batch_hit_count_row(
    *,
    configuration: str,
    y_value: str,
    batch_index: int,
    hit_count: int,
    param_cells: Optional[Mapping[str, str]] = None,
    plot_label: str = "",
    suite: str = "",
    session_dir: str = "",
) -> Dict[str, str]:
    """One batch row in the wide batch_hit_counts CSV schema."""
    row: Dict[str, str] = {
        "row_type": ROW_BATCH,
        "plot_label": plot_label,
        "configuration": configuration,
        "suite": suite,
        "session_dir": session_dir,
        "Y": y_value,
        "batch_index": str(int(batch_index)),
        "hit_index": NAN,
        "source_run_index": NAN,
        "hit_count": str(int(hit_count)),
        "n_hit_again": NAN,
        "hit_rate": NAN,
    }
    cells = dict(_blank_param_cells())
    if param_cells:
        for name in PARAM_COLUMNS:
            val = param_cells.get(name)
            if val is not None and str(val).strip():
                cells[name] = str(val)
    row.update(cells)
    return row


primary_batch_paper_row = batch_hit_count_row


def primary_hit_count_row(
    *,
    configuration: str,
    y_value: str,
    batch_index: int,
    hit_index: int,
    source_run_index: int,
    param_cells: Optional[Mapping[str, str]] = None,
    plot_label: str = "",
    suite: str = "",
    session_dir: str = "",
) -> Dict[str, str]:
    """One primary-hit row in the session hit-counts CSV schema."""
    row: Dict[str, str] = {
        "plot_label": plot_label,
        "configuration": configuration,
        "suite": suite,
        "session_dir": session_dir,
        "Y": y_value,
        "batch_index": str(int(batch_index)),
        "hit_index": str(int(hit_index)),
        "source_run_index": str(int(source_run_index)),
        "n_seeds": NAN,
        "n_hit_again": NAN,
    }
    cells = dict(_blank_param_cells())
    if param_cells:
        for name in PARAM_COLUMNS:
            val = param_cells.get(name)
            if val is not None and str(val).strip():
                cells[name] = str(val)
    row.update(cells)
    return row


def _param_vector_key(vec: Sequence[Any], *, decimals: int = 8) -> str:
    parts: List[str] = []
    for x in vec:
        try:
            v = float(x)
            parts.append(f"{round(v, decimals):.{decimals}f}" if math.isfinite(v) else "nan")
        except (TypeError, ValueError):
            parts.append("nan")
    return "|".join(parts)


def build_primary_hit_csv_rows(
    *,
    offload_records: Sequence[Mapping[str, Any]],
    param_names_list: Sequence[str],
    n_runs: int,
    configuration: str,
    y_value: str,
    plot_label: str = "",
    suite: str = "",
    session_dir: str = "",
    primary_numeric: Optional[Mapping[str, Any]] = None,
    primary_bounds: Optional[Mapping[str, Any]] = None,
    dedupe_params: bool = True,
) -> List[Dict[str, str]]:
    """
    Build one CSV row per primary hit, using the same hit_index assignment as
    ``collect_hit_specs`` (deduped param vectors by default).
    """
    from gui.apps.neutral_comparison.offload import param_dict_from_offload_vector
    from gui.apps.neutral_comparison.primary_event_chart import _primary_hit_from_record

    runs = max(1, int(n_runs))
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()
    for rec in offload_records:
        if not isinstance(rec, dict):
            continue
        if _primary_hit_from_record(rec) is not True:
            continue
        vec = rec.get("param_vector")
        if not isinstance(vec, list) or not vec:
            continue
        key = _param_vector_key(vec)
        if dedupe_params and key in seen:
            continue
        seen.add(key)
        try:
            run_index = int(rec.get("run_index") if rec.get("run_index") is not None else rec.get("run_id") or 0)
        except (TypeError, ValueError):
            run_index = len(rows)
        decoded = param_dict_from_offload_vector(vec, param_names_list, strict=False)
        param_cells = param_cells_for_csv_row(
            primary_numeric=primary_numeric,
            primary_bounds=primary_bounds,
            sampled_means=decoded,
        )
        # hit_index assignment matches collect_hit_specs when dedupe_params is the same.
        rows.append(
            primary_hit_count_row(
                configuration=configuration,
                y_value=y_value,
                batch_index=_batch_index_for_source_run(run_index, runs),
                hit_index=len(rows),
                source_run_index=run_index,
                param_cells=param_cells,
                plot_label=plot_label,
                suite=suite,
                session_dir=session_dir,
            )
        )
    return rows


def write_session_hit_counts_csv(
    *,
    folder: str,
    session_id: str,
    hit_counts: Sequence[int],
    primary_numeric: Optional[Mapping[str, Any]] = None,
    primary_toggles: Optional[Mapping[str, Any]] = None,
    primary_bounds: Optional[Mapping[str, Any]] = None,
    configuration: str = "",
    suite: str = "",
    y_value: str = "",
    plot_label: str = "",
    session_dir: str = "",
    param_names_list: Optional[Sequence[str]] = None,
    offload_records: Optional[Sequence[Mapping[str, Any]]] = None,
    n_runs: Optional[int] = None,
) -> Optional[str]:
    """
    Write one CSV row per primary hit for one session.

    Writes ``batch_hit_counts_<session>.csv`` (or a configuration-slug variant)
    with campaign metadata, hit identity columns, and parameter values.
    """
    if not hit_counts:
        return None
    # Per-hit rows require offload records so hit_index / source_run_index can be assigned.
    if not offload_records or not param_names_list or n_runs is None:
        return None
    out_dir = os.path.abspath(os.path.expanduser(folder))
    meta = resolve_paper_campaign_metadata(
        session_id=session_id,
        folder=out_dir,
        primary_numeric=primary_numeric,
        primary_toggles=primary_toggles,
        primary_bounds=primary_bounds,
        configuration=configuration,
        suite=suite,
        y_value=y_value,
        plot_label=plot_label,
        session_dir=session_dir,
    )
    rows = build_primary_hit_csv_rows(
        offload_records=offload_records,
        param_names_list=param_names_list,
        n_runs=int(n_runs),
        configuration=meta["configuration"],
        y_value=meta["y_value"],
        plot_label=meta["plot_label"],
        suite=meta["suite"],
        session_dir=meta["session_dir"],
        primary_numeric=primary_numeric,
        primary_bounds=primary_bounds,
    )
    if not rows:
        return None
    fname = session_hit_counts_csv_basename(session_id, configuration=meta["configuration"])
    out_abs = os.path.join(out_dir, fname)
    try:
        with open(out_abs, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(PAPER_HIT_COUNT_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        return out_abs
    except OSError:
        return None


def rescreen_row_type_for_screen_mode(screen_mode: str) -> str:
    mode = str(screen_mode or "").strip().lower()
    if mode in ("non_hits", "non-hits", "nonhits"):
        return ROW_RESCREEN_NON_HIT
    return ROW_RESCREEN_HIT


def _is_batch_summary_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("row_type") or "") in _LEGACY_BATCH_ROW_TYPES


def _is_data_hit_row(row: Mapping[str, Any]) -> bool:
    """True for primary-hit data rows (hit-only schema or ``row_type=hit``)."""
    row_type = str(row.get("row_type") or "").strip()
    if row_type in _RESCREEN_EXTRA_ROW_TYPES:
        return False
    if row_type in _LEGACY_BATCH_ROW_TYPES:
        return False
    return _rescreen_point_key(row) is not None


def _is_hit_row(row: Mapping[str, Any]) -> bool:
    return _is_data_hit_row(row)


def _is_nan_cell(value: Any) -> bool:
    # Only empty cells and the literal "nan" count as missing; hit_index=0 is valid.
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or text == "nan"


def _rescreen_point_key(row: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    if _is_nan_cell(row.get("hit_index")) or _is_nan_cell(row.get("source_run_index")):
        return None
    try:
        return int(row["hit_index"]), int(row["source_run_index"])
    except (TypeError, ValueError, KeyError):
        return None


def aggregate_rescreen_stats_by_batch(
    rescreen_rows: Sequence[Mapping[str, Any]],
    n_runs: int,
) -> Dict[int, Tuple[int, float]]:
    """Map batch_index -> (total n_hit_again, pooled hit_rate) for re-screen rows."""
    buckets: Dict[int, Dict[str, int]] = {}
    for row in rescreen_rows:
        batch_idx = _batch_index_for_source_run(row.get("source_run_index"), n_runs)
        bucket = buckets.setdefault(batch_idx, {"n_hit_again": 0, "n_trials": 0})
        try:
            bucket["n_hit_again"] += int(row.get("n_hit_again") or 0)
        except (TypeError, ValueError):
            pass
        try:
            seeds = int(row.get("n_seeds") or 0)
        except (TypeError, ValueError):
            seeds = 0
        if seeds > 0:
            bucket["n_trials"] += seeds
    out: Dict[int, Tuple[int, float]] = {}
    for batch_idx, bucket in buckets.items():
        trials = int(bucket["n_trials"])
        again = int(bucket["n_hit_again"])
        rate = float(again) / float(trials) if trials else 0.0
        out[int(batch_idx)] = (again, rate)
    return out


def index_rescreen_rows_by_point(
    rescreen_rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, int], Mapping[str, Any]]:
    out: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for row in rescreen_rows:
        key = _rescreen_point_key(row)
        if key is not None:
            out[key] = row
    return out


def _apply_rescreen_stats_to_row(
    row: Dict[str, str],
    *,
    n_seeds: int,
    n_hit_again: int,
) -> None:
    row["n_seeds"] = str(int(n_seeds))
    row["n_hit_again"] = str(int(n_hit_again))
    # Session CSV uses n_seeds/n_hit_again instead of per-row hit_rate.
    row.pop("hit_rate", None)


def _clear_rescreen_stats(row: Dict[str, str]) -> None:
    row["n_seeds"] = NAN
    row["n_hit_again"] = NAN
    row.pop("hit_rate", None)


def _rescreen_stats_from_row(rr: Mapping[str, Any]) -> Tuple[int, int]:
    try:
        n_seeds = int(rr.get("n_seeds") or 0)
    except (TypeError, ValueError):
        n_seeds = 0
    try:
        again = int(rr.get("n_hit_again") or 0)
    except (TypeError, ValueError):
        again = 0
    return n_seeds, again


def _param_cells_from_rescreen_row(rr: Mapping[str, Any]) -> Dict[str, str]:
    cells = _blank_param_cells()
    raw = rr.get("params_json")
    if not raw:
        return cells
    try:
        import json

        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return cells
    if not isinstance(decoded, dict):
        return cells
    for name in PARAM_COLUMNS:
        if name in decoded:
            cells[name] = format_csv_param_value(decoded[name])
    return cells


def _new_hit_row_from_rescreen(
    rr: Mapping[str, Any],
    *,
    template: Optional[Mapping[str, Any]],
    n_runs: int,
) -> Dict[str, str]:
    """Create a hit row when the session CSV has no row for this point."""
    tmpl = dict(template or {})
    try:
        hit_index = int(rr.get("hit_index") or 0)
    except (TypeError, ValueError):
        hit_index = 0
    try:
        source_run_index = int(rr.get("source_run_index") or 0)
    except (TypeError, ValueError):
        source_run_index = 0
    n_seeds, again = _rescreen_stats_from_row(rr)
    row = primary_hit_count_row(
        configuration=str(tmpl.get("configuration") or ""),
        y_value=str(tmpl.get("Y") or ""),
        batch_index=_batch_index_for_source_run(source_run_index, n_runs),
        hit_index=hit_index,
        source_run_index=source_run_index,
        param_cells=_param_cells_from_rescreen_row(rr),
        plot_label=str(tmpl.get("plot_label") or ""),
        suite=str(tmpl.get("suite") or ""),
        session_dir=str(tmpl.get("session_dir") or ""),
    )
    _apply_rescreen_stats_to_row(row, n_seeds=n_seeds, n_hit_again=again)
    return row


def resolve_session_hit_counts_csv_path(session_dir: str, session_id: str = "") -> Optional[str]:
    """Locate the batch hit-counts CSV written for a finished campaign session."""
    root = os.path.abspath(os.path.expanduser(session_dir))
    sid = str(session_id or "").strip()
    try:
        from headless.primary_batch_session_meta import load_campaign_summary

        campaign = load_campaign_summary(root)
    except Exception:
        campaign = None
    if isinstance(campaign, dict):
        name = str(campaign.get("hit_counts_csv") or "").strip()
        if name:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
        if not sid:
            sid = str(campaign.get("full_save_session_id") or "").strip()
    patterns: List[str] = []
    if sid:
        patterns.extend(
            (
                f"batch_hit_counts_*_{sid}.csv",
                f"batch_hit_counts_{sid}.csv",
                f"primary_batch_hit_counts_{sid}.csv",
            )
        )
    patterns.extend(("batch_hit_counts_*.csv", "primary_batch_hit_counts_*.csv"))
    for pattern in patterns:
        matches = sorted(Path(root).glob(pattern))
        if matches:
            return str(matches[0])
    return None


def _batch_index_for_source_run(source_run_index: Any, n_runs: int) -> int:
    """Map a 0-based offload run_index to a 1-based primary batch index."""
    try:
        src = int(source_run_index)
    except (TypeError, ValueError):
        return 1
    runs = max(1, int(n_runs))
    # Batch 1 covers run_index 0..n_runs-1, batch 2 covers n_runs..2*n_runs-1, etc.
    return src // runs + 1


def _campaign_n_runs(session_dir: str, session_id: str) -> int:
    try:
        from headless.primary_batch_session_meta import load_campaign_summary

        campaign = load_campaign_summary(session_dir)
        if isinstance(campaign, dict) and campaign.get("n_runs") is not None:
            return max(1, int(campaign["n_runs"]))
    except Exception:
        pass
    return 1


def update_session_hit_counts_csv_from_rescreen(
    *,
    session_dir: str,
    session_id: str,
    rescreen_rows: Sequence[Mapping[str, Any]],
    screen_mode: str,
    n_runs: Optional[int] = None,
) -> Optional[str]:
    """
    Merge re-screen results into the session hit-counts CSV.

    Updates existing rows in place (``n_seeds``, ``n_hit_again``) matched by
    ``(hit_index, source_run_index)``. Batch-summary and ``rescreen_*`` rows are
    dropped. Missing hit rows are appended from re-screen output.
    """
    if not rescreen_rows:
        return None
    csv_path = resolve_session_hit_counts_csv_path(session_dir, session_id)
    if not csv_path or not os.path.isfile(csv_path):
        return None

    with open(csv_path, newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))

    fieldnames = list(PAPER_HIT_COUNT_FIELDS)
    # Keep only per-hit rows; drop batch-summary and rescreen sidecar rows from older CSVs.
    merged = [
        dict(row)
        for row in existing
        if _is_data_hit_row(row)
    ]

    runs = max(1, int(n_runs)) if n_runs is not None else _campaign_n_runs(session_dir, session_id)
    mode = rescreen_row_type_for_screen_mode(screen_mode)
    if mode == ROW_RESCREEN_HIT:
        by_point = index_rescreen_rows_by_point(rescreen_rows)
        template = merged[0] if merged else next(
            (row for row in existing if not str(row.get("row_type") or "") in _RESCREEN_EXTRA_ROW_TYPES),
            {},
        )
        existing_hit_keys = {
            key
            for row in merged
            for key in [_rescreen_point_key(row)]
            if key is not None
        }
        # Patch n_seeds / n_hit_again on rows that already exist in the session CSV.
        for row in merged:
            point_key = _rescreen_point_key(row)
            if point_key is None or point_key not in by_point:
                continue
            n_seeds, again = _rescreen_stats_from_row(by_point[point_key])
            _apply_rescreen_stats_to_row(row, n_seeds=n_seeds, n_hit_again=again)
        # Append rows for hits that appear in re-screen output but not in the session CSV.
        for point_key, rr in by_point.items():
            if point_key in existing_hit_keys:
                continue
            merged.append(_new_hit_row_from_rescreen(rr, template=template, n_runs=runs))

    if not merged and mode != ROW_RESCREEN_HIT:
        legacy_only = [
            row for row in existing
            if str(row.get("row_type") or "") not in _RESCREEN_EXTRA_ROW_TYPES
        ]
        if not legacy_only:
            return None

    def _sort_key(row: Mapping[str, Any]) -> Tuple[int, int]:
        try:
            batch_idx = int(row.get("batch_index") or 0)
        except (TypeError, ValueError):
            batch_idx = 0
        try:
            hit_idx = int(row.get("hit_index") or 0)
        except (TypeError, ValueError):
            hit_idx = 0
        if _is_nan_cell(row.get("hit_index")):
            hit_idx = 0
        return batch_idx, hit_idx

    merged.sort(key=_sort_key)
    _write_hit_counts_csv_atomic(csv_path, fieldnames, merged)

    # Some sessions also keep a generic batch_hit_counts.csv copy beside the slugged filename.
    legacy_copy = os.path.join(os.path.dirname(csv_path), "batch_hit_counts.csv")
    if legacy_copy != csv_path and os.path.isfile(legacy_copy):
        _write_hit_counts_csv_atomic(legacy_copy, fieldnames, merged)

    return csv_path if merged else None


def _write_hit_counts_csv_atomic(path: str, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".batch_hit_counts_",
        suffix=".csv",
        dir=os.path.dirname(out_path) or ".",
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames if name != "hit_rate"})
        os.replace(tmp_path, out_path)
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
