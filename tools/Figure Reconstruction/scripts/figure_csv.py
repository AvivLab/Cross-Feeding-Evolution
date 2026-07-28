#!/usr/bin/env python3
"""Load paper figure tables from data/figure_reproduction/batch_hit_counts.csv."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from plot_hit_rescreen_panel import BATCHES_PER_CAMPAIGN, N_RESCREEN, N_SIMS_PER_BATCH
from plot_primary_batch_violins import PAPER_CONFIG_ORDER
from plot_ratio_supplementary import SUITE_ORDER

FIGURE_DATA_CSV = Path(__file__).resolve().parents[1] / "data" / "figure_reproduction" / "batch_hit_counts.csv"

ROW_PRIMARY_BATCH = "primary_batch"
ROW_HIT = "hit"
ROW_NON_HIT = "non_hit"

PAPER_CONFIG_KEYS = frozenset(key for key, _, _ in PAPER_CONFIG_ORDER)
PAPER_Y_VALS = frozenset(y for _, y, _ in SUITE_ORDER)


def _suite_tag(y: str) -> str:
    y_f = float(y)
    if y_f == int(y_f):
        return f"Fixed_{int(y_f)}_ratio"
    return f"Fixed_{y}_ratio"


def _y_in_paper(y: str) -> bool:
    try:
        return float(y) in PAPER_Y_VALS
    except ValueError:
        return False


def _read_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _coerce_int(raw: object) -> int | None:
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    return value


def _mode_positive(values: Sequence[int], default: int) -> int:
    vals = [int(v) for v in values if int(v) > 0]
    if not vals:
        return int(default)
    return int(Counter(vals).most_common(1)[0][0])


def infer_n_sims_from_rows(csv_path: Path) -> int:
    """Infer simulations-per-batch from assembled or Summary compare CSV."""
    vals: List[int] = []
    for row in _read_rows(csv_path):
        for key in ("n_sims_per_batch", "n_runs"):
            value = _coerce_int(row.get(key))
            if value is not None and value > 0:
                vals.append(value)
                break
    return _mode_positive(vals, N_SIMS_PER_BATCH)


def infer_campaign_shape(csv_path: Path) -> Tuple[int, int, int]:
    """Return ``(n_sims_per_batch, n_batches, n_rescreen)`` from assembled CSV columns."""
    rows = _read_rows(csv_path)
    n_sims_vals: List[int] = []
    n_rescreen_vals: List[int] = []
    batch_counts: Dict[Tuple[str, str], set[int]] = defaultdict(set)

    for row in rows:
        n_sims = _coerce_int(row.get("n_sims_per_batch") or row.get("n_runs"))
        if n_sims is not None and n_sims > 0:
            n_sims_vals.append(n_sims)
        n_rescreen = _coerce_int(row.get("n_rescreen") or row.get("n_seeds"))
        if n_rescreen is not None and n_rescreen > 0:
            n_rescreen_vals.append(n_rescreen)
        if row.get("row_type") == ROW_PRIMARY_BATCH:
            configuration = str(row.get("configuration") or "")
            y = str(row.get("Y") or "")
            batch_idx = _coerce_int(row.get("batch_index"))
            if configuration and y and batch_idx is not None:
                batch_counts[(configuration, y)].add(batch_idx)
        n_hit = _coerce_int(row.get("n_hit_again"))
        # Fallback when n_rescreen column is missing: use max observed re-hit count.
        if n_hit is not None and n_hit > 0 and not n_rescreen_vals:
            n_rescreen_vals.append(n_hit)

    n_sims = _mode_positive(n_sims_vals, N_SIMS_PER_BATCH)
    if n_rescreen_vals:
        # Explicit n_rescreen/n_seeds preferred; otherwise max observed n_hit_again.
        explicit = [
            v
            for r in rows
            for v in [_coerce_int(r.get("n_rescreen") or r.get("n_seeds"))]
            if v is not None and v > 0
        ]
        n_rescreen = _mode_positive(explicit, max(n_rescreen_vals)) if explicit else max(n_rescreen_vals)
    else:
        n_rescreen = N_RESCREEN
    if batch_counts:
        n_batches = _mode_positive([len(v) for v in batch_counts.values()], BATCHES_PER_CAMPAIGN)
    else:
        n_batches = BATCHES_PER_CAMPAIGN
    return n_sims, n_batches, int(n_rescreen)


def _batch_index_for_row(row: dict, *, sims_per_batch: int) -> int:
    src = row.get("source_run_index")
    if src and str(src).strip() and str(src).strip().lower() != "nan":
        try:
            return int(src) // max(1, int(sims_per_batch))
        except (TypeError, ValueError):
            pass
    batch_idx = _coerce_int(row.get("batch_index"))
    if batch_idx is None:
        return 0
    return int(batch_idx)


def load_by_suite(csv_path: Path) -> Dict[str, Dict[str, List[int]]]:
    """Fig. 3 panel (a): suite -> configuration -> batch hit counts."""
    out: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for row in _read_rows(csv_path):
        if row.get("row_type") != ROW_PRIMARY_BATCH:
            continue
        configuration = row["configuration"]
        y = row["Y"]
        if configuration not in PAPER_CONFIG_KEYS or not _y_in_paper(y):
            continue
        out[_suite_tag(y)][configuration].append(int(row["hit_count"]))
    return {suite: dict(cfgs) for suite, cfgs in out.items()}


def load_rescreen_batch_rates_by_suite(
    csv_path: Path,
    *,
    sims_per_batch: int | None = None,
    n_batches: int | None = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Fig. 3 panel (b): suite -> configuration -> per-batch mean re-screen rates."""
    if sims_per_batch is None or n_batches is None:
        inferred_sims, inferred_batches, _ = infer_campaign_shape(csv_path)
        sims_per_batch = inferred_sims if sims_per_batch is None else sims_per_batch
        n_batches = inferred_batches if n_batches is None else n_batches

    batches: Dict[str, Dict[str, Dict[int, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in _read_rows(csv_path):
        if row.get("row_type") != ROW_HIT:
            continue
        configuration = row["configuration"]
        y = row["Y"]
        if configuration not in PAPER_CONFIG_KEYS or not _y_in_paper(y):
            continue
        rate = row.get("hit_rate")
        if not rate or rate == "nan":
            continue
        suite = _suite_tag(y)
        batch_index = _batch_index_for_row(row, sims_per_batch=int(sims_per_batch))
        batches[suite][configuration][batch_index].append(float(rate))

    out: Dict[str, Dict[str, List[float]]] = {}
    for suite, by_config in batches.items():
        out[suite] = {}
        for configuration, by_batch in by_config.items():
            out[suite][configuration] = [
                (sum(by_batch[bi]) / len(by_batch[bi])) if by_batch.get(bi) else 0.0
                for bi in range(int(n_batches))
            ]
    return out


def load_rescreen_batch_counts_by_suite(
    csv_path: Path,
    *,
    row_type: str,
    sims_per_batch: int | None = None,
    n_batches: int | None = None,
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Ridgelines: suite -> configuration -> per-batch n_hit_again lists."""
    if sims_per_batch is None or n_batches is None:
        inferred_sims, inferred_batches, _ = infer_campaign_shape(csv_path)
        sims_per_batch = inferred_sims if sims_per_batch is None else sims_per_batch
        n_batches = inferred_batches if n_batches is None else n_batches

    batches: Dict[str, Dict[str, Dict[int, List[int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in _read_rows(csv_path):
        if row.get("row_type") != row_type:
            continue
        configuration = row["configuration"]
        y = row["Y"]
        if configuration not in PAPER_CONFIG_KEYS or not _y_in_paper(y):
            continue
        n_hit = row.get("n_hit_again")
        if not n_hit or n_hit == "nan":
            continue
        suite = _suite_tag(y)
        batch_index = _batch_index_for_row(row, sims_per_batch=int(sims_per_batch))
        batches[suite][configuration][batch_index].append(int(n_hit))

    out: Dict[str, Dict[str, List[List[int]]]] = {}
    for suite, by_config in batches.items():
        out[suite] = {}
        for configuration, by_batch in by_config.items():
            out[suite][configuration] = [by_batch.get(bi, []) for bi in range(int(n_batches))]
    return out
