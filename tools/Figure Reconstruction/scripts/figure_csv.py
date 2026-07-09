#!/usr/bin/env python3
"""Load paper figure tables from data/figure_reproduction/batch_hit_counts.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from plot_hit_rescreen_panel import BATCHES_PER_CAMPAIGN, N_SIMS_PER_BATCH
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
    sims_per_batch: int = N_SIMS_PER_BATCH,
    n_batches: int = BATCHES_PER_CAMPAIGN,
) -> Dict[str, Dict[str, List[float]]]:
    """Fig. 3 panel (b): suite -> configuration -> per-batch mean re-screen rates."""
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
        src = row.get("source_run_index")
        rate = row.get("hit_rate")
        if not src or not rate or rate == "nan":
            continue
        suite = _suite_tag(y)
        batch_index = int(src) // int(sims_per_batch)
        batches[suite][configuration][batch_index].append(float(rate))

    out: Dict[str, Dict[str, List[float]]] = {}
    for suite, by_config in batches.items():
        out[suite] = {}
        for configuration, by_batch in by_config.items():
            out[suite][configuration] = [
                (sum(by_batch[bi]) / len(by_batch[bi])) if by_batch.get(bi) else 0.0
                for bi in range(n_batches)
            ]
    return out


def load_rescreen_batch_counts_by_suite(
    csv_path: Path,
    *,
    row_type: str,
    sims_per_batch: int = N_SIMS_PER_BATCH,
    n_batches: int = BATCHES_PER_CAMPAIGN,
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Ridgelines: suite -> configuration -> per-batch n_hit_again lists."""
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
        src = row.get("source_run_index")
        n_hit = row.get("n_hit_again")
        if not src or not n_hit or n_hit == "nan":
            continue
        suite = _suite_tag(y)
        batch_index = int(src) // int(sims_per_batch)
        batches[suite][configuration][batch_index].append(int(n_hit))

    out: Dict[str, Dict[str, List[List[int]]]] = {}
    for suite, by_config in batches.items():
        out[suite] = {}
        for configuration, by_batch in by_config.items():
            out[suite][configuration] = [
                by_batch.get(bi, []) for bi in range(n_batches)
            ]
    return out
