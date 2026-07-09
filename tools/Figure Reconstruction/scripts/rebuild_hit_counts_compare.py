#!/usr/bin/env python3
"""Rebuild Summary compare CSVs from per-session primary_batch_hit_counts_*.csv files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from plot_primary_batch_violins import workspaces_output

PREFIX_TO_KEY = {
    "a": "Neutral",
    "aa": "trueNeutral2",
    "b": "binary_death",
    "c": "justDeath",
    "d": "justDup",
    "e": "Death+Dup",
}

_HIT_COUNTS_RE = re.compile(
    r"^primary_batch_hit_counts_(?P<prefix>a|aa|b|c|d|e)_(?P<rest>.+)\.csv$"
)
_JOB_DIR_RE = re.compile(r"^job(?P<job_id>\d+)-")

SUITE_DIRS = (
    "Fixed_0.1_ratio",
    "Fixed_0.5_ratio",
    "Fixed_1_ratio",
    "Fixed_3_ratio",
    "Fixed_5_ratio",
    "Fixed_7_ratio",
    "Fixed_10_ratio",
    "Fixed_20_ratio",
    "Variable_Ratio",
)


def _job_sort_key(session_dir: Path) -> Tuple[int, str]:
    m = _JOB_DIR_RE.match(session_dir.name)
    if m:
        return int(m.group("job_id")), session_dir.name
    return 0, session_dir.name


def _load_hit_counts(csv_path: Path) -> List[int]:
    counts: List[int] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            counts.append(int(row["hit_count"]))
    if not counts:
        raise ValueError(f"no hit counts in {csv_path}")
    return counts


def collect_campaigns(output_root: Path) -> List[Tuple[str, str, str, str, List[int]]]:
    """Return rows as (plot_label, configuration, suite, session_dir, hit_counts)."""
    best: Dict[Tuple[str, str], Tuple[Path, str, List[int]]] = {}
    for suite in SUITE_DIRS:
        suite_dir = output_root / suite
        if not suite_dir.is_dir():
            continue
        for session_dir in suite_dir.iterdir():
            if not session_dir.is_dir() or not session_dir.name.startswith("job"):
                continue
            hit_csvs = sorted(session_dir.glob("primary_batch_hit_counts_*.csv"))
            if not hit_csvs:
                continue
            hit_csv = hit_csvs[0]
            m = _HIT_COUNTS_RE.match(hit_csv.name)
            if not m:
                continue
            prefix = m.group("prefix")
            configuration = PREFIX_TO_KEY[prefix]
            plot_label = f"{m.group('rest')} [{suite}]"
            key = (suite, configuration)
            candidate = (session_dir, plot_label, _load_hit_counts(hit_csv))
            prev = best.get(key)
            if prev is None or _job_sort_key(session_dir) > _job_sort_key(prev[0]):
                best[key] = candidate

    rows: List[Tuple[str, str, str, str, List[int]]] = []
    for suite in SUITE_DIRS:
        for configuration in PREFIX_TO_KEY.values():
            entry = best.get((suite, configuration))
            if entry is None:
                continue
            session_dir, plot_label, counts = entry
            rows.append((plot_label, configuration, suite, str(session_dir.resolve()), counts))
    return rows


def write_compare_csv(
    path: Path,
    campaigns: Sequence[Tuple[str, str, str, str, List[int]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["plot_label", "configuration", "suite", "session_dir", "batch_index", "hit_count"]
        )
        for plot_label, configuration, suite, session_dir, counts in campaigns:
            for batch_idx, hit_count in enumerate(counts, start=1):
                writer.writerow(
                    [plot_label, configuration, suite, session_dir, batch_idx, int(hit_count)]
                )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspaces_output(),
        help="Workspaces/Output root",
    )
    parser.add_argument(
        "--write-ratio",
        type=Path,
        default=None,
        help="Summary_Ratio CSV path (default: <output-root>/Summary/Summary_Ratio/...)",
    )
    parser.add_argument(
        "--write-simple",
        type=Path,
        default=None,
        help="Summary_Simple CSV path (default: <output-root>/Summary/Summary_Simple/...)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_root = args.output_root.resolve()
    campaigns = collect_campaigns(output_root)
    if not campaigns:
        raise SystemExit(f"No primary_batch_hit_counts_*.csv found under {output_root}")

    ratio_path = args.write_ratio or (
        output_root / "Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv"
    )
    simple_path = args.write_simple or (
        output_root / "Summary/Summary_Simple/primary_batch_compare_hit_counts.csv"
    )
    write_compare_csv(ratio_path, campaigns)
    simple_campaigns = [row for row in campaigns if row[1] != "trueNeutral2"]
    write_compare_csv(simple_path, simple_campaigns)

    print(f"Output root: {output_root}")
    print(f"Campaigns: {len(campaigns)}")
    print(f"Wrote: {ratio_path}")
    print(f"Wrote: {simple_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
