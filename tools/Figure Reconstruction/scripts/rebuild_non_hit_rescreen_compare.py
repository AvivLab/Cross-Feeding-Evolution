#!/usr/bin/env python3
"""Rebuild Re-Runs-NonHits/primary_non_hit_rescreen_compare.csv from session folders."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from plot_hit_rescreen_panel import workspaces_non_hit_re_runs
from plot_primary_batch_violins import workspaces_output

PREFIX_TO_KEY = {
    "a": "Neutral",
    "aa": "trueNeutral2",
    "b": "binary_death",
    "c": "justDeath",
    "d": "justDup",
    "e": "Death+Dup",
}

COMPARE_FIELDS = [
    "session_id",
    "plot_label",
    "config_key",
    "suite_tag",
    "n_hits_screened",
    "n_seeds_per_hit",
    "mean_hit_rate",
    "std_hit_rate",
    "median_hit_rate",
    "hits_always",
    "hits_never",
    "csv_path",
    "json_path",
]


def parse_session_dir(name: str) -> Tuple[str, str, str, str] | None:
    m = re.fullmatch(r"(a|aa|b|c|d|e)_(.+)", name)
    if not m:
        return None
    prefix, rest = m.group(1), m.group(2)
    config_key = PREFIX_TO_KEY[prefix]
    if "_Fixed_" in rest:
        _, y_part = rest.split("_Fixed_", 1)
        suite_tag = f"Fixed_{y_part}_ratio"
        plot_label = f"{config_key}_Fixed_{y_part} [{suite_tag}]"
    else:
        suite_tag = "Variable_Ratio"
        plot_label = f"{config_key} [{suite_tag}]"
    return name, config_key, suite_tag, plot_label


def summarize_session_csv(csv_path: Path) -> Tuple[int, float, float, float, int, int, int]:
    rates: List[float] = []
    n_seeds_per_hit = 1
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rates.append(float(row["hit_rate"]))
            if row.get("n_seeds"):
                n_seeds_per_hit = int(row["n_seeds"])
    n = len(rates)
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 0, 0, n_seeds_per_hit
    mean_rate = sum(rates) / n
    std_rate = statistics.stdev(rates) if n > 1 else 0.0
    median_rate = statistics.median(rates)
    hits_always = sum(1 for r in rates if r >= 1.0 - 1e-12)
    hits_never = sum(1 for r in rates if r <= 1e-12)
    return n, mean_rate, std_rate, median_rate, hits_always, hits_never, n_seeds_per_hit


def rebuild_compare_rows(sessions_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not sessions_dir.is_dir():
        return rows
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_session_dir(session_dir.name)
        if parsed is None:
            continue
        session_id, config_key, suite_tag, plot_label = parsed
        csvs = sorted(session_dir.glob("primary_non_hit_rescreen_*.csv"))
        if not csvs:
            continue
        csv_path = csvs[0]
        n, mean_rate, std_rate, median_rate, hits_always, hits_never, n_seeds = summarize_session_csv(
            csv_path
        )
        rows.append(
            {
                "session_id": session_id,
                "plot_label": plot_label,
                "config_key": config_key,
                "suite_tag": suite_tag,
                "n_hits_screened": str(n),
                "n_seeds_per_hit": str(n_seeds),
                "mean_hit_rate": f"{mean_rate:.4f}",
                "std_hit_rate": f"{std_rate:.4f}",
                "median_hit_rate": f"{median_rate:.4f}",
                "hits_always": str(hits_always),
                "hits_never": str(hits_never),
                "csv_path": str(csv_path.resolve()),
                "json_path": "",
            }
        )
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=workspaces_non_hit_re_runs() / "sessions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspaces_non_hit_re_runs() / "primary_non_hit_rescreen_compare.csv",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = rebuild_compare_rows(args.sessions_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMPARE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output root: {workspaces_output()}")
    print(f"Sessions: {args.sessions_dir}")
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
