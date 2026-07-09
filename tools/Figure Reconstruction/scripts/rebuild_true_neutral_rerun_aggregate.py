#!/usr/bin/env python3
"""Combine neutral primary-hit re-screen CSVs into one flat table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from plot_primary_batch_violins import workspaces_output

REScreen_FIELDS = (
    "session_id",
    "suite_tag",
    "hit_index",
    "source_run_index",
    "original_seed",
    "param_key",
    "hit_rate",
    "seeds_tested",
    "trial_hits",
    "total_deaths",
    "total_duplications",
    "total_mutations",
    "total_outflows",
    "n_seeds",
    "n_hit_again",
)

PARAM_FIELDS = (
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
)

OUTPUT_FIELDS = REScreen_FIELDS[:2] + PARAM_FIELDS + REScreen_FIELDS[2:]


def workspaces_re_runs() -> Path:
    return workspaces_output() / "Re-Runs"


def default_output_path() -> Path:
    return workspaces_output() / "Neutral" / "true_neutral_hit_rescreen_all_ratios.csv"


def default_sorted_output_path() -> Path:
    return (
        workspaces_output()
        / "Neutral"
        / "true_neutral_hit_rescreen_all_ratios_sorted.csv"
    )


def parse_true_neutral_session(name: str) -> Tuple[str, str] | None:
    """Return (session_id, suite_tag) for ``a_trueNeutral*`` folders."""
    if not name.startswith("a_trueNeutral") or name.startswith("aa_"):
        return None
    if "_Fixed_" in name:
        y_part = name.split("_Fixed_", 1)[1]
        return name, f"Fixed_{y_part}_ratio"
    return name, "Variable_Ratio"


def load_session_rows(session_dir: Path, *, suite_tag: str) -> List[Dict[str, str]]:
    csvs = sorted(session_dir.glob("primary_hit_rescreen_*.csv"))
    if not csvs:
        return []
    rows: List[Dict[str, str]] = []
    with csvs[0].open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            params = json.loads(row["params_json"])
            flat = {
                "session_id": row["session_id"],
                "suite_tag": suite_tag,
                "hit_index": row["hit_index"],
                "source_run_index": row.get("source_run_index", ""),
                "original_seed": row.get("original_seed", ""),
                "param_key": row.get("param_key", ""),
                "n_seeds": row["n_seeds"],
                "n_hit_again": row["n_hit_again"],
                "hit_rate": row["hit_rate"],
                "seeds_tested": row.get("seeds_tested", ""),
                "trial_hits": row.get("trial_hits", ""),
                "total_deaths": row.get("total_deaths", ""),
                "total_duplications": row.get("total_duplications", ""),
                "total_mutations": row.get("total_mutations", ""),
                "total_outflows": row.get("total_outflows", ""),
            }
            for name in PARAM_FIELDS:
                flat[name] = params.get(name, "")
            rows.append(flat)
    return rows


def write_rows(output_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def rebuild_true_neutral_rerun_aggregate(
    *,
    sessions_dir: Path,
    output_path: Path,
    sorted_output_path: Path,
) -> int:
    all_rows: List[Dict[str, str]] = []
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_true_neutral_session(session_dir.name)
        if parsed is None:
            continue
        _, suite_tag = parsed
        all_rows.extend(load_session_rows(session_dir, suite_tag=suite_tag))

    write_rows(output_path, all_rows)
    sorted_rows = sorted(
        all_rows,
        key=lambda row: int(row["n_hit_again"]),
        reverse=True,
    )
    write_rows(sorted_output_path, sorted_rows)
    return len(all_rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=workspaces_re_runs() / "sessions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="Combined CSV path (default: Workspaces/Output/Neutral/true_neutral_hit_rescreen_all_ratios.csv)",
    )
    parser.add_argument(
        "--sorted-output",
        type=Path,
        default=default_sorted_output_path(),
        help="Sorted CSV path (default: .../true_neutral_hit_rescreen_all_ratios_sorted.csv)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    n_rows = rebuild_true_neutral_rerun_aggregate(
        sessions_dir=args.sessions_dir,
        output_path=args.output,
        sorted_output_path=args.sorted_output,
    )
    print(f"Sessions dir: {args.sessions_dir}")
    print(f"Wrote: {args.output} ({n_rows:,} rows)")
    print(f"Wrote: {args.sorted_output} ({n_rows:,} rows, sorted by n_hit_again desc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
