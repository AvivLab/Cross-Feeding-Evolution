#!/usr/bin/env python3
"""Assemble data/figure_reproduction/batch_hit_counts.csv for paper figure plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, Sequence

from figure_param_specs import ALL_CONFIG_KEYS, PARAM_SPECS
from plot_hit_rescreen_panel import (
    N_SIMS_PER_BATCH,
    parse_rescreen_session_dir,
    workspaces_non_hit_re_runs,
    workspaces_re_runs,
)
from plot_primary_batch_violins import PAPER_CONFIG_ORDER, workspaces_output
from plot_ratio_supplementary import MIN_FIXED_Y, SUITE_ORDER

PAPER_CONFIG_KEYS = frozenset(key for key, _, _ in PAPER_CONFIG_ORDER)
PAPER_SUITE_TAGS = frozenset(suite for suite, _, _ in SUITE_ORDER)
_SUITE_Y_RE = re.compile(r"^Fixed_(?P<y>.+)_ratio$")

ROW_PRIMARY_BATCH = "primary_batch"
ROW_HIT = "hit"
ROW_NON_HIT = "non_hit"

PARAM_COLUMNS: Sequence[str] = (
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

CSV_FIELDS: Sequence[str] = (
    "row_type",
    "configuration",
    "Y",
    "batch_index",
    "hit_index",
    "source_run_index",
    "hit_count",
    "n_hit_again",
    "hit_rate",
    *PARAM_COLUMNS,
)

NAN = "nan"

PARAM_APPLICABILITY: Dict[str, FrozenSet[str]] = {
    name: configs for name, _, _, configs in PARAM_SPECS
}
for name in PARAM_COLUMNS:
    PARAM_APPLICABILITY.setdefault(name, ALL_CONFIG_KEYS)

LEGACY_DATA_FILES = (
    "hit_rescreen.csv",
    "non_hit_hist.csv",
    "figure_data.xlsx",
)


def _suite_y(suite: str) -> str:
    m = _SUITE_Y_RE.match(suite)
    if not m:
        return ""
    return m.group("y")


def _paper_y(y_val: str) -> bool:
    try:
        return float(y_val) >= MIN_FIXED_Y
    except ValueError:
        return False


def _paper_campaign(configuration: str, suite: str) -> bool:
    y = _suite_y(suite)
    return configuration in PAPER_CONFIG_KEYS and suite in PAPER_SUITE_TAGS and _paper_y(y)


def _batch_index(source_run_index: str) -> str:
    return str(int(source_run_index) // N_SIMS_PER_BATCH)


def _blank_params(configuration: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in PARAM_COLUMNS:
        if configuration not in PARAM_APPLICABILITY[name]:
            out[name] = NAN
        else:
            out[name] = NAN
    return out


def _expand_params(params_json: str, configuration: str) -> Dict[str, str]:
    try:
        params = json.loads(params_json)
    except (json.JSONDecodeError, TypeError):
        params = {}
    out: Dict[str, str] = {}
    for name in PARAM_COLUMNS:
        if configuration not in PARAM_APPLICABILITY[name]:
            out[name] = NAN
            continue
        raw = params.get(name)
        out[name] = NAN if raw is None or raw == "" else str(raw)
    return out


def _base_row(
    *,
    row_type: str,
    configuration: str,
    y_val: str,
    batch_idx: str,
) -> Dict[str, str]:
    row: Dict[str, str] = {
        "row_type": row_type,
        "configuration": configuration,
        "Y": y_val,
        "batch_index": batch_idx,
        "hit_index": NAN,
        "source_run_index": NAN,
        "hit_count": NAN,
        "n_hit_again": NAN,
        "hit_rate": NAN,
    }
    row.update(_blank_params(configuration))
    return row


def _iter_primary_batch_rows(summary_csv: Path) -> Iterator[Dict[str, str]]:
    if not summary_csv.is_file():
        return
    with summary_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            configuration = row["configuration"]
            suite = row["suite"]
            if configuration not in PAPER_CONFIG_KEYS or suite not in PAPER_SUITE_TAGS:
                continue
            y_val = _suite_y(suite)
            if not _paper_y(y_val):
                continue
            out = _base_row(
                row_type=ROW_PRIMARY_BATCH,
                configuration=configuration,
                y_val=y_val,
                batch_idx=row["batch_index"],
            )
            out["hit_count"] = row["hit_count"]
            yield out


def _iter_rescreen_rows(
    sessions_dir: Path,
    *,
    row_type: str,
    csv_glob: str,
) -> Iterator[Dict[str, str]]:
    if not sessions_dir.is_dir():
        return
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_rescreen_session_dir(session_dir.name)
        if parsed is None:
            continue
        configuration, suite = parsed
        if not _paper_campaign(configuration, suite):
            continue
        csvs = sorted(session_dir.glob(csv_glob))
        if not csvs:
            continue
        y_val = _suite_y(suite)
        with csvs[0].open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                src = row.get("source_run_index")
                if not src:
                    continue
                out = _base_row(
                    row_type=row_type,
                    configuration=configuration,
                    y_val=y_val,
                    batch_idx=_batch_index(src),
                )
                out["hit_index"] = row.get("hit_index", NAN) or NAN
                out["source_run_index"] = src
                out["n_hit_again"] = row.get("n_hit_again", NAN) or NAN
                out["hit_rate"] = row.get("hit_rate", NAN) or NAN
                if row.get("params_json"):
                    out.update(_expand_params(row["params_json"], configuration))
                yield out


def _write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def assemble_figure_data(*, data_dir: Path) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = (
        workspaces_output() / "Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv"
    )

    def all_rows() -> Iterator[Dict[str, str]]:
        yield from _iter_primary_batch_rows(summary_csv)
        yield from _iter_rescreen_rows(
            workspaces_re_runs() / "sessions",
            row_type=ROW_HIT,
            csv_glob="primary_hit_rescreen_*.csv",
        )
        yield from _iter_rescreen_rows(
            workspaces_non_hit_re_runs() / "sessions",
            row_type=ROW_NON_HIT,
            csv_glob="primary_non_hit_rescreen_*.csv",
        )

    return _write_csv(data_dir / "batch_hit_counts.csv", all_rows())


def _remove_legacy_outputs(data_dir: Path) -> None:
    legacy_csv = data_dir / "figure_data.csv"
    if legacy_csv.is_file():
        legacy_csv.unlink()
        print(f"Removed legacy {legacy_csv.name}")
    legacy_top_csv = data_dir.parent / "batch_hit_counts.csv"
    if legacy_top_csv.is_file():
        legacy_top_csv.unlink()
        print(f"Removed legacy {legacy_top_csv}")
    for name in LEGACY_DATA_FILES:
        path = data_dir / name
        if path.is_file():
            path.unlink()
            print(f"Removed legacy {path.name}")


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "figure_reproduction"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=data_dir,
        help="Output directory (default: data/figure_reproduction under this folder)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    n_rows = assemble_figure_data(data_dir=args.data_dir.resolve())
    print(f"Paper configs: {', '.join(sorted(PAPER_CONFIG_KEYS))}")
    print(f"Fixed-Y suites (Y >= {MIN_FIXED_Y:g}): {', '.join(sorted(PAPER_SUITE_TAGS))}")
    path = args.data_dir / "batch_hit_counts.csv"
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  batch_hit_counts: {n_rows:,} rows -> {path.name} ({size_mb:.1f} MiB)")

    _remove_legacy_outputs(args.data_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
