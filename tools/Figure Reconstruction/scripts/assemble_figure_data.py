#!/usr/bin/env python3
"""Assemble data/figure_reproduction/batch_hit_counts.csv for paper figure plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Iterator, Sequence, Tuple

from figure_param_specs import ALL_CONFIG_KEYS, PARAM_SPECS
from plot_hit_rescreen_panel import (
    N_SIMS_PER_BATCH,
    parse_rescreen_session_dir,
)
from plot_primary_batch_violins import PAPER_CONFIG_ORDER, workspaces_output
from plot_ratio_supplementary import MAX_FIXED_Y, MIN_FIXED_Y, SUITE_ORDER

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
    "n_sims_per_batch",
    "n_rescreen",
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
        y = float(y_val)
        return MIN_FIXED_Y <= y <= MAX_FIXED_Y
    except ValueError:
        return False


def _paper_campaign(configuration: str, suite: str) -> bool:
    y = _suite_y(suite)
    return configuration in PAPER_CONFIG_KEYS and suite in PAPER_SUITE_TAGS and _paper_y(y)


def _batch_index(source_run_index: str, *, n_sims_per_batch: int) -> str:
    n = max(1, int(n_sims_per_batch))
    return str(int(source_run_index) // n)


def _coerce_positive_int(raw: object, default: int = 0) -> int:
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _load_summary_n_runs(summary_csv: Path) -> Dict[Tuple[str, str], int]:
    """Map (configuration, suite) -> n_runs from the Summary compare CSV."""
    out: Dict[Tuple[str, str], int] = {}
    if not summary_csv.is_file():
        return out
    with summary_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            configuration = str(row.get("configuration") or "").strip()
            suite = str(row.get("suite") or "").strip()
            n_runs = _coerce_positive_int(row.get("n_runs"), 0)
            if not configuration or not suite or n_runs < 1:
                continue
            out[(configuration, suite)] = n_runs
    return out


def _rescreen_n_seeds(session_dir: Path, csv_path: Path) -> int:
    """Prefer n_seeds from the rescreen CSV row / sidecar JSON."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        sample = next(csv.DictReader(fh), None)
    if sample:
        n = _coerce_positive_int(sample.get("n_seeds"), 0)
        if n > 0:
            return n
    for path in sorted(session_dir.glob("primary_*rescreen_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        n = _coerce_positive_int(data.get("n_seeds_per_hit"), 0)
        if n > 0:
            return n
    return 0


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
    n_sims_per_batch: str = NAN,
    n_rescreen: str = NAN,
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
        "n_sims_per_batch": n_sims_per_batch,
        "n_rescreen": n_rescreen,
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
            n_sims = _coerce_positive_int(row.get("n_runs"), N_SIMS_PER_BATCH)
            out = _base_row(
                row_type=ROW_PRIMARY_BATCH,
                configuration=configuration,
                y_val=y_val,
                batch_idx=row["batch_index"],
                n_sims_per_batch=str(n_sims),
            )
            out["hit_count"] = row["hit_count"]
            yield out


def _iter_rescreen_rows(
    sessions_dir: Path,
    *,
    row_type: str,
    csv_glob: str,
    n_runs_by_campaign: Dict[Tuple[str, str], int] | None = None,
) -> Iterator[Dict[str, str]]:
    if not sessions_dir.is_dir():
        return
    n_runs_by_campaign = n_runs_by_campaign or {}
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
        n_sims = n_runs_by_campaign.get((configuration, suite), N_SIMS_PER_BATCH)
        n_rescreen = _rescreen_n_seeds(session_dir, csvs[0])
        with csvs[0].open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                src = row.get("source_run_index")
                if not src:
                    continue
                row_n_rescreen = _coerce_positive_int(row.get("n_seeds"), n_rescreen)
                out = _base_row(
                    row_type=row_type,
                    configuration=configuration,
                    y_val=y_val,
                    batch_idx=_batch_index(src, n_sims_per_batch=n_sims),
                    n_sims_per_batch=str(n_sims),
                    n_rescreen=str(row_n_rescreen) if row_n_rescreen > 0 else NAN,
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


def assemble_figure_data(
    *,
    data_dir: Path,
    output_root: Path | None = None,
    summary_csv: Path | None = None,
) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)
    root = (output_root or workspaces_output()).resolve()
    summary = summary_csv or (
        root / "Summary" / "Summary_Ratio" / "primary_batch_compare_hit_counts.csv"
    )
    re_runs = root / "Re-Runs" / "sessions"
    re_runs_non = root / "Re-Runs-NonHits" / "sessions"

    def all_rows() -> Iterator[Dict[str, str]]:
        n_runs_by_campaign = _load_summary_n_runs(summary)
        yield from _iter_primary_batch_rows(summary)
        yield from _iter_rescreen_rows(
            re_runs,
            row_type=ROW_HIT,
            csv_glob="primary_hit_rescreen_*.csv",
            n_runs_by_campaign=n_runs_by_campaign,
        )
        yield from _iter_rescreen_rows(
            re_runs_non,
            row_type=ROW_NON_HIT,
            csv_glob="primary_non_hit_rescreen_*.csv",
            n_runs_by_campaign=n_runs_by_campaign,
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Campaign output root containing Summary/ and Re-Runs/ "
            "(default: Workspaces/Output if present)"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Override path to primary_batch_compare_hit_counts.csv",
    )
    parser.add_argument(
        "--build-summary",
        action="store_true",
        help="Run build_summary_hit_counts.py for --output-root before assembling",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_root = args.output_root.resolve() if args.output_root else None
    if args.build_summary:
        if output_root is None:
            output_root = workspaces_output()
        from build_summary_hit_counts import main as build_summary_main

        rc = build_summary_main(["--output-root", str(output_root)])
        if rc:
            return rc

    n_rows = assemble_figure_data(
        data_dir=args.data_dir.resolve(),
        output_root=output_root,
        summary_csv=args.summary_csv.resolve() if args.summary_csv else None,
    )
    print(f"Paper configs: {', '.join(sorted(PAPER_CONFIG_KEYS))}")
    print(
        f"Fixed-Y suites ({MIN_FIXED_Y:g} <= Y <= {MAX_FIXED_Y:g}): "
        f"{', '.join(sorted(PAPER_SUITE_TAGS))}"
    )
    path = args.data_dir / "batch_hit_counts.csv"
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  batch_hit_counts: {n_rows:,} rows -> {path.name} ({size_mb:.1f} MiB)")

    _remove_legacy_outputs(args.data_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
