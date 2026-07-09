#!/usr/bin/env python3
"""Supplementary figure: 2x2 ridgeline panels for non-hit re-screens."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from figure_csv import ROW_NON_HIT
from plot_hit_rescreen_panel import (
    N_RESCREEN,
    plot_rescreen_ridgeline_figure,
    workspaces_non_hit_re_runs,
)


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help="Unified figure CSV (data/figure_reproduction/batch_hit_counts.csv)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=workspaces_non_hit_re_runs() / "sessions",
        help="Re-Runs-NonHits session folders (ignored if --data-csv is set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/Extra/supplementary_non_hit_rescreen_ridgelines.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=11.0,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=10.5,
        help="Figure height in inches",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    plot_rescreen_ridgeline_figure(
        sessions_dir=None if args.data_csv else args.sessions_dir,
        data_csv=args.data_csv,
        data_row_type=ROW_NON_HIT,
        output_path=args.output,
        figsize=(args.fig_width, args.fig_height),
        csv_glob="primary_non_hit_rescreen_*.csv",
        xlabel=f"Re-run successes per non-hit (out of {N_RESCREEN})",
        errorbar_peak_floor=0.03,
    )
    if args.data_csv is not None:
        print(f"Data CSV: {args.data_csv}")
    else:
        print(f"Sessions: {args.sessions_dir}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
