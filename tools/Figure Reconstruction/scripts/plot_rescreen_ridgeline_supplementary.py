#!/usr/bin/env python3
"""Supplementary figure: ridgeline panels (one regime per panel)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from figure_csv import ROW_HIT
from plot_hit_rescreen_panel import plot_rescreen_ridgeline_figure, workspaces_re_runs
from plot_primary_batch_violins import MAIN_CONFIG_ORDER, PAPER_CONFIG_ORDER


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
        default=workspaces_re_runs() / "sessions",
        help="Re-Runs session folders (ignored if --data-csv is set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --configs)",
    )
    parser.add_argument(
        "--configs",
        choices=("main", "all"),
        default="main",
        help="main: Neutral + Selection (default); all: four configurations",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_order = MAIN_CONFIG_ORDER if args.configs == "main" else PAPER_CONFIG_ORDER
    if args.output is None:
        if args.configs == "main":
            args.output = root / "figures/Used/supplementary_rescreen_ridgelines.png"
        else:
            args.output = root / "figures/Extra/supplementary_rescreen_ridgelines_four_configs.png"
    figsize = None
    if args.fig_width is not None or args.fig_height is not None:
        figsize = (
            11.0 if args.fig_width is None else args.fig_width,
            (5.4 if args.configs == "main" else 10.5)
            if args.fig_height is None
            else args.fig_height,
        )

    plot_rescreen_ridgeline_figure(
        sessions_dir=None if args.data_csv else args.sessions_dir,
        data_csv=args.data_csv,
        data_row_type=ROW_HIT,
        output_path=args.output,
        figsize=figsize,
        csv_glob="primary_hit_rescreen_*.csv",
        config_order=config_order,
    )
    if args.data_csv is not None:
        print(f"Data CSV: {args.data_csv}")
    else:
        print(f"Sessions: {args.sessions_dir}")
    print(f"Configs: {args.configs}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
