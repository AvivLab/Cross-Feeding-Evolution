#!/usr/bin/env python3
"""Supplementary figure S4: events scatter (a--d) plus neutral zoom (e)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from plot_hit_rescreen_panel import plot_rescreen_events_figure, workspaces_re_runs


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=workspaces_re_runs() / "sessions",
        help="Re-Runs session folders with per-hit rescreen CSVs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/Extra/supplementary_rescreen_events.png",
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
        default=13.5,
        help="Figure height in inches",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="Max scatter points per panel (random subsample if exceeded)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    plot_rescreen_events_figure(
        sessions_dir=args.sessions_dir,
        output_path=args.output,
        figsize=(args.fig_width, args.fig_height),
        max_points=args.max_points,
    )
    print(f"Sessions: {args.sessions_dir}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
