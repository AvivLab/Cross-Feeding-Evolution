#!/usr/bin/env python3
"""Supplementary panel (e): neutral events vs re-screen hits (16--20 zoom)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from plot_hit_rescreen_panel import plot_true_neutral_events_zoom_figure, workspaces_re_runs


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
        default=root / "figures/Extra/html/figures/supplementary_rescreen_events_true_neutral_zoom.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        default=16,
        help="Minimum re-screen success count (inclusive)",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=20,
        help="Maximum re-screen success count (inclusive)",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=7.5,
        help="Figure width in inches",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.5,
        help="Figure height in inches",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    plot_true_neutral_events_zoom_figure(
        sessions_dir=args.sessions_dir,
        output_path=args.output,
        x_min=args.x_min,
        x_max=args.x_max,
        figsize=(args.fig_width, args.fig_height),
    )
    print(f"Sessions: {args.sessions_dir}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
