#!/usr/bin/env python3
"""Figure 3 panel A: batch hit counts vs fixed task energy yield ratio Y (all regimes)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_primary_batch_violins import (
    MAIN_CONFIG_ORDER,
    PAPER_CONFIG_ORDER,
    hit_stats,
    workspaces_output,
)

N_SIMS_PER_BATCH = 1000
MIN_FIXED_Y = 0.0001  # paper figures: all completed fixed-Y campaigns
MAX_FIXED_Y = 10.0  # temporarily omit Y=20 from paper figures

_ALL_SUITE_ORDER: Sequence[Tuple[str, float, str]] = (
    ("Fixed_0.0001_ratio", 0.0001, r"$Y=10^{-4}$"),
    ("Fixed_0.1_ratio", 0.1, r"$Y=0.1$"),
    ("Fixed_0.25_ratio", 0.25, r"$Y=0.25$"),
    ("Fixed_0.5_ratio", 0.5, r"$Y=0.5$"),
    ("Fixed_0.75_ratio", 0.75, r"$Y=0.75$"),
    ("Fixed_1_ratio", 1.0, r"$Y=1$"),
    ("Fixed_3_ratio", 3.0, r"$Y=3$"),
    ("Fixed_5_ratio", 5.0, r"$Y=5$"),
    ("Fixed_7_ratio", 7.0, r"$Y=7$"),
    ("Fixed_10_ratio", 10.0, r"$Y=10$"),
    ("Fixed_20_ratio", 20.0, r"$Y=20$"),
)

SUITE_ORDER: Sequence[Tuple[str, float, str]] = tuple(
    entry
    for entry in _ALL_SUITE_ORDER
    if MIN_FIXED_Y <= entry[1] <= MAX_FIXED_Y
)

# Main-text Figure 3 typography (legible at single-column width in the paper).
FIG3_LABELSIZE = 12
FIG3_TICKSIZE = 11
FIG3_LEGENDSIZE = 10
FIG3_PANELSIZE = 14


def apply_figure3_axis_typography(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=FIG3_TICKSIZE)
    ax.xaxis.label.set_size(FIG3_LABELSIZE)
    ax.yaxis.label.set_size(FIG3_LABELSIZE)


def format_y_tick(y: float) -> str:
    if y == 0.0001:
        return r"$10^{-4}$"
    return f"{y:g}" if y < 1 or y != int(y) else f"{int(y)}"


def x_positions_for_y_vals(
    y_vals: Sequence[float],
    *,
    even_spacing: bool = False,
) -> np.ndarray:
    """Map fixed-Y values to x coordinates; optional equal spacing for categorical axes."""
    if even_spacing:
        return np.arange(len(y_vals), dtype=float)
    return np.asarray(y_vals, dtype=float)


def panel_label(
    ax: plt.Axes,
    letter: str,
    *,
    x: float = -0.12,
    y: float = 1.0,
) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FIG3_PANELSIZE,
        fontweight="bold",
        clip_on=False,
        zorder=10,
    )


def apply_panel_figure_layout(fig: plt.Figure, *, nrows: int = 1) -> None:
    """Leave left margin for panel letters outside the axes."""
    if nrows == 1:
        bottom, hspace = 0.10, 0.0
    elif nrows == 2:
        bottom, hspace = 0.07, 0.22
    else:
        bottom, hspace = 0.06, 0.30
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=bottom, hspace=hspace)


def load_by_suite(csv_path: Path) -> Dict[str, Dict[str, List[int]]]:
    out: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["suite"]][row["configuration"]].append(int(row["hit_count"]))
    return {suite: dict(cfgs) for suite, cfgs in out.items()}


def suite_means(
    by_suite: Dict[str, Dict[str, List[int]]],
    *,
    n_sims: int,
    config_order: Sequence[Tuple[str, str, str]] = PAPER_CONFIG_ORDER,
) -> Tuple[List[float], List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    y_vals: List[float] = []
    x_labels: List[str] = []
    mean_hits: Dict[str, List[float]] = {key: [] for key, _, _ in config_order}
    std_hits: Dict[str, List[float]] = {key: [] for key, _, _ in config_order}
    for suite, y_val, label in SUITE_ORDER:
        cfgs = by_suite.get(suite)
        if cfgs is None:
            raise KeyError(f"suite {suite!r} missing from CSV")
        y_vals.append(y_val)
        x_labels.append(label)
        for key, _, _ in config_order:
            vals = cfgs.get(key)
            if not vals:
                raise KeyError(f"{key!r} missing for suite {suite!r}")
            mean, std, _, _ = hit_stats(vals, n_sims=n_sims)
            mean_hits[key].append(mean)
            std_hits[key].append(std)
    return y_vals, x_labels, mean_hits, std_hits


def plot_hit_counts_panel(
    ax: plt.Axes,
    *,
    y_vals: Sequence[float],
    mean_hits: Dict[str, List[float]],
    std_hits: Dict[str, List[float]],
    show_legend: bool = True,
    panel_letter: str | None = "a",
    show_xlabel: bool = True,
    show_xticklabels: bool = True,
    even_x_spacing: bool = False,
    n_sims: int = N_SIMS_PER_BATCH,
    config_order: Sequence[Tuple[str, str, str]] = PAPER_CONFIG_ORDER,
) -> None:
    y_max = 0.0
    x_arr = x_positions_for_y_vals(y_vals, even_spacing=even_x_spacing)
    for key, label, color in config_order:
        means = np.asarray(mean_hits[key], dtype=float)
        stds = np.asarray(std_hits[key], dtype=float)
        y_max = max(y_max, float(np.max(means + stds)))
        ax.errorbar(
            x_arr,
            means,
            yerr=stds,
            marker="o",
            markersize=6,
            linewidth=2.0,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            label=label,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )

    ax.set_xticks(list(x_arr))
    if show_xticklabels:
        ax.set_xticklabels([format_y_tick(y) for y in y_vals])
    else:
        ax.set_xticklabels([])
    if show_xlabel:
        ax.set_xlabel("Fixed task energy yield ratio $Y$")
    ax.set_ylabel(f"Mean batch hit count $H_b$ (out of {int(n_sims)})")
    ax.set_ylim(0, y_max * 1.08)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    apply_figure3_axis_typography(ax)
    if show_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=FIG3_LEGENDSIZE)
    if panel_letter:
        panel_label(ax, panel_letter)


def plot_supplementary(
    *,
    y_vals: Sequence[float],
    mean_hits: Dict[str, List[float]],
    std_hits: Dict[str, List[float]],
    output_path: Path,
    n_sims: int = N_SIMS_PER_BATCH,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    plot_hit_counts_panel(
        ax,
        y_vals=y_vals,
        mean_hits=mean_hits,
        std_hits=std_hits,
        show_legend=True,
        panel_letter=None,
        n_sims=n_sims,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=workspaces_output()
        / "Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv",
        help="Summary_Ratio batch hit-count CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/Extra/html/figures/supplementary_fig1_ratio_hit_counts.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Simulations per batch (default: infer from CSV n_runs / n_sims_per_batch)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    by_suite = load_by_suite(args.csv)
    n_sims = args.n_sims
    if n_sims is None:
        from figure_csv import infer_n_sims_from_rows

        n_sims = infer_n_sims_from_rows(args.csv)
    y_vals, x_labels, mean_hits, std_hits = suite_means(by_suite, n_sims=n_sims)

    print(f"CSV: {args.csv}")
    for label, idx in zip(x_labels, range(len(x_labels))):
        print(f"  {label}:")
        for key, cfg_label, _ in PAPER_CONFIG_ORDER:
            print(
                f"    {cfg_label}: {mean_hits[key][idx]:.1f} ± {std_hits[key][idx]:.1f}"
            )

    plot_supplementary(
        y_vals=y_vals,
        mean_hits=mean_hits,
        std_hits=std_hits,
        output_path=args.output,
        n_sims=n_sims,
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
