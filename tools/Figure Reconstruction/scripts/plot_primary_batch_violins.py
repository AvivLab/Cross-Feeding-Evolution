#!/usr/bin/env python3
"""Plot primary-batch hit-count violins from a summary CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PAPER_CONFIG_ORDER: Sequence[Tuple[str, str, str]] = (
    ("Neutral", "Neutral", "#59a14f"),
    ("justDeath", "Death Selection", "#e15759"),
    ("justDup", "Duplication Selection", "#4e79a7"),
    ("Death+Dup", "Death and Duplication Selection", "#b07aa1"),
)
PANEL_LETTERS = "abcd"
N_SIMS_PER_BATCH = 1000


def workspaces_output() -> Path:
    """Workspaces/Output next to the MCCP_Enzymes repo (under the shared GIT folder)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "Minimal":
            return parent.parent.parent / "Workspaces" / "Output"
        if (parent / "main.py").is_file() and (parent / "simulation").is_dir():
            return parent.parent / "Workspaces" / "Output"
    return here.parents[3] / "Workspaces" / "Output"



def hit_stats(vals: Sequence[int], *, n_sims: int) -> Tuple[float, float, float, float]:
    """Return mean and sample std for batch hit counts and hit-rate percents.

    Each entry in ``vals`` is one batch hit count H_b (b = 1..B).

        H_bar = (1/B) * sum_b H_b
        s     = sqrt( (1/(B-1)) * sum_b (H_b - H_bar)^2 )    # sample std, ddof=1

    The percent line scales by 100 / N_sim (hits per simulation as a percentage).
    """
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    pct_scale = 100.0 / float(n_sims)
    return mean, std, mean * pct_scale, std * pct_scale


def format_hit_label(mean: float, std: float, mean_pct: float, std_pct: float) -> str:
    return (
        f"{mean:.1f} ± {std:.1f}\n"
        f"({mean_pct:.1f}% ± {std_pct:.1f}%)"
    )


def load_hit_counts(csv_path: Path, *, suite: str | None) -> Dict[str, List[int]]:
    by_config: Dict[str, List[int]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if suite is not None and row["suite"] != suite:
                continue
            by_config[row["configuration"]].append(int(row["hit_count"]))
    return dict(by_config)


def ordered_series(
    by_config: Dict[str, List[int]],
    config_order: Sequence[Tuple[str, str, str]],
) -> Tuple[List[str], List[str], List[List[int]]]:
    labels: List[str] = []
    colors: List[str] = []
    series: List[List[int]] = []
    for key, label, color in config_order:
        if key not in by_config:
            raise KeyError(f"configuration {key!r} missing from CSV")
        labels.append(label)
        colors.append(color)
        series.append(by_config[key])
    return labels, colors, series


def plot_compare_violins(
    *,
    labels: Sequence[str],
    colors: Sequence[str],
    series: Sequence[Sequence[int]],
    output_path: Path,
    title: str,
    ylabel: str,
    n_sims: int = N_SIMS_PER_BATCH,
    use_legend: bool = False,
) -> None:
    n = len(labels)
    fig_w = 10.5 if use_legend else 9.5
    fig_h = 6.4 if use_legend else 5.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    positions = np.arange(1, n + 1)
    parts = ax.violinplot(
        [np.asarray(vals, dtype=float) for vals in series],
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    violin_colors = list(colors) if use_legend else ["#4C72B0"] * n
    for body, color in zip(parts["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.88 if use_legend else 0.85)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)

    label_ys: List[float] = []
    for pos, vals in zip(positions, series):
        mean, std, mean_pct, std_pct = hit_stats(vals, n_sims=n_sims)
        top = float(np.max(vals))
        label_y = top + 8.0
        label_ys.append(label_y + 14.0)
        ax.text(
            pos,
            label_y,
            format_hit_label(mean, std, mean_pct, std_pct),
            ha="center",
            va="bottom",
            fontsize=9,
            linespacing=1.15,
        )

    ax.set_xticks(positions)
    if use_legend:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=4)
        legend_handles = [
            Patch(facecolor=color, edgecolor="black", alpha=0.88, label=f"({PANEL_LETTERS[i]}) {label}")
            for i, (color, label) in enumerate(zip(colors, labels))
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.30, 1.0, 0.24),
            bbox_transform=ax.transAxes,
            mode="expand",
            ncol=2,
            fontsize=12,
            frameon=False,
            handlelength=2.8,
            handleheight=1.5,
            columnspacing=3.5,
            labelspacing=0.9,
            borderaxespad=0.0,
        )
    else:
        ax.set_xticklabels(labels, rotation=28, ha="right")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.4, n + 0.6)
    ymax = max(max(vals) for vals in series)
    label_top = max(label_ys) if label_ys else ymax
    ax.set_ylim(0, max(250, label_top * 1.12, ymax * 1.15))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if use_legend:
        fig.subplots_adjust(bottom=0.38)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    else:
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_means(labels: Sequence[str], series: Sequence[Sequence[int]], *, n_sims: int) -> None:
    for label, vals in zip(labels, series):
        mean, std, mean_pct, std_pct = hit_stats(vals, n_sims=n_sims)
        print(
            f"  {label}: n={len(vals)} "
            f"mean={mean:.2f} std={std:.2f} "
            f"({mean_pct:.2f}% ± {std_pct:.2f}%)"
        )


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    figures_dir = root / "figures" / "Used"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=workspaces_output()
        / "Summary/Summary_Simple/primary_batch_compare_hit_counts.csv",
        help="Summary CSV with batch hit counts",
    )
    parser.add_argument(
        "--suite",
        default="Variable_Ratio",
        help="Suite filter (default: Variable_Ratio). Use 'all' for every suite combined.",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=N_SIMS_PER_BATCH,
        help="Simulations per batch (for percent labels; default: 1000)",
    )
    parser.add_argument(
        "--style",
        choices=("plain", "legend", "both"),
        default="both",
        help="plain: blue violins with rotated labels; legend: colored violins with (a)--(e) and legend",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=figures_dir / "primary_batch_compare_violins.png",
        help="Output PNG for plain style",
    )
    parser.add_argument(
        "--output-legend",
        type=Path,
        default=figures_dir / "primary_batch_compare_violins_legend.png",
        help="Output PNG for colored legend style",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    suite = None if args.suite == "all" else args.suite
    by_config = load_hit_counts(args.csv, suite=suite)
    labels, colors, series = ordered_series(by_config, PAPER_CONFIG_ORDER)

    print(f"CSV: {args.csv}")
    print(f"Suite: {args.suite}")
    print_means(labels, series, n_sims=args.n_sims)

    title = "Primary batch hit counts by configuration"
    ylabel = "Hit count per batch (N simulations)"
    if args.style in ("plain", "both"):
        plot_compare_violins(
            labels=labels,
            colors=colors,
            series=series,
            output_path=args.output,
            title=title,
            ylabel=ylabel,
            n_sims=args.n_sims,
            use_legend=False,
        )
        print(f"Wrote: {args.output}")
    if args.style in ("legend", "both"):
        plot_compare_violins(
            labels=labels,
            colors=colors,
            series=series,
            output_path=args.output_legend,
            title=title,
            ylabel=ylabel,
            n_sims=args.n_sims,
            use_legend=True,
        )
        print(f"Wrote: {args.output_legend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
