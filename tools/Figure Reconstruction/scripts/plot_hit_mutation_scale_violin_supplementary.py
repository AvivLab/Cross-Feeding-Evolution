#!/usr/bin/env python3
"""Supplementary figure: sampled hit parameters by configuration and fixed Y (boxplots)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu

from plot_hit_rescreen_panel import parse_rescreen_session_dir, workspaces_re_runs
from plot_primary_batch_violins import PAPER_CONFIG_ORDER
from plot_ratio_supplementary import SUITE_ORDER, x_positions_for_y_vals

ALL_CONFIG_KEYS = frozenset(key for key, _, _ in PAPER_CONFIG_ORDER)

# Display labels for section headers (plain text + mathtext symbols only; no LaTeX spacing commands).
PARAM_SPECS: Tuple[Tuple[str, str, Tuple[float, float], FrozenSet[str]], ...] = (
    ("Constant Probability", "Constant probability $p$", (0.0, 0.5), frozenset({"justDeath"})),
    ("Death Decay Rate", "Death decay $\\lambda$", (0.0, 20.0), frozenset({"justDeath", "Death+Dup"})),
    (
        "Duplication Sigmoid Intensity",
        "Duplication steepness $\\varepsilon$",
        (0.0, 10.0),
        frozenset({"justDup", "Death+Dup"}),
    ),
    (
        "Duplication Sigmoid Midpoint",
        "Duplication midpoint $\\omega$",
        (0.0, 5.0),
        frozenset({"justDup", "Death+Dup"}),
    ),
    ("Cost of Life", "Cost of life $c_{m}$", (0.0, 0.05), ALL_CONFIG_KEYS),
    ("Flow Percentage", "Flow fraction $\\phi$ (%)", (0.0, 50.0), ALL_CONFIG_KEYS),
    ("Mutation Rate", "Mutation rate $\\mu_{m}$", (0.0, 0.005), ALL_CONFIG_KEYS),
    ("Initial A", "Initial Task A $A_0$", (0.0, 1.0), ALL_CONFIG_KEYS),
    ("Initial Energy", "Initial energy $E_{\\mathrm{init}}$", (0.0, 20.0), ALL_CONFIG_KEYS),
    ("Mutation Scale", "Mutation scale $\\sigma_{m}$", (0.0, 1.0), ALL_CONFIG_KEYS),
)

PARAM_KEYS = tuple(spec[0] for spec in PARAM_SPECS)
SPLIT_ROW_INDEX = len(PARAM_SPECS) // 2
ROW_HEIGHT_IN = 2.75
HEADER_HEIGHT_IN = 1.2
LEGEND_HEIGHT_IN = 0.48

# Bracket colors by BH-adjusted significance tier.
SIG_COLORS: Tuple[Tuple[str, str], ...] = (
    ("p05", "#f1c40f"),
    ("p01", "#e67e22"),
    ("p001", "#c0392b"),
)


def load_all_hit_params_by_suite(
    sessions_dir: Path,
    *,
    csv_glob: str = "primary_hit_rescreen_*.csv",
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Return suite_tag -> config_key -> param_name -> hit-level sampled values."""
    out: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    if not sessions_dir.is_dir():
        return {}
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_rescreen_session_dir(session_dir.name)
        if parsed is None:
            continue
        config_key, suite_tag = parsed
        if config_key not in ALL_CONFIG_KEYS:
            continue
        if not suite_tag.startswith("Fixed_"):
            continue
        csvs = sorted(session_dir.glob(csv_glob))
        if not csvs:
            continue
        with csvs[0].open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                params = json.loads(row["params_json"])
                for param_name in PARAM_KEYS:
                    raw = params.get(param_name)
                    if raw is None or raw == "":
                        continue
                    val = float(raw)
                    if np.isfinite(val):
                        out[suite_tag][config_key][param_name].append(val)
    return {
        suite: {cfg: dict(params) for cfg, params in cfgs.items()}
        for suite, cfgs in out.items()
    }


def values_by_y_for_cell(
    by_suite: Dict[str, Dict[str, Dict[str, List[float]]]],
    *,
    config_key: str,
    param_name: str,
    y_vals: Sequence[float],
    suite_to_y: Dict[str, float],
) -> Dict[float, List[float]]:
    values_by_y: Dict[float, List[float]] = {float(y): [] for y in y_vals}
    for suite, y_val in suite_to_y.items():
        y_key = float(y_val)
        if y_key not in values_by_y:
            continue
        for val in by_suite.get(suite, {}).get(config_key, {}).get(param_name, []):
            values_by_y[y_key].append(float(val))
    return values_by_y


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Return Benjamini-Hochberg adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(p_values[idx] * m / (rank + 1), prev)
        prev = val
        adjusted[idx] = val
    return adjusted.tolist()


def sig_color(p_adj: float) -> str | None:
    """Return bracket color for a BH-adjusted p-value, or None if not significant."""
    if p_adj >= 0.05:
        return None
    if p_adj < 0.001:
        return SIG_COLORS[2][1]
    if p_adj < 0.01:
        return SIG_COLORS[1][1]
    return SIG_COLORS[0][1]


def adjusted_p_all_pairs(
    values_by_y: Dict[float, Sequence[float]],
    y_vals: Sequence[float],
    x_pos: Sequence[float],
) -> List[Tuple[float, float, float]]:
    """Mann-Whitney p-values for all fixed-Y pairs, BH-FDR adjusted within the panel."""
    pair_keys: List[Tuple[float, float]] = []
    p_raw: List[float] = []
    for i, y_left in enumerate(y_vals):
        left_vals = np.asarray(values_by_y.get(float(y_left), []), dtype=float)
        left_vals = left_vals[np.isfinite(left_vals)]
        for j in range(i + 1, len(y_vals)):
            y_right = float(y_vals[j])
            right_vals = np.asarray(values_by_y.get(y_right, []), dtype=float)
            right_vals = right_vals[np.isfinite(right_vals)]
            if left_vals.size < 2 or right_vals.size < 2:
                p_value = 1.0
            else:
                _, p_value = mannwhitneyu(left_vals, right_vals, alternative="two-sided")
                p_value = float(p_value)
            pair_keys.append((float(x_pos[i]), float(x_pos[j])))
            p_raw.append(p_value)
    if not p_raw:
        return []
    adjusted = benjamini_hochberg(p_raw)
    return [(x1, x2, p_adj) for (x1, x2), p_adj in zip(pair_keys, adjusted)]


def _overlap_lane_count(
    group: Sequence[Tuple[int, int, int, float, float, float]],
) -> Tuple[Dict[Tuple[int, int], int], int]:
    """Assign overlap lanes within a left-anchor tier (longest span first)."""
    ordered = sorted(group, key=lambda item: -item[2])
    lanes: List[List[Tuple[int, int]]] = []
    lane_by_pair: Dict[Tuple[int, int], int] = {}
    for left_idx, right_idx, _span, _x1, _x2, _p_adj in ordered:
        lane = 0
        while lane < len(lanes) and any(
            not (right_idx < existing_left or left_idx > existing_right)
            for existing_left, existing_right in lanes[lane]
        ):
            lane += 1
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append((left_idx, right_idx))
        lane_by_pair[(left_idx, right_idx)] = lane
    return lane_by_pair, len(lanes)


def assign_sorted_bracket_levels(
    sig_pairs: Sequence[Tuple[float, float, float]],
    x_pos: Sequence[float],
) -> List[Tuple[float, float, float, int, int]]:
    """Stack tiers by left anchor: leftmost group on top, then each next anchor below."""
    x_to_idx = {float(x): idx for idx, x in enumerate(x_pos)}
    indexed: List[Tuple[int, int, int, float, float, float]] = []
    for x1, x2, p_adj in sig_pairs:
        left_idx, right_idx = sorted((x_to_idx[float(x1)], x_to_idx[float(x2)]))
        indexed.append((
            left_idx,
            right_idx,
            right_idx - left_idx,
            float(x_pos[left_idx]),
            float(x_pos[right_idx]),
            float(p_adj),
        ))

    by_left: Dict[int, List[Tuple[int, int, int, float, float, float]]] = defaultdict(list)
    for item in indexed:
        by_left[item[0]].append(item)

    lanes_per_tier: Dict[int, int] = {}
    lane_within_tier: Dict[Tuple[int, int], int] = {}
    for left_idx in range(len(x_pos)):
        group = by_left.get(left_idx, [])
        if not group:
            lanes_per_tier[left_idx] = 0
            continue
        lane_map, n_lanes = _overlap_lane_count(group)
        lanes_per_tier[left_idx] = n_lanes
        lane_within_tier.update(lane_map)

    tier_base: Dict[int, int] = {}
    lanes_below = 0
    for left_idx in reversed(range(len(x_pos))):
        tier_base[left_idx] = lanes_below
        lanes_below += lanes_per_tier[left_idx]

    return [
        (
            x1,
            x2,
            p_adj,
            left_idx,
            tier_base[left_idx] + lane_within_tier[(left_idx, right_idx)],
        )
        for left_idx, right_idx, _span, x1, x2, p_adj in indexed
    ]


def box_upper_extent(vals: np.ndarray) -> float:
    """Upper whisker end for matplotlib's 1.5*IQR boxplot rule."""
    if vals.size == 0:
        return float("nan")
    q1, q3 = np.percentile(vals, [25, 75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    in_whisker = vals[vals <= upper_fence]
    if in_whisker.size == 0:
        return float(np.max(vals))
    return float(np.max(in_whisker))


def draw_significance_bracket(
    ax: plt.Axes,
    x_left: float,
    x_right: float,
    y: float,
    tick_h: float,
    *,
    p_adj: float,
) -> float:
    """Draw a significance bracket colored by adjusted p-value tier."""
    x_min, x_max = (x_left, x_right) if x_left <= x_right else (x_right, x_left)
    color = sig_color(p_adj)
    if color is None:
        return y
    ax.plot(
        [x_min, x_min, x_max, x_max],
        [y, y + tick_h, y + tick_h, y],
        color=color,
        linewidth=1.1,
        clip_on=False,
        zorder=11,
        solid_capstyle="butt",
    )
    return y + tick_h


def plot_param_boxplot_cell(
    ax: plt.Axes,
    *,
    color: str,
    y_vals: Sequence[float],
    values_by_y: Dict[float, Sequence[float]],
    ylim: Tuple[float, float],
) -> None:
    x_pos = x_positions_for_y_vals(y_vals, even_spacing=True)
    series = [np.asarray(values_by_y.get(float(y), []), dtype=float) for y in y_vals]
    parts = ax.boxplot(
        series,
        positions=x_pos,
        widths=0.78,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.9},
        whiskerprops={"color": "black", "linewidth": 0.75},
        capprops={"color": "black", "linewidth": 0.75},
        boxprops={"edgecolor": "black", "linewidth": 0.75},
    )
    for patch in parts["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    pair_results = adjusted_p_all_pairs(values_by_y, y_vals, x_pos)
    sig_pairs = [
        (x_left, x_right, p_adj)
        for x_left, x_right, p_adj in pair_results
        if sig_color(p_adj) is not None
    ]
    leveled = assign_sorted_bracket_levels(sig_pairs, x_pos)
    y_span = ylim[1] - ylim[0]
    tops = [box_upper_extent(vals) for vals in series if vals.size > 0]
    data_top = max(tops) if tops else ylim[0]
    bracket_base = data_top + 0.025 * y_span
    tick_h = 0.008 * y_span
    n_levels = max((level for *_, level in leveled), default=-1) + 1
    max_bracket_zone = 0.34 * y_span
    bracket_step = min(0.032 * y_span, max_bracket_zone / max(n_levels, 1))

    max_bracket_y = bracket_base
    for x_left, x_right, p_adj, left_idx, level in leveled:
        y_bracket = bracket_base + level * bracket_step
        label_top = draw_significance_bracket(
            ax,
            x_left,
            x_right,
            y_bracket,
            tick_h,
            p_adj=p_adj,
        )
        max_bracket_y = max(max_bracket_y, label_top)

    ax.set_xticks(x_pos)
    ax.set_xlim(float(x_pos[0]) - 0.55, float(x_pos[-1]) + 0.55)
    ax.set_ylim(ylim[0], max(ylim[1], max_bracket_y + 0.025 * y_span))
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5, pad=1.5)
    ax.set_xlabel("Fixed $Y$", fontsize=7.0, labelpad=4.0)
    ax.set_xticklabels([label for _, _, label in SUITE_ORDER], fontsize=6.0)


def setup_param_section(subfig: plt.SubFigure, label: str) -> None:
    """Draw parameter label; plot axes are placed manually per column."""
    subfig.patch.set_visible(False)
    subfig.text(
        0.015,
        0.93,
        label,
        transform=subfig.transSubfigure,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#222222",
        bbox={
            "boxstyle": "square,pad=0.42",
            "facecolor": "white",
            "edgecolor": "#555555",
            "linewidth": 1.0,
        },
        zorder=5,
        clip_on=False,
    )


def row_plot_axes(subfig: plt.SubFigure, n_cols: int) -> List[plt.Axes]:
    """Create a centered row of axes in subfigure coordinates."""
    left_margin = 0.05
    right_margin = 0.01
    col_gap = 0.08
    plot_height = 0.58
    plot_bottom = 0.5 - plot_height / 2.0 + 0.01
    usable_width = 1.0 - left_margin - right_margin - col_gap * (n_cols - 1)
    col_width = usable_width / n_cols
    axes: List[plt.Axes] = []
    for col_idx in range(n_cols):
        left = left_margin + col_idx * (col_width + col_gap)
        axes.append(subfig.add_axes([left, plot_bottom, col_width, plot_height]))
    return axes


def plot_blank_cell(ax: plt.Axes) -> None:
    ax.set_visible(False)


def add_significance_legend(legend_subfig: plt.SubFigure) -> None:
    """Place p-value color key below the bordered panel."""
    legend_subfig.patch.set_visible(False)
    ax = legend_subfig.add_axes([0.03, 0.02, 0.94, 0.96])
    ax.set_axis_off()

    sig_labels = (r"$P<0.05$", r"$P<0.01$", r"$P<0.001$")
    sig_handles = [
        Line2D([0, 1], [0, 0], color=color, linewidth=2.4, label=label)
        for (_key, color), label in zip(SIG_COLORS, sig_labels)
    ]
    ax.legend(
        handles=sig_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=6.8,
        title="Bracket color (BH-adjusted $P$)",
        title_fontsize=6.8,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    ax.text(
        0.5,
        0.08,
        "Pairwise fixed-$Y$ comparisons (Mann--Whitney, Benjamini--Hochberg FDR per panel)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#333333",
    )


def add_param_block_frame(param_block: plt.SubFigure) -> None:
    """Draw a full box frame; more reliable than the subfigure patch edge when saving tight."""
    transform = param_block.transSubfigure
    frame_color = "#555555"
    frame_lw = 1.1
    for x_vals, y_vals in (
        ([0.0, 1.0], [1.0, 1.0]),
        ([0.0, 1.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 1.0]),
        ([1.0, 1.0], [0.0, 1.0]),
    ):
        param_block.add_artist(
            Line2D(
                x_vals,
                y_vals,
                transform=transform,
                color=frame_color,
                linewidth=frame_lw,
                zorder=10,
                clip_on=False,
            )
        )


def default_figsize(n_rows: int, *, show_header: bool) -> Tuple[float, float]:
    header_pad = HEADER_HEIGHT_IN if show_header else 0.15
    return (13.5, ROW_HEIGHT_IN * n_rows + header_pad + LEGEND_HEIGHT_IN)


def plot_hit_param_boxplot_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
    param_specs: Sequence[Tuple[str, str, Tuple[float, float], FrozenSet[str]]] = PARAM_SPECS,
    show_header: bool = True,
    y_vals: Sequence[float] | None = None,
    figsize: Tuple[float, float] | None = None,
    csv_glob: str = "primary_hit_rescreen_*.csv",
) -> None:
    """Grid of boxplots for a slice of sampled parameters."""
    by_suite = load_all_hit_params_by_suite(sessions_dir, csv_glob=csv_glob)
    if y_vals is None:
        y_vals = [y for _, y, _ in SUITE_ORDER]
    suite_to_y = {suite: y_val for suite, y_val, _ in SUITE_ORDER}

    n_rows = len(param_specs)
    n_cols = len(PAPER_CONFIG_ORDER)
    if n_rows == 0:
        raise ValueError("param_specs must not be empty")
    if figsize is None:
        figsize = default_figsize(n_rows, show_header=show_header)

    if show_header:
        fig = plt.figure(figsize=figsize)
        outer_subfigs = np.atleast_1d(
            fig.subfigures(
                3,
                1,
                squeeze=False,
                hspace=0.02,
                height_ratios=[0.14, n_rows, 0.38],
            )
        ).ravel()
        header_subfig = outer_subfigs[0]
        param_block = outer_subfigs[1]
        legend_subfig = outer_subfigs[2]
        header_subfig.patch.set_visible(False)
        header_gs = header_subfig.add_gridspec(1, n_cols, wspace=0.32)
        for col_idx, (_, config_label, color) in enumerate(PAPER_CONFIG_ORDER):
            header_ax = header_subfig.add_subplot(header_gs[0, col_idx])
            header_ax.set_axis_off()
            header_ax.text(
                0.5,
                0.08,
                config_label,
                transform=header_ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10.0,
                fontweight="bold",
                color=color,
            )
        header_subfig.subplots_adjust(left=0.05, right=0.99, top=0.98, bottom=0.0)
    else:
        fig = plt.figure(figsize=figsize)
        outer_subfigs = np.atleast_1d(
            fig.subfigures(
                2,
                1,
                squeeze=False,
                hspace=0.02,
                height_ratios=[n_rows, 0.38],
            )
        ).ravel()
        param_block = outer_subfigs[0]
        legend_subfig = outer_subfigs[1]

    param_block.patch.set_linewidth(0)
    param_block.patch.set_edgecolor("#555555")
    param_block.patch.set_facecolor("#fbfbfb")

    param_subfigs = np.atleast_1d(
        param_block.subfigures(n_rows, 1, squeeze=False, hspace=0)
    ).ravel()
    for i in range(1, n_rows):
        y = 1.0 - (i / n_rows)
        param_block.add_artist(
            Line2D(
                [0.0, 1.0],
                [y, y],
                transform=param_block.transSubfigure,
                color="#555555",
                linewidth=1.1,
                zorder=10,
                clip_on=False,
            )
        )
    add_param_block_frame(param_block)
    add_significance_legend(legend_subfig)

    for row_idx, (subfig, (param_name, ylabel, ylim, active_configs)) in enumerate(
        zip(param_subfigs, param_specs)
    ):
        setup_param_section(subfig, ylabel)
        row_axes = row_plot_axes(subfig, n_cols)

        for ax, (config_key, _, color) in zip(row_axes, PAPER_CONFIG_ORDER):
            if config_key not in active_configs:
                plot_blank_cell(ax)
                continue
            values_by_y = values_by_y_for_cell(
                by_suite,
                config_key=config_key,
                param_name=param_name,
                y_vals=y_vals,
                suite_to_y=suite_to_y,
            )
            plot_param_boxplot_cell(
                ax,
                color=color,
                y_vals=y_vals,
                values_by_y=values_by_y,
                ylim=ylim,
            )

    fig.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def split_output_path(output_path: Path, part: str) -> Path:
    return output_path.with_name(f"{output_path.stem}_{part}{output_path.suffix}")


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
        default=root / "figures/Extra/hit_param_boxplots.png",
        help="Output PNG path stem (writes _a and _b page images)",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (default: auto from parameter count)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches (default: auto from parameter count)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    top_specs = PARAM_SPECS[:SPLIT_ROW_INDEX]
    bottom_specs = PARAM_SPECS[SPLIT_ROW_INDEX:]
    output_a = split_output_path(args.output, "a")
    output_b = split_output_path(args.output, "b")

    figsize_a = None
    figsize_b = None
    if args.fig_width is not None or args.fig_height is not None:
        width = 13.5 if args.fig_width is None else args.fig_width
        if args.fig_height is None:
            figsize_a = default_figsize(len(top_specs), show_header=True)
            figsize_b = default_figsize(len(bottom_specs), show_header=True)
        else:
            figsize_a = figsize_b = (width, args.fig_height)

    plot_hit_param_boxplot_figure(
        sessions_dir=args.sessions_dir,
        output_path=output_a,
        param_specs=top_specs,
        show_header=True,
        figsize=figsize_a,
    )
    plot_hit_param_boxplot_figure(
        sessions_dir=args.sessions_dir,
        output_path=output_b,
        param_specs=bottom_specs,
        show_header=True,
        figsize=figsize_b,
    )
    print(f"Sessions: {args.sessions_dir}")
    print(f"Parameters: {len(PARAM_SPECS)} ({len(top_specs)} + {len(bottom_specs)})")
    print(f"Wrote: {output_a}")
    print(f"Wrote: {output_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
