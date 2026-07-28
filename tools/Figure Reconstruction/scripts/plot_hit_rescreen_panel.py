#!/usr/bin/env python3
"""Panels B for Figure 3 and supplementary ridgeline plot of re-screen distributions."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_primary_batch_violins import MAIN_CONFIG_ORDER, PAPER_CONFIG_ORDER, workspaces_output
from plot_ratio_supplementary import SUITE_ORDER, apply_panel_figure_layout, apply_figure3_axis_typography, format_y_tick, panel_label, FIG3_LEGENDSIZE, N_SIMS_PER_BATCH, x_positions_for_y_vals

RescreenRow = Tuple[float, float, int]
BATCHES_PER_CAMPAIGN = 100
N_RESCREEN = 20
EVENT_COLS: Tuple[str, ...] = (
    "total_deaths",
    "total_duplications",
    "total_mutations",
    "total_outflows",
)
RescreenEventPoint = Tuple[int, int, float]  # n_hit_again, total_events, y_val

Y_SCATTER_COLORS: Dict[float, str] = {
    y_val: color
    for (_, y_val, _), color in zip(
        SUITE_ORDER,
        [
            "#dadaeb",
            "#bcbddc",
            "#9e9ac8",
            "#4e79a7",
            "#59a14f",
            "#e15759",
            "#b07aa1",
            "#f28e2b",
            "#76b7b2",
            "#edc948",
            "#ff9da7",
        ],
    )
}

PREFIX_TO_KEY = {
    "a": "Neutral",
    "aa": "trueNeutral2",
    "b": "binary_death",
    "c": "justDeath",
    "d": "justDup",
    "e": "Death+Dup",
}
_SESSION_DIR_RE = re.compile(r"^(a|aa|b|c|d|e)_(.+)$")


def workspaces_re_runs() -> Path:
    return workspaces_output() / "Re-Runs"


def workspaces_non_hit_re_runs() -> Path:
    return workspaces_output() / "Re-Runs-NonHits"


def parse_rescreen_session_dir(name: str) -> Tuple[str, str] | None:
    m = _SESSION_DIR_RE.fullmatch(name)
    if not m:
        return None
    prefix, rest = m.group(1), m.group(2)
    config_key = PREFIX_TO_KEY[prefix]
    if "_Fixed_" in rest:
        _, y_part = rest.split("_Fixed_", 1)
        suite_tag = f"Fixed_{y_part}_ratio"
    else:
        suite_tag = "Variable_Ratio"
    return config_key, suite_tag


def load_rescreen_distributions_by_suite(
    sessions_dir: Path,
    *,
    csv_glob: str = "primary_hit_rescreen_*.csv",
) -> Dict[str, Dict[str, List[int]]]:
    """Return suite -> config_key -> per-draw re-run success counts (0--N_RESCREEN)."""
    out: Dict[str, Dict[str, List[int]]] = {}
    if not sessions_dir.is_dir():
        return out
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_rescreen_session_dir(session_dir.name)
        if parsed is None:
            continue
        config_key, suite_tag = parsed
        if config_key not in {k for k, _, _ in PAPER_CONFIG_ORDER}:
            continue
        if not suite_tag.startswith("Fixed_"):
            continue
        csvs = sorted(session_dir.glob(csv_glob))
        if not csvs:
            continue
        counts: List[int] = []
        with csvs[0].open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                counts.append(int(row["n_hit_again"]))
        if counts:
            out.setdefault(suite_tag, {})[config_key] = counts
    return out


def _total_events(row: dict) -> int:
    return sum(int(row[col]) for col in EVENT_COLS)


def load_rescreen_events_by_config(
    sessions_dir: Path,
    *,
    csv_glob: str = "primary_hit_rescreen_*.csv",
    y_vals: Sequence[float] | None = None,
) -> Dict[str, List[RescreenEventPoint]]:
    """Return config_key -> list of (n_hit_again, total_events, y_val)."""
    if y_vals is None:
        y_vals = [y for _, y, _ in SUITE_ORDER]
    y_set = set(float(y) for y in y_vals)
    suite_to_y = {suite: y_val for suite, y_val, _ in SUITE_ORDER if y_val in y_set}
    out: Dict[str, List[RescreenEventPoint]] = {key: [] for key, _, _ in PAPER_CONFIG_ORDER}
    if not sessions_dir.is_dir():
        return out
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_rescreen_session_dir(session_dir.name)
        if parsed is None:
            continue
        config_key, suite_tag = parsed
        if config_key not in out:
            continue
        y_val = suite_to_y.get(suite_tag)
        if y_val is None:
            continue
        csvs = sorted(session_dir.glob(csv_glob))
        if not csvs:
            continue
        with csvs[0].open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not all(row.get(col) for col in EVENT_COLS):
                    continue
                out[config_key].append(
                    (int(row["n_hit_again"]), _total_events(row), float(y_val))
                )
    return out


def load_rescreen_by_suite(csv_path: Path) -> Dict[str, Dict[str, RescreenRow]]:
    """Return suite -> config_key -> (mean_hit_rate, std_hit_rate, n_hits_screened)."""
    rows: Dict[str, Dict[str, RescreenRow]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            suite = row["suite_tag"]
            key = row["config_key"]
            if key not in {k for k, _, _ in PAPER_CONFIG_ORDER}:
                continue
            if not suite.startswith("Fixed_"):
                continue
            rate = float(row["mean_hit_rate"])
            n = int(row["n_hits_screened"])
            if row.get("std_hit_rate"):
                std = float(row["std_hit_rate"])
            else:
                std = float(np.sqrt(rate * (1.0 - rate) / n)) if n > 0 else 0.0
            rows.setdefault(suite, {})[key] = (rate, std, n)
    return rows


def batch_rate_stats(vals: Sequence[float]) -> Tuple[float, float]:
    """Return mean and sample std of per-batch mean re-screen rates."""
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _batch_mean_rates_from_session_csv(
    csv_path: Path,
    *,
    sims_per_batch: int = N_SIMS_PER_BATCH,
    n_batches: int = BATCHES_PER_CAMPAIGN,
) -> List[float]:
    """Mean per-hit re-run rate within each primary batch (0..n_batches-1)."""
    batches: Dict[int, List[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            src = row.get("source_run_index")
            if not src:
                continue
            batch_index = int(src) // int(sims_per_batch)
            batches[batch_index].append(float(row["hit_rate"]))
    return [
        float(np.mean(batches[bi])) if batches.get(bi) else 0.0
        for bi in range(n_batches)
    ]


def load_rescreen_batch_rates_by_suite(
    sessions_dir: Path,
    *,
    sims_per_batch: int = N_SIMS_PER_BATCH,
    n_batches: int = BATCHES_PER_CAMPAIGN,
    csv_glob: str = "primary_hit_rescreen_*.csv",
) -> Dict[str, Dict[str, List[float]]]:
    """Return suite -> config_key -> per-batch mean re-screen rates (length n_batches)."""
    out: Dict[str, Dict[str, List[float]]] = {}
    if not sessions_dir.is_dir():
        return out
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_rescreen_session_dir(session_dir.name)
        if parsed is None:
            continue
        config_key, suite_tag = parsed
        if config_key not in {k for k, _, _ in PAPER_CONFIG_ORDER}:
            continue
        if not suite_tag.startswith("Fixed_"):
            continue
        csvs = sorted(session_dir.glob(csv_glob))
        if not csvs:
            continue
        batch_rates = _batch_mean_rates_from_session_csv(
            csvs[0],
            sims_per_batch=sims_per_batch,
            n_batches=n_batches,
        )
        out.setdefault(suite_tag, {})[config_key] = batch_rates
    return out


def suite_rescreen_batch_means(
    by_suite: Dict[str, Dict[str, List[float]]],
    *,
    config_order: Sequence[Tuple[str, str, str]] = PAPER_CONFIG_ORDER,
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]]]:
    """Aggregate per-batch re-screen rates into campaign means and s.d.s."""
    y_vals: List[float] = []
    mean_rates: Dict[str, List[float]] = {key: [] for key, _, _ in config_order}
    std_rates: Dict[str, List[float]] = {key: [] for key, _, _ in config_order}
    for suite, y_val, _ in SUITE_ORDER:
        cfgs = by_suite.get(suite)
        y_vals.append(y_val)
        for key, _, _ in config_order:
            if cfgs is None:
                mean_rates[key].append(float("nan"))
                std_rates[key].append(float("nan"))
                continue
            batch_rates = cfgs.get(key)
            if not batch_rates:
                mean_rates[key].append(float("nan"))
                std_rates[key].append(float("nan"))
                continue
            mean, std = batch_rate_stats(batch_rates)
            mean_rates[key].append(mean)
            std_rates[key].append(std)
    return y_vals, mean_rates, std_rates


def suite_rescreen_rates(
    by_suite: Dict[str, Dict[str, RescreenRow]],
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[int]]]:
    y_vals: List[float] = []
    mean_rates: Dict[str, List[float]] = {key: [] for key, _, _ in PAPER_CONFIG_ORDER}
    std_rates: Dict[str, List[float]] = {key: [] for key, _, _ in PAPER_CONFIG_ORDER}
    n_hits: Dict[str, List[int]] = {key: [] for key, _, _ in PAPER_CONFIG_ORDER}
    for suite, y_val, _ in SUITE_ORDER:
        cfgs = by_suite.get(suite)
        y_vals.append(y_val)
        for key, _, _ in PAPER_CONFIG_ORDER:
            if cfgs is None:
                mean_rates[key].append(float("nan"))
                std_rates[key].append(float("nan"))
                n_hits[key].append(0)
                continue
            entry = cfgs.get(key)
            if entry is None:
                mean_rates[key].append(float("nan"))
                std_rates[key].append(float("nan"))
                n_hits[key].append(0)
                continue
            rate, std, n = entry
            mean_rates[key].append(rate)
            std_rates[key].append(std)
            n_hits[key].append(n)
    return y_vals, mean_rates, std_rates, n_hits


def plot_rescreen_panel(
    ax: plt.Axes,
    *,
    y_vals: Sequence[float],
    mean_rates: Dict[str, List[float]],
    std_rates: Dict[str, List[float]] | None = None,
    show_legend: bool = True,
    panel_letter: str = "b",
    show_xlabel: bool = True,
    even_x_spacing: bool = False,
    config_order: Sequence[Tuple[str, str, str]] = PAPER_CONFIG_ORDER,
) -> None:
    y_max = 0.0
    x_arr = x_positions_for_y_vals(y_vals, even_spacing=even_x_spacing)
    for key, label, color in config_order:
        means = 100.0 * np.asarray(mean_rates[key], dtype=float)
        mask = np.isfinite(means)
        if not np.any(mask):
            continue
        if std_rates is not None:
            stds = 100.0 * np.asarray(std_rates[key], dtype=float)
            y_max = max(y_max, float(np.nanmax(means[mask] + stds[mask])))
            ax.errorbar(
                x_arr[mask],
                means[mask],
                yerr=stds[mask],
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
        else:
            y_max = max(y_max, float(np.nanmax(means[mask])))
            ax.plot(
                x_arr[mask],
                means[mask],
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=label,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )

    ax.set_xticks(list(x_arr))
    ax.set_xticklabels([format_y_tick(y) for y in y_vals])
    if show_xlabel:
        ax.set_xlabel("Fixed task energy yield ratio $Y$")
    ylabel = (
        "Mean batch per-hit re-run success rate (%)"
        if std_rates is not None
        else "Mean per-hit re-run success rate (%)"
    )
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    apply_figure3_axis_typography(ax)
    if show_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=FIG3_LEGENDSIZE)
    panel_label(ax, panel_letter)


def _batch_n_hit_again_lists_from_session_csv(
    csv_path: Path,
    *,
    sims_per_batch: int = N_SIMS_PER_BATCH,
    n_batches: int = BATCHES_PER_CAMPAIGN,
) -> List[List[int]]:
    """Per-batch lists of re-run success counts (length n_batches)."""
    batches: Dict[int, List[int]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            src = row.get("source_run_index")
            if not src:
                continue
            batch_index = int(src) // int(sims_per_batch)
            batches[batch_index].append(int(row["n_hit_again"]))
    return [batches.get(bi, []) for bi in range(n_batches)]


def load_rescreen_batch_counts_by_suite(
    sessions_dir: Path,
    *,
    sims_per_batch: int = N_SIMS_PER_BATCH,
    n_batches: int = BATCHES_PER_CAMPAIGN,
    csv_glob: str = "primary_hit_rescreen_*.csv",
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Return suite -> config_key -> per-batch n_hit_again lists."""
    out: Dict[str, Dict[str, List[List[int]]]] = {}
    if not sessions_dir.is_dir():
        return out
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        parsed = parse_rescreen_session_dir(session_dir.name)
        if parsed is None:
            continue
        config_key, suite_tag = parsed
        if config_key not in {k for k, _, _ in PAPER_CONFIG_ORDER}:
            continue
        if not suite_tag.startswith("Fixed_"):
            continue
        csvs = sorted(session_dir.glob(csv_glob))
        if not csvs:
            continue
        batch_lists = _batch_n_hit_again_lists_from_session_csv(
            csvs[0],
            sims_per_batch=sims_per_batch,
            n_batches=n_batches,
        )
        out.setdefault(suite_tag, {})[config_key] = batch_lists
    return out


def _per_batch_density_profiles(
    batch_count_lists: Sequence[Sequence[int]],
    *,
    max_count: int,
) -> np.ndarray:
    """Return (n_batches, max_count+1) density profiles for each batch."""
    profiles: List[np.ndarray] = []
    bin_edges = np.arange(-0.5, max_count + 1.5, 1.0)
    for counts in batch_count_lists:
        if not counts:
            profiles.append(np.zeros(max_count + 1, dtype=float))
            continue
        values = np.asarray(counts, dtype=float)
        hist, _ = np.histogram(values, bins=bin_edges, density=True)
        profiles.append(np.clip(hist, 0.0, None))
    return np.asarray(profiles, dtype=float)


def _ridge_histogram_batch_stats(
    batch_count_lists: Sequence[Sequence[int]],
    *,
    max_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean ridge heights with Q1–Q3 across batches, peak-normalized like pooled ridges."""
    x = np.arange(0, max_count + 1, dtype=float)
    nonempty_lists = [counts for counts in batch_count_lists if counts]
    profiles = _per_batch_density_profiles(nonempty_lists, max_count=max_count)
    if profiles.size == 0:
        z = np.zeros_like(x)
        return x, z, z, z
    mean_prof = profiles.mean(axis=0)
    if profiles.shape[0] > 1:
        q1_prof = np.percentile(profiles, 25, axis=0)
        q3_prof = np.percentile(profiles, 75, axis=0)
    else:
        q1_prof = mean_prof.copy()
        q3_prof = mean_prof.copy()
    peak = float(mean_prof.max())
    if peak > 0.0:
        mean_prof = mean_prof / peak
        q1_prof = q1_prof / peak
        q3_prof = q3_prof / peak
    return x, mean_prof, q1_prof, q3_prof


def _ridge_errorbar_yerr(
    mean_density: np.ndarray,
    q1_density: np.ndarray,
    q3_density: np.ndarray,
    *,
    ridge_height: float,
    peak_floor: float | None = None,
) -> np.ndarray:
    """Map batch quartile densities to asymmetric ridge-axis error bars (Q1–Q3)."""
    mean_arr = np.asarray(mean_density, dtype=float)
    q1_arr = np.asarray(q1_density, dtype=float)
    q3_arr = np.asarray(q3_density, dtype=float)
    lower = ridge_height * np.maximum(mean_arr - q1_arr, 0.0)
    upper = ridge_height * np.maximum(q3_arr - mean_arr, 0.0)
    yerr = np.vstack([lower, upper])
    if peak_floor is None or peak_floor <= 0.0:
        return yerr
    peak_yerr = float(np.max(yerr))
    if peak_yerr <= 0.0 or peak_yerr >= peak_floor:
        return yerr
    return yerr * (peak_floor / peak_yerr)


def _ridge_histogram(values: np.ndarray, *, max_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integer-bin histogram heights normalized to peak 1 (x = 0..max_count)."""
    x = np.arange(0, max_count + 1, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return x, np.zeros_like(x)
    bin_edges = np.arange(-0.5, max_count + 1.5, 1.0)
    counts, _ = np.histogram(values, bins=bin_edges, density=True)
    counts = np.clip(counts, 0.0, None)
    peak = float(counts.max())
    if peak > 0.0:
        counts = counts / peak
    return x, counts


def plot_regime_ridgeline_panel(
    ax: plt.Axes,
    *,
    config_key: str,
    config_label: str,
    color: str,
    y_vals: Sequence[float],
    batch_counts_by_suite: Dict[str, Dict[str, List[List[int]]]],
    ridge_spacing: float = 0.42,
    ridge_height: float = 0.36,
    n_rescreen: int = N_RESCREEN,
    show_xlabel: bool = True,
    show_xticklabels: bool = True,
    show_ylabel: bool = True,
    panel_letter: str | None = None,
    xlabel: str | None = None,
    hide_std_y_vals: Sequence[float] | None = None,
    errorbar_peak_floor: float | None = None,
) -> None:
    """Ridgelines for one regime: one ridge per fixed $Y$ value."""
    hide_std = set(hide_std_y_vals or ())
    ridges: List[Tuple[str, float, str, List[List[int]]]] = []
    for suite, y_val, y_label in SUITE_ORDER:
        if y_val not in y_vals:
            continue
        batch_lists = batch_counts_by_suite.get(suite, {}).get(config_key)
        if not batch_lists:
            continue
        ridges.append((y_label, y_val, suite, batch_lists))

    group_centers = [
        ridge_idx * ridge_spacing for ridge_idx in range(len(ridges))
    ]
    group_labels = [label for label, _, _, _ in ridges]

    for ridge_idx, (_, y_val, _suite, batch_lists) in enumerate(ridges):
        base_y = ridge_idx * ridge_spacing
        x_pts, mean_density, q1_density, q3_density = _ridge_histogram_batch_stats(
            batch_lists,
            max_count=n_rescreen,
        )
        top_y = base_y + ridge_height * mean_density
        ax.fill_between(
            x_pts,
            base_y,
            top_y,
            color=color,
            alpha=0.78,
            linewidth=0.0,
            zorder=2,
        )
        if y_val not in hide_std:
            yerr = _ridge_errorbar_yerr(
                mean_density,
                q1_density,
                q3_density,
                ridge_height=ridge_height,
                peak_floor=errorbar_peak_floor,
            )
            q1_y = base_y + ridge_height * q1_density
            q3_y = base_y + ridge_height * q3_density
            ax.fill_between(
                x_pts,
                q1_y,
                q3_y,
                color="#333333",
                alpha=0.12,
                linewidth=0.0,
                zorder=3,
            )
            ax.errorbar(
                x_pts,
                top_y,
                yerr=yerr,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.85,
                capsize=1.8,
                capthick=0.65,
                zorder=4,
            )
        ax.plot(
            x_pts,
            top_y,
            color="black",
            linewidth=0.65,
            alpha=0.9,
            marker="o",
            markersize=2.5,
            markerfacecolor="black",
            markeredgecolor="black",
            zorder=3,
        )

    y_top = max(len(ridges) - 1, 0) * ridge_spacing + ridge_height + 0.08
    ax.set_xlim(-0.35, float(n_rescreen) + 0.85)
    ax.set_ylim(-0.22, y_top)
    if show_xlabel:
        ax.set_xlabel(
            xlabel if xlabel is not None else f"Re-run successes per hit (out of {n_rescreen})"
        )
    if show_ylabel:
        ax.set_ylabel("Fixed task energy yield ratio $Y$")
    x_ticks = np.arange(0, n_rescreen + 1, 5)
    ax.set_xticks(x_ticks)
    if show_xticklabels:
        ax.set_xticklabels([f"{int(t)}" for t in x_ticks])
        ax.tick_params(axis="x", which="both", labelbottom=True, bottom=True)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(group_centers)
    ax.set_yticklabels(group_labels)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(config_label, fontsize=10.5, fontweight="bold", color=color, pad=6)
    if panel_letter is not None:
        panel_label(ax, panel_letter, x=-0.20, y=1.02)


def plot_rescreen_ridgeline_figure(
    *,
    sessions_dir: Path | None = None,
    data_csv: Path | None = None,
    data_row_type: str = "hit",
    output_path: Path,
    y_vals: Sequence[float] | None = None,
    figsize: Tuple[float, float] | None = None,
    ridge_spacing: float = 0.42,
    ridge_height: float = 0.36,
    csv_glob: str = "primary_hit_rescreen_*.csv",
    xlabel: str | None = None,
    hide_std_y_vals: Sequence[float] | None = None,
    errorbar_peak_floor: float | None = None,
    config_order: Sequence[Tuple[str, str, str]] = PAPER_CONFIG_ORDER,
) -> None:
    """Supplementary ridgeline figure: one panel per regime, ridges by fixed $Y$."""
    if data_csv is not None:
        from figure_csv import load_rescreen_batch_counts_by_suite as load_counts_csv

        batch_counts_by_suite = load_counts_csv(data_csv, row_type=data_row_type)
    else:
        if sessions_dir is None:
            raise ValueError("sessions_dir or data_csv is required")
        batch_counts_by_suite = load_rescreen_batch_counts_by_suite(
            sessions_dir,
            csv_glob=csv_glob,
        )
    if xlabel is None:
        xlabel = f"Re-run successes per draw (out of {N_RESCREEN})"
    if y_vals is None:
        y_vals = [y for _, y, _ in SUITE_ORDER]

    n_panels = len(config_order)
    if n_panels <= 2:
        nrows, ncols = 1, n_panels
        if figsize is None:
            figsize = (11.0, 5.4)
    else:
        nrows, ncols = 2, 2
        if figsize is None:
            figsize = (11.0, 10.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    panel_letters = "abcd"
    for panel_idx, ((key, label, color), letter) in enumerate(
        zip(config_order, panel_letters)
    ):
        ax = axes.ravel()[panel_idx]
        plot_regime_ridgeline_panel(
            ax,
            config_key=key,
            config_label=label,
            color=color,
            y_vals=y_vals,
            batch_counts_by_suite=batch_counts_by_suite,
            ridge_spacing=ridge_spacing,
            ridge_height=ridge_height,
            show_xlabel=True,
            show_xticklabels=True,
            show_ylabel=panel_idx % ncols == 0,
            panel_letter=letter,
            xlabel=xlabel,
            hide_std_y_vals=hide_std_y_vals,
            errorbar_peak_floor=errorbar_peak_floor,
        )
    for ax in axes.ravel()[n_panels:]:
        ax.set_axis_off()

    apply_panel_figure_layout(fig, nrows=nrows)
    fig.subplots_adjust(hspace=0.38, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_regime_events_panel(
    ax: plt.Axes,
    *,
    config_label: str,
    color: str,
    points: Sequence[RescreenEventPoint],
    n_rescreen: int = N_RESCREEN,
    max_points: int = 6000,
    x_min: int | None = None,
    x_max: int | None = None,
    events_scale: float = 1.0e6,
    point_alpha: float = 0.22,
    point_size: float = 10.0,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    panel_letter: str | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    """Scatter of primary-run demographic events vs re-screen success count for one regime."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = list(points)
    if x_min is not None or x_max is not None:
        lo = 0 if x_min is None else int(x_min)
        hi = n_rescreen if x_max is None else int(x_max)
        pts = [pt for pt in pts if lo <= pt[0] <= hi]
    if len(pts) > max_points:
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = [pts[i] for i in sorted(idx)]

    jitter_span = 0.15 if x_min is not None or x_max is not None else 0.18
    for n_hit, total_events, y_val in pts:
        jitter = float(rng.uniform(-jitter_span, jitter_span))
        ax.scatter(
            float(n_hit) + jitter,
            float(total_events) / events_scale,
            s=point_size,
            alpha=point_alpha,
            linewidths=0.0,
            color=Y_SCATTER_COLORS.get(y_val, "#888888"),
            zorder=2,
        )

    if x_min is not None or x_max is not None:
        lo = 0 if x_min is None else int(x_min)
        hi = n_rescreen if x_max is None else int(x_max)
        ax.set_xlim(float(lo) - 0.5, float(hi) + 0.5)
        ax.set_xticks(np.arange(lo, hi + 1, 1))
    else:
        ax.set_xlim(-0.5, float(n_rescreen) + 0.5)
        ax.set_xticks(np.arange(0, n_rescreen + 1, 5))
    if show_xlabel:
        ax.set_xlabel(f"Re-run successes per hit (out of {n_rescreen})")
    if show_ylabel:
        ylabel = "Total events in primary run"
        if events_scale == 1.0e6:
            ylabel += " (millions)"
        elif events_scale == 1.0e3:
            ylabel += " (thousands)"
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_title(config_label, fontsize=10.5, fontweight="bold", color=color, pad=6)
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def add_events_zoom_highlight(
    ax: plt.Axes,
    *,
    x_min: int,
    x_max: int,
    y_min: float | None = None,
    y_max: float | None = None,
) -> Rectangle:
    """Draw a dotted box for the zoomed region in axis data coordinates."""
    if y_min is None or y_max is None:
        y_min, y_max = ax.get_ylim()
    rect = Rectangle(
        (float(x_min) - 0.5, float(y_min)),
        float(x_max - x_min) + 1.0,
        float(y_max) - float(y_min),
        linewidth=1.1,
        edgecolor="#444444",
        linestyle=(0, (4, 3)),
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect)
    return rect


def plot_rescreen_events_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
    y_vals: Sequence[float] | None = None,
    figsize: Tuple[float, float] | None = None,
    csv_glob: str = "primary_hit_rescreen_*.csv",
    max_points: int = 6000,
    include_true_neutral_zoom: bool = True,
    zoom_x_min: int = 16,
    zoom_x_max: int = 20,
    zoom_max_points: int = 10000,
) -> None:
    """Supplementary scatter: re-screen hits vs primary-run demographic events."""
    by_config = load_rescreen_events_by_config(sessions_dir, csv_glob=csv_glob, y_vals=y_vals)
    if y_vals is None:
        y_vals = [y for _, y, _ in SUITE_ORDER]

    rng = np.random.default_rng(0)
    if include_true_neutral_zoom:
        if figsize is None:
            figsize = (11.0, 13.5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.78], hspace=0.38, wspace=0.26)
        axes = {
            "a": fig.add_subplot(gs[0, 0]),
            "b": fig.add_subplot(gs[0, 1]),
            "c": fig.add_subplot(gs[1, 0]),
            "d": fig.add_subplot(gs[1, 1]),
            "e": fig.add_subplot(gs[2, :]),
        }
        panel_layout = (
            ("a", PAPER_CONFIG_ORDER[0]),
            ("b", PAPER_CONFIG_ORDER[1]),
            ("c", PAPER_CONFIG_ORDER[2]),
            ("d", PAPER_CONFIG_ORDER[3]),
        )
    else:
        if figsize is None:
            figsize = (11.0, 10.5)
        fig, grid_axes = plt.subplots(2, 2, figsize=figsize)
        axes = {letter: ax for letter, ax in zip("abcd", grid_axes.ravel())}
        panel_layout = tuple(zip("abcd", PAPER_CONFIG_ORDER))

    for letter, (key, label, color) in panel_layout:
        ax = axes[letter]
        plot_regime_events_panel(
            ax,
            config_label=label,
            color=color,
            points=by_config.get(key, []),
            show_xlabel=True,
            show_ylabel=letter in {"a", "c"},
            panel_letter=letter,
            max_points=max_points,
            rng=rng,
        )

    if include_true_neutral_zoom:
        key, label, color = PAPER_CONFIG_ORDER[0]
        ax_e = axes["e"]
        zoom_events_scale = 1.0e3
        main_events_scale = 1.0e6
        plot_regime_events_panel(
            ax_e,
            config_label=label,
            color=color,
            points=by_config.get(key, []),
            x_min=zoom_x_min,
            x_max=zoom_x_max,
            events_scale=zoom_events_scale,
            point_alpha=0.35,
            point_size=12.0,
            max_points=zoom_max_points,
            show_xlabel=True,
            show_ylabel=True,
            panel_letter="e",
            rng=rng,
        )
        e_ymin, e_ymax = ax_e.get_ylim()
        scale_ratio = zoom_events_scale / main_events_scale
        add_events_zoom_highlight(
            axes["a"],
            x_min=zoom_x_min,
            x_max=zoom_x_max,
            y_min=e_ymin * scale_ratio,
            y_max=e_ymax * scale_ratio,
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=Y_SCATTER_COLORS[y_val],
            markeredgecolor=Y_SCATTER_COLORS[y_val],
            label=format_y_tick(y_val),
        )
        for _, y_val, _ in SUITE_ORDER
        if y_val in y_vals
    ]
    fig.legend(
        handles=legend_handles,
        title="Fixed $Y$",
        loc="lower center",
        ncol=min(6, len(legend_handles)),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    bottom = 0.12 if include_true_neutral_zoom else 0.14
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=bottom)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_true_neutral_events_zoom_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
    x_min: int = 16,
    x_max: int = 20,
    y_vals: Sequence[float] | None = None,
    figsize: Tuple[float, float] = (7.5, 5.5),
    csv_glob: str = "primary_hit_rescreen_*.csv",
    max_points: int = 10000,
) -> None:
    """Standalone panel (e) only; prefer ``plot_rescreen_events_figure`` for the full S4 layout."""
    by_config = load_rescreen_events_by_config(sessions_dir, csv_glob=csv_glob, y_vals=y_vals)
    if y_vals is None:
        y_vals = [y for _, y, _ in SUITE_ORDER]

    key, label, color = PAPER_CONFIG_ORDER[0]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_regime_events_panel(
        ax,
        config_label=label,
        color=color,
        points=by_config.get(key, []),
        x_min=x_min,
        x_max=x_max,
        events_scale=1.0e3,
        point_alpha=0.35,
        point_size=12.0,
        max_points=max_points,
        panel_letter="e",
        rng=rng,
    )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=Y_SCATTER_COLORS[y_val],
            markeredgecolor=Y_SCATTER_COLORS[y_val],
            label=format_y_tick(y_val),
        )
        for _, y_val, _ in SUITE_ORDER
        if y_val in y_vals
    ]
    ax.legend(
        handles=legend_handles,
        title="Fixed $Y$",
        loc="upper right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


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
        default=root / "figures/Extra/html/figures/figure3_panel_b_rescreen.png",
        help="Output PNG path",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    rescreen_by_suite = load_rescreen_batch_rates_by_suite(args.sessions_dir)
    y_vals, mean_rates, std_rates = suite_rescreen_batch_means(rescreen_by_suite)

    print(f"Sessions: {args.sessions_dir}")
    for idx, (_, y_val, _) in enumerate(SUITE_ORDER):
        print(f"  Y={y_val:g}:")
        for key, cfg_label, _ in PAPER_CONFIG_ORDER:
            rate = mean_rates[key][idx]
            std = std_rates[key][idx]
            if not np.isfinite(rate):
                print(f"    {cfg_label}: (no re-screen data)")
                continue
            print(
                f"    {cfg_label}: {100.0 * rate:.1f}% ± {100.0 * std:.1f}% "
                f"(batch mean ± s.d.)"
            )

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot_rescreen_panel(
        ax,
        y_vals=y_vals,
        mean_rates=mean_rates,
        std_rates=std_rates,
    )
    apply_panel_figure_layout(fig, nrows=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
