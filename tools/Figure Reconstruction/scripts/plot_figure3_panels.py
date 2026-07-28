#!/usr/bin/env python3
"""Figure 3: panel (a) hit counts and (b) mean re-screen rates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt

from plot_hit_rescreen_panel import (
    load_rescreen_batch_rates_by_suite,
    plot_rescreen_panel,
    suite_rescreen_batch_means,
    workspaces_re_runs,
)
from plot_primary_batch_violins import MAIN_CONFIG_ORDER, PAPER_CONFIG_ORDER
from plot_ratio_supplementary import (
    apply_panel_figure_layout,
    load_by_suite,
    plot_hit_counts_panel,
    suite_means,
    workspaces_output,
)

ConfigOrder = Sequence[Tuple[str, str, str]]


def plot_figure3_panels(
    *,
    data_csv: Path | None,
    hit_csv: Path | None,
    rescreen_sessions_dir: Path | None,
    output_path: Path,
    n_sims: int | None = None,
    config_order: ConfigOrder = MAIN_CONFIG_ORDER,
) -> None:
    if data_csv is not None:
        from figure_csv import infer_campaign_shape
        from figure_csv import load_by_suite as load_by_suite_csv
        from figure_csv import load_rescreen_batch_rates_by_suite as load_rates_csv

        by_suite = load_by_suite_csv(data_csv)
        rescreen_by_suite = load_rates_csv(data_csv)
        if n_sims is None:
            n_sims, _, _ = infer_campaign_shape(data_csv)
    else:
        if hit_csv is None or rescreen_sessions_dir is None:
            raise ValueError("Provide --data-csv or both --hit-csv and --rescreen-sessions-dir")
        by_suite = load_by_suite(hit_csv)
        rescreen_by_suite = load_rescreen_batch_rates_by_suite(rescreen_sessions_dir)
        if n_sims is None:
            from figure_csv import infer_n_sims_from_rows

            n_sims = infer_n_sims_from_rows(hit_csv)

    y_vals, _, mean_hits, std_hits = suite_means(
        by_suite, n_sims=n_sims, config_order=config_order
    )
    _, mean_rates, std_rates = suite_rescreen_batch_means(
        rescreen_by_suite, config_order=config_order
    )

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10.0, 8.2))
    plot_hit_counts_panel(
        ax_a,
        y_vals=y_vals,
        mean_hits=mean_hits,
        std_hits=std_hits,
        show_legend=True,
        panel_letter="a",
        show_xlabel=False,
        show_xticklabels=True,
        even_x_spacing=True,
        n_sims=n_sims,
        config_order=config_order,
    )
    plot_rescreen_panel(
        ax_b,
        y_vals=y_vals,
        mean_rates=mean_rates,
        std_rates=std_rates,
        show_legend=False,
        panel_letter="b",
        even_x_spacing=True,
        config_order=config_order,
    )
    xlim = (-0.5, len(y_vals) - 0.5)
    ax_a.set_xlim(xlim)
    ax_b.set_xlim(xlim)
    apply_panel_figure_layout(fig, nrows=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict = {"bbox_inches": "tight", "pad_inches": 0.04}
    if output_path.suffix.lower() == ".pdf":
        save_kw["format"] = "pdf"
    else:
        save_kw["dpi"] = 300
    fig.savefig(output_path, **save_kw)
    plt.close(fig)


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
        "--hit-csv",
        type=Path,
        default=workspaces_output()
        / "Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv",
        help="Summary_Ratio batch hit-count CSV (ignored if --data-csv is set)",
    )
    parser.add_argument(
        "--rescreen-sessions-dir",
        type=Path,
        default=workspaces_re_runs() / "sessions",
        help="Re-Runs session folders (ignored if --data-csv is set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/Used/figure3_hit_panels.pdf",
        help="Combined Figure 3 path (.pdf preferred; .png also supported)",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Simulations per batch (default: infer from data CSV)",
    )
    parser.add_argument(
        "--configs",
        choices=("main", "all"),
        default="main",
        help="main: Neutral + Death+Dup (default); all: four configurations",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_order = MAIN_CONFIG_ORDER if args.configs == "main" else PAPER_CONFIG_ORDER

    plot_figure3_panels(
        data_csv=args.data_csv,
        hit_csv=args.hit_csv,
        rescreen_sessions_dir=args.rescreen_sessions_dir,
        output_path=args.output,
        n_sims=args.n_sims,
        config_order=config_order,
    )
    if args.data_csv is not None:
        print(f"Data CSV: {args.data_csv}")
    else:
        print(f"Hit CSV: {args.hit_csv}")
        print(f"Re-screen sessions: {args.rescreen_sessions_dir}")
    print(f"Configs: {args.configs}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
