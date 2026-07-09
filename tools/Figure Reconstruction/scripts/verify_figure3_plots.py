#!/usr/bin/env python3
"""Cross-check Figure 3 and supplementary ridgeline metrics against raw CSVs."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, List

from plot_hit_rescreen_panel import (
    EVENT_COLS,
    N_RESCREEN,
    batch_rate_stats,
    load_rescreen_batch_rates_by_suite,
    load_rescreen_distributions_by_suite,
    load_rescreen_events_by_config,
    parse_rescreen_session_dir,
    suite_rescreen_batch_means,
    workspaces_non_hit_re_runs,
    workspaces_re_runs,
)
from plot_primary_batch_violins import PAPER_CONFIG_ORDER, hit_stats, workspaces_output
from plot_ratio_supplementary import SUITE_ORDER, load_by_suite, suite_means


def verify_ridgelines(
    *,
    sessions_dir,
    csv_glob: str,
    label: str,
) -> List[str]:
    issues: List[str] = []
    distributions = load_rescreen_distributions_by_suite(sessions_dir, csv_glob=csv_glob)
    session_map = {}
    for d in sessions_dir.iterdir():
        if d.is_dir():
            parsed = parse_rescreen_session_dir(d.name)
            if parsed:
                session_map[parsed] = d

    for suite, y_val, _ in SUITE_ORDER:
        for key, cfg_label, _ in PAPER_CONFIG_ORDER:
            counts = distributions.get(suite, {}).get(key, [])
            if not counts:
                issues.append(f"{label}: missing ridgeline data for {cfg_label} Y={y_val}")
                continue
            session_dir = session_map.get((key, suite))
            if session_dir is None:
                issues.append(f"{label}: missing session for {cfg_label} {suite}")
                continue
            csv_path = sorted(session_dir.glob(csv_glob))[0]
            session_counts = []
            with csv_path.open(encoding="utf-8") as fh:
                import csv

                for row in csv.DictReader(fh):
                    session_counts.append(int(row["n_hit_again"]))
            if counts != session_counts:
                issues.append(f"{label}: n_hit_again mismatch for {cfg_label} Y={y_val}")
            if any(c < 0 or c > N_RESCREEN for c in counts):
                issues.append(f"{label}: invalid count for {cfg_label} Y={y_val}")
    return issues


def verify_events(*, sessions_dir) -> List[str]:
    issues: List[str] = []
    by_config = load_rescreen_events_by_config(sessions_dir)
    session_map = {}
    for d in sessions_dir.iterdir():
        if d.is_dir():
            parsed = parse_rescreen_session_dir(d.name)
            if parsed:
                session_map[parsed] = d

    for suite, y_val, _ in SUITE_ORDER:
        for key, cfg_label, _ in PAPER_CONFIG_ORDER:
            session_dir = session_map.get((key, suite))
            if session_dir is None:
                issues.append(f"Events scatter: missing session for {cfg_label} {suite}")
                continue
            csv_path = sorted(session_dir.glob("primary_hit_rescreen_*.csv"))[0]
            session_points = []
            with csv_path.open(encoding="utf-8") as fh:
                import csv

                for row in csv.DictReader(fh):
                    if not all(row.get(col) for col in EVENT_COLS):
                        issues.append(
                            f"Events scatter: missing event columns for {cfg_label} Y={y_val}"
                        )
                        break
                    session_points.append(
                        (
                            int(row["n_hit_again"]),
                            sum(int(row[col]) for col in EVENT_COLS),
                            float(y_val),
                        )
                    )
                else:
                    loaded = [
                        pt for pt in by_config.get(key, []) if abs(pt[2] - float(y_val)) < 1e-9
                    ]
                    if len(loaded) != len(session_points):
                        issues.append(
                            f"Events scatter: point count mismatch for {cfg_label} Y={y_val}"
                        )
                    elif loaded != session_points:
                        issues.append(
                            f"Events scatter: point mismatch for {cfg_label} Y={y_val}"
                        )
    return issues


def verify(*, hit_csv, rescreen_csv, sessions_dir) -> List[str]:
    issues: List[str] = []

    by_suite = load_by_suite(hit_csv)
    y_vals, _, mean_hits, std_hits = suite_means(by_suite, n_sims=1000)
    rescreen_by = load_rescreen_batch_rates_by_suite(sessions_dir)
    _, mean_rates, std_rates = suite_rescreen_batch_means(rescreen_by)

    for suite, y_val, _ in SUITE_ORDER:
        idx = y_vals.index(y_val)
        for key, label, _ in PAPER_CONFIG_ORDER:
            vals = by_suite[suite][key]
            m, s, _, _ = hit_stats(vals, n_sims=1000)
            if abs(m - mean_hits[key][idx]) > 1e-9:
                issues.append(f"Panel (a) mean mismatch {label} Y={y_val}")
            if abs(s - std_hits[key][idx]) > 1e-9:
                issues.append(f"Panel (a) std mismatch {label} Y={y_val}")
            if len(vals) != 100:
                issues.append(f"Panel (a) expected 100 batches for {label} Y={y_val}, got {len(vals)}")

    session_map = {}
    for d in sessions_dir.iterdir():
        if d.is_dir():
            parsed = parse_rescreen_session_dir(d.name)
            if parsed:
                session_map[parsed] = d

    for suite, y_val, _ in SUITE_ORDER:
        idx = y_vals.index(y_val)
        for key, label, _ in PAPER_CONFIG_ORDER:
            batch_rates = rescreen_by.get(suite, {}).get(key)
            if batch_rates is None:
                issues.append(f"Panel (b) missing batch data for {label} {suite}")
                continue
            if len(batch_rates) != 100:
                issues.append(
                    f"Panel (b) expected 100 batches for {label} Y={y_val}, "
                    f"got {len(batch_rates)}"
                )
            mean, std = batch_rate_stats(batch_rates)
            if abs(mean - mean_rates[key][idx]) > 1e-9:
                issues.append(f"Panel (b) mean mismatch {label} Y={y_val}")
            if abs(std - std_rates[key][idx]) > 1e-9:
                issues.append(f"Panel (b) std mismatch {label} Y={y_val}")

            session_dir = session_map.get((key, suite))
            if session_dir is None:
                issues.append(f"Panel (b) missing session for {label} {suite}")
                continue
            csv_path = sorted(session_dir.glob("primary_hit_rescreen_*.csv"))[0]
            rates, counts, n_seeds_vals = [], [], []
            with csv_path.open(encoding="utf-8") as fh:
                import csv

                for row in csv.DictReader(fh):
                    rates.append(float(row["hit_rate"]))
                    counts.append(int(row["n_hit_again"]))
                    n_seeds_vals.append(int(row["n_seeds"]))
            if len(set(n_seeds_vals)) != 1 or n_seeds_vals[0] != N_RESCREEN:
                issues.append(f"Panel (b) unexpected n_seeds for {label} Y={y_val}")
            from_counts = sum(counts) / (len(counts) * N_RESCREEN)
            direct_mean = sum(rates) / len(rates)
            if abs(direct_mean - from_counts) > 1e-4:
                issues.append(f"Panel (b) hit_rate != n_hit_again/{N_RESCREEN} for {label} Y={y_val}")

    issues.extend(
        verify_ridgelines(
            sessions_dir=sessions_dir,
            csv_glob="primary_hit_rescreen_*.csv",
            label="Hit ridgelines",
        )
    )
    issues.extend(
        verify_ridgelines(
            sessions_dir=workspaces_non_hit_re_runs() / "sessions",
            csv_glob="primary_non_hit_rescreen_*.csv",
            label="Non-hit ridgelines",
        )
    )
    issues.extend(verify_events(sessions_dir=sessions_dir))

    return issues


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hit-csv",
        default=workspaces_output() / "Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv",
        type=str,
    )
    parser.add_argument(
        "--rescreen-csv",
        default=workspaces_re_runs() / "primary_hit_rescreen_compare.csv",
        type=str,
    )
    parser.add_argument(
        "--sessions-dir",
        default=workspaces_re_runs() / "sessions",
        type=str,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    issues = verify(
        hit_csv=args.hit_csv,
        rescreen_csv=args.rescreen_csv,
        sessions_dir=args.sessions_dir,
    )
    if issues:
        print(f"FAILED: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    print("OK: Figure 3 panels (a–b), supplementary ridgelines, and events scatter match raw CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
