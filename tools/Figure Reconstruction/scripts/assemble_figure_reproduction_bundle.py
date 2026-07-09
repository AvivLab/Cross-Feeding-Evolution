#!/usr/bin/env python3
"""Assemble data/figure_reproduction/: batch_hit_counts.csv, minimal scripts, TikZ sources."""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "data" / "figure_reproduction"
SCRIPTS_SRC = ROOT / "scripts"
FIGURES_SRC = ROOT / "figures" / "Used"

PYTHON_SCRIPTS: Sequence[str] = (
    "figure_csv.py",
    "plot_figure3_panels.py",
    "plot_ratio_supplementary.py",
    "plot_hit_rescreen_panel.py",
    "plot_primary_batch_violins.py",
    "plot_rescreen_ridgeline_supplementary.py",
    "plot_non_hit_ridgeline_supplementary.py",
)

SHELL_SCRIPTS: Sequence[str] = (
    "build_mccp_chain_figure.sh",
    "build_mccp_four_conditions_figure.sh",
    "build_simulation_loop_combined_figure.sh",
    "build_chemostat_snapshot_pdfs.sh",
)

TIKZ_FILES: Sequence[str] = (
    "mccp_chain_tikz.tex",
    "mccp_chain_metab_arrows.tex",
    "mccp_chain_standalone.tex",
    "mccp_four_conditions_tikz.tex",
    "mccp_four_conditions_standalone.tex",
    "simulation_loop_combined_grid.tex",
    "simulation_loop_combined_tikz.tex",
    "simulation_loop_tikz.tex",
    "chemostat_snapshot_tikz.tex",
    "chemostat_snap_standalone.tex",
)

ROOT_TEX_FILES: Sequence[str] = ("simulation_loop_combined_standalone.tex",)

def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_requirements(bundle_dir: Path) -> None:
    (bundle_dir / "requirements.txt").write_text(
        "matplotlib>=3.7\nnumpy>=1.24\n",
        encoding="utf-8",
    )


def _write_readme(bundle_dir: Path) -> None:
    text = """SecondPaperDraft — figure reproduction (minimal)
=============================================

Regenerate every figure in the main manuscript and SI PDF:

  cd data/figure_reproduction
  pip install -r requirements.txt
  ./rebuild_all_figures.sh

Data
----
  batch_hit_counts.csv
    Single table with row_type:
      primary_batch — Fig. 3 panel (a) batch hit counts
      hit             — Fig. 3 panel (b) and Supp. Fig. S2
      non_hit         — Supp. Fig. S3

Rebuild the CSV from Workspaces/Output (from the repo root):

  python3 scripts/assemble_figure_data.py

Outputs
-------
  output/figures/mccp_chain.pdf
  output/figures/mccp_four_conditions.pdf
  output/figures/figure3_hit_panels.png
  output/figures/simulation_loop_combined.pdf
  output/supplementary/figures/supplementary_rescreen_ridgelines.png
  output/supplementary/figures/supplementary_non_hit_rescreen_ridgelines.png

Requirements: Python 3.10+, matplotlib, numpy; pdfLaTeX + latexmk + pdfcrop for TikZ figures.
"""
    (bundle_dir / "README.txt").write_text(text, encoding="utf-8")


def _write_rebuild_script(bundle_dir: Path) -> None:
    text = r"""#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_CSV="$BUNDLE_ROOT/batch_hit_counts.csv"
SCRIPTS="$BUNDLE_ROOT/scripts"
OUT="$BUNDLE_ROOT/output"

mkdir -p "$OUT/figures" "$OUT/supplementary/figures"
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"

echo "== TikZ figures =="
"$SCRIPTS/build_mccp_chain_figure.sh"
cp -f "$BUNDLE_ROOT/figures/mccp_chain.pdf" "$OUT/figures/mccp_chain.pdf"
"$SCRIPTS/build_mccp_four_conditions_figure.sh"
cp -f "$BUNDLE_ROOT/figures/mccp_four_conditions.pdf" "$OUT/figures/mccp_four_conditions.pdf"
"$SCRIPTS/build_simulation_loop_combined_figure.sh"
cp -f "$BUNDLE_ROOT/figures/simulation_loop_combined.pdf" "$OUT/figures/simulation_loop_combined.pdf"

echo "== Data figures =="
python3 "$SCRIPTS/plot_figure3_panels.py" \
  --data-csv "$DATA_CSV" \
  --output "$OUT/figures/figure3_hit_panels.png"
python3 "$SCRIPTS/plot_rescreen_ridgeline_supplementary.py" \
  --data-csv "$DATA_CSV" \
  --output "$OUT/supplementary/figures/supplementary_rescreen_ridgelines.png"
python3 "$SCRIPTS/plot_non_hit_ridgeline_supplementary.py" \
  --data-csv "$DATA_CSV" \
  --output "$OUT/supplementary/figures/supplementary_non_hit_rescreen_ridgelines.png"

echo "Done. Figures under $OUT"
"""
    path = bundle_dir / "rebuild_all_figures.sh"
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def assemble_bundle(*, bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = bundle_dir / "scripts"
    figures_dir = bundle_dir / "figures"

    if scripts_dir.exists():
        shutil.rmtree(scripts_dir)
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    stale_output = bundle_dir / "output"
    if stale_output.exists():
        shutil.rmtree(stale_output)
    scripts_dir.mkdir()
    figures_dir.mkdir()

    csv_path = bundle_dir / "batch_hit_counts.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}; run scripts/assemble_figure_data.py first."
        )

    for name in PYTHON_SCRIPTS:
        _copy_file(SCRIPTS_SRC / name, scripts_dir / name)
    for name in SHELL_SCRIPTS:
        dst = scripts_dir / name
        _copy_file(SCRIPTS_SRC / name, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for name in TIKZ_FILES:
        _copy_file(FIGURES_SRC / name, figures_dir / name)
    for name in ROOT_TEX_FILES:
        _copy_file(FIGURES_SRC / name, bundle_dir / name)

    _write_requirements(bundle_dir)
    _write_readme(bundle_dir)
    _write_rebuild_script(bundle_dir)

    size_mb = (bundle_dir / "batch_hit_counts.csv").stat().st_size / (1024 * 1024)
    print(f"Bundle: {bundle_dir}")
    print(f"  batch_hit_counts.csv ({size_mb:.1f} MiB)")
    print(f"  {len(PYTHON_SCRIPTS)} Python + {len(SHELL_SCRIPTS)} shell scripts")
    print(f"  {len(TIKZ_FILES)} TikZ sources")
    print(f"  Run: {bundle_dir / 'rebuild_all_figures.sh'}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLE_DIR,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    assemble_bundle(bundle_dir=args.bundle_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
