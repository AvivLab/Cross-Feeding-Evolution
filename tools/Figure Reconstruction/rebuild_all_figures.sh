#!/usr/bin/env bash
# Regenerate all figures in the main manuscript and Supplementary Information PDFs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
DATA_CSV="$ROOT/data/figure_reproduction/batch_hit_counts.csv"
OUT="$ROOT/output"
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$DATA_CSV" ]]; then
  echo "Missing $DATA_CSV" >&2
  echo "Run: python3 scripts/assemble_figure_data.py" >&2
  echo "Or place an existing batch_hit_counts.csv under data/figure_reproduction/." >&2
  exit 1
fi

mkdir -p "$OUT/figures" "$OUT/supplementary/figures"

echo "== TikZ figures =="
"$SCRIPTS/build_mccm_chain_figure.sh"
cp -f "$ROOT/figures/Used/mccm_chain.pdf" "$OUT/figures/mccm_chain.pdf"
"$SCRIPTS/build_mccm_two_conditions_figure.sh"
cp -f "$ROOT/figures/Used/mccm_two_conditions.pdf" "$OUT/figures/mccm_two_conditions.pdf"
"$SCRIPTS/build_simulation_loop_combined_figure.sh"
cp -f "$ROOT/figures/Used/simulation_loop_combined.pdf" "$OUT/figures/simulation_loop_combined.pdf"

echo "== Data-driven figures =="
python3 "$SCRIPTS/plot_figure3_panels.py" \
  --data-csv "$DATA_CSV" \
  --configs main \
  --output "$OUT/figures/figure3_hit_panels.png"
python3 "$SCRIPTS/plot_rescreen_ridgeline_supplementary.py" \
  --data-csv "$DATA_CSV" \
  --configs main \
  --output "$OUT/supplementary/figures/supplementary_rescreen_ridgelines.png"

echo "Done. Figures under $OUT"
