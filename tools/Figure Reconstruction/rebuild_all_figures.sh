#!/usr/bin/env bash
# Regenerate manuscript and supplementary figures from batch_hit_counts.csv + TikZ sources.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$ROOT/scripts"
DATA_CSV="$ROOT/data/figure_reproduction/batch_hit_counts.csv"
OUT="$ROOT/output"
export PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$DATA_CSV" ]]; then
  echo "Missing $DATA_CSV" >&2
  echo "Obtain batch_hit_counts.csv or run: python3 scripts/assemble_figure_data.py" >&2
  exit 1
fi

mkdir -p "$OUT/figures" "$OUT/figures/Used" "$OUT/figures/Extra" "$OUT/supplementary/figures"

echo "== TikZ figures =="
"$SCRIPTS/build_mccp_chain_figure.sh"
cp -f "$ROOT/figures/Used/mccp_chain.pdf" "$OUT/figures/mccp_chain.pdf"
"$SCRIPTS/build_mccp_four_conditions_figure.sh"
cp -f "$ROOT/figures/Used/mccp_four_conditions.pdf" "$OUT/figures/mccp_four_conditions.pdf"
"$SCRIPTS/build_simulation_loop_combined_figure.sh"
cp -f "$ROOT/figures/Used/simulation_loop_combined.pdf" "$OUT/figures/simulation_loop_combined.pdf"

echo "== Data-driven figures =="
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
echo ""
echo "Session-folder plots (require Workspaces/Output Re-Runs/sessions):"
echo "  plot_rescreen_events_supplementary.py"
echo "  plot_rescreen_events_true_neutral_zoom.py"
echo "  plot_true_neutral_param_embedding.py"
echo "  plot_hit_mutation_scale_violin_supplementary.py"
