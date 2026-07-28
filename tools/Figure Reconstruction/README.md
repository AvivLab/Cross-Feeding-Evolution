# Figure Reconstruction

Scripts to regenerate every figure in the main manuscript and Supplementary Information PDF for *No Trade-Offs Required: Cross-Feeding From Survival Alone*.

| Output | Figure |
|--------|--------|
| `output/figures/mccm_chain.pdf` | Main text Fig. 1 |
| `output/figures/mccm_four_conditions.pdf` | Main text Fig. 2 |
| `output/figures/figure3_hit_panels.png` | Main text Fig. 3 |
| `output/figures/simulation_loop_combined.pdf` | Supp. Fig. S1 |
| `output/supplementary/figures/supplementary_rescreen_ridgelines.png` | Supp. Fig. S2 |

## Requirements

- Python 3.11+ with packages from the Minimal bundle (`pip install -r requirements.txt`)
- **pdfLaTeX**, **latexmk**, and **pdfcrop** for TikZ schematic figures

## Data

The data-driven figures read `data/figure_reproduction/batch_hit_counts.csv`. Place that file under `data/figure_reproduction/` before running the rebuild script.

To build the CSV from aggregated campaign outputs, collect:

- `Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv` — primary-batch hit counts for all suites (Neutral and Death+Duplication)
- `Re-Runs/sessions/` — hit re-screen CSVs (one folder per campaign session)
- `Re-Runs-NonHits/sessions/` — optional non-hit re-screens (included in the CSV but not used by the five published figures)

By default the script resolves these paths from the lab’s shared campaign output directory. If you mirror that tree locally, run:

```bash
cd "tools/Figure Reconstruction"
export PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/assemble_figure_data.py
```

This writes `data/figure_reproduction/batch_hit_counts.csv`.

## Quick rebuild

```bash
cd "tools/Figure Reconstruction"
./rebuild_all_figures.sh
```

Outputs land in `output/`. TikZ PDFs are also written under `figures/Used/`.

## Scripts

| Script | Output |
|--------|--------|
| `build_mccm_chain_figure.sh` | Fig. 1 — MCCM chain |
| `build_mccm_four_conditions_figure.sh` | Fig. 2 — life-cycle mode schematic (four modes illustrated; campaigns use Neutral + Death+Duplication) |
| `build_simulation_loop_combined_figure.sh` | Supp. Fig. S1 — simulation loop |
| `build_chemostat_snapshot_pdfs.sh` | Chemostat panels embedded in S1 (called by the script above) |
| `assemble_figure_data.py` | `batch_hit_counts.csv` from aggregated campaign outputs |
| `plot_figure3_panels.py` | Fig. 3 panels (a–b); defaults to Neutral + Death+Duplication |
| `plot_rescreen_ridgeline_supplementary.py` | Supp. Fig. S2 — hit re-screen ridgelines (same two configs by default) |

Run individual scripts with `--help` for paths and options. Set `PYTHONPATH=scripts` when invoking from this folder.
