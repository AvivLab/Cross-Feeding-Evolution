# Figure Reconstruction

Scripts to regenerate every figure in the main manuscript and Supplementary Information PDF for *No Trade-Offs Required: Cross-Feeding From Survival Alone*.

| Output | Figure |
|--------|--------|
| `output/figures/mccm_chain.pdf` | Main text Fig. 1 |
| `output/figures/mccm_two_conditions.pdf` | Main text Fig. 2 |
| `output/figures/figure3_hit_panels.png` | Main text Fig. 3 |
| `output/figures/simulation_loop_combined.pdf` | Supp. Fig. S1 |
| `output/supplementary/figures/supplementary_rescreen_ridgelines.png` | Supp. Fig. S2 |

## Requirements

- Python 3.11+ with packages from the Minimal bundle (`pip install -r requirements.txt`)
- **pdfLaTeX**, **latexmk**, and **pdfcrop** for TikZ schematic figures

## Data

The data-driven figures read `data/figure_reproduction/batch_hit_counts.csv`. That file is **not** shipped in the download. Build it from your campaign sessions:

1. Run paper Batch Runner / headless campaigns (and re-screens) into one output root.
2. Aggregate and assemble:

```bash
cd "tools/Figure Reconstruction"
export PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/build_summary_hit_counts.py --output-root /path/to/OUTPUT_ROOT
python3 scripts/assemble_figure_data.py --output-root /path/to/OUTPUT_ROOT
```

`build_summary_hit_counts.py` writes:

- `Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv` — per-batch hit counts for Neutral and Death+Duplication across paper Y suites
- `Re-Runs/sessions/` (and optional `Re-Runs-NonHits/sessions/`) — staged from each session’s in-folder re-screen outputs

`assemble_figure_data.py` then writes `data/figure_reproduction/batch_hit_counts.csv`.

You can combine those steps with `--build-summary` on `assemble_figure_data.py`.

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
| `build_mccm_two_conditions_figure.sh` | Fig. 2 — Neutral vs Death+Duplication (Selection) rate curves |
| `build_simulation_loop_combined_figure.sh` | Supp. Fig. S1 — simulation loop |
| `build_chemostat_snapshot_pdfs.sh` | Chemostat panels embedded in S1 (called by the script above) |
| `build_summary_hit_counts.py` | `Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv` (+ stage `Re-Runs/sessions/`) |
| `assemble_figure_data.py` | `batch_hit_counts.csv` from Summary + Re-Runs |
| `plot_figure3_panels.py` | Fig. 3 panels (a–b); defaults to Neutral + Death+Duplication |
| `plot_rescreen_ridgeline_supplementary.py` | Supp. Fig. S2 — hit re-screen ridgelines (same two configs by default) |

Run individual scripts with `--help` for paths and options. Set `PYTHONPATH=scripts` when invoking from this folder.
