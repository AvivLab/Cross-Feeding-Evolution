# Figure Reconstruction

Scripts to regenerate figures from *No Trade-Offs Required: Cross-Feeding From Survival Alone* (SecondPaperDraft). Copied from `Papers/SecondPaperDraft/scripts`; refresh with:

```bash
python3 tools/sync_figure_reconstruction_to_minimal.py
```

(from the MCCP_Enzymes repository root)

## Requirements

- Python 3.11+ with packages from the Minimal bundle (`pip install -r requirements.txt`)
- **pdfLaTeX**, **latexmk**, and **pdfcrop** for TikZ schematic figures

## Data

Most plots read `data/figure_reproduction/batch_hit_counts.csv` (not included here — large HPC export). Place that file under `data/figure_reproduction/` before running the rebuild script.

To assemble the CSV from raw HPC session folders (requires `Workspaces/Output` next to the GIT folder):

```bash
export PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/assemble_figure_data.py
```

## Quick rebuild

```bash
cd "tools/Figure Reconstruction"
./rebuild_all_figures.sh
```

Outputs land in `output/`. TikZ PDFs are also written under `figures/Used/`.

## Scripts

| Script | Output |
|--------|--------|
| `build_mccp_chain_figure.sh` | Fig. 1 — MCCP chain |
| `build_mccp_four_conditions_figure.sh` | Four selection regimes |
| `build_simulation_loop_combined_figure.sh` | Supp. simulation loop |
| `plot_figure3_panels.py` | Fig. 3 panels (a–b) |
| `plot_rescreen_ridgeline_supplementary.py` | Supp. hit re-screen ridgelines |
| `plot_non_hit_ridgeline_supplementary.py` | Supp. non-hit ridgelines |
| `plot_rescreen_events_supplementary.py` | Supp. re-screen events |
| `plot_rescreen_events_true_neutral_zoom.py` | Neutral zoom panel |
| `plot_true_neutral_param_embedding.py` | Supp. parameter embedding (+ optional HTML explorer) |
| `plot_hit_mutation_scale_violin_supplementary.py` | Hit parameter violins |
| `rebuild_hit_counts_compare.py` | Rebuild primary-batch summary CSV from sessions |
| `rebuild_rescreen_compare.py` | Rebuild hit re-screen compare CSV |
| `rebuild_non_hit_rescreen_compare.py` | Rebuild non-hit re-screen compare CSV |
| `verify_figure3_plots.py` | Cross-check Fig. 3 metrics against raw CSVs |

Run individual scripts with `--help` for paths and options. Set `PYTHONPATH=scripts` when invoking from this folder.
