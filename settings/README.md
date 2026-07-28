# Paper batch settings

Batch Runner JSON settings for the paper campaigns in *No Trade-Offs Required: Cross-Feeding From Survival Alone*.

## Layout

Each **suite folder** fixes the task-energy yield ratio (Acetate Ratio):

| Folder | Acetate Ratio (Y) |
|--------|-------------------|
| `Fixed_0.0001_ratio` | 0.0001 |
| `Fixed_0.1_ratio` | 0.1 |
| `Fixed_0.25_ratio` | 0.25 |
| `Fixed_0.5_ratio` | 0.5 |
| `Fixed_0.75_ratio` | 0.75 |
| `Fixed_1_ratio` | 1 |
| `Fixed_3_ratio` | 3 |
| `Fixed_5_ratio` | 5 |
| `Fixed_7_ratio` | 7 |
| `Fixed_10_ratio` | 10 |

Each suite contains the two **paper configurations** used in the quantitative campaigns (100 batches × 1000 simulations per batch):

| File prefix | Paper name |
|-------------|------------|
| `a_` | Neutral (`trueNeutral`) |
| `e_` | Death+Duplication (`Death+Dup`) |

All four metric filters A–C must pass for a hit (filter D is inactive). See any JSON file for bounds, toggles, and thresholds.

## Batch Runner (GUI)

1. Open **Batch Runner** from the launcher.
2. Use **Load JSON Settings…** and pick a file (e.g. `Fixed_3_ratio/e_Death+Dup_Fixed_3.json`).
3. Set a **Save folder** and click **Run Batch**.

Repeat for each suite and configuration you need.

## Headless batch runner

From the Minimal bundle root:

```bash
python headless/primary_batch_campaign.py \
  settings/Fixed_3_ratio/e_Death+Dup_Fixed_3.json \
  --output-dir OUTPUT_FOLDER
```

Optional: `--session-id NAME`, `--progress-every N`.

## After primary batches

Load each finished session in **Batch Re-Runner** and re-screen hits (the paper used 20 seeds per hit). Then assemble figures — see the main README, *Recreating the paper results* and *Reconstructing paper figures*.
