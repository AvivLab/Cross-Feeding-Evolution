<p align="center">
  <img src="docs/bergman_lab_logo.png" alt="Bergman Lab logo" width="480"/>
</p>

# MCCP_Enzymes

Simulate microbial evolution of a single population in a chemostat to investigate the **MCCP** model (Minimal Chemostat Cross-feeding Problem). This download includes four desktop apps for interactive runs, parameter search, and batch campaigns, plus optional command-line tools for batch runs and re-screening.

## About

This software was developed in the **Bergman Lab** at the **Department of Systems and Computational Biology**, Albert Einstein College of Medicine, Bronx, NY 10461, USA.

It implements the in silico chemostat evolution model used in *No Trade-Offs Required: Cross-Feeding From Survival Alone* (Samuel Rosean & Aviv Bergman).

Aviv Bergman is also affiliated with the Santa Fe Institute, Santa Fe, NM 87501, USA.

## Requirements

- **Python 3.11 or newer**
- **Tkinter** — required for the desktop GUIs. It usually comes with Python. If launching the GUI fails with a tkinter-related error, reinstall Python from [python.org](https://www.python.org/downloads/) and include Tcl/Tk in the install.

## Install and run

1. Unzip or copy this folder to wherever you want on your computer.
2. Open a terminal in **this folder** (the one that contains `main.py` and `requirements.txt`).
3. Install the listed packages:

   ```bash
   pip install -r requirements.txt
   ```

   Using a virtual environment in this folder is optional but recommended:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   On Windows, activate with: `.venv\Scripts\activate`

4. Start the launcher:

   ```bash
   python main.py
   ```

Choose one of the four tools from the launcher window.

---

## Individual Simulation

Run one chemostat simulation at a time, inspect trajectories, and compute metrics on the final population.

<p align="center">
  <img src="docs/screenshots/individual_simulation.png" alt="Individual Simulation" width="720"/>
</p>

### Layout

- **Left column** — simulation parameters, toggles (diffusion, chemostat flow, death/duplication modes, traits), **Run Simulation**, and status/progress.
- **Center column** — tabbed plots. **Main Plots** shows trait heatmaps, population size, mean energy, metabolite flux, and deaths/duplications per generation. Other tabs add **M1 Metabolite**, **M2 Metabolite**, **Budgets**, and **Budget 2** detail views. The rightmost tab, **Metrics**, holds the metric selector, **Run** (score the last simulation), optional **Sweep Seeds** (repeat the same settings with different seeds), and seed-sweep output.
- **Right column** — diagnostic figures: death and duplication probability curves vs energy, the investment-function plot, and a **Simulation Pathway Diagram** that updates with your toggles.

### Typical workflow

1. Set **Number of Generations**, population size, inflow, and metabolic parameters in the scrollable sections.
2. Choose transport and life-cycle toggles (e.g. pooled vs diffusion-limited nutrients, binary vs constant death).
3. Optionally enter a **Random Seed** for reproducibility (leave blank for a random seed each run).
4. Click **Run Simulation** and wait for the progress bar to finish.
5. Browse the center tabs (**Main Plots** first, then M1/M2 or budget tabs as needed). Hover tooltips on labels explain individual knobs.
6. Open the **Metrics** tab, pick a metric, and click **Run** to score the finished run (e.g. task specialization, trait diversity).

Use this app to build intuition for a parameter set before scaling up to batch or optimization workflows.

---

## Gradient Descent Optimization

Search parameter space to improve a chosen metric using finite-difference gradient steps from multiple random starting points.

<p align="center">
  <img src="docs/screenshots/gradient_descent.png" alt="Gradient Descent Optimization" width="720"/>
</p>

### Layout

- **Left** — optimization controls (metric, maximize/minimize, learning rate, iterations, random starts, replicates) and the list of parameters to optimize (fix/unfix, min/max bounds).
- **Right** — fixed parameters and simulation toggles that stay constant during the search.
- **Bottom** — run controls, progress, and buttons to save/load results.

### Typical workflow

1. Choose the **Metric to Optimize** and **Optimization Goal** (maximize or minimize).
2. Under **Parameters to Optimize**, leave parameters unfixed and set min/max bounds for those you want the search to adjust; fix others at known values.
3. Set **Number of Random Starts** and **Gradient Descents per Start** (use 0 descents with 0 max iterations for random-start sampling only).
4. Adjust **Learning Rate**, **Gradient Step Size**, and **Max Iterations** as needed (tooltips describe each control).
5. Set a **Full Save Folder** if you want large runs offloaded to disk instead of kept entirely in memory.
6. Click **Run Optimization** and monitor progress.
7. When finished, open **Analyze Results** (or equivalent analysis window) for tables, histograms, and optional parameter heatmaps / UMAP of evaluated points.
8. Use **Save Dataset** to export a portable snapshot, or **Load Dataset** / **Load Full Save Session** to continue from a previous run.

Each metric evaluation may run several simulation replicates with derived seeds, so results can be noisy; increase replicates for smoother objectives.

---

## Batch Runner

Run many independent Monte Carlo batches over a parameter box and count how often simulations pass your metric filters (“hits”).

<p align="center">
  <img src="docs/screenshots/batch_runner.png" alt="Batch Runner" width="720"/>
</p>

### Layout

Two tabs:

- **Setup & metrics** — parameter panel (same style as optimization: fixed values vs sampled ranges), batch size, metric filters A–D, save folder, and **Run Batch**.
- **Results** — hit-count histogram and primary-event charts after a campaign finishes or is loaded.

### Typical workflow

1. On **Setup & metrics**, set **Runs per batch (N)** and **Number of batches**.
2. In the parameter panel, mark which parameters are fixed and which are drawn uniformly from min/max bounds each simulation.
3. Configure **Metric A** (required) and optionally **B–C–D**; every active filter must pass for a hit (AND logic).
4. Choose a **Save folder** for the campaign output.
5. Click **Run Batch**. Use **Pause** / **Resume** (or type `pause` / `resume` in the terminal) for long jobs.
6. When complete, open the **Results** tab for hit-count plots and charts.
7. Optional: **Save JSON Settings…** exports the setup for the headless batch runner; **Load Campaign Summary** reloads a finished session.

### Outputs (inside your save folder)

Each campaign creates a **session folder** containing simulation offload batches, `primary_batch_campaign_<session>.json`, and a hit-count CSV. Keep that folder path for Batch Re-Runner or command-line re-screening.

---

## Batch Re-Runner

Re-test hits or non-hits from a finished Batch Runner campaign with fresh random seeds to estimate how often they pass the metric filters again.

<p align="center">
  <img src="docs/screenshots/batch_rerunner.png" alt="Batch Re-Runner" width="720"/>
</p>

### Layout

- **Load campaign** — pick the session folder from a completed Batch Runner run; the panel shows session id and hit/non-hit counts.
- **Re-screen options** — hits only, non-hits only, or both; seeds per point, worker count, optional deduplication of identical parameter vectors.
- **Loaded session** (right) — summary of metric filters and bounds from the original campaign.

### Typical workflow

1. Click **Choose Session Folder** and select the Batch Runner session directory (the folder that contains offload batches and the campaign summary JSON).
2. Confirm hit and non-hit counts in the status line.
3. Choose **Re-screen** mode (usually **Hits Only** first).
4. Set **Seeds per point (N)** — how many fresh simulations to run for each unique parameter vector (e.g. 20).
5. Click **Run Re-screen** and wait for completion.
6. Results are written under `Re-Runs/` (hits) or `Re-Runs-NonHits/` inside the session folder, and the session hit-count CSV is updated with re-screen statistics.

Use **Non-Hits Only** or **Both** to compare how often parameter sets that failed initially would pass on retry. Limit max non-hits when the non-hit pool is very large.

---

## Batch campaigns from the terminal (optional)

You can run a batch campaign without the GUI. In Batch Runner, use **Save JSON Settings…** to write a settings file, then from this folder:

```bash
python headless/primary_batch_campaign.py SETTINGS.json --output-dir OUTPUT_FOLDER
```

Replace `SETTINGS.json` with your saved settings file and `OUTPUT_FOLDER` with the folder where you want the session written.

Optional flags:

- `--session-id NAME` — name the session (default: a timestamp)
- `--progress-every N` — print progress every N simulations per batch

The run creates the same kinds of outputs as the GUI: offload batches, a hit-count CSV, and `primary_batch_campaign_<session>.json` inside `OUTPUT_FOLDER`.

---

## Re-screening from the terminal (optional)

After a batch campaign finishes, you can re-screen hits without the GUI. With your terminal still in this folder:

```bash
python headless/primary_hit_rescreen.py SESSION_FOLDER --n-seeds 20
python headless/primary_hit_rescreen.py SESSION_FOLDER --non-hits --max-hits 500
```

Replace `SESSION_FOLDER` with the batch session folder from Batch Runner or the headless batch runner.

Re-screen outputs are written inside that session folder under `Re-Runs/` or `Re-Runs-NonHits/`, and the session hit-counts CSV there is updated in place.

---

## Files in this download

- `main.py` — starts the GUI launcher
- `requirements.txt` — packages to install with pip
- `gui/` — desktop applications
- `headless/primary_batch_campaign.py` — command-line batch campaigns
- `headless/primary_hit_rescreen.py` — command-line re-screening
- `simulation/` — simulation engine used by the apps above
- `docs/bergman_lab_logo.png` — Bergman Lab logo
- `docs/screenshots/` — GUI screenshots referenced above

---

## Correspondence

**Correspondence:** Aviv Bergman — [aviv@einsteinmed.edu](mailto:aviv@einsteinmed.edu)

**Technical correspondence:** Samuel Rosean — [samuel.rosean@einsteinmed.edu](mailto:samuel.rosean@einsteinmed.edu)
