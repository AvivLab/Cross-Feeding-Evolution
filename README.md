# MCCP_Enzymes

Simulate microbial evolution in a chemostat using the **MCCP** model (Minimal Chemostat Cross-feeding Problem). This download includes desktop apps for interactive runs and batch campaigns, plus optional command-line tools for batch runs and re-screening.

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

## Desktop apps

The launcher opens four tools:

- **Individual Simulation** — one simulation at a time, with plots and change history
- **Gradient Descent Optimization** — search parameter space to improve a chosen metric
- **Batch Runner** — large Monte Carlo campaigns with hit counts, offload, and CSV output
- **Batch Re-Runner** — open a finished batch session and re-screen hits or non-hits in the GUI

When Batch Runner saves a campaign, it writes a **session folder**. Note that folder if you want to re-screen results later.

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

## Re-screening from the terminal (optional)

After a batch campaign finishes, you can re-screen hits without the GUI. With your terminal still in this folder:

```bash
python headless/primary_hit_rescreen.py SESSION_FOLDER --n-seeds 20
python headless/primary_hit_rescreen.py SESSION_FOLDER --non-hits --max-hits 500
```

Replace `SESSION_FOLDER` with the batch session folder from Batch Runner or the headless batch runner.

Re-screen outputs are written inside that session folder under `Re-Runs/` or `Re-Runs-NonHits/`, and the session hit-counts CSV there is updated in place.

## Files in this download

- `main.py` — starts the GUI launcher
- `requirements.txt` — packages to install with pip
- `gui/` — desktop applications
- `headless/primary_batch_campaign.py` — command-line batch campaigns
- `headless/primary_hit_rescreen.py` — command-line re-screening
- `simulation/` — simulation engine used by the apps above
