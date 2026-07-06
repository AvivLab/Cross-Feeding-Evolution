"""
Batch Runner GUI: run many independent simulation batches over a parameter box.

Each batch runs N simulations with parameters drawn uniformly from unfixed bounds;
fixed parameters and toggles stay constant. Metric filters A–D define hits (all active
filters must pass). Results are saved as simulation records, a campaign summary,
hit-count tables, and charts.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.shared.monte_carlo_panel import MonteCarloGdPanel, build_monte_carlo_gd_panel
from gui.persistence.full_save import full_save_manifest_path_json, full_save_settings_path_json
from gui.persistence.json_io import make_read_json_maybe_gz_fn, make_write_json_maybe_gz_atomic_fn
from gui.models.registry import OptimizationModelSpec
from gui.apps.neutral_comparison.offload import NeutralComparisonOffloadWriter
from gui.apps.neutral_comparison.batch import run_hit_count_batch, simulation_light_tracking_plan
from gui.apps.neutral_comparison.primary_event_chart import (
    draw_primary_event_rates_on_axes,
    load_primary_offload_records,
    summarize_primary_events_by_hit,
    write_primary_event_rates_png,
)
from gui.apps.neutral_comparison.parameter_heatmap import (
    collect_primary_hit_param_values,
    draw_parameter_heatmaps_on_figure,
    load_param_names_from_offload,
    resolve_heatmap_param_names,
    write_parameter_heatmap_png,
)
from gui.common.model_diagram import SimulationModelDiagram
from gui.common.simulation_settings import (
    apply_toggles_to_monte_carlo_panel,
    normalize_simulation_params,
    prune_irrelevant_numeric_parameters_for_export,
)
from gui.apps.batch_runner.csv_output import normalize_configuration_name, write_session_hit_counts_csv
from gui.common.terminal_pause_listener import register_terminal_pause_hooks
from gui.common.tooltips import BATCH_RUNNER_TOOLTIPS
from gui.common.widgets import CreateToolTip

_write_results_json = make_write_json_maybe_gz_atomic_fn(indent=2)
_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)


def _filedialog_open_json_kwargs() -> Dict[str, Any]:
    if sys.platform == "darwin":
        return {"filetypes": [("All files", "*")]}
    return {"filetypes": [("JSON", "*.json"), ("JSON gzip", "*.json.gz"), ("All files", "*")]}


def _safe_set_entry(entry: tk.Entry, value: str) -> None:
    try:
        entry.delete(0, tk.END)
        entry.insert(0, value)
    except tk.TclError:
        pass


def _parse_saved_bounds_pair(raw: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    lo_raw, hi_raw = raw[0], raw[1]
    if lo_raw in ("-inf", "-Infinity", None):
        lo = float("-inf")
    else:
        try:
            lo = float(lo_raw)
        except (TypeError, ValueError):
            return None
    try:
        hi = float(hi_raw)
    except (TypeError, ValueError):
        return None
    return lo, hi


def _format_bound_lo(lo: float) -> str:
    return "" if lo == float("-inf") else str(lo)


def _apply_neutral_comparison_save_to_panel(
    panel: MonteCarloGdPanel,
    *,
    bounds_json: Any,
    numeric_json: Any,
    toggles_json: Any,
) -> List[str]:
    """Restore bounds / toggles / numeric scalars from a saved summary JSON. Returns human-readable warnings."""
    warns: List[str] = []
    toggles_json = toggles_json if isinstance(toggles_json, dict) else {}
    numeric_json = dict(numeric_json) if isinstance(numeric_json, dict) else {}

    if toggles_json:
        apply_toggles_to_monte_carlo_panel(panel, toggles_json)

    if toggles_json or numeric_json:
        merged = normalize_simulation_params({**numeric_json, **toggles_json})
        numeric_json = {k: merged[k] for k in panel.param_names if k in merged}

    bounds_d = bounds_json if isinstance(bounds_json, dict) else {}
    for pname in panel.param_names:
        if pname in panel.hidden_params:
            continue
        if pname in bounds_d:
            pair = _parse_saved_bounds_pair(bounds_d[pname])
            if pair is None:
                warns.append(f"Skipped bounds for unknown format: {pname!r}")
                continue
            lo, hi = pair
            panel.param_fix_checkboxes[pname].set(False)
            if pname in panel.param_min_entries:
                _safe_set_entry(panel.param_min_entries[pname], _format_bound_lo(lo))
            if pname in panel.param_max_entries:
                _safe_set_entry(panel.param_max_entries[pname], str(hi))
        else:
            panel.param_fix_checkboxes[pname].set(True)

    panel.refresh_optimizable_params()
    panel.refresh_fixed_params()

    if isinstance(numeric_json, dict):
        for k, v in numeric_json.items():
            ks = str(k)
            try:
                txt = str(float(v))
            except (TypeError, ValueError):
                txt = str(v)
            if ks in panel.fixed_entries:
                _safe_set_entry(panel.fixed_entries[ks], txt)
            if ks in panel.param_initial_entries:
                _safe_set_entry(panel.param_initial_entries[ks], txt)
    return warns


def _default_dialog_dir() -> str:
    try:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    except Exception:
        return os.getcwd()


def _bounds_jsonable(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for k, (lo, hi) in bounds.items():
        lo_j: Any = "-inf" if lo == float("-inf") else float(lo)
        out[str(k)] = [lo_j, float(hi)]
    return out


def _jsonable_value(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {str(kk): _jsonable_value(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable_value(x) for x in v]
    if isinstance(v, int):
        return int(v)
    try:
        if hasattr(v, "item"):
            return _jsonable_value(v.item())
    except Exception:
        pass
    try:
        x = float(v)
        if x != x:
            return None
        return x
    except Exception:
        return str(v)


def _jsonable_params(d: Dict[str, Any]) -> Dict[str, Any]:
    from gui.common.simulation_settings import strip_internal_simulation_runtime_params

    return {str(k): _jsonable_value(v) for k, v in strip_internal_simulation_runtime_params(d).items()}


# Combobox sentinel for optional metric slots B–D (not stored as a real metric name).
METRIC_NONE_LABEL = "(No metric)"


def _metric_slot_active(name: str) -> bool:
    s = (name or "").strip()
    return bool(s) and s != METRIC_NONE_LABEL


def _primary_batch_seed(base_seed: Optional[int], batch_index: int) -> Optional[int]:
    if base_seed is None:
        return None
    return int(base_seed) + 19 + int(batch_index) * 10_007


def batch_runner_gui(
    win: tk.Toplevel | tk.Tk,
    root: tk.Tk,
    model_spec: OptimizationModelSpec,
) -> None:
    if getattr(model_spec, "key", "") != "simulation":
        messagebox.showerror("Model", "Batch Runner is only available for the Simulation model.")
        return

    win.title(f"Batch Runner — {getattr(model_spec, 'label', 'Simulation')}")
    win.geometry("1280x860")
    try:
        win.withdraw()
    except tk.TclError:
        pass

    base_names = list(getattr(model_spec, "metric_names", []) or [])

    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=6, pady=6)

    tab_setup = ttk.Frame(nb, padding=4)
    tab_plot = ttk.Frame(nb, padding=8)
    nb.add(tab_setup, text="Setup & metrics")
    nb.add(tab_plot, text="Results")

    # Grid: intro + params full width; bottom row is controls (left) + pathway diagram (right).
    tab_setup.grid_columnconfigure(0, weight=3)
    tab_setup.grid_columnconfigure(1, weight=2)
    tab_setup.grid_rowconfigure(0, weight=0)
    tab_setup.grid_rowconfigure(1, weight=1)
    tab_setup.grid_rowconfigure(2, weight=0, minsize=280)

    intro = ttk.Label(
        tab_setup,
        text=(
            "Set parameter ranges and simulation toggles in the panel below. "
            "Each batch runs N independent simulations with unfixed parameters drawn uniformly "
            "from their bounds; fixed values and toggles stay constant within a batch. "
            "Run many independent batches to see how often simulations pass your metric filters "
            "(A required; B–D optional, choose '"
            + METRIC_NONE_LABEL
            + "'). A hit is a simulation where every active filter passes. "
            "Results are saved in the folder you choose: simulation records, a campaign summary, "
            "hit-count tables, and charts."
        ),
        wraplength=1240,
        justify="left",
    )
    intro.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

    params_shell = ttk.Frame(tab_setup)
    params_shell.grid(row=1, column=0, columnspan=2, sticky="nsew")

    _metric_widgets: Dict[str, Any] = {
        "metric_combos": None,
        "metric_vars": None,
    }

    def _refresh_metric_combos() -> None:
        if not base_names:
            return
        combos = _metric_widgets.get("metric_combos") or []
        vars_ = _metric_widgets.get("metric_vars") or []
        if len(combos) != 4 or len(vars_) != 4 or any(c is None for c in combos):
            return
        allowed = list(primary_panel.filtered_metric_names(base_names))
        optional_values = [METRIC_NONE_LABEL] + allowed
        for i, combo in enumerate(combos):
            combo.configure(values=allowed if i == 0 else optional_values)
        for i, var in enumerate(vars_):
            cur = var.get().strip()
            choices = allowed if i == 0 else optional_values
            if i == 0:
                if cur and cur not in allowed:
                    var.set(allowed[0] if allowed else "")
            elif cur not in choices:
                var.set(METRIC_NONE_LABEL)

    primary_panel: MonteCarloGdPanel = build_monte_carlo_gd_panel(
        params_shell,
        model_spec,
        title="Simulation parameters",
        on_settings_change=_refresh_metric_combos,
    )
    primary_panel.outer.pack(fill="both", expand=True)

    bottom = ttk.LabelFrame(tab_setup, text="Batch size, metrics, and run", padding=4)
    bottom.grid(row=2, column=0, sticky="nsew", pady=(4, 0), padx=(0, 6))

    row0 = ttk.Frame(bottom)
    row0.pack(fill="x", pady=(0, 2))
    nruns_label = ttk.Label(row0, text="Runs per batch (N):")
    nruns_label.pack(side="left")
    nruns_var = tk.StringVar(value="1000")
    nruns_entry = ttk.Entry(row0, textvariable=nruns_var, width=10)
    nruns_entry.pack(side="left", padx=6)
    CreateToolTip(nruns_label, BATCH_RUNNER_TOOLTIPS["Runs per batch"])
    CreateToolTip(nruns_entry, BATCH_RUNNER_TOOLTIPS["Runs per batch"])
    pbatches_label = ttk.Label(row0, text="Number of batches:")
    pbatches_label.pack(side="left", padx=(20, 0))
    primary_batches_var = tk.StringVar(value="100")
    pbatches_entry = ttk.Entry(row0, textvariable=primary_batches_var, width=10)
    pbatches_entry.pack(side="left", padx=6)
    CreateToolTip(pbatches_label, BATCH_RUNNER_TOOLTIPS["Primary batches"])
    CreateToolTip(pbatches_entry, BATCH_RUNNER_TOOLTIPS["Primary batches"])
    seed_label = ttk.Label(row0, text="Random seed (optional):")
    seed_label.pack(side="left", padx=(20, 0))
    seed_var = tk.StringVar(value="")
    seed_entry = ttk.Entry(row0, textvariable=seed_var, width=12)
    seed_entry.pack(side="left", padx=6)
    CreateToolTip(seed_label, BATCH_RUNNER_TOOLTIPS["Random seed"])
    CreateToolTip(seed_entry, BATCH_RUNNER_TOOLTIPS["Random seed"])

    configuration_name_var = tk.StringVar(value="")
    config_row = ttk.Frame(bottom)
    config_row.pack(fill="x", pady=(2, 0))
    config_label = ttk.Label(config_row, text="Configuration name:")
    config_label.pack(side="left")
    config_entry = ttk.Entry(config_row, textvariable=configuration_name_var, width=36)
    config_entry.pack(side="left", padx=6)
    CreateToolTip(config_label, BATCH_RUNNER_TOOLTIPS["Configuration name"])
    CreateToolTip(config_entry, BATCH_RUNNER_TOOLTIPS["Configuration name"])

    save_dir_var = tk.StringVar(value="")
    # True only after «Set save folder…» picks a directory or a save is loaded that sets the path.
    # Prevents an empty field from resolving to the process cwd via os.path.abspath("").
    save_dir_explicit_var = tk.BooleanVar(value=False)
    # Last path for a neutral_set_comparison job JSON (Load saved… / Save JSON Settings…); recorded in manifest.
    comparison_job_settings_json_path: List[Optional[str]] = [None]

    mg = ttk.Frame(bottom)
    mg.pack(fill="x", pady=(2, 2))
    for combo_col in (1, 5):
        mg.grid_columnconfigure(combo_col, weight=1)
    op_values = [">", ">=", "<", "<="]

    m_a_var, m_b_var = tk.StringVar(), tk.StringVar()
    m_c_var, m_d_var = tk.StringVar(), tk.StringVar()
    op_a_var, op_b_var = tk.StringVar(value=">"), tk.StringVar(value=">")
    op_c_var, op_d_var = tk.StringVar(value=">"), tk.StringVar(value=">")
    thr_a_var, thr_b_var = tk.StringVar(value="0.3"), tk.StringVar(value="70")
    thr_c_var, thr_d_var = tk.StringVar(value="0"), tk.StringVar(value="0")

    m_a_combo = ttk.Combobox(mg, textvariable=m_a_var, values=[], width=22, state="readonly")
    m_b_combo = ttk.Combobox(mg, textvariable=m_b_var, values=[], width=22, state="readonly")
    m_c_combo = ttk.Combobox(mg, textvariable=m_c_var, values=[], width=22, state="readonly")
    m_d_combo = ttk.Combobox(mg, textvariable=m_d_var, values=[], width=22, state="readonly")

    def _place_metric_block(row: int, col_group: int, letter: str, m_var: tk.StringVar, m_combo: ttk.Combobox, op_var: tk.StringVar, thr_var: tk.StringVar, pady: Tuple[int, int]) -> None:
        base = col_group * 4
        metric_label = ttk.Label(mg, text=f"Metric {letter}")
        metric_label.grid(row=row, column=base + 0, sticky="w", padx=(0, 4), pady=pady)
        CreateToolTip(metric_label, BATCH_RUNNER_TOOLTIPS["Metric filter"])
        m_combo.grid(row=row, column=base + 1, sticky="ew", padx=4, pady=pady)
        CreateToolTip(m_combo, BATCH_RUNNER_TOOLTIPS["Metric filter"])
        op_combo = ttk.Combobox(mg, textvariable=op_var, values=op_values, width=4, state="readonly")
        op_combo.grid(row=row, column=base + 2, padx=2, pady=pady)
        CreateToolTip(op_combo, BATCH_RUNNER_TOOLTIPS["Metric operator"])
        thr_entry = ttk.Entry(mg, textvariable=thr_var, width=10)
        thr_entry.grid(row=row, column=base + 3, padx=4, pady=pady)
        CreateToolTip(thr_entry, BATCH_RUNNER_TOOLTIPS["Metric threshold"])

    _place_metric_block(0, 0, "A", m_a_var, m_a_combo, op_a_var, thr_a_var, (0, 4))
    _place_metric_block(0, 1, "B", m_b_var, m_b_combo, op_b_var, thr_b_var, (0, 4))
    _place_metric_block(1, 0, "C", m_c_var, m_c_combo, op_c_var, thr_c_var, (0, 0))
    _place_metric_block(1, 1, "D", m_d_var, m_d_combo, op_d_var, thr_d_var, (0, 0))

    _metric_widgets["metric_vars"] = [m_a_var, m_b_var, m_c_var, m_d_var]
    _metric_widgets["metric_combos"] = [m_a_combo, m_b_combo, m_c_combo, m_d_combo]

    for nm, tgt in (("Task Specialization Index (|share-0.5|)", m_a_var),):
        if nm in base_names:
            tgt.set(nm)
    m_b_var.set(METRIC_NONE_LABEL)
    m_c_var.set(METRIC_NONE_LABEL)
    m_d_var.set(METRIC_NONE_LABEL)

    _refresh_metric_combos()

    run_controls = ttk.Frame(bottom)
    run_controls.pack(fill="x", pady=(2, 0))
    prog = ttk.Progressbar(run_controls, mode="determinate", maximum=100.0)
    prog.pack(fill="x", pady=(0, 1))
    status_var = tk.StringVar(value="Idle.")
    status_label = ttk.Label(run_controls, textvariable=status_var, wraplength=680)
    status_label.pack(anchor="w")

    diagram_frame = tk.LabelFrame(tab_setup, text="Simulation Pathway Diagram", padx=4, pady=4)
    diagram_frame.grid(row=2, column=1, sticky="nsew", pady=(4, 0), padx=(6, 0))
    diagram_frame.grid_rowconfigure(0, weight=1)
    diagram_frame.grid_columnconfigure(0, weight=1)
    batch_runner_diagram = SimulationModelDiagram(diagram_frame, width=520, height=260)
    batch_runner_diagram.canvas.grid(row=0, column=0, sticky="nsew")
    batch_runner_diagram.bind_to_vars(
        enable_m1_diffusion=primary_panel.m1_diffusion_var,
        enable_m1_facilitated_diffusion=primary_panel.m1_facilitation_var,
        enable_m2_diffusion=primary_panel.m2_diffusion_var,
        enable_m1_porin_diffusion=primary_panel.m1_porin_diffusion_var,
        enable_diffusion_mutation=primary_panel.diffusion_mutation_var,
        homogeneous_initial_diffusion_const=primary_panel.homogeneous_initial_diffusion_const_var,
        homogeneous_population=primary_panel.homogeneous_mode_var,
        independent_traits=primary_panel.independent_traits_var,
        enable_chemostat_flow=primary_panel.enable_chemostat_flow_var,
        enable_initial_energy=primary_panel.enable_initial_energy_var,
        enable_intermediate_costs=primary_panel.enable_intermediate_costs_var,
        enable_acetate_addition=primary_panel.enable_acetate_addition_var,
    )

    # --- Results tab (batch hit counts + primary event rates) ---
    hist_controls = ttk.Frame(tab_plot)
    hist_controls.pack(fill="x", padx=4, pady=(4, 0))
    hist_fig, (hist_ax, events_ax) = plt.subplots(2, 1, figsize=(8.5, 8.6), dpi=100)
    hist_fig.subplots_adjust(hspace=0.38)
    hist_canvas = FigureCanvasTkAgg(hist_fig, master=tab_plot)
    hist_canvas.get_tk_widget().pack(fill="both", expand=True)
    last_hist: Dict[str, Any] = {}

    def _show_parameter_heatmaps() -> None:
        sf = str(last_hist.get("save_folder") or "").strip()
        fsid = str(last_hist.get("full_save_session_id") or "").strip()
        if not sf or not fsid:
            messagebox.showwarning(
                "No Data",
                "Run a batch campaign first, or load a finished campaign summary "
                "that includes simulation records.",
                parent=win,
            )
            return
        try:
            records = load_primary_offload_records(sf, fsid, _read_json_maybe_gz)
            param_names = load_param_names_from_offload(sf, fsid, _read_json_maybe_gz)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=win)
            return
        if not records:
            messagebox.showwarning(
                "No Data",
                "No simulation records found for this campaign.",
                parent=win,
            )
            return
        if not param_names:
            param_names = list(primary_panel.param_names)
        primary_bounds = last_hist.get("primary_bounds")
        if not isinstance(primary_bounds, dict):
            try:
                primary_bounds = primary_panel.read_optimizable_bounds()
            except Exception:
                primary_bounds = {}
        heatmap_params = resolve_heatmap_param_names(param_names, primary_bounds)
        if not heatmap_params:
            messagebox.showwarning("No Parameters", "No sampled parameters to plot.", parent=win)
            return

        top = tk.Toplevel(win)
        top.title("Parameter Heatmaps")
        top.geometry("1200x800")

        controls = tk.Frame(top)
        controls.pack(side="top", fill="x", padx=10, pady=10)
        row1 = tk.Frame(controls)
        row1.pack(side="top", fill="x")
        row2 = tk.Frame(controls)
        row2.pack(side="top", fill="x", pady=(6, 0))

        tk.Label(row1, text="Bins:").pack(side="left")
        bins_var = tk.StringVar(value="60")
        bins_entry = tk.Entry(row1, textvariable=bins_var, width=6)
        bins_entry.pack(side="left", padx=(5, 15))

        log_scale_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row2, text="Log scale axes", variable=log_scale_var).pack(side="left", padx=(0, 15))

        info_var = tk.StringVar(value="")
        tk.Label(row2, textvariable=info_var, fg="gray").pack(side="left", padx=(10, 0))

        plot_container = tk.Frame(top)
        plot_container.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))
        plot_state: Dict[str, Any] = {"canvas": None}

        def rebuild_plot() -> None:
            try:
                num_bins = int(bins_var.get().strip())
                if num_bins < 5:
                    raise ValueError()
            except Exception:
                messagebox.showerror("Invalid Bins", "Bins must be an integer >= 5.", parent=top)
                return
            values_by_param, n_hit, n_primary = collect_primary_hit_param_values(
                records, param_names, heatmap_params
            )
            info_var.set(
                f"Using {n_hit} threshold-crossing / {n_primary} primary runs — "
                f"{len(heatmap_params)} parameter(s)"
            )
            fig_h = max(4.0, 1.15 * len(heatmap_params))
            fig = plt.figure(figsize=(12, fig_h), constrained_layout=True)
            if not draw_parameter_heatmaps_on_figure(
                fig,
                values_by_param=values_by_param,
                primary_bounds=primary_bounds,
                num_bins=num_bins,
                use_log=bool(log_scale_var.get()),
                n_hit=n_hit,
                n_primary=n_primary,
            ):
                plt.close(fig)
                messagebox.showwarning("No Data", "Could not draw heatmaps.", parent=top)
                return
            if plot_state["canvas"] is not None:
                plot_state["canvas"].get_tk_widget().destroy()
            canvas_local = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_local.draw()
            canvas_local.get_tk_widget().pack(fill="both", expand=True)
            plot_state["canvas"] = canvas_local

        tk.Button(row2, text="Update", command=rebuild_plot).pack(side="left", padx=(0, 10))
        tk.Button(row2, text="Close", command=top.destroy).pack(side="left")
        rebuild_plot()

    param_heatmaps_btn = ttk.Button(hist_controls, text="Parameter Heatmaps", command=_show_parameter_heatmaps)
    param_heatmaps_btn.pack(side="left")

    def _draw_histogram() -> None:
        hist_ax.clear()
        events_ax.clear()
        if not last_hist:
            for ax in (hist_ax, events_ax):
                ax.set_axis_off()
            hist_ax.text(
                0.5,
                0.65,
                "Run a batch campaign on the Setup tab to populate the plots.",
                ha="center",
                va="center",
                transform=hist_ax.transAxes,
                fontsize=11,
            )
        else:
            batch_counts: List[int] = list(last_hist.get("batch_hit_counts") or [])
            hist_ax.set_axis_on()
            if batch_counts:
                parts = hist_ax.violinplot([batch_counts], positions=[1], showmeans=True, showmedians=True)
                for pc in parts.get("bodies", []) or []:
                    pc.set_facecolor("#0072B2")
                    pc.set_alpha(0.85)
                hist_ax.set_xticks([1])
                hist_ax.set_xticklabels(["batches"])
            hist_ax.set_ylabel("Hit count per batch")
            mean_hits = float(np.mean(batch_counts)) if batch_counts else float("nan")
            if batch_counts and np.isfinite(mean_hits):
                hist_ax.set_title(f"Hit counts per batch — mean: {mean_hits:.1f} ({len(batch_counts)} batches)")
            else:
                hist_ax.set_title("Hit counts per batch")

            ev_summary = last_hist.get("primary_event_summary")
            if isinstance(ev_summary, dict):
                draw_primary_event_rates_on_axes(events_ax, ev_summary)
            else:
                events_ax.set_axis_off()
                events_ax.text(
                    0.5,
                    0.5,
                    "Event chart unavailable — run or load a campaign with simulation records.",
                    ha="center",
                    va="center",
                    transform=events_ax.transAxes,
                    fontsize=10,
                )
        hist_fig.tight_layout()
        hist_canvas.draw_idle()

    _draw_histogram()

    save_row = ttk.Frame(bottom)
    save_row.pack(fill="x", pady=(4, 2))
    save_row.grid_columnconfigure(1, weight=1)

    save_folder_label = ttk.Label(save_row, text="Save folder:")
    save_folder_label.grid(row=0, column=0, sticky="w")

    def _choose_save_folder() -> None:
        sel = filedialog.askdirectory(
            title="Select folder for Batch Runner results",
            initialdir=_default_dialog_dir(),
            parent=win,
        )
        if sel:
            save_dir_var.set(sel)
            save_dir_explicit_var.set(True)

    save_entry = ttk.Entry(save_row, textvariable=save_dir_var)
    save_entry.grid(row=0, column=1, sticky="ew", padx=6)
    CreateToolTip(save_folder_label, BATCH_RUNNER_TOOLTIPS["Save folder"])
    CreateToolTip(save_entry, BATCH_RUNNER_TOOLTIPS["Save folder"])
    set_save_btn = ttk.Button(save_row, text="Choose Save Folder", command=_choose_save_folder)
    set_save_btn.grid(row=0, column=2, padx=(0, 6))

    action_row = ttk.Frame(bottom)
    action_row.pack(fill="x", pady=(0, 0))
    load_action_row = ttk.Frame(action_row)
    load_action_row.pack(fill="x", pady=(0, 4))
    run_action_row = ttk.Frame(action_row)
    run_action_row.pack(fill="x")

    def _apply_metric_filters_from_save(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        by_label: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            lab = str(r.get("label", "") or "").strip().upper()
            if lab:
                by_label[lab] = r
        slots = [
            ("A", m_a_var, op_a_var, thr_a_var),
            ("B", m_b_var, op_b_var, thr_b_var),
            ("C", m_c_var, op_c_var, thr_c_var),
            ("D", m_d_var, op_d_var, thr_d_var),
        ]
        for lab, mv, ov, tv in slots:
            row = by_label.get(lab)
            if not row:
                if lab != "A":
                    mv.set(METRIC_NONE_LABEL)
                continue
            active = bool(row.get("active", row.get("metric") is not None))
            mval = row.get("metric")
            if lab != "A" and (not active or not mval):
                mv.set(METRIC_NONE_LABEL)
                continue
            if mval:
                mv.set(str(mval))
            opv = row.get("operator")
            if opv:
                ov.set(str(opv))
            th = row.get("threshold")
            if th is not None:
                try:
                    tv.set(str(float(th)))
                except (TypeError, ValueError):
                    tv.set(str(th))

    def _apply_batch_settings_from_dict(
        data: Dict[str, Any],
        *,
        update_save_folder: bool = True,
        save_folder_fallback_dir: str = "",
    ) -> List[str]:
        """Restore batch controls and parameter box from a job or summary JSON. Returns warnings."""
        warns: List[str] = []
        if update_save_folder:
            sf = str(data.get("save_folder") or "").strip()
            if not sf and save_folder_fallback_dir:
                sf = os.path.abspath(save_folder_fallback_dir)
            if sf:
                save_dir_var.set(sf)
                save_dir_explicit_var.set(True)
        if "n_runs" in data:
            nruns_var.set(str(int(data["n_runs"])))
        if "n_primary_batches" in data:
            primary_batches_var.set(str(int(data["n_primary_batches"])))
        elif "n_neutral_batches" in data:
            primary_batches_var.set(str(int(data["n_neutral_batches"])))
        bs = data.get("base_seed")
        seed_var.set("" if bs is None else str(int(bs)))
        if "configuration_name" in data:
            configuration_name_var.set(str(data.get("configuration_name") or ""))

        pb = data.get("primary_bounds")
        pn = data.get("primary_numeric_parameters")
        pt = data.get("primary_toggles")
        saved_panel_cb = primary_panel._on_settings_change
        primary_panel._on_settings_change = None
        try:
            if isinstance(pb, dict) and isinstance(pn, dict) and isinstance(pt, dict):
                warns.extend(
                    _apply_neutral_comparison_save_to_panel(
                        primary_panel, bounds_json=pb, numeric_json=pn, toggles_json=pt
                    )
                )
            else:
                warns.append("Primary bounds/numeric/toggles missing; parameter box unchanged.")
        finally:
            primary_panel._on_settings_change = saved_panel_cb

        _refresh_metric_combos()
        rows = data.get("metric_filters")
        if rows is not None:
            _apply_metric_filters_from_save(rows)
        _refresh_metric_combos()

        try:
            n0, t0 = primary_panel.read_numeric_and_toggles()
            _ = normalize_simulation_params({**n0, **t0})
        except Exception as exc:
            warns.append(f"Restored parameters do not normalize cleanly yet: {exc}")
        return warns

    def _load_json_settings() -> None:
        path = filedialog.askopenfilename(
            parent=win,
            title="Load Batch Runner settings JSON",
            initialdir=_default_dialog_dir(),
            **_filedialog_open_json_kwargs(),
        )
        if not path:
            return
        try:
            data = _read_json_maybe_gz(path)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not read file:\n{exc}", parent=win)
            return
        if not isinstance(data, dict):
            messagebox.showerror("Load failed", "File is not a JSON object.", parent=win)
            return
        kind = str(data.get("kind", "") or "")
        if kind != "primary_batch_campaign":
            messagebox.showerror(
                "Load failed",
                "Not a Batch Runner settings file (expected kind 'primary_batch_campaign').\n"
                "Use «Load saved…» for campaign summary JSON with hit-count results.",
                parent=win,
            )
            return
        try:
            wp = _apply_batch_settings_from_dict(data, update_save_folder=False)
            comparison_job_settings_json_path[0] = os.path.abspath(path)
            msg = f"Loaded settings: {os.path.basename(path)}"
            if wp:
                msg += "\n\n" + "\n".join(wp)
            status_var.set(msg)
            try:
                nb.select(tab_setup)
            except tk.TclError:
                pass
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=win)

    def _load_saved_dataset() -> None:
        path = filedialog.askopenfilename(
            parent=win,
            title="Load Batch Runner save",
            initialdir=_default_dialog_dir(),
            **_filedialog_open_json_kwargs(),
        )
        if not path:
            return
        try:
            data = _read_json_maybe_gz(path)
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not read file:\n{exc}", parent=win)
            return
        if not isinstance(data, dict):
            messagebox.showerror("Load failed", "File is not a JSON object.", parent=win)
            return
        kind = str(data.get("kind", "") or "")
        accepted_kinds = (
            "primary_batch_campaign",
            "primary_batch_campaign_session",
            "neutral_set_comparison",
            "neutral_set_comparison_session",
        )
        if kind not in accepted_kinds:
            messagebox.showerror(
                "Load failed",
                "Not a Batch Runner file (expected kind 'primary_batch_campaign' or session settings).",
                parent=win,
            )
            return
        try:
            sf = str(data.get("save_folder") or "").strip() or os.path.dirname(os.path.abspath(path))

            wp: List[str] = []
            if kind in ("primary_batch_campaign", "neutral_set_comparison"):
                wp = _apply_batch_settings_from_dict(
                    data,
                    update_save_folder=True,
                    save_folder_fallback_dir=sf,
                )
            else:
                wp = ["Session settings file: parameter box was not changed (use a summary JSON for full UI)."]
                if "n_runs" in data:
                    nruns_var.set(str(int(data["n_runs"])))
                if "n_primary_batches" in data:
                    primary_batches_var.set(str(int(data["n_primary_batches"])))
                bs = data.get("base_seed")
                seed_var.set("" if bs is None else str(int(bs)))
                rows = data.get("metric_filters")
                if rows is not None:
                    _apply_metric_filters_from_save(rows)
                _refresh_metric_combos()

            if kind in ("primary_batch_campaign", "neutral_set_comparison"):
                hits = data.get("primary_hit_counts")
                if hits is None and "primary_hit_count" in data:
                    hits = [int(data["primary_hit_count"])]
                if isinstance(hits, list) and hits:
                    last_hist.clear()
                    last_hist["batch_hit_counts"] = [int(x) for x in hits]
                    last_hist["n_runs"] = int(data.get("n_runs", 0) or 0)
                    last_hist["n_primary_batches"] = len(hits)
                    last_hist["save_folder"] = sf
                    fsid = str(data.get("full_save_session_id") or "").strip()
                    if fsid:
                        last_hist["full_save_session_id"] = fsid
                    csv_path = data.get("hit_counts_csv")
                    if csv_path:
                        last_hist["hit_counts_csv"] = csv_path
                    pb = data.get("primary_bounds")
                    if isinstance(pb, dict):
                        last_hist["primary_bounds"] = pb
                    pes = data.get("primary_event_summary")
                    if isinstance(pes, dict):
                        last_hist["primary_event_summary"] = pes
                    elif fsid:
                        try:
                            _recs = load_primary_offload_records(sf, fsid, _read_json_maybe_gz)
                            last_hist["primary_event_summary"] = summarize_primary_events_by_hit(_recs)
                        except Exception:
                            pass
                    _draw_histogram()

            comparison_job_settings_json_path[0] = os.path.abspath(path)
            msg = f"Loaded: {os.path.basename(path)}"
            if wp:
                msg += "\n\n" + "\n".join(wp)
            status_var.set(msg)
            try:
                nb.select(tab_setup)
            except tk.TclError:
                pass
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=win)

    load_saved_btn = ttk.Button(load_action_row, text="Load Campaign Summary", command=_load_saved_dataset)
    load_saved_btn.pack(side="left")

    def _collect_batch_job_dict(*, require_save_folder: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Parse and validate the Setup tab; same checks as a run (save folder optional for JSON export)."""
        try:
            n_runs = int(nruns_var.get().strip())
            n_primary = int(primary_batches_var.get().strip())
        except ValueError:
            return None, "Runs per batch and number of batches must be integers."
        if n_runs < 1 or n_runs > 500_000:
            return None, "Runs per batch must be in [1, 500000]."
        if n_primary < 1 or n_primary > 10_000:
            return None, "Number of batches must be in [1, 10000]."

        try:
            primary_bounds = primary_panel.read_optimizable_bounds()
        except (ValueError, RuntimeError) as exc:
            return None, str(exc)

        ma = m_a_var.get().strip()
        mb = m_b_var.get().strip()
        mc = m_c_var.get().strip()
        md = m_d_var.get().strip()
        if not ma:
            return None, "Select metric A (required)."
        try:
            thr_a = float(thr_a_var.get())
            thr_b = float(thr_b_var.get()) if _metric_slot_active(mb) else 0.0
            thr_c = float(thr_c_var.get()) if _metric_slot_active(mc) else 0.0
            thr_d = float(thr_d_var.get()) if _metric_slot_active(md) else 0.0
        except ValueError:
            return None, "Thresholds must be numbers (only checked for active metrics)."

        folder_raw = save_dir_var.get().strip()
        folder = ""
        if require_save_folder:
            if not folder_raw:
                return None, (
                    "Choose a results folder with «Set save folder…» before running. "
                    "The path cannot be left empty."
                )
            if not save_dir_explicit_var.get():
                return None, (
                    "Use «Set save folder…» to select where results will be written, "
                    "or load a saved session (which restores the save folder). "
                    "This avoids accidentally writing to the current working directory."
                )
            folder = os.path.abspath(os.path.expanduser(folder_raw))
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as exc:
                return None, f"Cannot create or use folder:\n{exc}"
        elif folder_raw:
            folder = os.path.abspath(os.path.expanduser(folder_raw))

        seed_txt = seed_var.get().strip()
        base_seed = None
        if seed_txt != "":
            try:
                base_seed = int(seed_txt)
            except ValueError:
                return None, "Seed must be an integer or empty."

        try:
            num_p, tog_p = primary_panel.read_numeric_and_toggles()
            _ = normalize_simulation_params({**num_p, **tog_p})
        except Exception as exc:
            return None, str(exc)

        allowed = set(primary_panel.filtered_metric_names(base_names))
        for letter, mname in (("A", ma), ("B", mb), ("C", mc), ("D", md)):
            if letter != "A" and not _metric_slot_active(mname):
                continue
            if mname not in allowed:
                return None, (
                    f"Metric {letter} ({mname!r}) is not available for the current simulation settings, or is unknown."
                )

        metric_checks: List[Tuple[str, str, float]] = [(ma, op_a_var.get(), thr_a)]
        if _metric_slot_active(mb):
            metric_checks.append((mb, op_b_var.get(), thr_b))
        if _metric_slot_active(mc):
            metric_checks.append((mc, op_c_var.get(), thr_c))
        if _metric_slot_active(md):
            metric_checks.append((md, op_d_var.get(), thr_d))

        def _metric_filter_row(label: str, mname: str, op: str, thr: float) -> Dict[str, Any]:
            active = label == "A" or _metric_slot_active(mname)
            return {
                "label": label,
                "active": active,
                "metric": mname if active else None,
                "operator": str(op) if active else None,
                "threshold": float(thr) if active else None,
            }

        metric_filters_snapshot = [
            _metric_filter_row("A", ma, op_a_var.get(), thr_a),
            _metric_filter_row("B", mb, op_b_var.get(), thr_b),
            _metric_filter_row("C", mc, op_c_var.get(), thr_c),
            _metric_filter_row("D", md, op_d_var.get(), thr_d),
        ]
        sim_light_used, sim_light_canon = simulation_light_tracking_plan(model_spec, metric_checks)
        configuration_name = normalize_configuration_name(configuration_name_var.get())

        return {
            "n_runs": int(n_runs),
            "n_primary": int(n_primary),
            "folder": folder,
            "base_seed": base_seed,
            "configuration_name": configuration_name,
            "primary_bounds": primary_bounds,
            "num_p": num_p,
            "tog_p": tog_p,
            "metric_checks": metric_checks,
            "metric_filters_snapshot": metric_filters_snapshot,
            "sim_light_used": bool(sim_light_used),
            "sim_light_canon": list(sim_light_canon),
        }, None

    def _save_json_settings() -> None:
        collected, err = _collect_batch_job_dict(require_save_folder=False)
        if err is not None or collected is None:
            messagebox.showerror("Cannot save settings", err or "Unknown error.", parent=win)
            return
        default_name = f"batch_{time.strftime('%Y%m%d')}.json"
        path = filedialog.asksaveasfilename(
            parent=win,
            title="Save Batch Runner settings as JSON",
            initialdir=_default_dialog_dir(),
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        job: Dict[str, Any] = {
            "kind": "primary_batch_campaign",
            "version": 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model": {"key": model_spec.key, "label": getattr(model_spec, "label", "")},
            "save_folder": collected["folder"],
            "n_runs": collected["n_runs"],
            "n_primary_batches": collected["n_primary"],
            "base_seed": collected["base_seed"],
            "configuration_name": collected["configuration_name"],
            "metric_filters": list(collected["metric_filters_snapshot"]),
            "metric_filters_logical": "AND",
            "evaluation_note": (
                "Within each simulation, active metric conditions are evaluated in an order sorted by "
                "estimated metric compute cost (cheaper checks first). When simulation_light_tracking is true, "
                "each run used store_history=False and minimal_tracking=True."
            ),
            "simulation_light_tracking": collected["sim_light_used"],
            "simulation_light_tracking_metrics": collected["sim_light_canon"],
            "primary_bounds": _bounds_jsonable(collected["primary_bounds"]),
            "primary_numeric_parameters": _jsonable_params(
                prune_irrelevant_numeric_parameters_for_export(collected["num_p"], collected["tog_p"])
            ),
            "primary_toggles": _jsonable_params(collected["tog_p"]),
        }
        try:
            _write_results_json(path, job)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=win)
            return
        comparison_job_settings_json_path[0] = os.path.abspath(path)
        status_var.set(f"Saved settings JSON: {path}")

    load_json_btn = ttk.Button(load_action_row, text="Load JSON Settings", command=_load_json_settings)
    load_json_btn.pack(side="left", padx=(6, 0))

    save_json_btn = ttk.Button(load_action_row, text="Save JSON Settings", command=_save_json_settings)
    save_json_btn.pack(side="left", padx=(6, 0))

    def _run() -> None:
        collected, err = _collect_batch_job_dict(require_save_folder=True)
        if err is not None or collected is None:
            messagebox.showerror("Invalid input", err or "Unknown error.", parent=win)
            return

        n_runs = int(collected["n_runs"])
        n_primary = int(collected["n_primary"])
        folder = str(collected["folder"])
        base_seed = collected["base_seed"]
        primary_bounds = collected["primary_bounds"]
        num_p = collected["num_p"]
        tog_p = collected["tog_p"]
        metric_checks = collected["metric_checks"]
        metric_filters_snapshot = collected["metric_filters_snapshot"]
        configuration_name = str(collected.get("configuration_name") or "")

        run_btn.configure(state="disabled")
        pause_btn.configure(state="normal")
        resume_btn.configure(state="disabled")
        prog["value"] = 0.0
        q: queue.Queue = queue.Queue()
        resume_ev = threading.Event()
        resume_ev.set()
        resume_evt_holder[0] = resume_ev

        def _imm_pause_ns() -> None:
            ev = resume_evt_holder[0]
            if ev is not None and ev.is_set():
                ev.clear()

        def _imm_resume_ns() -> None:
            ev = resume_evt_holder[0]
            if ev is not None:
                ev.set()

        prev_unreg = terminal_pause_unreg_holder[0]
        if prev_unreg is not None:
            try:
                prev_unreg()
            except Exception:
                pass
        terminal_pause_unreg_holder[0] = register_terminal_pause_hooks(
            win,
            immediate_pause=_imm_pause_ns,
            immediate_resume=_imm_resume_ns,
            main_thread_after_pause=_pause_run,
            main_thread_after_resume=_resume_run,
        )

        session_id = time.strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(folder, f"primary_batch_campaign_{session_id}.json")
        param_names_list_snap = list(primary_panel.param_names)

        def worker() -> None:
            offload_writer: Optional[NeutralComparisonOffloadWriter] = None
            sim_light_used, sim_light_canon = simulation_light_tracking_plan(model_spec, metric_checks)
            try:
                offload_writer = NeutralComparisonOffloadWriter(
                    folder,
                    session_id,
                    model_key=str(getattr(model_spec, "key", "simulation")),
                    param_names_list=param_names_list_snap,
                    metric_name_at_offload=str(metric_checks[0][0]),
                    job_settings_json_path=comparison_job_settings_json_path[0],
                )
            except Exception as exc:
                q.put(("error", f"Could not initialize simulation record storage in folder:\n{exc}"))
                return
            try:
                settings_snapshot: Dict[str, Any] = {
                    "kind": "primary_batch_campaign_session",
                    "version": 1,
                    "session_id": session_id,
                    "model": {"key": model_spec.key, "label": getattr(model_spec, "label", "")},
                    "n_runs": int(n_runs),
                    "n_primary_batches": int(n_primary),
                    "base_seed": base_seed,
                    "configuration_name": configuration_name,
                    "metric_filters": list(metric_filters_snapshot),
                    "simulation_light_tracking": bool(sim_light_used),
                    "simulation_light_tracking_metrics": list(sim_light_canon),
                    "full_save_folder": folder,
                    "saved_at_epoch": int(time.time()),
                }
                try:
                    _write_results_json(full_save_settings_path_json(folder, session_id), settings_snapshot)
                except Exception:
                    pass

                n_p = int(n_primary)
                stage_width_pct = 100.0 / float(max(1, n_p))

                def overall_pct(batch_index: int, sim_done: int) -> float:
                    return min(
                        100.0,
                        batch_index * stage_width_pct
                        + (float(sim_done) / float(max(1, n_runs))) * stage_width_pct,
                    )

                def push_progress(batch_index: int, sim_done: int, hits_so_far: int, phase: str) -> None:
                    pct = overall_pct(batch_index, sim_done)
                    q.put(
                        (
                            "progress",
                            {
                                "pct": pct,
                                "status": (
                                    f"Batch {batch_index + 1}/{n_p}: {phase} — "
                                    f"simulation {sim_done}/{n_runs} (hits so far: {hits_so_far})"
                                ),
                            },
                        )
                    )

                batch_hit_counts: List[int] = []
                for r in range(n_p):
                    def batch_progress(
                        done: int,
                        total: int,
                        hits_so_far: int,
                        _r: int = r,
                    ) -> None:
                        push_progress(_r, done, hits_so_far, f"Primary Monte Carlo batch {_r + 1}")

                    q.put(
                        (
                            "progress",
                            {
                                "pct": overall_pct(r, 0),
                                "status": (
                                    f"Batch {r + 1}/{n_p}: starting {n_runs} simulations…"
                                ),
                            },
                        )
                    )
                    local_seed = _primary_batch_seed(base_seed, r)
                    hc = run_hit_count_batch(
                        model_spec=model_spec,
                        n_runs=n_runs,
                        bounds=primary_bounds,
                        numeric_base=num_p,
                        toggles=tog_p,
                        metric_checks=metric_checks,
                        base_seed=local_seed,
                        progress_callback=batch_progress,
                        resume_event=resume_ev,
                        offload_writer=offload_writer,
                        offload_stage="primary",
                    )
                    batch_hit_counts.append(int(hc))

                offload_writer.finalize()

                _prec: List[Dict[str, Any]] = []
                try:
                    _prec = load_primary_offload_records(folder, session_id, _read_json_maybe_gz)
                except Exception:
                    _prec = []

                export_numeric = _jsonable_params(
                    prune_irrelevant_numeric_parameters_for_export(num_p, tog_p)
                )
                hit_counts_csv_abs = write_session_hit_counts_csv(
                    folder=folder,
                    session_id=session_id,
                    hit_counts=batch_hit_counts,
                    primary_numeric=export_numeric,
                    primary_toggles=tog_p,
                    primary_bounds=_bounds_jsonable(primary_bounds),
                    configuration=str(configuration_name),
                    session_dir=folder,
                    plot_label=(
                        configuration_name.strip()
                        or (
                            os.path.splitext(os.path.basename(comparison_job_settings_json_path[0]))[0]
                            if comparison_job_settings_json_path[0]
                            else session_id
                        )
                    ),
                    param_names_list=param_names_list_snap,
                    offload_records=_prec,
                    n_runs=int(n_runs),
                )

                primary_event_summary: Optional[Dict[str, Any]] = None
                try:
                    primary_event_summary = summarize_primary_events_by_hit(_prec)
                except Exception:
                    primary_event_summary = None

                save_payload: Dict[str, Any] = {
                    "kind": "primary_batch_campaign",
                    "version": 1,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "full_save_session_id": session_id,
                    "full_save_manifest": os.path.basename(full_save_manifest_path_json(folder, session_id)),
                    "full_save_settings": os.path.basename(full_save_settings_path_json(folder, session_id)),
                    "offload_batches_written": len(offload_writer.manifest.get("batches", [])),
                    "save_folder": folder,
                    "output_path": out_path,
                    "n_runs": int(n_runs),
                    "n_primary_batches": int(n_p),
                    "base_seed": base_seed,
                    "configuration_name": configuration_name,
                    "metric_filters": list(metric_filters_snapshot),
                    "metric_filters_logical": "AND",
                    "simulation_light_tracking": bool(sim_light_used),
                    "simulation_light_tracking_metrics": list(sim_light_canon),
                    "primary_hit_counts": [int(x) for x in batch_hit_counts],
                    "primary_bounds": _bounds_jsonable(primary_bounds),
                    "primary_numeric_parameters": _jsonable_params(
                        prune_irrelevant_numeric_parameters_for_export(num_p, tog_p)
                    ),
                    "primary_toggles": _jsonable_params(tog_p),
                }
                if hit_counts_csv_abs:
                    save_payload["hit_counts_csv"] = os.path.basename(hit_counts_csv_abs)
                if primary_event_summary is not None:
                    save_payload["primary_event_summary"] = primary_event_summary
                    _pe_png = write_primary_event_rates_png(
                        folder=folder,
                        session_id=session_id,
                        summary=primary_event_summary,
                    )
                    if _pe_png:
                        save_payload["primary_events_png"] = os.path.basename(_pe_png)
                if _prec:
                    _ph_png = write_parameter_heatmap_png(
                        folder=folder,
                        session_id=session_id,
                        records=_prec,
                        param_names_list=param_names_list_snap,
                        primary_bounds=_bounds_jsonable(primary_bounds),
                    )
                    if _ph_png:
                        save_payload["parameter_heatmap_png"] = os.path.basename(_ph_png)
                saved_path: Optional[str] = None
                save_error: Optional[str] = None
                try:
                    _write_results_json(out_path, save_payload)
                    saved_path = out_path
                except Exception as exc:
                    save_error = str(exc)

                q.put(
                    (
                        "done",
                        {
                            "batch_hit_counts": batch_hit_counts,
                            "n_runs": n_runs,
                            "n_primary_batches": n_p,
                            "saved_path": saved_path,
                            "save_error": save_error,
                            "hit_counts_csv": hit_counts_csv_abs,
                            "full_save_session_id": session_id,
                            "save_folder": folder,
                            "offload_batches_written": len(offload_writer.manifest.get("batches", [])),
                            "full_save_manifest": os.path.basename(
                                full_save_manifest_path_json(folder, session_id)
                            ),
                            "primary_event_summary": primary_event_summary,
                            "primary_events_png": (
                                save_payload.get("primary_events_png") if primary_event_summary else None
                            ),
                            "primary_bounds": _bounds_jsonable(primary_bounds),
                            "parameter_heatmap_png": save_payload.get("parameter_heatmap_png"),
                        },
                    )
                )
            except Exception as exc:
                q.put(("error", str(exc)))
            finally:
                if offload_writer is not None:
                    try:
                        offload_writer.finalize()
                    except Exception:
                        pass
                u_fin = terminal_pause_unreg_holder[0]
                if u_fin is not None:
                    try:
                        u_fin()
                    except Exception:
                        pass
                    terminal_pause_unreg_holder[0] = None

        threading.Thread(target=worker, daemon=True).start()

        def _clear_terminal_hooks_for_this_run() -> None:
            u = terminal_pause_unreg_holder[0]
            if u is not None:
                try:
                    u()
                except Exception:
                    pass
                terminal_pause_unreg_holder[0] = None

        def poll() -> None:
            nonlocal last_hist
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == "progress":
                        prog["value"] = float(payload.get("pct", 0.0))
                        status_var.set(str(payload.get("status", "")))
                    elif kind == "status":
                        status_var.set(str(payload))
                    elif kind == "error":
                        status_var.set("Error.")
                        messagebox.showerror("Run failed", payload, parent=win)
                        _clear_terminal_hooks_for_this_run()
                        resume_evt_holder[0] = None
                        resume_ev.set()
                        run_btn.configure(state="normal")
                        pause_btn.configure(state="disabled")
                        resume_btn.configure(state="disabled")
                        return
                    elif kind == "done":
                        prog["value"] = 100.0
                        sp = payload.get("saved_path")
                        se = payload.get("save_error")
                        csv_path = payload.get("hit_counts_csv")
                        if sp and csv_path:
                            status_var.set(f"Done. Saved: {sp} and {csv_path}")
                        elif sp:
                            status_var.set(f"Done. Saved: {sp} — open the Results tab for the plot.")
                        elif se:
                            status_var.set(f"Done (save failed: {se}). Open the Results tab for the plot.")
                        else:
                            status_var.set("Done. Open the Results tab for the plot.")
                        last_hist = dict(payload)
                        counts = payload.get("batch_hit_counts") or []
                        mean_hits = float(np.mean(counts)) if counts else float("nan")
                        if np.isfinite(mean_hits):
                            status_var.set(f"{status_var.get()} Mean hits per batch: {mean_hits:.2f}.")
                        _draw_histogram()
                        _clear_terminal_hooks_for_this_run()
                        resume_evt_holder[0] = None
                        resume_ev.set()
                        run_btn.configure(state="normal")
                        pause_btn.configure(state="disabled")
                        resume_btn.configure(state="disabled")
                        try:
                            nb.select(tab_plot)
                        except tk.TclError:
                            pass
                        return
            except queue.Empty:
                pass
            win.after(150, poll)

        poll()

    resume_evt_holder: list[threading.Event | None] = [None]
    terminal_pause_unreg_holder: list[Optional[Callable[[], None]]] = [None]

    run_btn = ttk.Button(run_action_row, text="Run Batch", command=_run)
    run_btn.pack(side="left", padx=(0, 6))
    pause_btn = ttk.Button(run_action_row, text="Pause", state="disabled")
    pause_btn.pack(side="left", padx=(0, 4))
    resume_btn = ttk.Button(run_action_row, text="Resume", state="disabled")
    resume_btn.pack(side="left", padx=(0, 6))

    def _pause_run() -> None:
        ev = resume_evt_holder[0]
        if ev is None:
            return
        if ev.is_set():
            ev.clear()
        status_var.set("Paused — click Resume or type 'resume' in the terminal.")
        pause_btn.configure(state="disabled")
        resume_btn.configure(state="normal")

    def _resume_run() -> None:
        ev = resume_evt_holder[0]
        if ev is not None:
            ev.set()
        pause_btn.configure(state="normal")
        resume_btn.configure(state="disabled")

    pause_btn.configure(command=_pause_run)
    resume_btn.configure(command=_resume_run)

    def _close() -> None:
        u = terminal_pause_unreg_holder[0]
        if u is not None:
            try:
                u()
            except Exception:
                pass
            terminal_pause_unreg_holder[0] = None
        try:
            plt.close(hist_fig)
        except Exception:
            pass
        try:
            root.deiconify()
        except Exception:
            pass
        win.destroy()

    ttk.Button(run_action_row, text="← Back", command=_close).pack(side="left")

    try:
        win.update_idletasks()
        win.deiconify()
    except tk.TclError:
        pass


def neutral_set_comparison_gui(
    win: tk.Toplevel | tk.Tk,
    root: tk.Tk,
    model_spec: OptimizationModelSpec,
) -> None:
    """Open the Batch Runner GUI."""
    batch_runner_gui(win, root, model_spec)
