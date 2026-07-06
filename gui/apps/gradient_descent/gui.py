"""
Gradient Descent Optimization GUI for parameter optimization.

This module provides a GUI for finding optimal parameters using gradient descent
to maximize or minimize a particular metric.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import gzip
import threading
import queue
import time
import os
import re
import sys
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor
from gui.common.colors import OKABE_ITO, DEFAULT_HEATMAP_CMAP
from gui.apps.gradient_descent.runtime import (
    RunningStats,
    build_run_summary_stats,
    configure_optimization_warnings,
    reset_deleted_point_stats,
)

configure_optimization_warnings(os.environ)
# Check for available dimension reduction libraries
HAS_SKLEARN = False
HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    # Some environments raise RuntimeError during umap import (e.g., numba cache issues).
    umap = None
    HAS_UMAP = False
from gui.models.registry import OptimizationModelSpec, get_model_by_key
from gui.common.terminal_pause_listener import register_terminal_pause_hooks
from gui.apps.individual.gui import individual_gui
from gui.common.model_diagram import SimulationModelDiagram
from gui.common.data_utils import to_json_serializable
from gui.persistence.dataset_load import (
    DATASET_FILETYPES,
    apply_loaded_fixed_params_to_widgets,
    extract_loaded_fixed_params,
    invalid_dataset_file_message,
    is_supported_dataset_file,
    load_dataset_payload,
    prepare_decoded_aligned_dataset,
    resolve_loaded_best_run_index,
    run_post_loaded_fixed_refreshes,
    restore_param_ui_state_and_fixed_values,
    restore_results_summary_text,
    restore_ui_state,
    set_loaded_dataset_button_states,
    show_loaded_dataset_info_dialog,
    warn_if_model_mismatch,
)
from gui.persistence.full_save import (
    discover_full_save_session_ids,
    full_save_dataset_snapshot_path,
    full_save_manifest_path_json,
    full_save_settings_path_gz,
    full_save_settings_path_json,
    is_full_save_dataset_snapshot_basename,
    is_full_save_manifest_filename,
    load_full_save_manifest,
    manifest_declared_batch_path_count,
    ordered_existing_batch_paths_from_manifest,
)
from gui.persistence.offload_reconstruct import (
    list_offload_batch_files as list_offload_batch_files_shared,
    parse_offload_filename,
    reconstruct_from_offload_batches,
)
from gui.persistence.metric_sidecar import (
    build_metric_sidecar_payload,
    compute_metric_from_cached_input as compute_metric_from_cached_input_shared,
    decode_metric_sidecar_values,
    extract_metric_digest_from_sidecar_filename,
    metric_column_sidecar_path,
    metric_name_digest,
    normalize_metric_input as normalize_metric_input_shared,
)
from gui.persistence.json_io import make_read_json_maybe_gz_fn, make_write_json_maybe_gz_atomic_fn
from gui.persistence.seed_sweep import (
    build_seed_sweep_results_payload,
    load_seed_sweep_from_scratch_dir,
    load_seed_sweep_results,
    save_seed_sweep_results,
    suggested_seed_sweep_filename,
)
from gui.metrics import (
    filter_metric_options_for_simulation_settings,
    metric_supports_seed_sweep_light_mode,
    normalize_metric_name,
)
from gui.common.seed_policy import choose_launch_seed, descent_seed_for_run, parse_optional_seed, replicate_seed_for_run
from gui.common.simulation_settings import (
    NO_DEATH,
    both_constant_death_and_duplication,
    normalize_simulation_params,
    simulation_toggles_from_ui_state,
)
from gui.common.tooltips import (
    PARAMETER_TOOLTIPS, 
    GRADIENT_DESCENT_TOOLTIPS,
    SIMULATION_SETTINGS_TOOLTIPS,
    HOMOGENEOUS_TOOLTIP
)
from gui.common.widgets import CreateToolTip
from gui.shared.monte_carlo_panel import (
    _bind_vertical_mousewheel,
    _bind_vertical_mousewheel_descendants,
)


_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)
_write_json_maybe_gz = make_write_json_maybe_gz_atomic_fn(indent=2)


def _filedialog_filetypes_kwarg_macos_safe(filetypes):
    """
    macOS Tk 8.6 + Cocoa can abort with NSInvalidArgumentException in setAllowedFileTypes
    for some filters (e.g. '*.*', multi-glob strings like '*.json.gz *.json', mid-string globs).

    Passing no filetypes can still fail on some macOS Tk builds; a single ('*') group is
    the stable option. Callers should validate extensions after selection.
    """
    if sys.platform == "darwin":
        return {"filetypes": [("All files", "*")]}
    return {"filetypes": filetypes}


def gradient_descent_gui(win, root, model_spec: OptimizationModelSpec | None = None):
    """Gradient descent optimization GUI for finding optimal parameters."""

    if model_spec is None:
        model_spec = get_model_by_key("simulation")
    
    win.title("Gradient Descent Optimization")
    
    # Set window size
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    max_width = int(screen_width * 0.9)
    max_height = int(screen_height * 0.9)
    win.geometry(f"{max_width}x{max_height}+0+0")
    
    # Default parameter values come from the selected model spec
    default_params = dict(model_spec.default_params)
    # Optional cross-model parameter (hidden unless enabled in Simulation Settings)
    default_params.setdefault("Initial Energy", 0.0)
    default_params.setdefault("Intermediate Costs", 0.0)
    default_params.setdefault("Average In_Flow (Acetate)", 0.0)
    
    param_names = list(default_params.keys())
    
    # Short names for display (to prevent cutoff)
    param_short_names = {
        "M1 Facilitated Diffusion Constant": "M1 Facil. Diff.",
        "M2 Simple Diffusion Constant": "M2 Simple Diff.",
        "Duplication Sigmoid Midpoint": "Dup. Sigmoid Mid.",
        "Duplication Sigmoid Intensity": "Dup. Sigmoid Int.",
        "Constant Probability": "Const. Prob.",
        "Duplication Threshold Multiplier": "Dup. Thresh. Mult.",
        "Intermediate Storage Cost": "Storage Cost",
        "Number of Generations": "Num. Generations",
        "Initial Organism Count": "Init. Org. Count",
        "M1 Saturation Constant": "M1 Sat. Const.",
        "Transport Rate Constant": "Transport Rate",
        "Cost of Transport": "Transport Cost",
        "Initial Facilitation": "Init. Facil.",
        "Initial Energy": "Init. Energy",
        "Intermediate Costs": "Intermed. Costs",
        "Average In_Flow (Acetate)": "Acetate In-Flow",
        "Flow Percentage": "Flow %",
    }
    
    def get_display_name(param_name):
        """Get short display name if available, otherwise full name."""
        return param_short_names.get(param_name, param_name)
    
    # Default min/max bounds based on parameter default values
    def get_default_bounds(param_name, default_val):
        """Get reasonable default min/max bounds based on the default value."""
        if param_name == "Flow Percentage":
            return ("0.0", "50.0")
        if isinstance(default_val, (int, float)):
            if default_val == 0:
                return ("0.0", "10.0")
            elif default_val < 1:
                # Small values: allow range around default
                return (str(max(0, default_val * 0.1)), str(default_val * 10))
            elif default_val < 10:
                # Medium values: allow range around default
                return (str(max(0, default_val * 0.5)), str(default_val * 2))
            else:
                # Large values: allow range around default
                return (str(max(0, default_val * 0.5)), str(default_val * 2))
        else:
            return ("0.0", "10.0")
    
    # Left panel: Controls (split into two columns)
    left_panel = tk.Frame(win)
    left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    # Right panel: Results and plots
    right_panel = tk.Frame(win)
    right_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    
    # Set minimum width for left panel and allow it to grow slightly
    win.grid_columnconfigure(0, weight=1, minsize=800)
    win.grid_columnconfigure(1, weight=2)
    win.grid_rowconfigure(0, weight=1)
    
    # Left column frame (scrollable — keeps Optimization Settings reachable on short screens)
    left_col_shell = tk.Frame(left_panel)
    left_col_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    left_col_shell.grid_rowconfigure(0, weight=1)
    left_col_shell.grid_columnconfigure(0, weight=1)

    left_scroll_canvas = tk.Canvas(left_col_shell, highlightthickness=0)
    left_scrollbar = ttk.Scrollbar(left_col_shell, orient="vertical", command=left_scroll_canvas.yview)
    left_col = tk.Frame(left_scroll_canvas)

    def _on_left_col_configure(_event=None) -> None:
        try:
            left_scroll_canvas.configure(scrollregion=left_scroll_canvas.bbox("all"))
        except tk.TclError:
            pass

    left_col.bind("<Configure>", _on_left_col_configure)
    _left_inner_win = left_scroll_canvas.create_window((0, 0), window=left_col, anchor="nw")

    def _on_left_canvas_configure(event) -> None:
        try:
            left_scroll_canvas.itemconfig(_left_inner_win, width=max(1, int(event.width)))
        except tk.TclError:
            pass

    left_scroll_canvas.bind("<Configure>", _on_left_canvas_configure)
    left_scroll_canvas.configure(yscrollcommand=left_scrollbar.set)
    left_scroll_canvas.grid(row=0, column=0, sticky="nsew")
    left_scrollbar.grid(row=0, column=1, sticky="ns")
    _bind_vertical_mousewheel(left_scroll_canvas)
    _bind_vertical_mousewheel_descendants(left_scroll_canvas, left_col)
    
    # Right column frame
    right_col = tk.Frame(left_panel)
    right_col.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    
    left_panel.grid_columnconfigure(0, weight=1)
    left_panel.grid_columnconfigure(1, weight=1)
    left_panel.grid_rowconfigure(0, weight=1)
    
    # === Optimization Settings ===
    opt_frame = tk.LabelFrame(left_col, text="Optimization Settings", padx=5, pady=5)
    opt_frame.pack(fill="x", pady=(0, 10))
    opt_frame.grid_columnconfigure(1, weight=1)

    # Metric selection
    metric_label = tk.Label(opt_frame, text="Metric to Optimize:")
    metric_label.grid(row=0, column=0, sticky="w", pady=2)
    if model_spec.metric_names and "Task2 Share Weighted Prob. Mean" in model_spec.metric_names:
        default_metric = "Task2 Share Weighted Prob. Mean"
    else:
        default_metric = model_spec.metric_names[0] if model_spec.metric_names else ""
    metric_var = tk.StringVar(value=default_metric)
    metric_combo = ttk.Combobox(
        opt_frame,
        textvariable=metric_var,
        values=model_spec.metric_names,
        width=25,
        state="readonly",
    )
    metric_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    CreateToolTip(metric_label, GRADIENT_DESCENT_TOOLTIPS["Metric to Optimize"])
    CreateToolTip(metric_combo, GRADIENT_DESCENT_TOOLTIPS["Metric to Optimize"])
    # Full Save folder var must be initialized early because metric option refresh can reference manifests.
    full_save_dir_var = tk.StringVar(value="")

    # Maximize or minimize
    opt_goal_label = tk.Label(opt_frame, text="Optimization Goal:")
    opt_goal_label.grid(row=1, column=0, sticky="w", pady=2)
    opt_goal_var = tk.StringVar(value="Maximize")
    opt_goal_combo = ttk.Combobox(
        opt_frame,
        textvariable=opt_goal_var,
        values=["Maximize", "Minimize"],
        width=25,
        state="readonly",
    )
    opt_goal_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    CreateToolTip(opt_goal_label, GRADIENT_DESCENT_TOOLTIPS["Optimization Goal"])
    CreateToolTip(opt_goal_combo, GRADIENT_DESCENT_TOOLTIPS["Optimization Goal"])
    
    # Learning rate
    learning_rate_label = tk.Label(opt_frame, text="Learning Rate:")
    learning_rate_label.grid(row=2, column=0, sticky="w", pady=2)
    learning_rate_entry = tk.Entry(opt_frame, width=10)
    learning_rate_entry.insert(0, "0.01")
    learning_rate_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(learning_rate_label, GRADIENT_DESCENT_TOOLTIPS["Learning Rate"])
    CreateToolTip(learning_rate_entry, GRADIENT_DESCENT_TOOLTIPS["Learning Rate"])
    
    # Number of iterations
    max_iterations_label = tk.Label(opt_frame, text="Max Iterations:")
    max_iterations_label.grid(row=3, column=0, sticky="w", pady=2)
    max_iterations_entry = tk.Entry(opt_frame, width=10)
    max_iterations_entry.insert(0, "20")
    max_iterations_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(max_iterations_label, GRADIENT_DESCENT_TOOLTIPS["Max Iterations"])
    CreateToolTip(max_iterations_entry, GRADIENT_DESCENT_TOOLTIPS["Max Iterations"])
    
    # Convergence threshold
    convergence_threshold_label = tk.Label(opt_frame, text="Convergence Threshold:")
    convergence_threshold_label.grid(row=4, column=0, sticky="w", pady=2)
    convergence_threshold_entry = tk.Entry(opt_frame, width=10)
    convergence_threshold_entry.insert(0, "0.001")
    convergence_threshold_entry.grid(row=4, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(convergence_threshold_label, GRADIENT_DESCENT_TOOLTIPS["Convergence Threshold"])
    CreateToolTip(convergence_threshold_entry, GRADIENT_DESCENT_TOOLTIPS["Convergence Threshold"])
    
    # Gradient step size (for numerical differentiation)
    gradient_step_label = tk.Label(opt_frame, text="Gradient Step Size:")
    gradient_step_label.grid(row=5, column=0, sticky="w", pady=2)
    gradient_step_entry = tk.Entry(opt_frame, width=10)
    gradient_step_entry.insert(0, "0.01")
    gradient_step_entry.grid(row=5, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(gradient_step_label, GRADIENT_DESCENT_TOOLTIPS["Gradient Step Size"])
    CreateToolTip(gradient_step_entry, GRADIENT_DESCENT_TOOLTIPS["Gradient Step Size"])
    
    # Number of random starts
    num_starts_label = tk.Label(opt_frame, text="Number of Random Starts:")
    num_starts_label.grid(row=6, column=0, sticky="w", pady=2)
    num_starts_entry = tk.Entry(opt_frame, width=10)
    num_starts_entry.insert(0, "10000")
    num_starts_entry.grid(row=6, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(num_starts_label, GRADIENT_DESCENT_TOOLTIPS["Number of Random Starts"])
    CreateToolTip(num_starts_entry, GRADIENT_DESCENT_TOOLTIPS["Number of Random Starts"])
    
    # Number of gradient descents per start
    descents_per_start_label = tk.Label(opt_frame, text="Gradient Descents per Start:")
    descents_per_start_label.grid(row=7, column=0, sticky="w", pady=2)
    descents_per_start_entry = tk.Entry(opt_frame, width=10)
    descents_per_start_entry.insert(0, "1")
    descents_per_start_entry.grid(row=7, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(descents_per_start_label, GRADIENT_DESCENT_TOOLTIPS["Gradient Descents per Start"])
    CreateToolTip(descents_per_start_entry, GRADIENT_DESCENT_TOOLTIPS["Gradient Descents per Start"])
    
    # Storage sample rate (for memory management)
    sample_rate_label = tk.Label(opt_frame, text="Storage Sample Rate (1=all):")
    sample_rate_label.grid(row=8, column=0, sticky="w", pady=2)
    sample_rate_entry = tk.Entry(opt_frame, width=10)
    sample_rate_entry.insert(0, "1")
    sample_rate_entry.grid(row=8, column=1, sticky="w", padx=5, pady=2)
    tooltip_text = "Store every Nth iteration for visualization (1=all, 5=every 5th). Higher values use less memory but lower visualization detail."
    CreateToolTip(sample_rate_label, tooltip_text)
    CreateToolTip(sample_rate_entry, tooltip_text)

    # Random seed for optimization runs.
    seed_label = tk.Label(opt_frame, text="Random Seed:")
    seed_label.grid(row=9, column=0, sticky="w", pady=2)
    seed_entry = tk.Entry(opt_frame, width=10)
    seed_entry.grid(row=9, column=1, sticky="w", padx=5, pady=2)
    if "Random Seed (optional)" in PARAMETER_TOOLTIPS:
        CreateToolTip(seed_label, PARAMETER_TOOLTIPS["Random Seed (optional)"])
        CreateToolTip(seed_entry, PARAMETER_TOOLTIPS["Random Seed (optional)"])

    # Replicate count per optimization start.
    replicates_label = tk.Label(opt_frame, text="Number of Replicates:")
    replicates_label.grid(row=10, column=0, sticky="w", pady=2)
    replicates_entry = tk.Entry(opt_frame, width=10)
    replicates_entry.insert(0, "1")
    replicates_entry.grid(row=10, column=1, sticky="w", padx=5, pady=2)
    CreateToolTip(replicates_label, GRADIENT_DESCENT_TOOLTIPS["Number of Replicates"])
    CreateToolTip(replicates_entry, GRADIENT_DESCENT_TOOLTIPS["Number of Replicates"])

    viz_frame = tk.LabelFrame(opt_frame, text="Visualization metrics (heatmaps / histograms)", padx=4, pady=4)
    viz_frame.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    viz_frame.grid_columnconfigure(1, weight=1)

    viz_metric_label = tk.Label(viz_frame, text="Metric to Visualize:")
    viz_metric_label.grid(row=0, column=0, sticky="w", pady=2)
    viz_metric_var = tk.StringVar(value="")
    viz_metric_combo = ttk.Combobox(viz_frame, textvariable=viz_metric_var, values=[], width=25, state="readonly")
    viz_metric_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    CreateToolTip(viz_metric_label, "Metric used by Parameter Heatmaps and Metric Histogram.")
    CreateToolTip(viz_metric_combo, "Metric used by Parameter Heatmaps and Metric Histogram.")
    viz_metric2_label = tk.Label(viz_frame, text="2nd Metric to Visualize:")
    viz_metric2_label.grid(row=1, column=0, sticky="w", pady=2)
    viz_metric2_var = tk.StringVar(value="")
    viz_metric2_combo = ttk.Combobox(viz_frame, textvariable=viz_metric2_var, values=[], width=25, state="readonly")
    viz_metric2_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
    CreateToolTip(viz_metric2_label, "Secondary metric for Metric Histogram X/Y heatmap (e.g., Metric 1 vs Metric 2).")
    CreateToolTip(viz_metric2_combo, "Selecting this metric preloads it for faster Metric Histogram opening.")
    
    # === Parameters to Optimize ===
    param_opt_frame = tk.LabelFrame(left_col, text="Parameters to Optimize", padx=5, pady=5)
    param_opt_frame.pack(fill="x", pady=(0, 10))
    param_opt_frame.grid_rowconfigure(0, weight=1)
    param_opt_frame.grid_columnconfigure(0, weight=1)
    
    # Scrollable frame for parameter selection
    param_canvas = tk.Canvas(param_opt_frame, height=220, highlightthickness=0)
    param_scrollbar = ttk.Scrollbar(param_opt_frame, orient="vertical", command=param_canvas.yview)
    param_scrollable_frame = tk.Frame(param_canvas)
    
    param_scrollable_frame.bind(
        "<Configure>",
        lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all"))
    )
    
    _param_inner_win = param_canvas.create_window((0, 0), window=param_scrollable_frame, anchor="nw")

    def _on_param_canvas_configure(event) -> None:
        try:
            param_canvas.itemconfig(_param_inner_win, width=max(1, int(event.width)))
        except tk.TclError:
            pass

    param_canvas.bind("<Configure>", _on_param_canvas_configure)
    param_canvas.configure(yscrollcommand=param_scrollbar.set)
    
    param_canvas.grid(row=0, column=0, sticky="nsew")
    param_scrollbar.grid(row=0, column=1, sticky="ns")
    _bind_vertical_mousewheel(param_canvas)
    _bind_vertical_mousewheel_descendants(param_canvas, param_scrollable_frame)
    
    # Column headers (add first at row 0) - Fix checkbox on the left
    tk.Label(param_scrollable_frame, text="Fix", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=2, pady=2)
    tk.Label(param_scrollable_frame, text="Parameter", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky="w", padx=2, pady=2)
    tk.Label(param_scrollable_frame, text="Min (optional)", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=2, pady=2)
    tk.Label(param_scrollable_frame, text="Max", font=("Arial", 9, "bold")).grid(row=0, column=3, padx=2, pady=2)
    
    # Parameter entries and fix checkboxes (start at row 1)
    # All parameters start as optimizable by default
    param_fix_checkboxes = {}  # True = fixed, False = optimizable
    param_initial_entries = {}
    param_min_entries = {}
    param_max_entries = {}
    param_widgets = {}  # Store all widgets for a parameter so we can hide/show them
    hidden_params = set()  # Parameters temporarily hidden (model-dependent toggles)
    # Hide Initial Energy unless enabled in Simulation Settings
    hidden_params.add("Initial Energy")
    # Hide Intermediate Costs unless enabled in Simulation Settings
    hidden_params.add("Intermediate Costs")
    # Hide acetate inflow unless enabled in Simulation Settings
    hidden_params.add("Average In_Flow (Acetate)")
    # Hide Flow Percentage unless Chemostat Flow is enabled
    hidden_params.add("Flow Percentage")
    hidden_params.add("Constant Probability")
    # Independent Traits is controlled via Simulation Settings
    hidden_params.add("Independent Traits")
    # Initial B is only relevant when Independent Traits + Homogeneous Population are enabled
    hidden_params.add("Initial B")
    
    def _tooltip_for_param_name(param_name):
        alias = {
            "Average In_Flow": "Average In-Flow",
            "Average In-Flow": "Average In-Flow",
        }
        key = alias.get(param_name, param_name)
        return PARAMETER_TOOLTIPS.get(key, f"{param_name}\n\nModel parameter.")

    def refresh_optimizable_params():
        """Refresh the optimizable parameters section, showing only non-fixed parameters."""
        # Clear all existing widgets
        for widgets in param_widgets.values():
            for widget in widgets:
                widget.grid_remove()
        
        # Get fixed parameters
        fixed_params = [p for p, var in param_fix_checkboxes.items() if var.get() and (p not in hidden_params)]
        
        # Show only optimizable (non-fixed) parameters
        optimizable_params = [p for p in param_names if (p not in fixed_params) and (p not in hidden_params)]
        
        for i, param_name in enumerate(optimizable_params):
            row = i + 1  # Start from row 1, after headers
            widgets = param_widgets[param_name]
            
            # Show all widgets for this parameter - Fix checkbox on the left
            widgets[4].grid(row=row, column=0, padx=2, pady=1, sticky="w")  # Fix checkbox
            widgets[0].grid(row=row, column=1, sticky="w", padx=2, pady=1)  # Label
            widgets[2].grid(row=row, column=2, padx=2, pady=1)  # Min
            widgets[3].grid(row=row, column=3, padx=2, pady=1)  # Max
        try:
            param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        except tk.TclError:
            pass
    
    def refresh_fixed_params():
        """Refresh fixed parameters, showing only fixed parameters."""
        fixed_values = {name: entry.get() for name, entry in fixed_entries.items()}
        for widget in fixed_scrollable_frame.winfo_children():
            widget.destroy()
        fixed_entries.clear()
        
        # Get fixed parameters
        fixed_params = [p for p, var in param_fix_checkboxes.items() if var.get() and (p not in hidden_params)]
        
        # Add header row if there are fixed parameters
        if len(fixed_params) > 0:
            tk.Label(fixed_scrollable_frame, text="Unfix", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=2, pady=2)
            tk.Label(fixed_scrollable_frame, text="Parameter", font=("Arial", 9, "bold")).grid(row=0, column=1, padx=2, pady=2)
            tk.Label(fixed_scrollable_frame, text="Value", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=2, pady=2)
        
        for i, param_name in enumerate(fixed_params):
            row = i + 1  # Start from row 1, after header
            # Unfix checkbox (unchecking moves parameter back to optimization)
            unfix_var = param_fix_checkboxes[param_name]
            unfix_checkbox = tk.Checkbutton(fixed_scrollable_frame, text="", variable=unfix_var,
                                           command=lambda pn=param_name: on_fix_toggle(pn))
            unfix_checkbox.grid(row=row, column=0, padx=2, pady=1, sticky="w")
            CreateToolTip(
                unfix_checkbox,
                f"{GRADIENT_DESCENT_TOOLTIPS['Unfix Parameter']}\n\nParameter: {param_name}"
            )
            
            # Parameter label (use short name)
            param_label = tk.Label(fixed_scrollable_frame, text=get_display_name(param_name) + ":")
            param_label.grid(row=row, column=1, sticky="e", padx=2, pady=1)
            entry = tk.Entry(fixed_scrollable_frame, width=10)
            # Use the value from the optimizable section's initial entry
            if param_name in fixed_values:
                entry.insert(0, fixed_values[param_name])
                if param_name in param_initial_entries:
                    param_initial_entries[param_name].delete(0, tk.END)
                    param_initial_entries[param_name].insert(0, fixed_values[param_name])
            elif param_name in param_initial_entries:
                entry.insert(0, param_initial_entries[param_name].get())
            else:
                entry.insert(0, str(default_params[param_name]))
            entry.grid(row=row, column=2, padx=2, pady=1)
            fixed_entries[param_name] = entry
            tt = _tooltip_for_param_name(param_name)
            CreateToolTip(param_label, tt)
            CreateToolTip(entry, tt)
    
    def on_fix_toggle(param_name):
        """Called when a parameter's fix checkbox is toggled."""
        refresh_optimizable_params()
        refresh_fixed_params()
    
    # Create widgets for all parameters
    # Default-fixed parameters (names must exist in model_spec.default_params).
    fixed_by_default = [
        "Number of Generations",
        "Initial Organism Count",
        "Average In_Flow",
        "Investment Modifier",
        "Initial A",
        "Initial B",
        "Initial Energy",
        "Intermediate Costs",
        "Average In_Flow (Acetate)",
    ]
    for i, param_name in enumerate(param_names):
        row = i + 1  # Start from row 1, after headers
        # Mark certain parameters as fixed by default
        should_fix = param_name in fixed_by_default
        var = tk.BooleanVar(value=should_fix)  # True = fixed, False = optimizable
        param_fix_checkboxes[param_name] = var
        
        # Fix checkbox (checking this moves parameter to fixed section) - on the left
        # Use a more visible checkbox with text
        fix_checkbox = tk.Checkbutton(param_scrollable_frame, text="", variable=var,
                                      command=lambda pn=param_name: on_fix_toggle(pn),
                                      font=("Arial", 8))
        fix_checkbox.grid(row=row, column=0, padx=2, pady=1, sticky="w")
        CreateToolTip(
            fix_checkbox,
            f"{GRADIENT_DESCENT_TOOLTIPS['Fix Parameter']}\n\nParameter: {param_name}"
        )
        
        # Parameter label (use short name for display)
        label = tk.Label(param_scrollable_frame, text=get_display_name(param_name))
        label.grid(row=row, column=1, sticky="w", padx=2, pady=1)
        
        # Hidden internal start/default value store (off-screen; powers save/load without an Initial column).
        initial_entry = tk.Entry(param_scrollable_frame, width=8)
        initial_entry.insert(0, str(default_params[param_name]))
        param_initial_entries[param_name] = initial_entry
        
        # Min/Max values - use smart defaults for max, but default min to 0
        default_val = default_params[param_name]
        min_default, max_default = get_default_bounds(param_name, default_val)
        
        # Min value - default to 0 (but can be left empty for no lower bound)
        # Special cases for minimum values
        min_entry = tk.Entry(param_scrollable_frame, width=8)
        if param_name == "Chemostat Volume":
            min_entry.insert(0, "10000")
        elif param_name == "Average In_Flow":
            min_entry.insert(0, "200.0")
        elif param_name == "Average In_Flow (Acetate)":
            min_entry.insert(0, "0.0")
        elif param_name == "Cost of Life":
            # Simulation requires Cost of Life > 0 to avoid runaway growth.
            # Some models may legitimately explore 0.
            if getattr(model_spec, "key", "") in {"simulation"}:
                min_entry.insert(0, "0.001")
            else:
                min_entry.insert(0, "0.0")
        elif param_name == "Intermediate Costs":
            min_entry.insert(0, "0.0")
        elif param_name == "Investment Modifier":
            # Default range: [0.2, 5.0]
            min_entry.insert(0, "0.2")
        elif param_name == "Acetate Ratio":
            # Acetate ratio range [0.1, 10]
            min_entry.insert(0, "0.1")
        elif param_name == "Duplication Sigmoid Intensity":
            # Intensity range is [0, 10] by default
            min_entry.insert(0, "0.0")
        elif param_name == "Constant Probability":
            min_entry.insert(0, "0.0")
        else:
            min_entry.insert(0, "0.0")
        min_entry.grid(row=row, column=2, padx=2, pady=1)
        param_min_entries[param_name] = min_entry
        
        # Max value - use smart defaults (override special cases)
        max_entry = tk.Entry(param_scrollable_frame, width=8)
        if param_name == "Cost of Life":
            # Cost of Life: cap to a reasonable upper bound
            max_entry.insert(0, "0.2")
        elif param_name == "Degradation Rate":
            # Degradation Rate is a fraction in [0,1], default max=1
            max_entry.insert(0, "1.0")
        elif param_name in ["Initial A", "Initial B", "Initial Facilitation"]:
            max_entry.insert(0, "1.0")
        elif param_name == "Mutation Rate":
            # Default maximum mutation rate
            max_entry.insert(0, "0.05")
        elif param_name == "Average In_Flow":
            # Average In_Flow max is 400
            max_entry.insert(0, "400.0")
        elif param_name == "Average In_Flow (Acetate)":
            max_entry.insert(0, "400.0")
        elif param_name == "Investment Modifier":
            # Investment modifier max
            max_entry.insert(0, "5.0")
        elif param_name == "Diffusion Constant":
            # Diffusion constant is a fraction in [0,1]
            max_entry.insert(0, "1.0")
        elif param_name == "Cost of Transport":
            # Transport cost is a fraction in [0,1]
            max_entry.insert(0, "1.0")
        elif param_name == "Intermediate Costs":
            # Intermediate storage penalty coefficient
            max_entry.insert(0, "1.0")
        elif param_name == "Acetate Ratio":
            # Default max acetate ratio
            max_entry.insert(0, "10.0")
        elif param_name == "Initial Organism Count":
            max_entry.insert(0, "1000")
        elif param_name == "Chemostat Volume":
            max_entry.insert(0, "20000")
        elif param_name == "Duplication Sigmoid Intensity":
            max_entry.insert(0, "10.0")
        elif param_name == "Constant Probability":
            max_entry.insert(0, "1.0")
        else:
            max_entry.insert(0, max_default)
        max_entry.grid(row=row, column=3, padx=2, pady=1)
        param_max_entries[param_name] = max_entry
        
        # Store all widgets for this parameter
        param_widgets[param_name] = [label, initial_entry, min_entry, max_entry, fix_checkbox]
        
        # Add tooltips to all optimizable parameter widgets
        tt = _tooltip_for_param_name(param_name)
        CreateToolTip(label, tt)
        CreateToolTip(initial_entry, tt)
        CreateToolTip(min_entry, tt)
        CreateToolTip(max_entry, tt)
    
    # Initial refresh
    refresh_optimizable_params()
    
    # === Fixed Parameters ===
    fixed_frame = tk.LabelFrame(right_col, text="Fixed Parameters", padx=5, pady=5)
    fixed_frame.pack(fill="both", expand=True, pady=(0, 10))
    
    fixed_entries = {}
    fixed_canvas = tk.Canvas(fixed_frame, height=150)
    fixed_scrollbar = ttk.Scrollbar(fixed_frame, orient="vertical", command=fixed_canvas.yview)
    fixed_scrollable_frame = tk.Frame(fixed_canvas)
    
    fixed_scrollable_frame.bind(
        "<Configure>",
        lambda e: fixed_canvas.configure(scrollregion=fixed_canvas.bbox("all"))
    )
    
    fixed_canvas.create_window((0, 0), window=fixed_scrollable_frame, anchor="nw")
    fixed_canvas.configure(yscrollcommand=fixed_scrollbar.set)
    
    fixed_canvas.pack(side="left", fill="both", expand=True)
    fixed_scrollbar.pack(side="right", fill="y")
    
    # Initial refresh of fixed parameters (should be empty initially)
    refresh_fixed_params()
    
    # === Other Settings ===
    other_frame = tk.LabelFrame(right_col, text="Simulation Settings", padx=5, pady=5)
    other_frame.pack(fill="x", pady=(0, 10))
    
    # Simulation diffusion switches
    m1_diffusion_var = tk.BooleanVar(value=False)
    m2_diffusion_var = tk.BooleanVar(value=True)
    diffusion_mutation_var = tk.BooleanVar(value=False)
    homogeneous_initial_diffusion_const_var = tk.BooleanVar(value=False)
    m1_facilitation_var = tk.BooleanVar(value=False)
    m1_porin_diffusion_var = tk.BooleanVar(value=False)
    m1_diffusion_check = tk.Checkbutton(other_frame, text="Enable M1 Diffusion (simple)", variable=m1_diffusion_var)
    m1_diffusion_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
    m2_diffusion_check = tk.Checkbutton(other_frame, text="Enable M2 Diffusion", variable=m2_diffusion_var)
    m2_diffusion_check.grid(row=1, column=0, columnspan=2, sticky="w")
    diffusion_mutation_check = tk.Checkbutton(other_frame, text="Enable Diffusion Mutation", variable=diffusion_mutation_var)
    diffusion_mutation_check.grid(row=2, column=0, columnspan=2, sticky="w")
    homogeneous_initial_diffusion_const_check = tk.Checkbutton(
        other_frame,
        text="Homogeneous Initial Diffusion Const.",
        variable=homogeneous_initial_diffusion_const_var,
    )
    homogeneous_initial_diffusion_const_check.grid(row=3, column=0, columnspan=2, sticky="w")
    m1_facilitation_check = tk.Checkbutton(other_frame, text="Enable M1 Facilitated Diffusion", variable=m1_facilitation_var)
    m1_facilitation_check.grid(row=4, column=0, columnspan=2, sticky="w")
    m1_porin_diffusion_check = tk.Checkbutton(
        other_frame,
        text="Enable M1 Porin Diffusion (import-only)",
        variable=m1_porin_diffusion_var,
    )
    m1_porin_diffusion_check.grid(row=5, column=0, columnspan=2, sticky="w")
    CreateToolTip(
        m1_diffusion_check,
        GRADIENT_DESCENT_TOOLTIPS["Enable M1 Diffusion (simple)"]
    )
    CreateToolTip(
        m2_diffusion_check,
        GRADIENT_DESCENT_TOOLTIPS["Enable M2 Diffusion"]
    )
    CreateToolTip(
        diffusion_mutation_check,
        GRADIENT_DESCENT_TOOLTIPS["Enable Diffusion Mutation"]
    )
    CreateToolTip(
        homogeneous_initial_diffusion_const_check,
        GRADIENT_DESCENT_TOOLTIPS["Homogeneous Initial Diffusion Const."]
    )
    CreateToolTip(
        m1_facilitation_check,
        GRADIENT_DESCENT_TOOLTIPS["Enable M1 Facilitated Diffusion"]
    )
    CreateToolTip(
        m1_porin_diffusion_check,
        GRADIENT_DESCENT_TOOLTIPS["Enable M1 Porin Diffusion"]
    )

    def refresh_diffusion_mutation_state():
        m1_effective_on = bool(m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get())
        if m2_diffusion_var.get() or m1_effective_on:
            diffusion_mutation_check.configure(state="normal")
        else:
            diffusion_mutation_var.set(False)
            diffusion_mutation_check.configure(state="disabled")
        if diffusion_mutation_var.get():
            homogeneous_initial_diffusion_const_check.configure(state="normal")
        else:
            homogeneous_initial_diffusion_const_var.set(False)
            homogeneous_initial_diffusion_const_check.configure(state="disabled")

    m1_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    m2_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    diffusion_mutation_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    refresh_diffusion_mutation_state()

    def refresh_diffusion_constant_visibility():
        if getattr(model_spec, "key", "") != "simulation":
            return
        show = m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get() or m2_diffusion_var.get()
        for param_name in ("Diffusion Constant", "Chemostat Volume"):
            if param_name not in param_names:
                continue
            if show:
                hidden_params.discard(param_name)
            else:
                hidden_params.add(param_name)
                if param_name in param_fix_checkboxes:
                    param_fix_checkboxes[param_name].set(True)
        refresh_optimizable_params()
        refresh_fixed_params()

    updating_m1_mode = False

    def refresh_m1_diffusion_mode():
        nonlocal updating_m1_mode
        if getattr(model_spec, "key", "") != "simulation":
            return
        if updating_m1_mode:
            return
        updating_m1_mode = True
        if m1_facilitation_var.get():
            m1_diffusion_var.set(False)
            m1_porin_diffusion_var.set(False)
            m1_diffusion_check.configure(state="disabled")
            m1_porin_diffusion_check.configure(state="disabled")
            m1_facilitation_check.configure(state="normal")
        elif m1_porin_diffusion_var.get():
            m1_diffusion_var.set(False)
            m1_facilitation_var.set(False)
            m1_diffusion_check.configure(state="disabled")
            m1_facilitation_check.configure(state="disabled")
            m1_porin_diffusion_check.configure(state="normal")
        elif m1_diffusion_var.get():
            m1_porin_diffusion_var.set(False)
            m1_facilitation_var.set(False)
            m1_porin_diffusion_check.configure(state="disabled")
            m1_facilitation_check.configure(state="disabled")
            m1_diffusion_check.configure(state="normal")
        else:
            m1_diffusion_check.configure(state="normal")
            m1_facilitation_check.configure(state="normal")
            m1_porin_diffusion_check.configure(state="normal")
        updating_m1_mode = False

    def refresh_m1_diffusion_ui():
        refresh_m1_diffusion_mode()
        refresh_diffusion_mutation_state()
        refresh_diffusion_constant_visibility()
        refresh_facilitation_param_visibility()

    m1_diffusion_var.trace_add("write", lambda *_: refresh_m1_diffusion_ui())
    m1_facilitation_var.trace_add("write", lambda *_: refresh_m1_diffusion_ui())
    m1_porin_diffusion_var.trace_add("write", lambda *_: refresh_m1_diffusion_ui())
    m2_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_constant_visibility())
    refresh_diffusion_constant_visibility()
    refresh_m1_diffusion_mode()
    
    if getattr(model_spec, "key", "") != "simulation":
        m1_diffusion_var.set(False)
        m2_diffusion_var.set(True)
        diffusion_mutation_var.set(False)
        homogeneous_initial_diffusion_const_var.set(False)
        m1_facilitation_var.set(False)
        m1_porin_diffusion_var.set(False)
        m1_diffusion_check.grid_remove()
        m2_diffusion_check.grid_remove()
        diffusion_mutation_check.grid_remove()
        homogeneous_initial_diffusion_const_check.grid_remove()
        m1_facilitation_check.grid_remove()
        m1_porin_diffusion_check.grid_remove()
    
    # Homogeneous population (disabled by default; default init is random)
    homogeneous_mode_var = tk.BooleanVar(value=False)
    homogeneous_checkbox = tk.Checkbutton(other_frame, 
                                          text="Use Homogeneous Initial Genotype", 
                                          variable=homogeneous_mode_var)
    # Place below diffusion controls.
    homogeneous_checkbox.grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
    CreateToolTip(homogeneous_checkbox, HOMOGENEOUS_TOOLTIP)

    independent_traits_var = tk.BooleanVar(value=False)
    independent_traits_checkbox = tk.Checkbutton(
        other_frame,
        text="Independent A/B Traits",
        variable=independent_traits_var,
    )
    independent_traits_checkbox.grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
    CreateToolTip(
        independent_traits_checkbox,
        GRADIENT_DESCENT_TOOLTIPS["Independent A/B Traits"]
    )

    def refresh_metric_options():
        """Update metric dropdown based on independent trait mode."""
        metrics = list(model_spec.metric_names or [])
        if getattr(model_spec, "key", "") == "simulation":
            try:
                m2_transport_on = bool(m2_diffusion_var.get())
            except Exception:
                m2_transport_on = True
            try:
                m1_transport_on = bool(
                    m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get()
                )
            except Exception:
                m1_transport_on = False
            try:
                intermediate_on = bool(enable_intermediate_costs_var.get())
            except NameError:
                intermediate_on = False
            metrics = filter_metric_options_for_simulation_settings(
                metrics,
                enable_m2_diffusion=m2_transport_on,
                enable_m1_diffusion=m1_transport_on,
                enable_intermediate_costs=intermediate_on,
            )
        if independent_traits_var.get():
            metrics = [m for m in metrics if m not in {"Trait Std Dev (Coupled)", "Trait Std Dev (Neutral Perc.)", "Trait Entropy (Neutral Perc.)"}]
        # Visualization metric should only offer metrics already present in cached data.
        try:
            param_vectors_ref = all_param_vectors
            results_ref = all_results
            metric_inputs_ref = all_metric_inputs
            metric_values_ref = all_metric_values
            lightweight_mode_ref = full_save_lightweight_mode
            lightweight_metric_name_ref = full_save_lightweight_metric_name
        except NameError:
            param_vectors_ref = []
            results_ref = []
            metric_inputs_ref = []
            metric_values_ref = []
            lightweight_mode_ref = False
            lightweight_metric_name_ref = None

        # Include metrics discovered from cached dataset payloads, even if they are
        # not currently exposed by the active model/UI filter set.
        cached_metric_names = set()
        for metric_input in metric_inputs_ref:
            rec = normalize_metric_input_shared(metric_input)
            if not isinstance(rec, dict):
                continue
            metric_map = rec.get("metric_values_by_name", None)
            if not isinstance(metric_map, dict):
                continue
            for m_name in metric_map.keys():
                m_clean = str(m_name or "").strip()
                if m_clean:
                    cached_metric_names.add(m_clean)
        if lightweight_mode_ref and lightweight_metric_name_ref:
            m_clean = str(lightweight_metric_name_ref or "").strip()
            if m_clean:
                cached_metric_names.add(m_clean)
        if cached_metric_names:
            metrics = list(dict.fromkeys(list(metrics) + sorted(cached_metric_names)))

        metric_combo["values"] = metrics
        try:
            metric_combo.configure(postcommand=refresh_metric_options)
        except Exception:
            pass

        available_viz = set()
        if not lightweight_mode_ref:
            for metric_input in metric_inputs_ref:
                rec = normalize_metric_input_shared(metric_input)
                if not isinstance(rec, dict):
                    continue
                metric_map = rec.get("metric_values_by_name", None)
                if not isinstance(metric_map, dict):
                    continue
                for m_name in metric_map.keys():
                    if m_name in metrics:
                        available_viz.add(m_name)
        # Lightweight full-save loads may only have all_metric_values for the optimization metric.
        metric_values_metric_name = metric_var.get()
        if lightweight_mode_ref and lightweight_metric_name_ref:
            metric_values_metric_name = lightweight_metric_name_ref
        if metric_values_metric_name in metrics:
            for mv in metric_values_ref:
                try:
                    if not np.isnan(float(mv)):
                        available_viz.add(metric_values_metric_name)
                        break
                except Exception:
                    continue
        # Include metrics known to be cached in full-save manifests.
        # This is cheap and allows lightweight full-save sessions to surface reanalyzed metrics.
        available_viz.update(_available_metrics_from_manifests(set(metrics)))
        # Keep dropdown empty until there is at least one visualizable metric.
        viz_metrics = [m for m in metrics if m in available_viz]
        viz_metric_combo["values"] = viz_metrics
        viz_metric2_combo["values"] = viz_metrics
        current = metric_var.get()
        if current not in metrics:
            if "Task2 Share Weighted Prob. Mean" in metrics:
                metric_var.set("Task2 Share Weighted Prob. Mean")
            elif metrics:
                metric_var.set(metrics[0])
            else:
                metric_var.set("")
        viz_current = viz_metric_var.get()
        if viz_current not in viz_metrics:
            if current in viz_metrics:
                viz_metric_var.set(current)
            elif viz_metrics:
                viz_metric_var.set(viz_metrics[0])
            else:
                viz_metric_var.set("")
        viz2_current = viz_metric2_var.get()
        if viz2_current not in viz_metrics:
            if viz_current in viz_metrics:
                viz_metric2_var.set(viz_current)
            elif current in viz_metrics:
                viz_metric2_var.set(current)
            elif viz_metrics:
                viz_metric2_var.set(viz_metrics[0])
            else:
                viz_metric2_var.set("")
        try:
            viz_metric_combo.configure(postcommand=refresh_metric_options)
        except Exception:
            pass

    def _selected_visualization_metric():
        m = viz_metric_var.get().strip()
        if m:
            return m
        return metric_var.get().strip()

    def _aligned_point_count(include_metric_inputs=True):
        """Return the shortest aligned length across per-point arrays."""
        lengths = [
            len(all_param_vectors),
            len(all_metric_values),
            len(all_replicate_info),
            len(all_run_ids),
            len(all_run_indices),
        ]
        if include_metric_inputs:
            lengths.append(len(all_metric_inputs))
        if not lengths:
            return 0
        return int(min(lengths))

    def _get_point_metric_value_for_viz(idx, metric_name):
        # Lightweight full-save mode:
        # - all_metric_values corresponds to full_save_lightweight_metric_name
        # - any other selected metric must come from per-metric cache.
        if full_save_lightweight_mode:
            if (
                full_save_lightweight_metric_name
                and metric_name == full_save_lightweight_metric_name
                and idx < len(all_metric_values)
            ):
                try:
                    return float(all_metric_values[idx])
                except Exception:
                    return np.nan
            if metric_name not in lightweight_metric_cache:
                _build_lightweight_metric_cache(metric_name)
            col = lightweight_metric_cache.get(metric_name, [])
            if idx < len(col):
                try:
                    return float(col[idx])
                except Exception:
                    return np.nan
            return np.nan
        metric_input = all_metric_inputs[idx] if idx < len(all_metric_inputs) else None
        if metric_input is None:
            metric_input = _lazy_load_metric_input_for_idx(idx)
        metric_input = normalize_metric_input_shared(metric_input)
        if isinstance(metric_input, dict):
            metric_map = metric_input.get("metric_values_by_name", None)
            if isinstance(metric_map, dict) and metric_name in metric_map:
                mv = metric_map.get(metric_name)
                if mv is None:
                    return np.nan
                try:
                    return float(mv)
                except Exception:
                    return np.nan
        # Non-lightweight fallback: use the scalar column only when no per-point
        # metric cache is available for this point/metric.
        if metric_name == metric_var.get() and idx < len(all_metric_values):
            try:
                return float(all_metric_values[idx])
            except Exception:
                return np.nan
        return np.nan

    def _metric_sidecar_cache_path(metric_name):
        """Path for per-metric lightweight cache column for the currently loaded full-save session."""
        folder = str(full_save_dir_var.get() or "").strip()
        session_id = str(loaded_full_save_session_id or "").strip()
        metric_name_clean = str(metric_name or "").strip()
        if not folder or not session_id or not metric_name_clean:
            return None
        return metric_column_sidecar_path(folder, session_id, metric_name_clean)

    def _load_metric_column_sidecar(metric_name, expected_len):
        path = _metric_sidecar_cache_path(metric_name)
        if not path or not os.path.exists(path):
            return None
        try:
            payload = _read_json_maybe_gz(path)
            return decode_metric_sidecar_values(payload, expected_len, missing_value=np.nan)
        except Exception:
            return None

    def _save_metric_column_sidecar(metric_name, values):
        path = _metric_sidecar_cache_path(metric_name)
        if not path or not isinstance(values, list):
            return
        try:
            payload = build_metric_sidecar_payload(
                session_id=str(loaded_full_save_session_id or ""),
                metric_name=str(metric_name),
                values=values,
                saved_at_epoch=int(time.time()),
            )
            _write_json_maybe_gz(path, payload)
        except Exception:
            pass

    def _build_lightweight_metric_cache(metric_name):
        """Build one float metric column from offload batches without keeping heavy payloads in RAM."""
        if metric_name in lightweight_metric_cache:
            return
        n = len(all_param_vectors)
        col = [np.nan] * n
        if n == 0:
            lightweight_metric_cache[metric_name] = col
            return
        sidecar_col = _load_metric_column_sidecar(metric_name, n)
        if sidecar_col is not None:
            lightweight_metric_cache[metric_name] = sidecar_col
            return
        if not offload_point_sources:
            lightweight_metric_cache[metric_name] = col
            return

        # Group point indices by batch path so we only read each gzip once.
        by_batch = {}
        for point_idx, src in enumerate(offload_point_sources):
            if point_idx >= n or not src:
                continue
            try:
                batch_path, rec_idx = src
            except Exception:
                continue
            if not batch_path:
                continue
            by_batch.setdefault(batch_path, []).append((point_idx, rec_idx))

        for batch_path, mappings in by_batch.items():
            try:
                payload = _read_json_maybe_gz(batch_path)
            except Exception:
                continue
            records = payload.get("records", [])
            payload_metric_name = str(payload.get("metric_name_current", "") or "")
            if not isinstance(records, list):
                continue
            for point_idx, rec_idx in mappings:
                if rec_idx >= len(records):
                    continue
                rec = records[rec_idx]
                mv = np.nan
                metric_input = normalize_metric_input_shared(rec.get("metric_input", None))
                if isinstance(metric_input, dict):
                    metric_map = metric_input.get("metric_values_by_name", None)
                    if isinstance(metric_map, dict) and metric_name in metric_map:
                        try:
                            v = metric_map.get(metric_name)
                            mv = np.nan if v is None else float(v)
                        except Exception:
                            mv = np.nan
                col[point_idx] = mv

        lightweight_metric_cache[metric_name] = col
        _save_metric_column_sidecar(metric_name, col)

    def _lazy_load_metric_input_for_idx(idx):
        """Load metric_input for one point on demand from offload files."""
        nonlocal offload_last_batch_path, offload_last_batch_payload
        if idx < 0:
            return None
        if idx < len(all_metric_inputs) and all_metric_inputs[idx] is not None:
            return all_metric_inputs[idx]
        if idx >= len(offload_point_sources):
            return None
        src = offload_point_sources[idx]
        if not src:
            return None
        try:
            batch_path, rec_idx = src
        except Exception:
            return None
        if not batch_path:
            return None
        try:
            if offload_last_batch_path != batch_path or not isinstance(offload_last_batch_payload, dict):
                offload_last_batch_payload = _read_json_maybe_gz(batch_path)
                offload_last_batch_path = batch_path
            records = offload_last_batch_payload.get("records", [])
            if not isinstance(records, list) or rec_idx >= len(records):
                return None
            metric_input = records[rec_idx].get("metric_input", None)
            if idx < len(all_metric_inputs):
                all_metric_inputs[idx] = metric_input
            return metric_input
        except Exception:
            return None

    def _available_metrics_from_manifests(metrics_scope):
        """Return metric names reported as cached in full-save manifests."""
        try:
            folder = full_save_dir_var.get().strip()
        except Exception:
            return set()
        if not folder or not os.path.isdir(folder):
            return set()
        try:
            manifest_files = [
                os.path.join(folder, fn)
                for fn in os.listdir(folder)
                if is_full_save_manifest_filename(fn)
            ]
            sidecar_files = [
                os.path.join(folder, fn)
                for fn in os.listdir(folder)
                if (
                    fn.startswith("full_save_metric_column_")
                    and fn.endswith(".json.gz")
                    and (not fn.startswith("full_save_metric_column_shard_"))
                )
            ]
        except Exception:
            return set()
        # Cache parsed manifest metric availability by (folder + file mtimes) to avoid UI stalls.
        cache = getattr(_available_metrics_from_manifests, "_cache", None)
        if not isinstance(cache, dict):
            cache = {}
        sig = []
        for mp in sorted(manifest_files):
            try:
                sig.append((mp, os.path.getmtime(mp)))
            except Exception:
                sig.append((mp, None))
        for sp in sorted(sidecar_files):
            try:
                sig.append((sp, os.path.getmtime(sp)))
            except Exception:
                sig.append((sp, None))
        sig = tuple(sig)

        cached_folder = cache.get("folder")
        cached_sig = cache.get("sig")
        cached_available = cache.get("available_all")
        if cached_folder == folder and cached_sig == sig and isinstance(cached_available, set):
            return set(m for m in cached_available if m in metrics_scope)

        available_all = set()
        for manifest_path in manifest_files:
            try:
                manifest = _read_json_maybe_gz(manifest_path)
            except Exception:
                continue
            batches = manifest.get("batches", []) if isinstance(manifest, dict) else []
            if not isinstance(batches, list):
                continue
            for b in batches:
                if not isinstance(b, dict):
                    continue
                cached_counts = b.get("cached_metric_counts", {})
                if not isinstance(cached_counts, dict):
                    continue
                for m_name, m_count in cached_counts.items():
                    try:
                        if float(m_count) > 0:
                            available_all.add(m_name)
                    except Exception:
                        continue

        # Sidecar columns are GUI-consumable metric caches even when offload payloads
        # are intentionally immutable (headless sidecar write mode).
        sidecar_digests = set()
        for sp in sidecar_files:
            try:
                digest = extract_metric_digest_from_sidecar_filename(sp)
                if digest:
                    sidecar_digests.add(digest)
            except Exception:
                continue
        if sidecar_digests:
            for m_name in metrics_scope:
                try:
                    md = metric_name_digest(str(m_name))
                except Exception:
                    continue
                if md in sidecar_digests:
                    available_all.add(m_name)

        _available_metrics_from_manifests._cache = {
            "folder": folder,
            "sig": sig,
            "available_all": available_all,
        }
        return set(m for m in available_all if m in metrics_scope)
    
    def toggle_homogeneous_params():
        """Show/hide model-specific initial-condition parameters when homogeneous mode is toggled."""
        homogeneous_enabled = homogeneous_mode_var.get()
        
        # Model-specific "initial condition" parameters to toggle
        initial_enzyme_params = list(getattr(model_spec, "homogeneous_param_names", []) or [])
        
        if homogeneous_enabled:
            # These parameters become relevant: show them and make them optimizable by default
            for param in initial_enzyme_params:
                hidden_params.discard(param)
            # Unfix these parameters so they become optimizable (appear in parameter list)
            for param in initial_enzyme_params:
                if param in param_fix_checkboxes:
                    param_fix_checkboxes[param].set(False)
        else:
            # These parameters are irrelevant: fully hide them (not in optimizable or fixed lists)
            for param in initial_enzyme_params:
                hidden_params.add(param)
            # Also mark them as fixed so they can never be selected for optimization
            for param in initial_enzyme_params:
                if param in param_fix_checkboxes:
                    param_fix_checkboxes[param].set(True)
        
        # Refresh the parameter displays
        refresh_optimizable_params()
        refresh_fixed_params()

        refresh_facilitation_param_visibility()
        refresh_independent_traits_visibility()

    def refresh_independent_traits_visibility():
        """Show/hide parameters based on independent A/B trait mode."""
        independent_enabled = independent_traits_var.get()
        if "Independent Traits" in param_fix_checkboxes:
            param_fix_checkboxes["Independent Traits"].set(True)

        # Investment Modifier is irrelevant when traits are independent
        if "Investment Modifier" in param_names:
            if independent_enabled:
                hidden_params.add("Investment Modifier")
                if "Investment Modifier" in param_fix_checkboxes:
                    param_fix_checkboxes["Investment Modifier"].set(True)
            else:
                hidden_params.discard("Investment Modifier")

        # Initial B only applies when independent traits + homogeneous mode are enabled
        if "Initial B" in param_names:
            if independent_enabled and homogeneous_mode_var.get():
                hidden_params.discard("Initial B")
                if "Initial B" in param_fix_checkboxes:
                    param_fix_checkboxes["Initial B"].set(False)
            else:
                hidden_params.add("Initial B")
                if "Initial B" in param_fix_checkboxes:
                    param_fix_checkboxes["Initial B"].set(True)

        refresh_optimizable_params()
        refresh_fixed_params()
        refresh_metric_options()

    def refresh_facilitation_param_visibility():
        if getattr(model_spec, "key", "") != "simulation":
            return
        show = m1_facilitation_var.get()
        # Cost of Transport is only relevant when facilitation is enabled.
        if "Cost of Transport" in param_names:
            if show:
                hidden_params.discard("Cost of Transport")
                if "Cost of Transport" in param_fix_checkboxes:
                    param_fix_checkboxes["Cost of Transport"].set(False)
            else:
                hidden_params.add("Cost of Transport")
                if "Cost of Transport" in param_fix_checkboxes:
                    param_fix_checkboxes["Cost of Transport"].set(True)
        # Initial Facilitation is only relevant when facilitation + homogeneous mode are enabled.
        if "Initial Facilitation" in param_names:
            if show and homogeneous_mode_var.get():
                hidden_params.discard("Initial Facilitation")
                if "Initial Facilitation" in param_fix_checkboxes:
                    param_fix_checkboxes["Initial Facilitation"].set(False)
            else:
                hidden_params.add("Initial Facilitation")
                if "Initial Facilitation" in param_fix_checkboxes:
                    param_fix_checkboxes["Initial Facilitation"].set(True)
        refresh_optimizable_params()
        refresh_fixed_params()
    
    homogeneous_mode_var.trace_add('write', lambda *args: toggle_homogeneous_params())
    independent_traits_var.trace_add("write", lambda *_: refresh_independent_traits_visibility())
    m2_diffusion_var.trace_add("write", lambda *_: refresh_metric_options())
    m1_facilitation_var.trace_add("write", lambda *_: refresh_facilitation_param_visibility())
    toggle_homogeneous_params()
    refresh_facilitation_param_visibility()
    refresh_independent_traits_visibility()
    
    # Silent mode (default to True for faster batch runs)
    silent_mode_var = tk.BooleanVar(value=True)
    silent_checkbox = tk.Checkbutton(other_frame, 
                                     text="Silent Mode (suppress simulation progress)", 
                                     variable=silent_mode_var)
    silent_checkbox.grid(row=8, column=0, columnspan=2, sticky="w", pady=2)
    CreateToolTip(silent_checkbox, SIMULATION_SETTINGS_TOOLTIPS["Silent Mode"])

    # Initial energy supply (per organism)
    enable_initial_energy_var = tk.BooleanVar(value=False)
    enable_initial_energy_check = tk.Checkbutton(
        other_frame,
        text="Enable Initial Energy",
        variable=enable_initial_energy_var,
    )
    enable_initial_energy_check.grid(row=9, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_initial_energy_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Initial Energy"])

    def _refresh_initial_energy_state(*, preserve_fix_checkbox: bool = False):
        # Show/hide "Initial Energy" in the optimizable/fixed parameter table
        if enable_initial_energy_var.get():
            hidden_params.discard("Initial Energy")
            # Default new sessions to fixed. After loading a dataset, preserve_fix_checkbox=True
            # so param_ui_state is not overwritten.
            if (not preserve_fix_checkbox) and ("Initial Energy" in param_fix_checkboxes):
                param_fix_checkboxes["Initial Energy"].set(True)
            # Ensure the table has a sensible default if blank
            if "Initial Energy" in param_initial_entries and not param_initial_entries["Initial Energy"].get().strip():
                param_initial_entries["Initial Energy"].insert(0, "0.0")
            if "Initial Energy" in fixed_entries and not fixed_entries["Initial Energy"].get().strip():
                fixed_entries["Initial Energy"].insert(0, "0.0")
        else:
            hidden_params.add("Initial Energy")

        refresh_optimizable_params()
        refresh_fixed_params()

    _initial_energy_write_trace_ids: list[str] = []

    def _disconnect_initial_energy_write_traces():
        while _initial_energy_write_trace_ids:
            tid = _initial_energy_write_trace_ids.pop()
            try:
                enable_initial_energy_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _connect_initial_energy_write_trace():
        _disconnect_initial_energy_write_traces()

        def _on_initial_energy_write(*_):
            _refresh_initial_energy_state(preserve_fix_checkbox=False)

        _initial_energy_write_trace_ids.append(
            enable_initial_energy_var.trace_add("write", _on_initial_energy_write)
        )

    _connect_initial_energy_write_trace()
    _refresh_initial_energy_state()

    # Chemostat flow (removes a fraction of total volume each generation)
    enable_chemostat_flow_var = tk.BooleanVar(value=False)
    enable_chemostat_flow_check = tk.Checkbutton(
        other_frame,
        text="Enable Chemostat Flow",
        variable=enable_chemostat_flow_var,
    )
    enable_chemostat_flow_check.grid(row=10, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_chemostat_flow_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Chemostat Flow"])

    def _refresh_chemostat_flow_state(*, preserve_fix_checkbox: bool = False):
        # Show/hide "Flow Percentage" in the optimizable/fixed parameter table
        if enable_chemostat_flow_var.get():
            hidden_params.discard("Flow Percentage")
            if (not preserve_fix_checkbox) and ("Flow Percentage" in param_fix_checkboxes):
                param_fix_checkboxes["Flow Percentage"].set(True)
            if "Flow Percentage" in param_initial_entries and not param_initial_entries["Flow Percentage"].get().strip():
                param_initial_entries["Flow Percentage"].insert(0, "0.0")
            if "Flow Percentage" in fixed_entries and not fixed_entries["Flow Percentage"].get().strip():
                fixed_entries["Flow Percentage"].insert(0, "0.0")
        else:
            hidden_params.add("Flow Percentage")
        refresh_optimizable_params()
        refresh_fixed_params()
        # Keep No Death / Constant Duplication relationship in sync when flow changes.
        try:
            _refresh_constant_duplication_state(preserve_fix_checkbox=preserve_fix_checkbox)
        except NameError:
            # Early init: callback is defined before death/dup refresh helpers exist.
            pass

    _chemostat_flow_write_trace_ids: list[str] = []

    def _disconnect_chemostat_flow_write_traces():
        while _chemostat_flow_write_trace_ids:
            tid = _chemostat_flow_write_trace_ids.pop()
            try:
                enable_chemostat_flow_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _connect_chemostat_flow_write_trace():
        _disconnect_chemostat_flow_write_traces()

        def _on_chemostat_flow_write(*_):
            _refresh_chemostat_flow_state(preserve_fix_checkbox=False)

        _chemostat_flow_write_trace_ids.append(
            enable_chemostat_flow_var.trace_add("write", _on_chemostat_flow_write)
        )

    _connect_chemostat_flow_write_trace()
    _refresh_chemostat_flow_state()

    # Intermediate storage energetic penalty:
    # Energy cost each generation is proportional to internal M2 storage.
    enable_intermediate_costs_var = tk.BooleanVar(value=False)
    enable_intermediate_costs_check = tk.Checkbutton(
        other_frame,
        text="Enable Intermediate Costs",
        variable=enable_intermediate_costs_var,
    )
    enable_intermediate_costs_check.grid(row=11, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_intermediate_costs_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Intermediate Costs"])

    def _refresh_intermediate_costs_state(*, preserve_fix_checkbox: bool = False):
        if m2_diffusion_var.get():
            enable_intermediate_costs_check.configure(state="normal")
        else:
            enable_intermediate_costs_var.set(False)
            enable_intermediate_costs_check.configure(state="disabled")
        if enable_intermediate_costs_var.get():
            hidden_params.discard("Intermediate Costs")
            if (not preserve_fix_checkbox) and ("Intermediate Costs" in param_fix_checkboxes):
                param_fix_checkboxes["Intermediate Costs"].set(True)
            if "Intermediate Costs" in param_initial_entries and not param_initial_entries["Intermediate Costs"].get().strip():
                param_initial_entries["Intermediate Costs"].insert(0, "0.0")
            if "Intermediate Costs" in fixed_entries and not fixed_entries["Intermediate Costs"].get().strip():
                fixed_entries["Intermediate Costs"].insert(0, "0.0")
        else:
            hidden_params.add("Intermediate Costs")
        refresh_optimizable_params()
        refresh_fixed_params()

    _intermediate_costs_write_trace_ids: list[str] = []

    def _disconnect_intermediate_costs_write_traces():
        while _intermediate_costs_write_trace_ids:
            tid = _intermediate_costs_write_trace_ids.pop()
            try:
                enable_intermediate_costs_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _connect_intermediate_costs_write_trace():
        _disconnect_intermediate_costs_write_traces()

        def _on_intermediate_costs_write(*_):
            _refresh_intermediate_costs_state(preserve_fix_checkbox=False)

        _intermediate_costs_write_trace_ids.append(
            enable_intermediate_costs_var.trace_add("write", _on_intermediate_costs_write)
        )

    _connect_intermediate_costs_write_trace()
    _refresh_intermediate_costs_state()
    m2_diffusion_var.trace_add("write", lambda *_: _refresh_intermediate_costs_state())

    m1_diffusion_var.trace_add("write", lambda *_: refresh_metric_options())
    m1_facilitation_var.trace_add("write", lambda *_: refresh_metric_options())
    m1_porin_diffusion_var.trace_add("write", lambda *_: refresh_metric_options())
    enable_intermediate_costs_var.trace_add("write", lambda *_: refresh_metric_options())

    # Optional acetate inflow added each generation (independent of glucose inflow).
    enable_acetate_addition_var = tk.BooleanVar(value=False)
    enable_acetate_addition_check = tk.Checkbutton(
        other_frame,
        text="Enable Acetate Addition",
        variable=enable_acetate_addition_var,
    )
    enable_acetate_addition_check.grid(row=12, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_acetate_addition_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Acetate Addition"])

    def _refresh_acetate_addition_state(*, preserve_fix_checkbox: bool = False):
        if enable_acetate_addition_var.get():
            hidden_params.discard("Average In_Flow (Acetate)")
            if (not preserve_fix_checkbox) and ("Average In_Flow (Acetate)" in param_fix_checkboxes):
                param_fix_checkboxes["Average In_Flow (Acetate)"].set(True)
            if "Average In_Flow (Acetate)" in param_initial_entries and not param_initial_entries["Average In_Flow (Acetate)"].get().strip():
                param_initial_entries["Average In_Flow (Acetate)"].insert(0, "0.0")
            if "Average In_Flow (Acetate)" in fixed_entries and not fixed_entries["Average In_Flow (Acetate)"].get().strip():
                fixed_entries["Average In_Flow (Acetate)"].insert(0, "0.0")
        else:
            hidden_params.add("Average In_Flow (Acetate)")
        refresh_optimizable_params()
        refresh_fixed_params()

    _acetate_addition_write_trace_ids: list[str] = []

    def _disconnect_acetate_addition_write_traces():
        while _acetate_addition_write_trace_ids:
            tid = _acetate_addition_write_trace_ids.pop()
            try:
                enable_acetate_addition_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _connect_acetate_addition_write_trace():
        _disconnect_acetate_addition_write_traces()

        def _on_acetate_addition_write(*_):
            _refresh_acetate_addition_state(preserve_fix_checkbox=False)

        _acetate_addition_write_trace_ids.append(
            enable_acetate_addition_var.trace_add("write", _on_acetate_addition_write)
        )

    _connect_acetate_addition_write_trace()
    _refresh_acetate_addition_state()

    binary_death_at_zero_energy_var = tk.BooleanVar(value=False)
    binary_death_at_zero_energy_check = tk.Checkbutton(
        other_frame,
        text="Binary Death at Zero Energy",
        variable=binary_death_at_zero_energy_var,
    )
    binary_death_at_zero_energy_check.grid(row=13, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(binary_death_at_zero_energy_check, SIMULATION_SETTINGS_TOOLTIPS["Binary Death at Zero Energy"])

    no_death_var = tk.BooleanVar(value=False)
    no_death_check = tk.Checkbutton(
        other_frame,
        text=NO_DEATH,
        variable=no_death_var,
    )
    no_death_check.grid(row=14, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(no_death_check, SIMULATION_SETTINGS_TOOLTIPS[NO_DEATH])

    constant_death_probability_var = tk.BooleanVar(value=False)
    constant_death_probability_check = tk.Checkbutton(
        other_frame,
        text="Constant Death Probability",
        variable=constant_death_probability_var,
    )
    constant_death_probability_check.grid(row=15, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(
        constant_death_probability_check,
        SIMULATION_SETTINGS_TOOLTIPS["Constant Death Probability"],
    )

    constant_duplication_probability_var = tk.BooleanVar(value=False)
    constant_duplication_probability_check = tk.Checkbutton(
        other_frame,
        text="Constant Duplication Probability",
        variable=constant_duplication_probability_var,
    )
    constant_duplication_probability_check.grid(row=16, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(
        constant_duplication_probability_check,
        SIMULATION_SETTINGS_TOOLTIPS["Constant Duplication Probability"],
    )

    def _refresh_death_dup_constant_params_state(*, preserve_fix_checkbox: bool = False):
        if no_death_var.get() and constant_death_probability_var.get():
            constant_death_probability_var.set(False)
        constant_death = bool(constant_death_probability_var.get())
        constant_dup = bool(constant_duplication_probability_var.get())
        no_death = bool(no_death_var.get())
        any_constant = (constant_death or constant_dup) and not (no_death and constant_dup)

        if no_death or binary_death_at_zero_energy_var.get() or constant_death:
            hidden_params.add("Death Decay Rate")
        else:
            hidden_params.discard("Death Decay Rate")

        if any_constant:
            hidden_params.discard("Constant Probability")
            if (not preserve_fix_checkbox) and ("Constant Probability" in param_fix_checkboxes):
                param_fix_checkboxes["Constant Probability"].set(True)
        else:
            hidden_params.add("Constant Probability")

        if constant_dup:
            hidden_params.add("Duplication Sigmoid Midpoint")
            hidden_params.add("Duplication Sigmoid Intensity")
        else:
            hidden_params.discard("Duplication Sigmoid Midpoint")
            hidden_params.discard("Duplication Sigmoid Intensity")

        if not preserve_fix_checkbox:
            refresh_optimizable_params()
            refresh_fixed_params()

    def _refresh_binary_death_state(*, preserve_fix_checkbox: bool = False):
        _refresh_death_dup_constant_params_state(preserve_fix_checkbox=preserve_fix_checkbox)

    def _refresh_constant_death_state(*, preserve_fix_checkbox: bool = False):
        if bool(no_death_var.get()):
            if bool(constant_death_probability_var.get()):
                constant_death_probability_var.set(False)
            constant_death_probability_check.configure(state="disabled")
        else:
            constant_death_probability_check.configure(state="normal")
        if bool(constant_death_probability_var.get()):
            if bool(no_death_var.get()):
                no_death_var.set(False)
            no_death_check.configure(state="disabled")
        else:
            no_death_check.configure(state="normal")
        _refresh_death_dup_constant_params_state(preserve_fix_checkbox=preserve_fix_checkbox)

    def _refresh_constant_duplication_state(*, preserve_fix_checkbox: bool = False):
        if (
            bool(no_death_var.get())
            and bool(constant_duplication_probability_var.get())
            and (not bool(enable_chemostat_flow_var.get()))
        ):
            constant_duplication_probability_var.set(False)
        if bool(no_death_var.get()) and (not bool(enable_chemostat_flow_var.get())):
            constant_duplication_probability_check.configure(state="disabled")
        else:
            constant_duplication_probability_check.configure(state="normal")
        _refresh_death_dup_constant_params_state(preserve_fix_checkbox=preserve_fix_checkbox)

    _binary_death_write_trace_ids: list[str] = []
    _constant_death_write_trace_ids: list[str] = []
    _no_death_write_trace_ids: list[str] = []

    def _disconnect_binary_death_write_traces():
        while _binary_death_write_trace_ids:
            tid = _binary_death_write_trace_ids.pop()
            try:
                binary_death_at_zero_energy_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _disconnect_constant_death_write_traces():
        while _constant_death_write_trace_ids:
            tid = _constant_death_write_trace_ids.pop()
            try:
                constant_death_probability_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _disconnect_no_death_write_traces():
        while _no_death_write_trace_ids:
            tid = _no_death_write_trace_ids.pop()
            try:
                no_death_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _connect_binary_death_write_trace():
        _disconnect_binary_death_write_traces()

        def _on_binary_death_write(*_):
            _refresh_constant_death_state(preserve_fix_checkbox=False)

        _binary_death_write_trace_ids.append(
            binary_death_at_zero_energy_var.trace_add("write", _on_binary_death_write)
        )

    def _connect_constant_death_write_trace():
        _disconnect_constant_death_write_traces()

        def _on_constant_death_write(*_):
            _refresh_constant_death_state(preserve_fix_checkbox=False)

        _constant_death_write_trace_ids.append(
            constant_death_probability_var.trace_add("write", _on_constant_death_write)
        )

    def _connect_no_death_write_trace():
        _disconnect_no_death_write_traces()

        def _on_no_death_write(*_):
            _refresh_constant_death_state(preserve_fix_checkbox=False)

        _no_death_write_trace_ids.append(
            no_death_var.trace_add("write", _on_no_death_write)
        )

    _connect_binary_death_write_trace()
    _connect_constant_death_write_trace()
    _connect_no_death_write_trace()

    _constant_duplication_write_trace_ids: list[str] = []

    def _disconnect_constant_duplication_write_traces():
        while _constant_duplication_write_trace_ids:
            tid = _constant_duplication_write_trace_ids.pop()
            try:
                constant_duplication_probability_var.trace_remove("write", tid)
            except tk.TclError:
                pass

    def _connect_constant_duplication_write_trace():
        _disconnect_constant_duplication_write_traces()

        def _on_constant_duplication_write(*_):
            _refresh_constant_duplication_state(preserve_fix_checkbox=False)

        _constant_duplication_write_trace_ids.append(
            constant_duplication_probability_var.trace_add("write", _on_constant_duplication_write)
        )

    _connect_constant_duplication_write_trace()
    _refresh_death_dup_constant_params_state()

    def _disconnect_conditional_param_feature_write_traces():
        """Suspend traces that map enable-* toggles onto param fix checkboxes (dataset load/snapshot)."""
        _disconnect_initial_energy_write_traces()
        _disconnect_chemostat_flow_write_traces()
        _disconnect_intermediate_costs_write_traces()
        _disconnect_acetate_addition_write_traces()
        _disconnect_binary_death_write_traces()
        _disconnect_constant_death_write_traces()
        _disconnect_no_death_write_traces()
        _disconnect_constant_duplication_write_traces()

    def _connect_conditional_param_feature_write_traces():
        _connect_initial_energy_write_trace()
        _connect_chemostat_flow_write_trace()
        _connect_intermediate_costs_write_trace()
        _connect_acetate_addition_write_trace()
        _connect_binary_death_write_trace()
        _connect_constant_death_write_trace()
        _connect_no_death_write_trace()
        _connect_constant_duplication_write_trace()

    # Store history is disabled by default (no GUI toggle).
    store_history_var = tk.BooleanVar(value=False)
    
    # === Control Buttons ===
    button_frame = tk.Frame(left_panel)
    button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    
    # Three rows for buttons (to avoid crowding)
    button_row1 = tk.Frame(button_frame)
    button_row1.pack(side="top", fill="x", pady=(0, 4))
    button_row2 = tk.Frame(button_frame)
    button_row2.pack(side="top", fill="x", pady=(0, 4))
    button_row3 = tk.Frame(button_frame)
    button_row3.pack(side="top", fill="x", pady=(0, 6))
    
    run_button = tk.Button(button_row1, text="Start", command=lambda: start_optimization())
    run_button.pack(side="left", padx=5)
    
    pause_button = tk.Button(button_row1, text="Pause", command=lambda: toggle_pause_optimization(), state="disabled")
    pause_button.pack(side="left", padx=5)

    stop_button = tk.Button(button_row1, text="Stop", command=lambda: stop_optimization(), state="disabled")
    stop_button.pack(side="left", padx=5)
    
    back_button = tk.Button(button_row1, text="← Back", command=lambda: go_back())
    back_button.pack(side="left", padx=5)

    # Parameter heatmap button (enabled once results exist)
    param_heatmaps_button = tk.Button(button_row2, text="Parameter Heatmaps", state="disabled")
    param_heatmaps_button.pack(side="left", padx=5)

    # Metric histogram button (enabled once results exist)
    metric_hist_button = tk.Button(button_row2, text="Metric Histogram", state="disabled")
    metric_hist_button.pack(side="left", padx=5)

    # Memory management button
    def build_optimization_summary_text(all_results_local, best_run_index_local, final_params_local, final_metric_local):
        """Build the end-of-optimization summary text (also saved/loaded with datasets)."""
        lines = []
        lines.append(f"\n{'='*50}")
        lines.append("Optimization Complete!\n")

        # Run stats (prefer tracked values; fallback to all_results if needed)
        total_runs = run_summary_stats["total_runs"]
        nan_runs = run_summary_stats["nan_runs"]
        valid_runs = run_summary_stats["metric_stats"].n
        mean_metric = run_summary_stats["metric_stats"].get_mean()
        std_metric = run_summary_stats["metric_stats"].get_std(ddof=1)

        if (total_runs == 0) and all_results_local:
            total_runs = len(all_results_local)
            nan_runs = 0
            tmp_stats = RunningStats()
            for r in all_results_local:
                mv = r.get("best_metric", r.get("final_metric", np.nan))
                if np.isnan(mv):
                    nan_runs += 1
                else:
                    tmp_stats.add(mv)
            valid_runs = tmp_stats.n
            mean_metric = tmp_stats.get_mean()
            std_metric = tmp_stats.get_std(ddof=1)

        lines.append(f"Summary of {total_runs} runs:")
        lines.append(f"- Valid runs (non-NaN best metric): {valid_runs} / {total_runs}")
        lines.append(f"- NaN runs: {nan_runs} / {total_runs}")
        if valid_runs > 0:
            lines.append(f"- Mean best metric (valid runs): {mean_metric:.6f}")
            lines.append(f"- Std best metric (valid runs, sample): {std_metric:.6f}")
        else:
            lines.append("- Mean/Std best metric: N/A (no valid runs)")

        # Optional: report what auto-cleanup removed (point-level, not run-level)
        if deleted_point_stats["total_points"] > 0:
            deleted_total = deleted_point_stats["total_points"]
            deleted_nan = deleted_point_stats["nan_points"]
            deleted_valid = deleted_point_stats["metric_stats"].n
            lines.append("")
            lines.append("Auto-cleanup removed stored points (for memory):")
            lines.append(f"- Deleted points: {deleted_total} ({deleted_nan} NaN, {deleted_valid} valid)")
            if deleted_valid > 0:
                deleted_mean = deleted_point_stats["metric_stats"].get_mean()
                deleted_std = deleted_point_stats["metric_stats"].get_std(ddof=1)
                lines.append(f"- Mean metric of deleted valid points: {deleted_mean:.6f}")
                lines.append(f"- Std metric of deleted valid points (sample): {deleted_std:.6f}")
            else:
                lines.append("- Mean/Std metric of deleted valid points: N/A (no valid deleted points)")

        # Top K runs
        if all_results_local:
            maximize = opt_goal_var.get() == "Maximize"
            TOP_K_RUNS_TO_PRINT = 10

            def sort_key(x):
                metric = x.get("best_metric", x.get("final_metric", np.nan))
                if np.isnan(metric):
                    return (1, 0)
                return (0, -metric) if maximize else (0, metric)

            sorted_results = sorted(all_results_local, key=sort_key)
            best_result_obj = all_results_local[best_run_index_local] if best_run_index_local < len(all_results_local) else None

            printed = 0
            for result in sorted_results:
                metric_val = result.get("best_metric", result.get("final_metric", np.nan))
                if np.isnan(metric_val):
                    continue
                if printed >= TOP_K_RUNS_TO_PRINT:
                    break
                marker = "★" if (best_result_obj is not None and result is best_result_obj) else " "
                lines.append(f"{marker} Run {result.get('run_index', 0) + 1}: Best Metric = {metric_val:.6f}")
                printed += 1

            # Best-run line
            if best_run_index_local < len(all_results_local):
                best_result_metric = all_results_local[best_run_index_local].get(
                    "best_metric", all_results_local[best_run_index_local].get("final_metric", np.nan)
                )
                lines.append("")
                if np.isnan(best_result_metric):
                    lines.append(f"Best Run: Run {best_run_index_local + 1} (WARNING: Metric is NaN)")
                else:
                    lines.append(f"Best Run: Run {best_run_index_local + 1} (Best Metric: {best_result_metric:.6f})")

        lines.append(f"\n{'='*50}")
        lines.append(f"Best Metric Achieved: {final_metric_local:.6f}")
        lines.append("Best Parameters:")
        for param_name, value in (final_params_local or {}).items():
            try:
                lines.append(f"  {param_name}: {float(value):.6f}")
            except Exception:
                lines.append(f"  {param_name}: {value}")

        lines.append("")  # trailing newline for nicer UX
        return "\n".join(lines)

    def show_parameter_heatmaps():
        """Show 1D binned heatmaps per parameter for where metric is non-zero/above-threshold."""
        if not all_results:
            messagebox.showwarning("No Data", "No optimization results available yet. Run an optimization or load a dataset first.")
            return

        # Only use parameters that were actually optimized for this dataset/run
        if not param_names_list:
            messagebox.showwarning("No Optimized Parameters", "No optimized parameters found for this dataset.")
            return

        # Popup controls
        top = tk.Toplevel(win)
        top.title("Parameter Heatmaps (where metric > threshold)")
        top.geometry("1200x800")

        controls = tk.Frame(top)
        controls.pack(side="top", fill="x", padx=10, pady=10)
        controls_desc_row = tk.Frame(controls)
        controls_desc_row.pack(side="top", fill="x")
        controls_row1 = tk.Frame(controls)
        controls_row1.pack(side="top", fill="x", pady=(6, 0))
        controls_row2 = tk.Frame(controls)
        controls_row2.pack(side="top", fill="x", pady=(6, 0))

        viz_metric_info_var = tk.StringVar(value=f"Visualizing metric: {_selected_visualization_metric()}")
        tk.Label(controls_row1, textvariable=viz_metric_info_var, fg="gray").pack(side="right")

        tk.Label(controls_row1, text="Bins:").pack(side="left")
        bins_var = tk.StringVar(value="60")
        bins_entry = tk.Entry(controls_row1, textvariable=bins_var, width=6)
        bins_entry.pack(side="left", padx=(5, 15))

        tk.Label(controls_row1, text="Metric threshold (uses |threshold|; includes |metric| >= threshold):").pack(side="left")
        thresh_var = tk.StringVar(value="0.0")
        thresh_entry = tk.Entry(controls_row1, textvariable=thresh_var, width=8)
        thresh_entry.pack(side="left", padx=(5, 15))

        log_scale_var = tk.BooleanVar(value=False)
        log_scale_check = tk.Checkbutton(controls_row2, text="Log scale axes", variable=log_scale_var)
        log_scale_check.pack(side="left", padx=(0, 15))
        CreateToolTip(log_scale_check, GRADIENT_DESCENT_TOOLTIPS["Heatmap Log Scale Axes"])

        show_mean_var = tk.BooleanVar(value=False)
        show_mean_check = tk.Checkbutton(controls_row2, text="Show mean metric per bin", variable=show_mean_var)
        show_mean_check.pack(side="left", padx=(0, 15))
        CreateToolTip(show_mean_check, GRADIENT_DESCENT_TOOLTIPS["Heatmap Show Mean Metric"])

        info_label = tk.Label(controls_row2, text="", fg="gray")
        info_label.pack(side="left", padx=(10, 0))

        # Canvas area for plot
        plot_container = tk.Frame(top)
        plot_container.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        # We rebuild the figure on each "Update"
        plot_state = {"canvas": None, "fig": None}

        def rebuild_plot():
            viz_metric_info_var.set(f"Visualizing metric: {_selected_visualization_metric()}")
            # Parse settings
            try:
                num_bins = int(bins_var.get().strip())
                if num_bins < 5:
                    raise ValueError()
            except Exception:
                messagebox.showerror("Invalid Bins", "Bins must be an integer >= 5.")
                return

            try:
                metric_threshold = float(thresh_var.get().strip())
            except Exception:
                messagebox.showerror("Invalid Threshold", "Metric threshold must be a number.")
                return

            # Extract ALL data points from all runs' histories (not affected by point cleanup).
            # Fallback: if a run has no history points (e.g., max_iterations == 0), use best_params/best_metric.
            # Interpret threshold by magnitude so negative inputs behave intuitively.
            eps = abs(float(metric_threshold))

            param_values_by_name = {p: [] for p in param_names_list}
            metric_values = []
            included_points = 0
            total_points = 0
            viz_metric_name = _selected_visualization_metric()

            def add_point(metric_value, params_dict):
                nonlocal included_points, total_points
                total_points += 1
                if metric_value is None or np.isnan(metric_value):
                    return
                if abs(float(metric_value)) < eps:
                    return
                # Params dict should include the optimized params (param_names_list)
                row_vals = []
                for p in param_names_list:
                    if p not in params_dict:
                        return
                    row_vals.append(float(params_dict[p]))
                for p, v in zip(param_names_list, row_vals):
                    param_values_by_name[p].append(v)
                metric_values.append(float(metric_value))
                included_points += 1

            for r in all_results:
                hist = r.get("history", []) or []
                if hist:
                    for h in hist:
                        mv = h.get("metric", np.nan)
                        pd = h.get("params", {}) or {}
                        add_point(mv, pd)
                else:
                    mv = r.get("best_metric", r.get("final_metric", np.nan))
                    pd = r.get("best_params", r.get("final_params", {})) or {}
                    add_point(mv, pd)

            # For loaded/saved point data, prefer cached selected-visualization metric.
            if param_names_list and all_param_vectors:
                param_values_by_name = {p: [] for p in param_names_list}
                metric_values = []
                included_points = 0
                total_points = 0
                for idx, vec in enumerate(all_param_vectors):
                    if len(vec) != len(param_names_list):
                        continue
                    try:
                        params_dict = {p: float(v) for p, v in zip(param_names_list, vec)}
                    except Exception:
                        continue
                    mv = _get_point_metric_value_for_viz(idx, viz_metric_name)
                    add_point(mv, params_dict)

            info_label.config(text=f"Using {included_points} / {total_points} points (metric |x| >= {eps:g}) — {viz_metric_name}")
            # If no runs met the threshold, still show empty heatmaps (all zeros)
            no_matches = (included_points == 0)

            # Build plots (one row per parameter)
            n_params = len(param_names_list)
            fig_h = max(4.0, 1.2 * n_params)
            fig, axes = plt.subplots(n_params, 1, figsize=(12, fig_h), constrained_layout=True)
            if n_params == 1:
                axes = [axes]

            # Common colormap
            cmap = DEFAULT_HEATMAP_CMAP
            use_log = bool(log_scale_var.get())
            show_mean = bool(show_mean_var.get())

            for ax_i, pname in zip(axes, param_names_list):
                vals = np.array(param_values_by_name[pname], dtype=float)
                metrics_arr = np.array(metric_values, dtype=float)

                # Bounds from GUI fields (preferred), fall back to observed values if present,
                # otherwise to a safe [0, 1] range.
                pmin = None
                pmax = None
                try:
                    pmin = float(param_min_entries[pname].get().strip())
                except Exception:
                    pmin = None
                try:
                    pmax = float(param_max_entries[pname].get().strip())
                except Exception:
                    pmax = None

                if (pmin is None) or (pmax is None) or (not np.isfinite(pmin)) or (not np.isfinite(pmax)) or (pmax <= pmin):
                    if vals.size > 0:
                        pmin = float(np.min(vals))
                        pmax = float(np.max(vals))
                    else:
                        pmin, pmax = 0.0, 1.0

                # For log scale, restrict to positive range only.
                if use_log:
                    pos_mask = vals > 0
                    vals = vals[pos_mask]
                    metrics_arr = metrics_arr[pos_mask]
                    if (not np.isfinite(pmin)) or (not np.isfinite(pmax)) or (pmax <= pmin) or (pmin <= 0.0) or (pmax <= 0.0):
                        if vals.size > 0:
                            pmin = float(np.min(vals))
                            pmax = float(np.max(vals))
                        else:
                            messagebox.showerror(
                                "Log Scale Error",
                                f"Parameter '{pname}' has no positive values available for log scaling.",
                            )
                            return

                # If still invalid, force defaults
                if (not np.isfinite(pmin)) or (not np.isfinite(pmax)) or (pmax <= pmin):
                    pmin, pmax = 0.0, 1.0

                if vals.size > 0:
                    vals = np.clip(vals, pmin, pmax)

                # Bin counts or mean metric per bin (log uses log10-space bins)
                if use_log:
                    plot_min = np.log10(pmin)
                    plot_max = np.log10(pmax)
                    vals_plot = np.log10(vals) if vals.size > 0 else vals
                    bins = np.linspace(plot_min, plot_max, num_bins + 1)
                else:
                    plot_min = pmin
                    plot_max = pmax
                    vals_plot = vals
                    bins = np.linspace(plot_min, plot_max, num_bins + 1)

                if show_mean:
                    # Compute mean metric per bin (NaN for empty bins)
                    bin_indices = np.digitize(vals_plot, bins, right=False) - 1
                    bin_sums = np.zeros(num_bins, dtype=float)
                    bin_counts = np.zeros(num_bins, dtype=int)
                    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
                    if valid_mask.any():
                        np.add.at(bin_sums, bin_indices[valid_mask], metrics_arr[valid_mask])
                        np.add.at(bin_counts, bin_indices[valid_mask], 1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        counts = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)
                else:
                    counts, _ = np.histogram(vals_plot, bins=bins)

                # Also show observed non-zero region
                obs_min = float(np.min(vals)) if vals.size > 0 else np.nan
                obs_max = float(np.max(vals)) if vals.size > 0 else np.nan

                im = ax_i.imshow(
                    counts[np.newaxis, :],
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    extent=(plot_min, plot_max, 0, 1),
                )

                # Label + simple ticks
                ax_i.set_yticks([])
                ax_i.set_ylabel(pname, rotation=0, ha="right", va="center")
                ax_i.set_xlim(plot_min, plot_max)
                mid_tick = (plot_min + plot_max) / 2.0
                ax_i.set_xticks([plot_min, mid_tick, plot_max])
                if use_log:
                    ax_i.set_xticklabels([f"{pmin:g}", f"{(10 ** mid_tick):g}", f"{pmax:g}"])
                else:
                    ax_i.set_xticklabels([f"{plot_min:g}", f"{mid_tick:g}", f"{plot_max:g}"])

                # Observed region markers
                if np.isfinite(obs_min) and np.isfinite(obs_max):
                    if use_log:
                        ax_i.axvline(np.log10(obs_min), color="white", lw=1, alpha=0.9)
                        ax_i.axvline(np.log10(obs_max), color="white", lw=1, alpha=0.9)
                    else:
                        ax_i.axvline(obs_min, color="white", lw=1, alpha=0.9)
                        ax_i.axvline(obs_max, color="white", lw=1, alpha=0.9)

            # Single shared colorbar
            cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
            if show_mean:
                cbar.set_label(f"Mean metric (|metric| >= {eps:g}) — {viz_metric_name}")
            else:
                cbar.set_label(f"Count of runs with |metric| >= {eps:g} — {viz_metric_name}")

            if no_matches:
                # Add a clear banner when everything is empty (but keep the heatmaps visible)
                fig.suptitle("No runs met the metric threshold — showing empty heatmaps", fontsize=12)

            # Replace existing canvas
            if plot_state["canvas"] is not None:
                plot_state["canvas"].get_tk_widget().destroy()

            canvas_local = FigureCanvasTkAgg(fig, master=plot_container)
            canvas_local.draw()
            canvas_local.get_tk_widget().pack(fill="both", expand=True)

            plot_state["canvas"] = canvas_local
            plot_state["fig"] = fig

        update_btn = tk.Button(controls_row2, text="Update", command=rebuild_plot)
        update_btn.pack(side="left", padx=(0, 10))

        close_btn = tk.Button(controls_row2, text="Close", command=top.destroy)
        close_btn.pack(side="left")

        # Build once on open
        rebuild_plot()

    # Wire button
    param_heatmaps_button.config(command=show_parameter_heatmaps)

    def _build_fixed_params_from_loaded_snapshot():
        """Build fixed params fallback from the currently loaded full-save settings snapshot."""
        if not isinstance(loaded_full_save_settings_snapshot, dict):
            return {}
        ui_state = loaded_full_save_settings_snapshot.get("ui_state", {})
        fixed_vals = loaded_full_save_settings_snapshot.get("fixed_param_values", {})
        if not isinstance(ui_state, dict):
            ui_state = {}
        if not isinstance(fixed_vals, dict):
            fixed_vals = {}

        snap_params = {}
        for k, v in fixed_vals.items():
            snap_params[k] = v
        snap_params["silent"] = bool(ui_state.get("silent_mode", True))

        # Booleans/toggles mirrored from optimization evaluation settings.
        snap_params.update(simulation_toggles_from_ui_state(ui_state))

        if not snap_params.get("Enable Initial Energy", False):
            snap_params["Initial Energy"] = 0.0

        if not snap_params.get("Enable Chemostat Flow", False):
            snap_params["Flow Percentage"] = 0.0

        if not snap_params.get("Enable Intermediate Costs", False):
            snap_params["Intermediate Costs"] = 0.0

        if not snap_params.get("Enable Acetate Addition", False):
            snap_params["Average In_Flow (Acetate)"] = 0.0

        snap_params["Enable M1 Diffusion"] = bool(ui_state.get("enable_m1_diffusion", False))
        snap_params["Enable M2 Diffusion"] = bool(ui_state.get("enable_m2_diffusion", False))
        snap_params["Enable M1 Facilitated Diffusion"] = bool(ui_state.get("enable_m1_facilitated_diffusion", False))
        snap_params["Enable M1 Porin Diffusion"] = bool(
            ui_state.get(
                "enable_m1_porin_diffusion",
                False,
            )
        )
        if snap_params["Enable M1 Facilitated Diffusion"]:
            snap_params["Enable M1 Diffusion"] = True
        if snap_params["Enable M1 Porin Diffusion"]:
            snap_params["Enable M1 Diffusion"] = True
            snap_params["Enable M1 Facilitated Diffusion"] = False
        allow_diffusion_mutation = bool(snap_params["Enable M2 Diffusion"] or snap_params["Enable M1 Diffusion"])
        snap_params["Enable Diffusion Mutation"] = bool(ui_state.get("enable_diffusion_mutation", False)) if allow_diffusion_mutation else False
        snap_params["Homogeneous Initial Diffusion Const."] = (
            bool(ui_state.get("homogeneous_initial_diffusion_const", False)) if snap_params["Enable Diffusion Mutation"] else False
        )

        if getattr(model_spec, "key", "") in {"simulation"}:
            snap_params["store_history"] = False
            snap_params = normalize_simulation_params(snap_params)
        return snap_params

    def _seed_from_loaded_snapshot():
        """Return loaded session seed text when available, else empty string."""
        if not isinstance(loaded_full_save_settings_snapshot, dict):
            return ""
        ui_state = loaded_full_save_settings_snapshot.get("ui_state", {})
        if not isinstance(ui_state, dict):
            return ""
        seed_val = ui_state.get("seed", "")
        return str(seed_val).strip()

    def _derived_descent_seed_from_run_index(run_index_for_point):
        """
        Return the base seed used for descents in the current policy.
        Seeding is held fixed across starts/descents; only replicate index offsets
        the seed when num_replicates > 1.
        """
        try:
            return parse_optional_seed(_seed_from_loaded_snapshot() or seed_entry.get().strip())
        except Exception:
            return None

    def _launch_individual_from_param_vector(
        param_vector,
        run_index_for_point,
        point_index=None,
        source_metric_name="",
        source_metric_value=np.nan,
    ):
        """Launch Individual GUI for a selected parameter vector."""
        full_params = {}

        # Try to get fixed parameters from the stored result for this run
        stored_fixed_params = None
        stored_run_result = None
        if run_index_for_point is not None:
            for result in all_results:
                if result.get("run_index") == run_index_for_point:
                    stored_run_result = result
                    stored_fixed_params = result.get("fixed_params")
                    break

        # Use stored fixed parameters only when they are non-empty.
        # In lightweight full-save loads this can be {}, which should fall back.
        if isinstance(stored_fixed_params, dict) and len(stored_fixed_params) > 0:
            full_params.update(stored_fixed_params)
        else:
            # In lightweight full-save mode, per-run fixed_params may be unavailable.
            # Fall back to saved session settings before using live GUI state.
            full_params.update(_build_fixed_params_from_loaded_snapshot())
            for param_name, entry in fixed_entries.items():
                if param_name in full_params:
                    continue
                try:
                    full_params[param_name] = float(entry.get())
                except ValueError:
                    full_params[param_name] = entry.get()

        # Set seed value (individual GUI expects "Random Seed (optional)").
        # Prefer point-specific seeds whenever available to avoid stale run-level seeds.
        descent_seed = None
        if isinstance(stored_run_result, dict):
            descent_seed = stored_run_result.get("descent_seed", None)
        metric_input_seed = None
        if point_index is not None:
            try:
                pi = int(point_index)
                mi = None
                if 0 <= pi < len(all_metric_inputs):
                    mi = all_metric_inputs[pi]
                # Offloaded points may not have metric_input in RAM; load lazily so
                # launch uses the exact point seed when available.
                if mi is None:
                    mi = _lazy_load_metric_input_for_idx(pi)
                if isinstance(mi, dict):
                    metric_input_seed = mi.get("random_seed_used", None)
            except Exception:
                metric_input_seed = None
        derived_seed = _derived_descent_seed_from_run_index(run_index_for_point) if run_index_for_point is not None else None
        chosen_seed_text = choose_launch_seed(
            descent_seed=descent_seed,
            metric_input_seed=metric_input_seed,
            derived_seed=derived_seed,
            fallback_seed_text=(_seed_from_loaded_snapshot() or seed_entry.get().strip()),
        )
        if chosen_seed_text != "":
            full_params["Random Seed (optional)"] = chosen_seed_text
        elif "Random Seed (optional)" not in full_params:
            full_params["Random Seed (optional)"] = ""
        if "silent" not in full_params:
            full_params["silent"] = bool(silent_mode_var.get())

        # Add model-specific optional settings, but never overwrite stored run settings.
        # If fixed_params came from a saved run, those values are authoritative.
        if getattr(model_spec, "key", "") == "simulation":
            if "Enable M1 Diffusion" not in full_params:
                full_params["Enable M1 Diffusion"] = bool(m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get())
            if "Enable M2 Diffusion" not in full_params:
                full_params["Enable M2 Diffusion"] = bool(m2_diffusion_var.get())
            if "Enable M1 Facilitated Diffusion" not in full_params:
                full_params["Enable M1 Facilitated Diffusion"] = bool(m1_facilitation_var.get())
            if "Enable M1 Porin Diffusion" not in full_params:
                full_params["Enable M1 Porin Diffusion"] = bool(m1_porin_diffusion_var.get())
            full_params["Enable M1 Porin Diffusion"] = bool(full_params.get("Enable M1 Porin Diffusion", False))
            if full_params["Enable M1 Facilitated Diffusion"]:
                full_params["Enable M1 Porin Diffusion"] = False
                full_params["Enable M1 Diffusion"] = True
            elif full_params["Enable M1 Porin Diffusion"]:
                full_params["Enable M1 Diffusion"] = True
            allow_diffusion_mutation = bool(full_params.get("Enable M2 Diffusion", False) or full_params.get("Enable M1 Diffusion", False))
            if "Enable Diffusion Mutation" not in full_params:
                full_params["Enable Diffusion Mutation"] = bool(diffusion_mutation_var.get()) if allow_diffusion_mutation else False
            full_params["Enable Diffusion Mutation"] = bool(full_params.get("Enable Diffusion Mutation", False)) if allow_diffusion_mutation else False
            if "Homogeneous Initial Diffusion Const." not in full_params:
                full_params["Homogeneous Initial Diffusion Const."] = (
                    bool(homogeneous_initial_diffusion_const_var.get()) if full_params["Enable Diffusion Mutation"] else False
                )
            full_params["Homogeneous Initial Diffusion Const."] = (
                bool(full_params.get("Homogeneous Initial Diffusion Const.", False))
                if full_params["Enable Diffusion Mutation"] else False
            )
            if "Enable Intermediate Costs" not in full_params:
                full_params["Enable Intermediate Costs"] = bool(enable_intermediate_costs_var.get())
            if not full_params["Enable Intermediate Costs"]:
                full_params["Intermediate Costs"] = 0.0
            elif "Intermediate Costs" not in full_params:
                full_params["Intermediate Costs"] = 0.0
            if "Enable Acetate Addition" not in full_params:
                full_params["Enable Acetate Addition"] = bool(enable_acetate_addition_var.get())
            if not full_params["Enable Acetate Addition"]:
                full_params["Average In_Flow (Acetate)"] = 0.0
            elif "Average In_Flow (Acetate)" not in full_params:
                full_params["Average In_Flow (Acetate)"] = 0.0
        if "Homogeneous Population" not in full_params:
            full_params["Homogeneous Population"] = bool(homogeneous_mode_var.get())
        if "Independent Traits" not in full_params:
            full_params["Independent Traits"] = bool(independent_traits_var.get())
        if full_params["Independent Traits"] and full_params["Homogeneous Population"]:
            if "Initial B" not in full_params:
                try:
                    full_params["Initial B"] = float(param_initial_entries["Initial B"].get())
                except Exception:
                    full_params["Initial B"] = float(default_params.get("Initial B", 0.5))

        # Add optimized parameters from selected point
        if param_names_list and len(param_vector) == len(param_names_list):
            for param_name, param_val in zip(param_names_list, param_vector):
                full_params[param_name] = float(param_val)
        else:
            for i, param_val in enumerate(param_vector):
                full_params[f"Param {i+1}"] = float(param_val)

        if getattr(model_spec, "key", "") in {"simulation"}:
            full_params["store_history"] = False
            full_params["minimal_tracking"] = True
            full_params["keep_optional_final_arrays"] = True
            full_params = normalize_simulation_params(full_params)

        # Pass source-point metric context so the Individual GUI can show comparable values.
        src_rep_info = None
        try:
            if point_index is not None:
                pi = int(point_index)
                if 0 <= pi < len(all_replicate_info):
                    src_rep_info = all_replicate_info[pi]
        except Exception:
            src_rep_info = None
        if src_rep_info is None and run_index_for_point is not None:
            try:
                for i, ridx in enumerate(all_run_indices):
                    if int(ridx) == int(run_index_for_point) and i < len(all_replicate_info):
                        src_rep_info = all_replicate_info[i]
                        break
            except Exception:
                src_rep_info = None
        if src_rep_info is not None:
            try:
                full_params["_source_replicate_info"] = list(src_rep_info)
            except Exception:
                pass
        metric_name_src = str(source_metric_name or "").strip()
        if metric_name_src:
            full_params["_source_metric_name"] = metric_name_src
            try:
                mv = float(source_metric_value)
                if np.isfinite(mv):
                    full_params["_source_metric_value"] = mv
            except Exception:
                pass

        individual_win = tk.Toplevel(root)
        individual_gui(individual_win, root, preset_params=full_params, model_spec=model_spec)

    def show_metric_histogram():
        """Show a histogram of final/best metric values across all runs."""
        if not all_results:
            return

        viz_metric_name = _selected_visualization_metric()
        # Keep histogram metric choices aligned with metrics that are actually available in data/cache.
        try:
            refresh_metric_options()
        except Exception:
            pass
        metric_options = list(viz_metric_combo["values"]) if "values" in viz_metric_combo.keys() else []
        metric_options = [str(m) for m in metric_options if str(m).strip()]
        selected_metric2_default = viz_metric2_var.get().strip()
        if not metric_options:
            messagebox.showwarning(
                "No Metrics Available",
                "No cached/available metrics were found for this dataset.\n"
                "Load or compute a visualization metric first."
            )
            return
        # Collect per-run metric and parameter dictionaries once.
        run_rows = []
        if param_names_list and all_param_vectors:
            n_points = _aligned_point_count(include_metric_inputs=True)
            for idx in range(n_points):
                vec = all_param_vectors[idx]
                if len(vec) != len(param_names_list):
                    continue
                try:
                    params = {p: float(v) for p, v in zip(param_names_list, vec)}
                except Exception:
                    continue
                mv = _get_point_metric_value_for_viz(idx, viz_metric_name)
                run_idx = all_run_indices[idx] if idx < len(all_run_indices) else None
                try:
                    run_idx = int(run_idx) if run_idx is not None else None
                except Exception:
                    run_idx = None
                run_rows.append(
                    {
                        "params": params,
                        "run_index": run_idx,
                        "point_idx": idx,
                        "metric_cache": {viz_metric_name: mv},
                    }
                )
        else:
            for r in all_results:
                mv = r.get("best_metric", r.get("final_metric", np.nan))
                try:
                    mv = float(mv)
                except Exception:
                    mv = float("nan")
                params = r.get("best_params", r.get("final_params", {}))
                if not isinstance(params, dict):
                    params = {}
                run_idx = r.get("run_index", None)
                try:
                    run_idx = int(run_idx) if run_idx is not None else None
                except Exception:
                    run_idx = None
                run_rows.append(
                    {
                        "params": params,
                        "run_index": run_idx,
                        "point_idx": None,
                        "metric_cache": {viz_metric_name: mv},
                    }
                )

        top = tk.Toplevel(win)
        top.title("Metric Histogram")
        top.geometry("980x640+0+0")

        controls = tk.Frame(top)
        controls.pack(side="top", fill="x", padx=10, pady=10)
        controls_desc_row = tk.Frame(controls)
        controls_desc_row.pack(side="top", fill="x")
        controls_row1 = tk.Frame(controls)
        controls_row1.pack(side="top", fill="x", pady=(6, 0))
        controls_row2 = tk.Frame(controls)
        controls_row2.pack(side="top", fill="x", pady=(6, 0))
        controls_row3 = tk.Frame(controls)
        controls_row3.pack(side="top", fill="x", pady=(6, 0))
        controls_row4 = tk.Frame(controls)
        controls_row4.pack(side="top", fill="x", pady=(6, 0))
        controls_row5 = tk.Frame(controls)
        controls_row5.pack(side="top", fill="x", pady=(6, 0))
        controls_row6 = tk.Frame(controls)
        controls_row6.pack(side="top", fill="x", pady=(6, 0))
        controls_row7 = tk.Frame(controls)
        controls_row7.pack(side="top", fill="x", pady=(6, 0))

        viz_metric_info_var = tk.StringVar(value=f"Visualizing metric: {viz_metric_name}")
        tk.Label(controls_desc_row, textvariable=viz_metric_info_var, fg="gray").pack(side="right")
        info = tk.Label(controls_desc_row, text="", fg="gray")
        info.pack(side="left")

        tk.Label(controls_row1, text="Bins:").pack(side="left")
        bins_var = tk.StringVar(value="50")
        bins_entry = tk.Entry(controls_row1, textvariable=bins_var, width=6)
        bins_entry.pack(side="left", padx=(5, 15))
        tk.Label(controls_row1, text="Metric 1:").pack(side="left")
        metric1_var = tk.StringVar(value=viz_metric_name if viz_metric_name else (metric_options[0] if metric_options else ""))
        metric1_menu = ttk.Combobox(
            controls_row1,
            textvariable=metric1_var,
            values=metric_options,
            state="readonly",
            width=26,
        )
        metric1_menu.pack(side="left", padx=(5, 8))
        metric2_options = ["(none)"] + metric_options
        default_metric2 = selected_metric2_default if selected_metric2_default in metric_options else "(none)"
        metric2_var = tk.StringVar(value=default_metric2)
        tk.Label(controls_row1, text="Metric 2:").pack(side="left")
        metric2_menu = ttk.Combobox(
            controls_row1,
            textvariable=metric2_var,
            values=metric2_options,
            state="readonly",
            width=26,
        )
        metric2_menu.pack(side="left", padx=(5, 10))

        param_candidates = []
        if param_names_list:
            param_candidates = [p for p in param_names_list if any(p in rr["params"] for rr in run_rows)]
        if not param_candidates and run_rows:
            # Fallback: discover numeric keys from first run's params.
            discovered = []
            for k in run_rows[0]["params"].keys():
                try:
                    float(run_rows[0]["params"][k])
                    discovered.append(k)
                except Exception:
                    pass
            param_candidates = discovered

        filter_target_options = (
            ["(none)"]
            + [f"Param: {p}" for p in param_candidates]
            + [f"Metric: {m}" for m in metric_options]
        )

        # First filter row (parameter or metric).
        tk.Label(controls_row2, text="Filter 1:").pack(side="left", padx=(0, 4))
        filter_param_var = tk.StringVar(value="(none)")
        param_menu = ttk.Combobox(
            controls_row2,
            textvariable=filter_param_var,
            values=filter_target_options,
            state="readonly",
            width=30,
        )
        param_menu.pack(side="left", padx=(2, 6))

        tk.Label(controls_row2, text="Min:").pack(side="left")
        low_val_var = tk.StringVar(value="")
        low_entry = tk.Entry(controls_row2, textvariable=low_val_var, width=8)
        low_entry.pack(side="left", padx=(4, 8))

        tk.Label(controls_row2, text="Max:").pack(side="left")
        high_val_var = tk.StringVar(value="")
        high_entry = tk.Entry(controls_row2, textvariable=high_val_var, width=8)
        high_entry.pack(side="left", padx=(4, 8))

        # Second independent filter row (ANDed with first filter).
        tk.Label(controls_row3, text="Filter 2:").pack(side="left", padx=(0, 4))
        filter_param2_var = tk.StringVar(value="(none)")
        param2_menu = ttk.Combobox(
            controls_row3,
            textvariable=filter_param2_var,
            values=filter_target_options,
            state="readonly",
            width=30,
        )
        param2_menu.pack(side="left", padx=(2, 6))
        tk.Label(controls_row3, text="Min:").pack(side="left")
        low2_val_var = tk.StringVar(value="")
        low2_entry = tk.Entry(controls_row3, textvariable=low2_val_var, width=8)
        low2_entry.pack(side="left", padx=(4, 8))
        tk.Label(controls_row3, text="Max:").pack(side="left")
        high2_val_var = tk.StringVar(value="")
        high2_entry = tk.Entry(controls_row3, textvariable=high2_val_var, width=8)
        high2_entry.pack(side="left", padx=(4, 8))

        # Two-axis relationship heatmap controls.
        metric_axis_1_label = "Metric 1"
        metric_axis_2_label = "Metric 2"
        relation_axis_candidates = [metric_axis_1_label, metric_axis_2_label] + list(param_candidates)
        default_x = metric_axis_1_label
        default_y = metric_axis_2_label
        tk.Label(controls_row4, text="X Axis:").pack(side="left", padx=(0, 4))
        relation_x_var = tk.StringVar(value=default_x)
        relation_x_menu = ttk.Combobox(
            controls_row4,
            textvariable=relation_x_var,
            values=relation_axis_candidates,
            state="readonly",
            width=18,
        )
        relation_x_menu.pack(side="left", padx=(0, 6))
        tk.Label(controls_row4, text="Y Axis:").pack(side="left", padx=(6, 4))
        relation_y_var = tk.StringVar(value=default_y)
        relation_y_menu = ttk.Combobox(
            controls_row4,
            textvariable=relation_y_var,
            values=relation_axis_candidates,
            state="readonly",
            width=18,
        )
        relation_y_menu.pack(side="left", padx=(0, 8))
        tk.Label(controls_row4, text="X Bins:").pack(side="left", padx=(8, 4))
        heatmap_bins_x_var = tk.StringVar(value="60")
        heatmap_bins_x_entry = tk.Entry(controls_row4, textvariable=heatmap_bins_x_var, width=6)
        heatmap_bins_x_entry.pack(side="left", padx=(0, 6))
        tk.Label(controls_row4, text="Y Bins:").pack(side="left", padx=(6, 4))
        heatmap_bins_y_var = tk.StringVar(value="60")
        heatmap_bins_y_entry = tk.Entry(controls_row4, textvariable=heatmap_bins_y_var, width=6)
        heatmap_bins_y_entry.pack(side="left", padx=(0, 8))
        heatmap_log_counts_var = tk.BooleanVar(value=False)
        heatmap_log_counts_check = tk.Checkbutton(
            controls_row4,
            text="Log Counts",
            variable=heatmap_log_counts_var,
            command=lambda: rebuild(),
        )
        heatmap_log_counts_check.pack(side="left", padx=(8, 0))

        tk.Label(controls_row5, text="Param Hist 1:").pack(side="left", padx=(0, 4))
        lower_hist_param1_var = tk.StringVar(value="(none)")
        lower_hist_param1_menu = ttk.Combobox(
            controls_row5,
            textvariable=lower_hist_param1_var,
            values=["(none)"] + param_candidates,
            state="readonly",
            width=18,
        )
        lower_hist_param1_menu.pack(side="left", padx=(0, 6))
        tk.Label(controls_row5, text="Param Hist 2:").pack(side="left", padx=(6, 4))
        lower_hist_param2_var = tk.StringVar(value="(none)")
        lower_hist_param2_menu = ttk.Combobox(
            controls_row5,
            textvariable=lower_hist_param2_var,
            values=["(none)"] + param_candidates,
            state="readonly",
            width=18,
        )
        lower_hist_param2_menu.pack(side="left", padx=(0, 6))

        tk.Label(controls_row5, text="Max Plot Points:").pack(side="left", padx=(10, 4))
        max_plot_points_var = tk.StringVar(value="20000")
        max_plot_points_entry = tk.Entry(controls_row5, textvariable=max_plot_points_var, width=8)
        max_plot_points_entry.pack(side="left", padx=(0, 6))

        # Optional metric-value filter (applies to histogram + heatmap source rows).
        tk.Label(controls_row6, text="Metric 1 Min:").pack(side="left", padx=(0, 4))
        metric_min_var = tk.StringVar(value="")
        metric_min_entry = tk.Entry(controls_row6, textvariable=metric_min_var, width=8)
        metric_min_entry.pack(side="left", padx=(0, 6))
        tk.Label(controls_row6, text="Max:").pack(side="left")
        metric_max_var = tk.StringVar(value="")
        metric_max_entry = tk.Entry(controls_row6, textvariable=metric_max_var, width=8)
        metric_max_entry.pack(side="left", padx=(4, 6))

        metric2_filter_frame = tk.Frame(controls_row6)
        tk.Label(metric2_filter_frame, text="Metric 2 Min:").pack(side="left", padx=(12, 4))
        metric2_min_var = tk.StringVar(value="")
        metric2_min_entry = tk.Entry(metric2_filter_frame, textvariable=metric2_min_var, width=8)
        metric2_min_entry.pack(side="left", padx=(0, 6))
        tk.Label(metric2_filter_frame, text="Max:").pack(side="left")
        metric2_max_var = tk.StringVar(value="")
        metric2_max_entry = tk.Entry(metric2_filter_frame, textvariable=metric2_max_var, width=8)
        metric2_max_entry.pack(side="left", padx=(4, 6))

        tk.Button(controls_row6, text="Apply Metric", command=lambda: _apply_metric_filter()).pack(side="left", padx=(8, 0))
        tk.Button(controls_row6, text="Reset Metric", command=lambda: _reset_metric_filter()).pack(side="left", padx=(4, 0))

        tk.Label(controls_row7, text="UMAP Points:").pack(side="left", padx=(0, 4))
        umap_plot_points_var = tk.StringVar(value="20000")
        umap_plot_points_entry = tk.Entry(controls_row7, textvariable=umap_plot_points_var, width=8)
        umap_plot_points_entry.pack(side="left", padx=(0, 6))

        param_ranges = {}
        for pname in param_candidates:
            vals_p = []
            for rr in run_rows:
                if pname in rr["params"]:
                    try:
                        vals_p.append(float(rr["params"][pname]))
                    except Exception:
                        pass
            if vals_p:
                pmin = float(np.min(vals_p))
                pmax = float(np.max(vals_p))
                param_ranges[pname] = (pmin, pmax)

        fig = plt.Figure(figsize=(10.2, 7.8), dpi=100)
        gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
        axh = fig.add_subplot(gs[0, 0])   # Metric histogram
        axu = fig.add_subplot(gs[0, 1])   # Two-parameter relationship scatter
        axp1 = fig.add_subplot(gs[1, 0])  # Parameter histogram 1
        axp2 = fig.add_subplot(gs[1, 1])  # Parameter histogram 2
        canvas_h = FigureCanvasTkAgg(fig, master=top)
        canvas_h.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

        relation_cbar_ref = {"obj": None}
        relation_cax_ref = {"ax": None}
        axu_initial_position = axu.get_position().frozen()
        try:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axu)
            relation_cax_ref["ax"] = divider.append_axes("right", size="5%", pad=0.05)
        except Exception:
            relation_cax_ref["ax"] = None

        def _format_bound(v):
            return f"{float(v):.6g}"

        def _metric_value_for_row(row, metric_name):
            cache = row.get("metric_cache")
            if not isinstance(cache, dict):
                cache = {}
                row["metric_cache"] = cache
            if metric_name in cache:
                try:
                    return float(cache[metric_name])
                except Exception:
                    return np.nan
            point_idx = row.get("point_idx", None)
            if point_idx is None:
                cache[metric_name] = np.nan
                return np.nan
            val = _get_point_metric_value_for_viz(int(point_idx), metric_name)
            try:
                val = float(val)
            except Exception:
                val = np.nan
            cache[metric_name] = val
            return val

        def _selected_metric1():
            m = metric1_var.get().strip()
            return m if m else viz_metric_name

        def _selected_metric2():
            m = metric2_var.get().strip()
            if not m or m == "(none)":
                return ""
            return m

        def _decode_filter_target(selection):
            sel = str(selection or "").strip()
            if not sel or sel == "(none)":
                return None, None
            if sel.startswith("Param: "):
                return "param", sel[len("Param: "):]
            if sel.startswith("Metric: "):
                return "metric", sel[len("Metric: "):]
            return None, None

        def _value_for_filter_target(row, target_kind, target_name):
            if target_kind == "param":
                try:
                    return float(row["params"].get(target_name, np.nan))
                except Exception:
                    return np.nan
            if target_kind == "metric":
                return _metric_value_for_row(row, target_name)
            return np.nan

        def _range_for_filter_target(rows, target_kind, target_name):
            if target_kind == "param":
                vals = []
                for rr in rows:
                    if target_name not in rr["params"]:
                        continue
                    try:
                        vals.append(float(rr["params"][target_name]))
                    except Exception:
                        continue
                if not vals:
                    return None
                arr = np.asarray(vals, dtype=float)
                return float(np.min(arr)), float(np.max(arr))
            if target_kind == "metric":
                return _metric_range_for_rows(rows, target_name)
            return None

        def _metric_range_for_rows(rows, metric_name):
            vals = []
            for rr in rows:
                mv = _metric_value_for_row(rr, metric_name)
                if np.isnan(mv):
                    continue
                vals.append(mv)
            if not vals:
                return None
            arr = np.asarray(vals, dtype=float)
            return float(np.min(arr)), float(np.max(arr))

        def _get_max_plot_points():
            try:
                n = int(float(max_plot_points_var.get() or 0))
            except Exception:
                n = 20000
            if n <= 0:
                n = 1
            return min(n, 2_000_000)

        def _get_heatmap_bins():
            try:
                bx = int(float(heatmap_bins_x_var.get() or 0))
            except Exception:
                bx = 60
            try:
                by = int(float(heatmap_bins_y_var.get() or 0))
            except Exception:
                by = 60
            bx = min(max(bx, 2), 500)
            by = min(max(by, 2), 500)
            return bx, by

        def _get_umap_plot_points():
            try:
                n = int(float(umap_plot_points_var.get() or 0))
            except Exception:
                n = 20000
            if n < 3:
                n = 3
            return min(n, 2_000_000)

        def _configure_slider_for_param():
            target_kind, target_name = _decode_filter_target(filter_param_var.get())
            if target_kind is None or not target_name:
                low_entry.config(state="disabled")
                high_entry.config(state="disabled")
                low_val_var.set("")
                high_val_var.set("")
                return

            data_range = _range_for_filter_target(run_rows, target_kind, target_name)
            if data_range is None:
                low_entry.config(state="disabled")
                high_entry.config(state="disabled")
                low_val_var.set("")
                high_val_var.set("")
                return
            pmin, pmax = data_range
            if np.isclose(pmin, pmax):
                pmax = pmin + 1e-9
            low_entry.config(state="normal")
            high_entry.config(state="normal")
            low_val_var.set(_format_bound(pmin))
            high_val_var.set(_format_bound(pmax))

        def _current_filter_bounds():
            target_kind, target_name = _decode_filter_target(filter_param_var.get())
            if target_kind is None or not target_name:
                return None
            try:
                lo = float(low_val_var.get())
                hi = float(high_val_var.get())
            except Exception:
                return None
            if lo > hi:
                lo, hi = hi, lo
            return target_kind, target_name, lo, hi

        def _current_filter2_bounds():
            target_kind, target_name = _decode_filter_target(filter_param2_var.get())
            if target_kind is None or not target_name:
                return None
            try:
                lo = float(low2_val_var.get())
                hi = float(high2_val_var.get())
            except Exception:
                return None
            if lo > hi:
                lo, hi = hi, lo
            return target_kind, target_name, lo, hi

        def _apply_entry_bounds(_event=None):
            bounds = _current_filter_bounds()
            if bounds is None:
                rebuild()
                return
            target_kind, target_name, lo, hi = bounds
            data_range = _range_for_filter_target(run_rows, target_kind, target_name)
            if data_range is None:
                rebuild()
                return
            pmin, pmax = data_range
            lo = max(pmin, min(pmax, lo))
            hi = max(pmin, min(pmax, hi))
            if lo > hi:
                lo, hi = hi, lo
            low_val_var.set(_format_bound(lo))
            high_val_var.set(_format_bound(hi))
            rebuild()

        def _apply_entry2_bounds(_event=None):
            bounds = _current_filter2_bounds()
            if bounds is None:
                rebuild()
                return
            target_kind, target_name, lo, hi = bounds
            data_range = _range_for_filter_target(run_rows, target_kind, target_name)
            if data_range is None:
                rebuild()
                return
            pmin, pmax = data_range
            lo = max(pmin, min(pmax, lo))
            hi = max(pmin, min(pmax, hi))
            if lo > hi:
                lo, hi = hi, lo
            low2_val_var.set(_format_bound(lo))
            high2_val_var.set(_format_bound(hi))
            rebuild()

        low_entry.bind("<Return>", _apply_entry_bounds)
        high_entry.bind("<Return>", _apply_entry_bounds)
        low2_entry.bind("<Return>", _apply_entry2_bounds)
        high2_entry.bind("<Return>", _apply_entry2_bounds)

        def _filtered_rows():
            metric1_name = _selected_metric1()
            metric2_name = _selected_metric2()
            bounds = _current_filter_bounds()
            filtered = run_rows
            if bounds is not None:
                target_kind, target_name, lo, hi = bounds
                subset = []
                for rr in run_rows:
                    pv = _value_for_filter_target(rr, target_kind, target_name)
                    if not np.isfinite(pv):
                        continue
                    if lo <= pv <= hi:
                        subset.append(rr)
                filtered = subset

            bounds2 = _current_filter2_bounds()
            if bounds2 is not None:
                target_kind2, target_name2, lo2, hi2 = bounds2
                subset2 = []
                for rr in filtered:
                    pv2 = _value_for_filter_target(rr, target_kind2, target_name2)
                    if not np.isfinite(pv2):
                        continue
                    if lo2 <= pv2 <= hi2:
                        subset2.append(rr)
                filtered = subset2

            # Optional metric-value range filter (ignores NaN rows when active).
            metric_min = None
            metric_max = None
            metric_min_txt = metric_min_var.get().strip()
            metric_max_txt = metric_max_var.get().strip()
            if metric_min_txt:
                try:
                    metric_min = float(metric_min_txt)
                except Exception:
                    metric_min = None
            if metric_max_txt:
                try:
                    metric_max = float(metric_max_txt)
                except Exception:
                    metric_max = None
            if metric_min is not None and metric_max is not None and metric_min > metric_max:
                metric_min, metric_max = metric_max, metric_min
                metric_min_var.set(_format_bound(metric_min))
                metric_max_var.set(_format_bound(metric_max))

            if metric_min is not None or metric_max is not None:
                metric_subset = []
                for rr in filtered:
                    mv = _metric_value_for_row(rr, metric1_name)
                    if np.isnan(mv):
                        continue
                    if metric_min is not None and mv < metric_min:
                        continue
                    if metric_max is not None and mv > metric_max:
                        continue
                    metric_subset.append(rr)
                filtered = metric_subset

            # Optional metric-2 range filter (active only when Metric 2 is selected).
            if metric2_name:
                metric2_min = None
                metric2_max = None
                metric2_min_txt = metric2_min_var.get().strip()
                metric2_max_txt = metric2_max_var.get().strip()
                if metric2_min_txt:
                    try:
                        metric2_min = float(metric2_min_txt)
                    except Exception:
                        metric2_min = None
                if metric2_max_txt:
                    try:
                        metric2_max = float(metric2_max_txt)
                    except Exception:
                        metric2_max = None
                if metric2_min is not None and metric2_max is not None and metric2_min > metric2_max:
                    metric2_min, metric2_max = metric2_max, metric2_min
                    metric2_min_var.set(_format_bound(metric2_min))
                    metric2_max_var.set(_format_bound(metric2_max))
                if metric2_min is not None or metric2_max is not None:
                    metric2_subset = []
                    for rr in filtered:
                        mv2 = _metric_value_for_row(rr, metric2_name)
                        if np.isnan(mv2):
                            continue
                        if metric2_min is not None and mv2 < metric2_min:
                            continue
                        if metric2_max is not None and mv2 > metric2_max:
                            continue
                        metric2_subset.append(rr)
                    filtered = metric2_subset
            return filtered

        def rebuild():
            metric1_name = _selected_metric1()
            metric2_name = _selected_metric2()
            viz_metric_info_var.set(f"Metrics: 1) {metric1_name}    2) {metric2_name if metric2_name else '(none)'}")
            # Prevent cumulative axis shrink from repeated colorbar redraws.
            axu.set_position(axu_initial_position)
            axh.clear()
            axu.clear()
            axp1.clear()
            axp2.clear()
            if relation_cbar_ref["obj"] is not None:
                try:
                    relation_cbar_ref["obj"].remove()
                except Exception:
                    pass
                relation_cbar_ref["obj"] = None
            if relation_cax_ref["ax"] is not None:
                try:
                    relation_cax_ref["ax"].clear()
                except Exception:
                    pass
            try:
                bins = int(float(bins_var.get() or 50))
                bins = max(1, min(500, bins))
            except Exception:
                bins = 50

            filtered_rows = _filtered_rows()

            vals = []
            nan_count = 0
            for rr in filtered_rows:
                mv = _metric_value_for_row(rr, metric1_name)
                if np.isnan(mv):
                    nan_count += 1
                else:
                    vals.append(mv)

            if len(vals) == 0:
                axh.text(0.5, 0.5, "No valid (non-NaN) metric values.", ha="center", va="center", transform=axh.transAxes)
                axh.set_xticks([])
                axh.set_yticks([])
                info.config(text=f"Valid: 0 / {len(filtered_rows)} filtered runs   NaN: {nan_count}")
            else:
                v = np.asarray(vals, dtype=float)
                axh.hist(v, bins=bins, color=OKABE_ITO["blue"], alpha=0.85, edgecolor="white")
                axh.set_title(f"Final Metric Values — {metric1_name}", fontsize=11)
                axh.set_xlabel("Metric value", fontsize=9)
                axh.set_ylabel("Count", fontsize=9)
                axh.grid(alpha=0.25)
                info.config(text=f"Valid: {len(v)} / {len(filtered_rows)} filtered runs   NaN: {nan_count}")

            def _draw_param_hist(ax_local, selected_name, title_prefix):
                if selected_name not in param_candidates:
                    ax_local.text(
                        0.5, 0.5,
                        f"Select {title_prefix.lower()} parameter.",
                        ha="center", va="center", transform=ax_local.transAxes,
                    )
                    ax_local.set_title(f"{title_prefix} (No selection)", fontsize=10)
                    ax_local.set_xticks([])
                    ax_local.set_yticks([])
                    return
                values = []
                for rr in filtered_rows:
                    if selected_name not in rr["params"]:
                        continue
                    try:
                        values.append(float(rr["params"][selected_name]))
                    except Exception:
                        continue
                if len(values) == 0:
                    ax_local.text(
                        0.5, 0.5,
                        f"No values for '{selected_name}' in current subset.",
                        ha="center", va="center", transform=ax_local.transAxes,
                    )
                    ax_local.set_title(f"{title_prefix} — {selected_name}", fontsize=10)
                    ax_local.set_xticks([])
                    ax_local.set_yticks([])
                    return
                pv = np.asarray(values, dtype=float)
                ax_local.hist(pv, bins=bins, color=OKABE_ITO["green"], alpha=0.85, edgecolor="white")
                ax_local.set_title(f"{title_prefix} — {selected_name}", fontsize=10)
                ax_local.set_xlabel(selected_name, fontsize=9)
                ax_local.set_ylabel("Count", fontsize=9)
                ax_local.grid(alpha=0.25)

            _draw_param_hist(axp1, lower_hist_param1_var.get(), "Parameter Histogram 1")
            _draw_param_hist(axp2, lower_hist_param2_var.get(), "Parameter Histogram 2")

            def _axis_display(axis_name):
                if axis_name == metric_axis_1_label:
                    return metric1_name
                if axis_name == metric_axis_2_label:
                    return metric2_name if metric2_name else metric1_name
                return axis_name

            def _axis_value_for_row(row, axis_name):
                if axis_name == metric_axis_1_label:
                    return _metric_value_for_row(row, metric1_name)
                if axis_name == metric_axis_2_label:
                    if metric2_name:
                        return _metric_value_for_row(row, metric2_name)
                    return _metric_value_for_row(row, metric1_name)
                return row["params"].get(axis_name, np.nan)

            x_name = relation_x_var.get()
            y_name = relation_y_var.get()
            if x_name not in relation_axis_candidates or y_name not in relation_axis_candidates:
                axu.text(0.5, 0.5, "Select X and Y axes.", ha="center", va="center", transform=axu.transAxes)
                axu.set_title("Parameter Relationship", fontsize=10)
                axu.set_xticks([])
                axu.set_yticks([])
                canvas_h.draw_idle()
                return

            xs = []
            ys = []
            for rr in filtered_rows:
                try:
                    xv = float(_axis_value_for_row(rr, x_name))
                    yv = float(_axis_value_for_row(rr, y_name))
                except Exception:
                    continue
                if not np.isfinite(xv) or not np.isfinite(yv):
                    continue
                xs.append(xv)
                ys.append(yv)

            total_relation_points = len(xs)
            sampled_points = total_relation_points
            max_plot_points = _get_max_plot_points()
            if total_relation_points > max_plot_points:
                sample_idx = np.random.choice(total_relation_points, size=max_plot_points, replace=False)
                xs = [xs[i] for i in sample_idx]
                ys = [ys[i] for i in sample_idx]
                sampled_points = max_plot_points

            if len(xs) == 0:
                axu.text(
                    0.5, 0.5,
                    f"No rows with numeric values for '{x_name}' and '{y_name}'.",
                    ha="center", va="center", transform=axu.transAxes,
                )
                axu.set_title("Parameter Relationship", fontsize=10)
                axu.set_xticks([])
                axu.set_yticks([])
                canvas_h.draw_idle()
                return

            x_arr = np.asarray(xs, dtype=float)
            y_arr = np.asarray(ys, dtype=float)
            if np.isclose(np.ptp(x_arr), 0.0) or np.isclose(np.ptp(y_arr), 0.0):
                axu.text(
                    0.5,
                    0.5,
                    "Heatmap requires variation in both X and Y.",
                    ha="center",
                    va="center",
                    transform=axu.transAxes,
                )
                axu.set_xticks([])
                axu.set_yticks([])
                canvas_h.draw_idle()
                return

            heat_bins_x, heat_bins_y = _get_heatmap_bins()
            try:
                h2d, x_edges, y_edges = np.histogram2d(x_arr, y_arr, bins=[heat_bins_x, heat_bins_y])
                if bool(heatmap_log_counts_var.get()):
                    h2d_plot = np.log10(h2d + 1.0)
                    cbar_label = "log10(Point Count + 1)"
                else:
                    h2d_plot = h2d
                    cbar_label = "Point Count"
                mesh = axu.pcolormesh(
                    x_edges,
                    y_edges,
                    h2d_plot.T,
                    cmap=DEFAULT_HEATMAP_CMAP,
                    shading="auto",
                )
                if relation_cax_ref["ax"] is not None:
                    cbar = fig.colorbar(mesh, cax=relation_cax_ref["ax"])
                else:
                    cbar = fig.colorbar(mesh, ax=axu, fraction=0.046, pad=0.03)
                cbar.set_label(cbar_label, rotation=270, labelpad=12)
                relation_cbar_ref["obj"] = cbar
            except Exception:
                relation_cbar_ref["obj"] = None

            if sampled_points < total_relation_points:
                title_suffix = f" (sampled {sampled_points}/{total_relation_points})"
            else:
                title_suffix = f" (n={sampled_points})"
            x_label = _axis_display(x_name)
            y_label = _axis_display(y_name)
            axu.set_title(f"Parameter Heatmap: {x_label} vs {y_label}{title_suffix}", fontsize=10)
            axu.set_xlabel(x_label, fontsize=9)
            axu.set_ylabel(y_label, fontsize=9)
            axu.grid(alpha=0.25)
            canvas_h.draw_idle()

        def _open_filtered_umap_plot():
            if not HAS_UMAP or umap is None:
                messagebox.showwarning("UMAP Unavailable", "UMAP is not available in this environment.")
                return
            if not param_candidates:
                messagebox.showwarning("No Parameters", "No numeric parameter columns are available for UMAP.")
                return

            filtered_rows = _filtered_rows()
            if not filtered_rows:
                messagebox.showwarning("No Data", "No filtered points to project.")
                return

            matrix = []
            rows_for_umap = []
            for rr in filtered_rows:
                vec = []
                ok = True
                for pname in param_candidates:
                    try:
                        vv = float(rr["params"].get(pname, np.nan))
                    except Exception:
                        vv = np.nan
                    if not np.isfinite(vv):
                        ok = False
                        break
                    vec.append(vv)
                if ok:
                    matrix.append(vec)
                    rows_for_umap.append(rr)

            if len(matrix) < 3:
                messagebox.showwarning("Not Enough Points", "Need at least 3 valid points to build UMAP.")
                return

            umap_plot_points = _get_umap_plot_points()
            if len(matrix) > umap_plot_points:
                sample_idx = np.random.choice(len(matrix), size=umap_plot_points, replace=False)
                sample_idx = np.asarray(sample_idx, dtype=int)
                matrix = [matrix[i] for i in sample_idx]
                rows_for_umap = [rows_for_umap[i] for i in sample_idx]

            info.config(text="Building clickable UMAP projection...")
            top.update_idletasks()
            try:
                mat = np.asarray(matrix, dtype=float)
                n_neighbors = min(30, max(2, mat.shape[0] - 1))
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
                emb = reducer.fit_transform(mat)
            except Exception as e:
                info.config(text="")
                messagebox.showerror("UMAP Error", f"Failed to compute UMAP projection:\n{e}")
                return
            info.config(text="")

            metric1_name = _selected_metric1()
            color_vals = np.asarray([_metric_value_for_row(rr, metric1_name) for rr in rows_for_umap], dtype=float)

            umap_top = tk.Toplevel(top)
            umap_top.title("Filtered Parameter UMAP (Clickable)")
            umap_top.geometry("900x650")

            umap_fig = plt.Figure(figsize=(8.8, 6.2), dpi=100)
            umap_ax = umap_fig.add_subplot(111)
            if np.any(np.isfinite(color_vals)):
                sc = umap_ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    c=color_vals,
                    cmap=DEFAULT_HEATMAP_CMAP,
                    s=20,
                    alpha=0.9,
                    linewidths=0.0,
                )
                cbar = umap_fig.colorbar(sc, ax=umap_ax, fraction=0.046, pad=0.03)
                cbar.set_label(metric1_name, rotation=270, labelpad=12)
            else:
                sc = umap_ax.scatter(
                    emb[:, 0],
                    emb[:, 1],
                    color=OKABE_ITO["blue"],
                    s=20,
                    alpha=0.9,
                    linewidths=0.0,
                )
            umap_ax.set_title(f"Filtered Parameter UMAP (n={emb.shape[0]})", fontsize=11)
            umap_ax.set_xlabel("UMAP-1")
            umap_ax.set_ylabel("UMAP-2")
            umap_ax.grid(alpha=0.2)

            umap_canvas = FigureCanvasTkAgg(umap_fig, master=umap_top)
            umap_canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=8, pady=8)

            click_info_var = tk.StringVar(value="Click a point to inspect its parameters and metric values.")
            tk.Label(umap_top, textvariable=click_info_var, anchor="w", justify="left", fg="gray").pack(
                side="bottom", fill="x", padx=10, pady=(0, 8)
            )

            selected_marker = {"artist": None}

            def _extract_param_vector_from_row(row):
                params = row.get("params", {}) or {}
                if param_names_list:
                    try:
                        full_vec = [float(params[pname]) for pname in param_names_list]
                        if all(np.isfinite(vv) for vv in full_vec):
                            return full_vec
                    except Exception:
                        return None
                    return None
                fallback_vec = []
                for pname in param_candidates:
                    try:
                        vv = float(params.get(pname, np.nan))
                    except Exception:
                        vv = np.nan
                    if np.isfinite(vv):
                        fallback_vec.append(vv)
                return fallback_vec if fallback_vec else None

            def _on_umap_click(event):
                if event.inaxes != umap_ax or event.xdata is None or event.ydata is None:
                    return
                dx = emb[:, 0] - float(event.xdata)
                dy = emb[:, 1] - float(event.ydata)
                idx = int(np.argmin(dx * dx + dy * dy))
                rr = rows_for_umap[idx]
                run_idx = rr.get("run_index")
                metric1_val = _metric_value_for_row(rr, _selected_metric1())
                metric2_val = _metric_value_for_row(rr, _selected_metric2())
                params = rr.get("params", {})
                preview_names = list(param_candidates[:5])
                preview = ", ".join(f"{pn}={float(params.get(pn, np.nan)):.4g}" for pn in preview_names if pn in params)
                click_info_var.set(
                    f"Point {idx + 1}/{len(rows_for_umap)} | run_index={run_idx} | "
                    f"{_selected_metric1()}={metric1_val:.6g} | {_selected_metric2()}={metric2_val:.6g} | {preview}"
                )
                if selected_marker["artist"] is not None:
                    try:
                        selected_marker["artist"].remove()
                    except Exception:
                        pass
                selected_marker["artist"] = umap_ax.scatter(
                    [emb[idx, 0]],
                    [emb[idx, 1]],
                    s=90,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )
                umap_canvas.draw_idle()
                metric1_name = _selected_metric1()
                metric2_name = _selected_metric2()
                params_lines = []
                for pname in param_candidates:
                    if pname not in params:
                        continue
                    try:
                        pv = float(params[pname])
                    except Exception:
                        continue
                    if not np.isfinite(pv):
                        continue
                    params_lines.append(f"- {pname}: {pv:.8g}")
                max_param_lines = 14
                if len(params_lines) > max_param_lines:
                    shown = params_lines[:max_param_lines]
                    shown.append(f"... ({len(params_lines) - max_param_lines} more parameters)")
                    params_block = "\n".join(shown)
                else:
                    params_block = "\n".join(params_lines) if params_lines else "- (no numeric parameters found)"
                prompt_lines = [
                    f"Point {idx + 1}/{len(rows_for_umap)}",
                    f"run_index: {run_idx}",
                    f"{metric1_name}: {metric1_val:.8g}",
                ]
                if metric2_name != metric1_name:
                    prompt_lines.append(f"{metric2_name}: {metric2_val:.8g}")
                prompt_lines.extend(["", "Parameters:", params_block, "", "Launch individual simulation with these parameters?"])
                if messagebox.askyesno(
                    "Launch Individual Simulation?",
                    "\n".join(prompt_lines),
                ):
                    param_vector = _extract_param_vector_from_row(rr)
                    if not param_vector:
                        messagebox.showerror(
                            "Launch Failed",
                            "Could not build a full valid parameter vector from the selected point.",
                        )
                        return
                    _launch_individual_from_param_vector(
                        param_vector,
                        run_idx,
                        point_index=rr.get("point_idx"),
                        source_metric_name=metric1_name,
                        source_metric_value=metric1_val,
                    )

            umap_canvas.mpl_connect("button_press_event", _on_umap_click)
            umap_canvas.draw_idle()

        def _open_filtered_parameter_heatmaps():
            """Open per-parameter heatmaps using the currently filtered histogram subset."""
            if not param_candidates:
                messagebox.showwarning("No Parameters", "No numeric parameter columns are available.")
                return

            filtered_rows = _filtered_rows()
            if not filtered_rows:
                messagebox.showwarning("No Data", "No filtered points available for parameter heatmaps.")
                return

            heat_top = tk.Toplevel(top)
            heat_top.title("Filtered Parameter Heatmaps")
            heat_top.geometry("1100x760")

            controls_h = tk.Frame(heat_top)
            controls_h.pack(side="top", fill="x", padx=10, pady=10)

            tk.Label(controls_h, text="Bins:").pack(side="left")
            bins_h_var = tk.StringVar(value="60")
            bins_h_entry = tk.Entry(controls_h, textvariable=bins_h_var, width=6)
            bins_h_entry.pack(side="left", padx=(5, 12))

            log_h_var = tk.BooleanVar(value=False)
            log_h_check = tk.Checkbutton(controls_h, text="Log scale axes", variable=log_h_var)
            log_h_check.pack(side="left", padx=(0, 12))

            mean_h_var = tk.BooleanVar(value=False)
            mean_h_check = tk.Checkbutton(controls_h, text="Show mean Metric 1 per bin", variable=mean_h_var)
            mean_h_check.pack(side="left", padx=(0, 12))

            info_h_var = tk.StringVar(value="")
            tk.Label(controls_h, textvariable=info_h_var, fg="gray").pack(side="left", padx=(8, 0))

            plot_container_h = tk.Frame(heat_top)
            plot_container_h.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))
            plot_state_h = {"canvas": None, "fig": None}

            def _rebuild_filtered_heatmaps():
                metric1_name = _selected_metric1()
                rows = _filtered_rows()
                if not rows:
                    messagebox.showwarning("No Data", "No filtered points available for parameter heatmaps.")
                    return
                try:
                    num_bins = int(float(bins_h_var.get() or 0))
                except Exception:
                    num_bins = 60
                num_bins = min(max(num_bins, 5), 500)
                use_log = bool(log_h_var.get())
                show_mean = bool(mean_h_var.get())

                n_params = len(param_candidates)
                fig_h = max(4.0, 1.2 * n_params)
                fig, axes = plt.subplots(n_params, 1, figsize=(11.5, fig_h), constrained_layout=True)
                if n_params == 1:
                    axes = [axes]

                im_ref = None
                non_empty_params = 0
                metric_points_used = 0

                for ax_i, pname in zip(axes, param_candidates):
                    vals = []
                    metric_pairs = []
                    for rr in rows:
                        try:
                            pv = float(rr["params"].get(pname, np.nan))
                        except Exception:
                            pv = np.nan
                        if not np.isfinite(pv):
                            continue
                        vals.append(pv)
                        mv = _metric_value_for_row(rr, metric1_name)
                        if np.isfinite(mv):
                            metric_pairs.append((pv, float(mv)))

                    if not vals:
                        ax_i.text(
                            0.5,
                            0.5,
                            f"No numeric values for '{pname}' in reduced set.",
                            ha="center",
                            va="center",
                            transform=ax_i.transAxes,
                        )
                        ax_i.set_yticks([])
                        ax_i.set_xticks([])
                        ax_i.set_ylabel(pname, rotation=0, ha="right", va="center")
                        continue

                    non_empty_params += 1
                    arr = np.asarray(vals, dtype=float)
                    pmin = float(np.min(arr))
                    pmax = float(np.max(arr))
                    if np.isclose(pmin, pmax):
                        pmax = pmin + 1e-9

                    metric_arr = np.asarray([mm for _, mm in metric_pairs], dtype=float)
                    metric_param_arr = np.asarray([pp for pp, _ in metric_pairs], dtype=float)

                    if use_log:
                        pos_mask = arr > 0
                        arr = arr[pos_mask]
                        if show_mean and metric_param_arr.size > 0:
                            mp_mask = metric_param_arr > 0
                            metric_param_arr = metric_param_arr[mp_mask]
                            metric_arr = metric_arr[mp_mask]
                        if arr.size == 0:
                            ax_i.text(
                                0.5,
                                0.5,
                                f"'{pname}' has no positive values for log scaling.",
                                ha="center",
                                va="center",
                                transform=ax_i.transAxes,
                            )
                            ax_i.set_yticks([])
                            ax_i.set_xticks([])
                            ax_i.set_ylabel(pname, rotation=0, ha="right", va="center")
                            continue
                        pmin = float(np.min(arr))
                        pmax = float(np.max(arr))
                        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
                            ax_i.text(0.5, 0.5, "Invalid log range.", ha="center", va="center", transform=ax_i.transAxes)
                            ax_i.set_yticks([])
                            ax_i.set_xticks([])
                            ax_i.set_ylabel(pname, rotation=0, ha="right", va="center")
                            continue
                        plot_min = np.log10(pmin)
                        plot_max = np.log10(pmax)
                        arr_plot = np.log10(arr)
                        metric_plot = np.log10(metric_param_arr) if metric_param_arr.size > 0 else metric_param_arr
                    else:
                        plot_min = pmin
                        plot_max = pmax
                        arr_plot = arr
                        metric_plot = metric_param_arr

                    bins = np.linspace(plot_min, plot_max, num_bins + 1)
                    if show_mean:
                        if metric_plot.size == 0:
                            counts = np.full(num_bins, np.nan, dtype=float)
                        else:
                            metric_points_used += int(metric_arr.size)
                            bin_indices = np.digitize(metric_plot, bins, right=False) - 1
                            bin_sums = np.zeros(num_bins, dtype=float)
                            bin_counts = np.zeros(num_bins, dtype=int)
                            valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
                            if valid_mask.any():
                                np.add.at(bin_sums, bin_indices[valid_mask], metric_arr[valid_mask])
                                np.add.at(bin_counts, bin_indices[valid_mask], 1)
                            with np.errstate(invalid="ignore", divide="ignore"):
                                counts = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)
                    else:
                        counts, _ = np.histogram(arr_plot, bins=bins)

                    im = ax_i.imshow(
                        counts[np.newaxis, :],
                        aspect="auto",
                        interpolation="nearest",
                        cmap=DEFAULT_HEATMAP_CMAP,
                        extent=(plot_min, plot_max, 0, 1),
                    )
                    im_ref = im
                    ax_i.set_yticks([])
                    ax_i.set_ylabel(pname, rotation=0, ha="right", va="center")
                    ax_i.set_xlim(plot_min, plot_max)
                    mid_tick = (plot_min + plot_max) / 2.0
                    ax_i.set_xticks([plot_min, mid_tick, plot_max])
                    if use_log:
                        ax_i.set_xticklabels([f"{pmin:g}", f"{(10 ** mid_tick):g}", f"{pmax:g}"])
                    else:
                        ax_i.set_xticklabels([f"{plot_min:g}", f"{mid_tick:g}", f"{plot_max:g}"])

                if im_ref is not None:
                    cbar = fig.colorbar(im_ref, ax=axes, shrink=0.7, pad=0.01)
                    if show_mean:
                        cbar.set_label(f"Mean {metric1_name} per bin")
                    else:
                        cbar.set_label("Point count per bin")

                if plot_state_h["canvas"] is not None:
                    plot_state_h["canvas"].get_tk_widget().destroy()

                canvas_local = FigureCanvasTkAgg(fig, master=plot_container_h)
                canvas_local.draw()
                canvas_local.get_tk_widget().pack(fill="both", expand=True)
                plot_state_h["canvas"] = canvas_local
                plot_state_h["fig"] = fig

                info_h_var.set(
                    f"Reduced points: {len(rows)} | Parameters with data: {non_empty_params}/{len(param_candidates)}"
                    + (f" | Mean metric points: {metric_points_used}" if show_mean else "")
                )

            tk.Button(controls_h, text="Update", command=_rebuild_filtered_heatmaps).pack(side="right", padx=(8, 0))
            tk.Button(controls_h, text="Close", command=heat_top.destroy).pack(side="right")
            bins_h_entry.bind("<Return>", lambda _e: _rebuild_filtered_heatmaps())
            _rebuild_filtered_heatmaps()

        def _open_seed_sweep_means_histogram():
            """Run per-point seed sweeps on filtered rows and histogram the per-point means."""
            run_model_fn = getattr(model_spec, "run_simulation", None)
            compute_metric_fn = getattr(model_spec, "compute_metric", None)
            if not callable(run_model_fn) or not callable(compute_metric_fn):
                messagebox.showerror("Unavailable", "This model does not support seed-sweep metric analysis.")
                return

            rows_snapshot = _filtered_rows()
            if not rows_snapshot:
                messagebox.showwarning("No Data", "No filtered points available for seed sweep.")
                return

            top_sw = tk.Toplevel(top)
            top_sw.title("Filtered Seed-Sweep Means Histogram")
            top_sw.geometry("980x700")

            controls_sw = tk.Frame(top_sw)
            controls_sw.pack(side="top", fill="x", padx=10, pady=10)
            tk.Label(controls_sw, text="Metric:").pack(side="left")
            metric_sw_var = tk.StringVar(value=_selected_metric1())
            metric_sw_menu = ttk.Combobox(
                controls_sw,
                textvariable=metric_sw_var,
                values=metric_options,
                state="readonly",
                width=32,
            )
            metric_sw_menu.pack(side="left", padx=(6, 10))
            tk.Label(controls_sw, text="Seeds per Point:").pack(side="left")
            seeds_per_point_var = tk.StringVar(value="20")
            seeds_per_point_entry = tk.Entry(controls_sw, textvariable=seeds_per_point_var, width=8)
            seeds_per_point_entry.pack(side="left", padx=(4, 10))

            status_sw_var = tk.StringVar(value=f"Filtered points: {len(rows_snapshot)}")
            tk.Label(controls_sw, textvariable=status_sw_var, fg="gray").pack(side="left", padx=(8, 0))

            controls_sw_row2 = tk.Frame(top_sw)
            controls_sw_row2.pack(side="top", fill="x", padx=10, pady=(0, 8))
            progress_sw_var = tk.DoubleVar(value=0.0)
            progress_sw_bar = ttk.Progressbar(
                controls_sw_row2,
                mode="determinate",
                maximum=1.0,
                variable=progress_sw_var,
            )
            progress_sw_bar.pack(side="left", fill="x", expand=True)
            progress_sw_text = tk.StringVar(value="Idle")
            tk.Label(controls_sw_row2, textvariable=progress_sw_text).pack(side="left", padx=(8, 0))

            fig_sw = plt.Figure(figsize=(9.2, 8.8), dpi=100)
            gs_sw = fig_sw.add_gridspec(3, 1, height_ratios=[1.0, 1.1, 1.1], hspace=0.38)
            ax_cur = fig_sw.add_subplot(gs_sw[0, 0])
            ax_sw = fig_sw.add_subplot(gs_sw[1, 0])
            ax_all = fig_sw.add_subplot(gs_sw[2, 0])
            canvas_sw = FigureCanvasTkAgg(fig_sw, master=top_sw)
            canvas_sw.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10, pady=(0, 8))
            summary_sw_var = tk.StringVar(value="Run a sweep to compute per-point mean metric values.")
            tk.Label(
                top_sw,
                textvariable=summary_sw_var,
                anchor="w",
                justify="left",
                wraplength=940,
                padx=10,
                pady=0,
            ).pack(side="top", fill="x")

            run_btn_ref = {"btn": None}
            save_btn_ref = {"btn": None}
            worker_queue = queue.Queue()
            persisted_sweep = {
                "metric_name": "",
                "seeds_per_point": 0,
                "point_metadata": [],
                "per_point_values": [],
                "per_point_values_path": "",
                "failed_runs": 0,
                "source_path": None,
                "light_mode": None,
                "sweep_work_dir": "",
            }

            def _load_per_point_values_jsonl(path):
                """Rebuild per_point_values list from streamed seed-sweep JSONL."""
                rows_out = []
                with open(path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        vals = rec.get("values", [])
                        if not isinstance(vals, list):
                            vals = []
                        row = []
                        for x in vals:
                            try:
                                row.append(float(x))
                            except Exception:
                                continue
                        rows_out.append(row)
                return rows_out

            # Build a run-index -> fixed-params lookup so each row can be rerun faithfully.
            run_fixed_lookup = {}
            for run_item in all_results:
                if not isinstance(run_item, dict):
                    continue
                run_idx_raw = run_item.get("run_index", None)
                try:
                    run_idx = int(run_idx_raw) if run_idx_raw is not None else None
                except Exception:
                    run_idx = None
                if run_idx is None or run_idx in run_fixed_lookup:
                    continue
                fixed_payload = run_item.get("fixed_params", {})
                if isinstance(fixed_payload, dict):
                    run_fixed_lookup[run_idx] = dict(fixed_payload)

            def _build_base_params_for_row(row, sweep_metric_name: str):
                run_idx_raw = row.get("run_index", None)
                try:
                    run_idx = int(run_idx_raw) if run_idx_raw is not None else None
                except Exception:
                    run_idx = None
                base_params = {}
                if run_idx is not None and run_idx in run_fixed_lookup and run_fixed_lookup[run_idx]:
                    base_params.update(run_fixed_lookup[run_idx])
                else:
                    base_params.update(_build_fixed_params_from_loaded_snapshot())
                    for p_name, entry in fixed_entries.items():
                        if p_name in base_params:
                            continue
                        try:
                            base_params[p_name] = float(entry.get())
                        except Exception:
                            base_params[p_name] = entry.get()
                    if "Random Seed (optional)" not in base_params:
                        base_params["Random Seed (optional)"] = seed_entry.get().strip()
                params_row = row.get("params", {})
                if isinstance(params_row, dict):
                    for p_name, p_val in params_row.items():
                        try:
                            base_params[p_name] = float(p_val)
                        except Exception:
                            base_params[p_name] = p_val
                if getattr(model_spec, "key", "") == "simulation":
                    metric_key = normalize_metric_name(sweep_metric_name)
                    if metric_supports_seed_sweep_light_mode(metric_key):
                        base_params["store_history"] = False
                        base_params["minimal_tracking"] = True
                        base_params["tracking_metric_name"] = metric_key
                        base_params.pop("keep_optional_final_arrays", None)
                    else:
                        base_params["store_history"] = True
                        base_params["minimal_tracking"] = False
                        base_params.pop("tracking_metric_name", None)
                    base_params = normalize_simulation_params(base_params)
                return base_params

            def _render_current_histogram(metric_name, rows_local=None, values_override=None, title_prefix="Current Metric Histogram"):
                metric_name = str(metric_name or "").strip()
                vals = []
                if values_override is not None:
                    for mv in values_override:
                        try:
                            fv = float(mv)
                        except Exception:
                            continue
                        if np.isfinite(fv):
                            vals.append(fv)
                else:
                    if rows_local is None:
                        rows_local = _filtered_rows()
                    for rr in rows_local:
                        mv = _metric_value_for_row(rr, metric_name)
                        if np.isfinite(mv):
                            vals.append(float(mv))
                ax_cur.clear()
                if not metric_name:
                    ax_cur.text(0.5, 0.5, "Select a metric.", ha="center", va="center", transform=ax_cur.transAxes)
                    ax_cur.set_xticks([])
                    ax_cur.set_yticks([])
                    ax_cur.set_title(title_prefix)
                elif not vals:
                    ax_cur.text(
                        0.5,
                        0.5,
                        "No valid current metric values in filtered subset.",
                        ha="center",
                        va="center",
                        transform=ax_cur.transAxes,
                    )
                    ax_cur.set_xticks([])
                    ax_cur.set_yticks([])
                    ax_cur.set_title(f"{title_prefix}: {metric_name}")
                else:
                    arr = np.asarray(vals, dtype=float)
                    bins = min(80, max(12, int(np.sqrt(arr.size) * 2)))
                    ax_cur.hist(arr, bins=bins, color=OKABE_ITO["green"], alpha=0.85, edgecolor="white")
                    ax_cur.set_title(f"{title_prefix} (filtered points): {metric_name}")
                    ax_cur.set_xlabel("Metric value")
                    ax_cur.set_ylabel("Count of points")
                    ax_cur.grid(alpha=0.25)
                canvas_sw.draw_idle()

            def _render_empty(message, title_text):
                ax_sw.clear()
                ax_sw.set_title(title_text)
                ax_sw.text(0.5, 0.5, message, ha="center", va="center", transform=ax_sw.transAxes)
                ax_sw.set_xticks([])
                ax_sw.set_yticks([])
                ax_all.clear()
                ax_all.set_title("All Re-simulation Metric Values")
                ax_all.text(0.5, 0.5, message, ha="center", va="center", transform=ax_all.transAxes)
                ax_all.set_xticks([])
                ax_all.set_yticks([])
                canvas_sw.draw_idle()

            def _build_point_metadata(rows_local, metric_name):
                out = []
                for rr in rows_local:
                    params_payload = rr.get("params", {})
                    if isinstance(params_payload, dict):
                        params_payload = dict(params_payload)
                    else:
                        params_payload = {}
                    mv = _metric_value_for_row(rr, metric_name)
                    out.append(
                        {
                            "point_idx": rr.get("point_idx", None),
                            "run_index": rr.get("run_index", None),
                            "run_id": rr.get("run_id", None),
                            "params": params_payload,
                            "current_metric_value": float(mv) if np.isfinite(mv) else None,
                        }
                    )
                return out

            def _render_sweep_results(metric_name_done, means, all_vals, failed_runs, n_rows, seeds_done, light_mode_used=None):
                arr = np.asarray(means, dtype=float)
                valid = arr[np.isfinite(arr)]
                all_arr = np.asarray(all_vals, dtype=float)
                all_valid = all_arr[np.isfinite(all_arr)]
                ax_sw.clear()
                if valid.size == 0:
                    _render_empty(
                        "No valid per-point means were produced.",
                        f"Per-point Mean Histogram: {metric_name_done}",
                    )
                else:
                    bins = min(80, max(15, int(np.sqrt(valid.size) * 2)))
                    ax_sw.hist(valid, bins=bins, color=OKABE_ITO["blue"], alpha=0.85, edgecolor="white")
                    ax_sw.set_title(f"Histogram of per-point mean metric values: {metric_name_done}")
                    ax_sw.set_xlabel("Per-point mean metric value")
                    ax_sw.set_ylabel("Count of points")
                    ax_sw.grid(alpha=0.25)
                ax_all.clear()
                if all_valid.size == 0:
                    ax_all.set_title(f"All Re-simulation Metric Values: {metric_name_done}")
                    ax_all.text(
                        0.5,
                        0.5,
                        "No valid re-simulation metric values were produced.",
                        ha="center",
                        va="center",
                        transform=ax_all.transAxes,
                    )
                    ax_all.set_xticks([])
                    ax_all.set_yticks([])
                else:
                    bins_all = min(100, max(20, int(np.sqrt(all_valid.size) * 2)))
                    ax_all.hist(all_valid, bins=bins_all, color=OKABE_ITO["orange"], alpha=0.85, edgecolor="white")
                    ax_all.set_title(f"All re-simulation metric values: {metric_name_done}")
                    ax_all.set_xlabel("Metric value")
                    ax_all.set_ylabel("Count of runs")
                    ax_all.grid(alpha=0.25)
                canvas_sw.draw_idle()
                mean_of_means = float(np.mean(valid)) if valid.size > 0 else np.nan
                std_of_means = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
                light_note = ""
                if light_mode_used is True:
                    light_note = " | Simulation tracking: light (final-generation only)"
                elif light_mode_used is False:
                    light_note = " | Simulation tracking: full histories"
                summary_sw_var.set(
                    f"Points swept: {n_rows} | Seeds per point: {seeds_done} | "
                    f"Valid point means: {valid.size} | Failed runs/NaN: {failed_runs} | "
                    f"Valid re-sim values: {all_valid.size} | "
                    f"Mean of point means: {mean_of_means:.8g} | Std of point means: {std_of_means:.8g}"
                    f"{light_note}"
                )

            def _save_sweep_results():
                metric_name = str(persisted_sweep.get("metric_name") or "").strip()
                per_point_values = persisted_sweep.get("per_point_values", [])
                point_metadata = persisted_sweep.get("point_metadata", [])
                pp_path = str(persisted_sweep.get("per_point_values_path") or "").strip()
                if (not isinstance(per_point_values, list) or len(per_point_values) == 0) and pp_path and os.path.isfile(pp_path):
                    try:
                        per_point_values = _load_per_point_values_jsonl(pp_path)
                    except Exception as exc:
                        messagebox.showerror("Load Error", f"Could not read streamed per-point values:\n{exc}")
                        return
                if not metric_name or not isinstance(per_point_values, list) or len(per_point_values) == 0:
                    messagebox.showwarning("No Sweep Results", "Run or load sweep results before saving.")
                    return
                default_name = suggested_seed_sweep_filename(metric_name)
                initial_dir = str(full_save_dir_var.get() or "").strip() or _default_dialog_dir()
                save_path = filedialog.asksaveasfilename(
                    title="Save Seed Sweep Results",
                    initialdir=initial_dir,
                    initialfile=default_name,
                    defaultextension=".json.gz",
                    parent=top_sw,
                    **_filedialog_filetypes_kwarg_macos_safe(
                        [
                            ("Seed sweep results", "*.json.gz"),
                            ("JSON", "*.json"),
                            ("All files", "*"),
                        ]
                    ),
                )
                if not save_path:
                    return
                if not (save_path.endswith(".json") or save_path.endswith(".json.gz")):
                    save_path = save_path + ".json.gz"
                settings_payload = {
                    "metric_name": metric_name,
                    "seeds_per_point": int(persisted_sweep.get("seeds_per_point", 0) or 0),
                    "point_count": int(len(point_metadata)),
                    "model_key": str(getattr(model_spec, "key", "") or ""),
                    "source_full_save_dir": str(full_save_dir_var.get() or ""),
                    "source_session_id": str(loaded_full_save_session_id or ""),
                    "light_mode": persisted_sweep.get("light_mode"),
                    "sweep_scratch_dir": str(persisted_sweep.get("sweep_work_dir") or ""),
                }
                payload = build_seed_sweep_results_payload(
                    settings=settings_payload,
                    points=point_metadata,
                    per_point_values=per_point_values,
                    saved_at_epoch=int(time.time()),
                )
                try:
                    save_seed_sweep_results(save_path, payload)
                except Exception as exc:
                    messagebox.showerror("Save Error", f"Failed to save seed sweep results:\n{exc}")
                    return
                persisted_sweep["source_path"] = save_path
                status_sw_var.set(f"Saved sweep results: {os.path.basename(save_path)}")

            def _apply_loaded_seed_sweep_state(loaded, *, status_basename, persist_source_path=None, scratch_work_dir=""):
                """Apply decoded seed-sweep payload (export or scratch) to UI state."""
                settings_loaded = loaded.get("settings", {})
                metric_loaded = str(settings_loaded.get("metric_name", "") or "").strip()
                if metric_loaded:
                    metric_sw_var.set(metric_loaded)
                per_point_values = loaded.get("per_point_values", [])
                per_point_means = loaded.get("per_point_means", [])
                all_vals = loaded.get("all_values", [])
                points_loaded = loaded.get("points", [])
                if not isinstance(points_loaded, list):
                    points_loaded = []
                seeds_loaded = 0
                try:
                    seeds_loaded = int(settings_loaded.get("seeds_per_point", 0) or 0)
                except Exception:
                    seeds_loaded = 0
                if seeds_loaded > 0:
                    seeds_per_point_var.set(str(seeds_loaded))
                n_rows = len(points_loaded) if points_loaded else len(per_point_values)
                expected_runs = int(n_rows * max(0, seeds_loaded))
                failed_runs = max(0, expected_runs - int(len(all_vals))) if expected_runs > 0 else 0
                persisted_sweep["metric_name"] = metric_loaded
                persisted_sweep["seeds_per_point"] = int(seeds_loaded)
                persisted_sweep["point_metadata"] = points_loaded
                persisted_sweep["per_point_values"] = per_point_values
                persisted_sweep["per_point_values_path"] = ""
                persisted_sweep["failed_runs"] = failed_runs
                persisted_sweep["source_path"] = (
                    persist_source_path if persist_source_path is not None else status_basename
                )
                lm = settings_loaded.get("light_mode", None)
                persisted_sweep["light_mode"] = lm
                persisted_sweep["sweep_work_dir"] = str(scratch_work_dir or "").strip()
                run_btn_ref["btn"].configure(state="normal")
                if save_btn_ref["btn"] is not None:
                    save_btn_ref["btn"].configure(state="normal")
                progress_total = float(expected_runs) if expected_runs > 0 else float(max(1, len(all_vals)))
                progress_sw_bar.configure(maximum=progress_total)
                progress_sw_var.set(progress_total)
                progress_sw_text.set(f"{int(progress_total)} / {int(progress_total)}")
                status_sw_var.set(f"Loaded sweep results: {status_basename}")
                current_vals = []
                for pp in points_loaded:
                    if isinstance(pp, dict):
                        current_vals.append(pp.get("current_metric_value"))
                _render_current_histogram(
                    metric_loaded,
                    values_override=current_vals,
                    title_prefix="Saved Current Metric Histogram",
                )
                _render_sweep_results(
                    metric_loaded or "Metric",
                    per_point_means,
                    all_vals,
                    failed_runs,
                    n_rows,
                    seeds_loaded,
                    light_mode_used=persisted_sweep.get("light_mode"),
                )

            def _load_sweep_results():
                initial_dir = str(full_save_dir_var.get() or "").strip() or _default_dialog_dir()
                if not os.path.isdir(initial_dir):
                    initial_dir = os.path.expanduser("~")
                load_path = filedialog.askopenfilename(
                    title="Load exported seed sweep (.json / .json.gz) or select manifest.json inside a seed_sweep_work_* folder",
                    initialdir=initial_dir,
                    parent=top_sw,
                    **_filedialog_filetypes_kwarg_macos_safe(
                        [
                            ("Seed sweep results", "*.json.gz"),
                            ("JSON", "*.json"),
                            ("Scratch manifest", "manifest.json"),
                            ("All files", "*"),
                        ]
                    ),
                )
                if not load_path:
                    return
                loaded = None
                scratch_dir = ""
                try:
                    base = os.path.basename(str(load_path))
                    if base == "manifest.json":
                        scratch_dir = os.path.dirname(os.path.abspath(load_path))
                        loaded = load_seed_sweep_from_scratch_dir(scratch_dir)
                    else:
                        loaded = load_seed_sweep_results(load_path)
                except Exception as exc:
                    messagebox.showerror("Load Error", f"Failed to load seed sweep results:\n{exc}")
                    return
                disp = os.path.basename(load_path) if base != "manifest.json" else os.path.basename(scratch_dir.rstrip(os.sep))
                persist = load_path if not scratch_dir else scratch_dir
                _apply_loaded_seed_sweep_state(
                    loaded,
                    status_basename=disp,
                    persist_source_path=persist,
                    scratch_work_dir=scratch_dir if scratch_dir else "",
                )

            def _load_sweep_scratch_folder():
                initial_dir = str(full_save_dir_var.get() or "").strip() or _default_dialog_dir()
                if not os.path.isdir(initial_dir):
                    initial_dir = os.path.expanduser("~")
                folder = filedialog.askdirectory(
                    title="Select seed_sweep_work_* folder (contains manifest.json)",
                    initialdir=initial_dir,
                    parent=top_sw,
                )
                if not folder:
                    return
                try:
                    loaded = load_seed_sweep_from_scratch_dir(folder)
                except Exception as exc:
                    messagebox.showerror("Load Error", f"Failed to load scratch folder:\n{exc}")
                    return
                folder_abs = os.path.abspath(folder)
                _apply_loaded_seed_sweep_state(
                    loaded,
                    status_basename=os.path.basename(folder_abs.rstrip(os.sep)),
                    persist_source_path=folder_abs,
                    scratch_work_dir=folder_abs,
                )

            def _run_sweep():
                metric_name = metric_sw_var.get().strip()
                if not metric_name:
                    messagebox.showwarning("Metric Required", "Please select a metric first.")
                    return
                rows = _filtered_rows()
                if not rows:
                    messagebox.showwarning("No Data", "No filtered points available for seed sweep.")
                    return
                initial_dir = str(full_save_dir_var.get() or "").strip() or _default_dialog_dir()
                sweep_parent = filedialog.askdirectory(
                    title="Select folder for seed sweep scratch files (streams results to disk during the run)",
                    initialdir=initial_dir,
                    parent=top_sw,
                )
                if not sweep_parent:
                    return
                try:
                    sweep_work_dir = os.path.join(
                        sweep_parent,
                        f"seed_sweep_work_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}",
                    )
                    os.makedirs(sweep_work_dir, exist_ok=True)
                except Exception as exc:
                    messagebox.showerror("Folder Error", f"Could not create scratch folder:\n{exc}")
                    return

                try:
                    seeds_per_point = int(float(seeds_per_point_var.get() or 0))
                except Exception:
                    seeds_per_point = 20
                seeds_per_point = max(1, min(seeds_per_point, 500))
                seeds_per_point_var.set(str(seeds_per_point))

                use_light = (
                    getattr(model_spec, "key", "") == "simulation"
                    and metric_supports_seed_sweep_light_mode(metric_name)
                )

                total_runs = max(1, len(rows) * seeds_per_point)
                progress_sw_bar.configure(maximum=float(total_runs))
                progress_sw_var.set(0.0)
                progress_sw_text.set(f"0 / {total_runs}")
                light_txt = "light tracking (final generation only)" if use_light else "full-history tracking"
                summary_sw_var.set(
                    f"Running {seeds_per_point} seeds × {len(rows)} points on '{metric_name}' "
                    f"({light_txt}). Scratch: {sweep_work_dir}"
                )
                status_sw_var.set(f"Filtered points: {len(rows)} | Scratch: {os.path.basename(sweep_work_dir)}")
                point_metadata = _build_point_metadata(rows, metric_name)
                all_values_path = os.path.join(sweep_work_dir, "all_resim_values.f64")
                per_point_path = os.path.join(sweep_work_dir, "per_point_seeds.jsonl")
                try:
                    manifest = {
                        "kind": "gradient_gui_seed_sweep_scratch",
                        "metric_name": metric_name,
                        "seeds_per_point": int(seeds_per_point),
                        "n_rows": int(len(rows)),
                        "light_mode": bool(use_light),
                        "model_key": str(getattr(model_spec, "key", "") or ""),
                        "created_epoch": int(time.time()),
                    }
                    with open(os.path.join(sweep_work_dir, "manifest.json"), "w", encoding="utf-8") as mf:
                        json.dump(manifest, mf, indent=2)
                except Exception:
                    pass

                _render_current_histogram(metric_name, rows)
                _render_empty("Running seed sweeps...", f"Per-point Mean Histogram: {metric_name}")
                run_btn_ref["btn"].configure(state="disabled")
                if save_btn_ref["btn"] is not None:
                    save_btn_ref["btn"].configure(state="disabled")

                def worker():
                    rng_local = np.random.default_rng()
                    per_point_means = []
                    failed_runs = 0
                    completed = 0
                    try:
                        with open(all_values_path, "wb") as vals_fp, open(per_point_path, "w", encoding="utf-8") as pp_fp:
                            for row_i, rr in enumerate(rows):
                                base_params = _build_base_params_for_row(rr, metric_name)
                                vals_local = []
                                for _ in range(seeds_per_point):
                                    local_seed = int(rng_local.integers(0, 2**31 - 1))
                                    local_params = dict(base_params)
                                    local_params["Random Seed (optional)"] = int(local_seed)
                                    local_params["silent"] = True
                                    try:
                                        sim_res = run_model_fn(local_params)
                                        mv = float(compute_metric_fn(sim_res, metric_name))
                                        del sim_res
                                        if np.isnan(mv):
                                            failed_runs += 1
                                        else:
                                            vals_local.append(mv)
                                            vals_fp.write(np.float64(mv).tobytes())
                                    except Exception:
                                        failed_runs += 1
                                    completed += 1
                                    if completed % 8 == 0 or completed == total_runs:
                                        worker_queue.put(("progress", completed, failed_runs, row_i + 1, len(rows)))
                                if vals_local:
                                    per_point_means.append(float(np.mean(np.asarray(vals_local, dtype=float))))
                                else:
                                    per_point_means.append(np.nan)
                                pp_fp.write(json.dumps({"row_index": row_i, "values": [float(x) for x in vals_local]}) + "\n")
                                pp_fp.flush()
                    except Exception as exc:
                        worker_queue.put(("error", str(exc)))
                        return

                    try:
                        all_vals_np = np.fromfile(all_values_path, dtype=np.float64)
                    except Exception:
                        all_vals_np = np.asarray([], dtype=float)

                    worker_queue.put(
                        (
                            "done",
                            per_point_means,
                            all_vals_np,
                            point_metadata,
                            failed_runs,
                            len(rows),
                            metric_name,
                            seeds_per_point,
                            use_light,
                            per_point_path,
                            sweep_work_dir,
                        )
                    )

                def poll_queue():
                    finished = False
                    while True:
                        try:
                            event = worker_queue.get_nowait()
                        except queue.Empty:
                            break
                        etype = event[0]
                        if etype == "progress":
                            completed, failed_runs, row_done, row_total = event[1], event[2], event[3], event[4]
                            progress_sw_var.set(float(completed))
                            progress_sw_text.set(f"{completed} / {total_runs}")
                            status_sw_var.set(
                                f"Rows complete: {row_done}/{row_total} | failed runs: {failed_runs}"
                            )
                        elif etype == "error":
                            msg = str(event[1]) if len(event) > 1 else "Unknown error"
                            messagebox.showerror("Seed Sweep Error", msg)
                            run_btn_ref["btn"].configure(state="normal")
                            if save_btn_ref["btn"] is not None:
                                save_btn_ref["btn"].configure(state="disabled")
                            progress_sw_text.set("Error")
                            finished = True
                        elif etype == "done":
                            (
                                means,
                                all_vals_np,
                                point_meta,
                                failed_runs,
                                n_rows,
                                metric_name_done,
                                seeds_done,
                                light_used,
                                pp_path,
                                sweep_dir_ws,
                            ) = event[1:11]
                            _render_sweep_results(
                                metric_name_done,
                                means,
                                all_vals_np,
                                failed_runs,
                                n_rows,
                                seeds_done,
                                light_mode_used=light_used,
                            )
                            persisted_sweep["metric_name"] = metric_name_done
                            persisted_sweep["seeds_per_point"] = int(seeds_done)
                            persisted_sweep["point_metadata"] = list(point_meta) if isinstance(point_meta, list) else []
                            persisted_sweep["per_point_values"] = []
                            persisted_sweep["per_point_values_path"] = str(pp_path or "")
                            persisted_sweep["sweep_work_dir"] = str(sweep_dir_ws or "")
                            persisted_sweep["light_mode"] = light_used
                            persisted_sweep["failed_runs"] = int(failed_runs)
                            persisted_sweep["source_path"] = None
                            progress_sw_var.set(float(total_runs))
                            progress_sw_text.set(f"{total_runs} / {total_runs}")
                            status_sw_var.set("Complete.")
                            run_btn_ref["btn"].configure(state="normal")
                            if save_btn_ref["btn"] is not None:
                                save_btn_ref["btn"].configure(state="normal")
                            finished = True
                    if not finished:
                        top_sw.after(120, poll_queue)

                threading.Thread(target=worker, daemon=True).start()
                top_sw.after(120, poll_queue)

            run_btn = tk.Button(controls_sw, text="Run Sweep", command=_run_sweep)
            run_btn.pack(side="left", padx=(8, 0))
            run_btn_ref["btn"] = run_btn
            save_btn = tk.Button(controls_sw, text="Save Sweep Results", command=_save_sweep_results, state="disabled")
            save_btn.pack(side="left", padx=(8, 0))
            save_btn_ref["btn"] = save_btn
            tk.Button(controls_sw, text="Load Sweep Results", command=_load_sweep_results).pack(side="left", padx=(8, 0))
            tk.Button(controls_sw, text="Load Scratch Folder", command=_load_sweep_scratch_folder).pack(side="left", padx=(8, 0))
            tk.Button(controls_sw, text="Close", command=top_sw.destroy).pack(side="left", padx=(8, 0))
            def _on_metric_change(_event=None):
                m_now = metric_sw_var.get().strip()
                _render_current_histogram(m_now, _filtered_rows())
                _render_empty(
                    "Click 'Run Sweep' to compute per-point means across random seeds.",
                    f"Per-point Mean Histogram: {m_now or 'Metric'}",
                )
            metric_sw_menu.bind("<<ComboboxSelected>>", _on_metric_change)
            _render_current_histogram(metric_sw_var.get().strip(), rows_snapshot)
            _render_empty(
                "Click 'Run Sweep' to compute per-point means across random seeds.",
                f"Per-point Mean Histogram: {metric_sw_var.get().strip() or 'Metric'}",
            )

        def _save_filtered_dataset():
            """Save current filtered histogram subset as a standalone loadable dataset."""
            filtered_rows = _filtered_rows()
            if not filtered_rows:
                messagebox.showwarning("No Data", "No filtered points to save.")
                return

            metric1_name = _selected_metric1()
            metric2_name = _selected_metric2()
            metrics_to_export = []
            for m in metric_options:
                m_clean = str(m or "").strip()
                if m_clean:
                    metrics_to_export.append(m_clean)
            # Keep selected axes metrics first for readability/stability.
            prioritized = [x for x in [metric1_name, metric2_name] if str(x or "").strip()]
            metrics_to_export = list(dict.fromkeys(prioritized + metrics_to_export))
            # Also include any metric keys already cached in the filtered points'
            # metric_input payloads, even if they are not currently visible in
            # dropdown options.
            extra_cached_metric_keys = []
            for rr in filtered_rows:
                point_idx = rr.get("point_idx", None)
                metric_input_for_keys = None
                if isinstance(point_idx, (int, np.integer)):
                    pi = int(point_idx)
                    if 0 <= pi < len(all_metric_inputs):
                        metric_input_for_keys = all_metric_inputs[pi]
                    if metric_input_for_keys is None:
                        metric_input_for_keys = _lazy_load_metric_input_for_idx(pi)
                rec = normalize_metric_input_shared(metric_input_for_keys)
                if not isinstance(rec, dict):
                    continue
                mm = rec.get("metric_values_by_name", None)
                if not isinstance(mm, dict):
                    continue
                for mk in mm.keys():
                    mk_clean = str(mk or "").strip()
                    if mk_clean:
                        extra_cached_metric_keys.append(mk_clean)
            if extra_cached_metric_keys:
                metrics_to_export = list(dict.fromkeys(metrics_to_export + sorted(set(extra_cached_metric_keys))))

            subset_param_names = list(param_names_list) if param_names_list else []
            if not subset_param_names:
                discovered = set()
                for rr in filtered_rows:
                    params = rr.get("params", {})
                    if not isinstance(params, dict):
                        continue
                    for k, v in params.items():
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if np.isfinite(fv):
                            discovered.add(str(k))
                subset_param_names = sorted(discovered)
            if not subset_param_names:
                messagebox.showwarning(
                    "No Parameters",
                    "Filtered subset has no numeric parameters to save.",
                )
                return

            save_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                title="Save Filtered Dataset",
                initialdir=_default_dialog_dir(),
                initialfile="optimization_dataset_filtered_metric_histogram_dataset.json",
                **_filedialog_filetypes_kwarg_macos_safe(
                    [
                        ("JSON files", "*.json"),
                        ("All files", "*.*"),
                    ]
                ),
            )
            if not save_path:
                return
            # Keep filtered exports loader-compatible with the existing
            # load_dataset filename validator (optimization_dataset_*.json).
            save_dir = os.path.dirname(save_path) or _default_dialog_dir()
            base_name = os.path.basename(save_path).strip()
            if base_name.lower().endswith(".json.gz"):
                base_name = base_name[:-8]
            elif base_name.lower().endswith(".json"):
                base_name = base_name[:-5]
            base_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_name).strip("._-")
            if not base_name:
                base_name = "filtered_metric_histogram_dataset"
            if not base_name.startswith("optimization_dataset_"):
                base_name = f"optimization_dataset_{base_name}"
            save_path = os.path.join(save_dir, f"{base_name}.json")

            run_lookup = {}
            for r in all_results:
                if not isinstance(r, dict):
                    continue
                ridx = r.get("run_index", None)
                try:
                    ridx = int(ridx) if ridx is not None else None
                except Exception:
                    ridx = None
                if ridx is not None and ridx not in run_lookup:
                    run_lookup[ridx] = r

            subset_results = []
            subset_histories = []
            subset_param_vectors = []
            subset_metric_values = []
            subset_replicate_info = []
            subset_metric_inputs = []
            subset_run_ids = []
            subset_run_indices = []

            skipped_rows = 0
            next_run_id = (max(all_run_ids) + 1) if all_run_ids else 0
            for local_i, rr in enumerate(filtered_rows):
                params = rr.get("params", {})
                if not isinstance(params, dict):
                    skipped_rows += 1
                    continue

                vec = []
                bad_vec = False
                for pname in subset_param_names:
                    try:
                        pv = float(params.get(pname, np.nan))
                    except Exception:
                        pv = np.nan
                    if not np.isfinite(pv):
                        bad_vec = True
                        break
                    vec.append(pv)
                if bad_vec:
                    skipped_rows += 1
                    continue

                point_idx = rr.get("point_idx", None)
                metric_val = _metric_value_for_row(rr, metric1_name)

                run_index_val = rr.get("run_index", None)
                try:
                    run_index_val = int(run_index_val) if run_index_val is not None else local_i
                except Exception:
                    run_index_val = local_i

                replicate_info_val = [1, 0, 1]
                metric_input_val = None
                run_id_val = None
                if isinstance(point_idx, (int, np.integer)):
                    pi = int(point_idx)
                    if 0 <= pi < len(all_replicate_info):
                        replicate_info_val = all_replicate_info[pi]
                    if 0 <= pi < len(all_metric_inputs):
                        metric_input_val = all_metric_inputs[pi]
                    if metric_input_val is None:
                        metric_input_val = _lazy_load_metric_input_for_idx(pi)
                    if 0 <= pi < len(all_run_ids):
                        run_id_val = all_run_ids[pi]

                metric_input_copy = to_json_serializable(metric_input_val)
                if not isinstance(metric_input_copy, dict):
                    metric_input_copy = {"collapsed": False, "metric_values_by_name": {}}
                metric_map = metric_input_copy.get("metric_values_by_name")
                if not isinstance(metric_map, dict):
                    metric_map = {}
                    metric_input_copy["metric_values_by_name"] = metric_map
                # Persist all known cached visualization metrics for each retained point
                # so reloaded filtered datasets keep full metric dropdown coverage.
                row_metric_cache = rr.get("metric_cache", {})
                if not isinstance(row_metric_cache, dict):
                    row_metric_cache = {}
                for m_name in metrics_to_export:
                    m_cache_val = row_metric_cache.get(m_name, np.nan)
                    try:
                        m_cache_val = float(m_cache_val)
                    except Exception:
                        m_cache_val = np.nan
                    if np.isnan(m_cache_val):
                        m_cache_val = _metric_value_for_row(rr, m_name)
                    metric_map[m_name] = None if np.isnan(m_cache_val) else float(m_cache_val)

                source_run = run_lookup.get(run_index_val, {})
                if not isinstance(source_run, dict):
                    source_run = {}

                if run_id_val is None:
                    run_id_val = next_run_id
                    next_run_id += 1

                subset_param_vectors.append(np.array(vec, dtype=float))
                subset_metric_values.append(float(metric_val) if np.isfinite(metric_val) else np.nan)
                subset_replicate_info.append(to_json_serializable(replicate_info_val))
                subset_metric_inputs.append(metric_input_copy)
                subset_run_ids.append(int(run_id_val))
                subset_run_indices.append(int(run_index_val))
                subset_histories.append([])
                subset_results.append(
                    {
                        "start_index": int(run_index_val),
                        "descent_index": 0,
                        "run_index": int(run_index_val),
                        "descent_seed": source_run.get("descent_seed", None),
                        "final_params": {k: float(v) for k, v in zip(subset_param_names, vec)},
                        "final_metric": float(metric_val) if np.isfinite(metric_val) else np.nan,
                        "best_params": {k: float(v) for k, v in zip(subset_param_names, vec)},
                        "best_metric": float(metric_val) if np.isfinite(metric_val) else np.nan,
                        "history": [],
                        "fixed_params": source_run.get("fixed_params", {}) if isinstance(source_run.get("fixed_params", {}), dict) else {},
                    }
                )

            if not subset_results:
                messagebox.showwarning(
                    "No Valid Points",
                    "Filtered rows could not be converted into a valid dataset.",
                )
                return

            maximize = opt_goal_var.get() == "Maximize"
            valid_metric_pairs = []
            for i, m in enumerate(subset_metric_values):
                if not np.isnan(m):
                    valid_metric_pairs.append((i, float(m)))
            if valid_metric_pairs:
                subset_best_idx = max(valid_metric_pairs, key=lambda x: x[1])[0] if maximize else min(valid_metric_pairs, key=lambda x: x[1])[0]
            else:
                subset_best_idx = 0
            best_result = subset_results[subset_best_idx]
            subset_summary = build_optimization_summary_text(
                subset_results,
                subset_best_idx,
                best_result.get("best_params", {}),
                best_result.get("best_metric", np.nan),
            )

            dataset_payload = _build_dataset_payload(
                subset_results,
                subset_histories,
                subset_param_vectors,
                subset_metric_values,
                subset_replicate_info,
                subset_metric_inputs,
                subset_run_ids,
                subset_run_indices,
            )
            dataset_payload["best_run_index"] = int(subset_best_idx)
            dataset_payload["summary_text"] = subset_summary
            if isinstance(dataset_payload.get("ui_state"), dict):
                dataset_payload["ui_state"]["metric_name"] = metric1_name
                dataset_payload["ui_state"]["visualization_metric_name"] = metric1_name
                dataset_payload["ui_state"]["visualization_metric_name_2"] = metric2_name
            if isinstance(dataset_payload.get("metadata"), dict):
                dataset_payload["metadata"]["metric_name"] = metric1_name
                dataset_payload["metadata"]["num_runs"] = len(subset_results)
                dataset_payload["metadata"]["num_points"] = len(subset_param_vectors)

            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(dataset_payload, f, indent=2)
            except Exception as e:
                messagebox.showerror("Save Failed", f"Failed to save filtered dataset:\n{e}")
                return

            kept_count = len(subset_param_vectors)
            messagebox.showinfo(
                "Filtered Dataset Saved",
                f"Saved {kept_count} filtered points to:\n{save_path}\n\n"
                f"Metric: {metric1_name}\n"
                f"Skipped rows (missing parameter values): {skipped_rows}",
            )

        tk.Button(controls_row7, text="UMAP (Clickable)", command=_open_filtered_umap_plot).pack(side="left", padx=(8, 0))
        tk.Button(controls_row7, text="Parameter Heatmaps (Filtered)", command=_open_filtered_parameter_heatmaps).pack(side="left", padx=(8, 0))
        tk.Button(controls_row7, text="Seed Sweep Means (Filtered)", command=_open_seed_sweep_means_histogram).pack(side="left", padx=(8, 0))
        tk.Button(controls_row7, text="Save Filtered Dataset", command=_save_filtered_dataset).pack(side="left", padx=(8, 0))
        tk.Button(controls_row7, text="Update", command=rebuild).pack(side="left", padx=(10, 0))
        tk.Button(controls_row7, text="Close", command=top.destroy).pack(side="left", padx=(10, 0))

        def _reset_filter():
            _configure_slider_for_param()
            rebuild()

        tk.Button(controls_row1, text="Reset Range", command=_reset_filter).pack(side="left", padx=(6, 0))

        def _reset_filter2():
            target_kind2, target_name2 = _decode_filter_target(filter_param2_var.get())
            data_range = _range_for_filter_target(run_rows, target_kind2, target_name2)
            if data_range is not None:
                pmin2, pmax2 = data_range
                low2_val_var.set(_format_bound(pmin2))
                high2_val_var.set(_format_bound(pmax2))
            else:
                low2_val_var.set("")
                high2_val_var.set("")
            rebuild()

        tk.Button(controls_row3, text="Reset Range 2", command=_reset_filter2).pack(side="left", padx=(6, 0))

        def _refresh_metric2_filter_visibility():
            if _selected_metric2():
                metric2_filter_frame.pack(side="left", padx=(6, 0))
            else:
                metric2_filter_frame.pack_forget()
                metric2_min_var.set("")
                metric2_max_var.set("")

        def _apply_metric_filter(_event=None):
            # Normalize and clamp to observed metric range when possible.
            metric1_name = _selected_metric1()
            metric2_name = _selected_metric2()
            lo_txt = metric_min_var.get().strip()
            hi_txt = metric_max_var.get().strip()
            lo = None
            hi = None
            if lo_txt:
                try:
                    lo = float(lo_txt)
                except Exception:
                    lo = None
                    metric_min_var.set("")
            if hi_txt:
                try:
                    hi = float(hi_txt)
                except Exception:
                    hi = None
                    metric_max_var.set("")

            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo

            metric_data_range = _metric_range_for_rows(run_rows, metric1_name)
            if metric_data_range is not None:
                dmin, dmax = metric_data_range
                if lo is not None:
                    lo = max(dmin, min(dmax, lo))
                if hi is not None:
                    hi = max(dmin, min(dmax, hi))
                if lo is not None:
                    metric_min_var.set(_format_bound(lo))
                if hi is not None:
                    metric_max_var.set(_format_bound(hi))

            if metric2_name:
                lo2_txt = metric2_min_var.get().strip()
                hi2_txt = metric2_max_var.get().strip()
                lo2 = None
                hi2 = None
                if lo2_txt:
                    try:
                        lo2 = float(lo2_txt)
                    except Exception:
                        lo2 = None
                        metric2_min_var.set("")
                if hi2_txt:
                    try:
                        hi2 = float(hi2_txt)
                    except Exception:
                        hi2 = None
                        metric2_max_var.set("")
                if lo2 is not None and hi2 is not None and lo2 > hi2:
                    lo2, hi2 = hi2, lo2
                metric2_data_range = _metric_range_for_rows(run_rows, metric2_name)
                if metric2_data_range is not None:
                    dmin2, dmax2 = metric2_data_range
                    if lo2 is not None:
                        lo2 = max(dmin2, min(dmax2, lo2))
                    if hi2 is not None:
                        hi2 = max(dmin2, min(dmax2, hi2))
                    if lo2 is not None:
                        metric2_min_var.set(_format_bound(lo2))
                    if hi2 is not None:
                        metric2_max_var.set(_format_bound(hi2))

            rebuild()

        def _reset_metric_filter():
            metric1_name = _selected_metric1()
            metric2_name = _selected_metric2()
            metric_data_range = _metric_range_for_rows(run_rows, metric1_name)
            if metric_data_range is None:
                metric_min_var.set("")
                metric_max_var.set("")
            else:
                dmin, dmax = metric_data_range
                metric_min_var.set(_format_bound(dmin))
                metric_max_var.set(_format_bound(dmax))
            if metric2_name:
                metric2_data_range = _metric_range_for_rows(run_rows, metric2_name)
                if metric2_data_range is None:
                    metric2_min_var.set("")
                    metric2_max_var.set("")
                else:
                    dmin2, dmax2 = metric2_data_range
                    metric2_min_var.set(_format_bound(dmin2))
                    metric2_max_var.set(_format_bound(dmax2))
            else:
                metric2_min_var.set("")
                metric2_max_var.set("")
            rebuild()

        metric_min_entry.bind("<Return>", _apply_metric_filter)
        metric_max_entry.bind("<Return>", _apply_metric_filter)
        metric2_min_entry.bind("<Return>", _apply_metric_filter)
        metric2_max_entry.bind("<Return>", _apply_metric_filter)

        def _on_param_change(_event=None):
            _configure_slider_for_param()
            rebuild()

        def _on_param2_change(_event=None):
            target_kind2, target_name2 = _decode_filter_target(filter_param2_var.get())
            data_range = _range_for_filter_target(run_rows, target_kind2, target_name2)
            if data_range is not None:
                pmin2, pmax2 = data_range
                low2_val_var.set(_format_bound(pmin2))
                high2_val_var.set(_format_bound(pmax2))
            else:
                low2_val_var.set("")
                high2_val_var.set("")
            rebuild()

        param_menu.bind("<<ComboboxSelected>>", _on_param_change)
        param2_menu.bind("<<ComboboxSelected>>", _on_param2_change)
        metric1_menu.bind("<<ComboboxSelected>>", lambda _e: rebuild())
        metric2_menu.bind("<<ComboboxSelected>>", lambda _e: (_refresh_metric2_filter_visibility(), rebuild()))
        relation_x_menu.bind("<<ComboboxSelected>>", lambda _e: rebuild())
        relation_y_menu.bind("<<ComboboxSelected>>", lambda _e: rebuild())
        lower_hist_param1_menu.bind("<<ComboboxSelected>>", lambda _e: rebuild())
        lower_hist_param2_menu.bind("<<ComboboxSelected>>", lambda _e: rebuild())
        heatmap_bins_x_entry.bind("<Return>", lambda _e: rebuild())
        heatmap_bins_y_entry.bind("<Return>", lambda _e: rebuild())
        max_plot_points_entry.bind("<Return>", lambda _e: rebuild())
        _configure_slider_for_param()
        _on_param2_change()
        _refresh_metric2_filter_visibility()
        rebuild()

    metric_hist_button.config(command=show_metric_histogram)

    def show_memory_stats():
        """Show current memory usage statistics."""
        valid_run_count = run_summary_stats["metric_stats"].n
        nan_run_count = run_summary_stats["nan_runs"]
        total_run_count = run_summary_stats["total_runs"]
        mean_run_metric = run_summary_stats["metric_stats"].get_mean()
        std_run_metric = run_summary_stats["metric_stats"].get_std(ddof=1)

        deleted_points_total = deleted_point_stats["total_points"]
        deleted_points_nan = deleted_point_stats["nan_points"]
        deleted_points_valid = deleted_point_stats["metric_stats"].n
        deleted_points_mean = deleted_point_stats["metric_stats"].get_mean()
        deleted_points_std = deleted_point_stats["metric_stats"].get_std(ddof=1)

        run_metric_lines = (
            f"- Mean metric (valid runs): {mean_run_metric:.6f}\n"
            f"- Std metric (valid runs, sample): {std_run_metric:.6f}\n"
        ) if valid_run_count > 0 else "- Mean/Std metric (valid runs): N/A (no valid runs)\n"

        deleted_metric_lines = (
            f"- Mean metric of deleted valid points: {deleted_points_mean:.6f}\n"
            f"- Std metric of deleted valid points (sample): {deleted_points_std:.6f}\n"
        ) if deleted_points_valid > 0 else "- Mean/Std metric of deleted valid points: N/A (no valid deleted points)\n"

        stats_msg = (
            f"Memory Usage Stats:\n"
            f"- Stored parameter vectors: {len(all_param_vectors)}\n"
            f"- Stored metric values: {len(all_metric_values)}\n"
            f"- Run histories: {len(all_runs_history)}\n"
            f"- Max capacity: {MAX_STORED_POINTS}\n"
            f"- Sample rate: 1 in {STORAGE_SAMPLE_RATE}\n"
            f"- Full Save enabled: True\n"
            f"- Full Save folder: {full_save_dir_var.get()}\n"
            f"- Full Save offloaded points (this session): {full_save_offloaded_points}\n"
            f"\nRun Stats (this optimization):\n"
            f"- Runs completed: {total_run_count}\n"
            f"- Valid runs (non-NaN): {valid_run_count} / {total_run_count}\n"
            f"- NaN runs: {nan_run_count} / {total_run_count}\n"
            f"{run_metric_lines}"
            f"\nDeleted/Swept Data (auto-cleanup only):\n"
            f"- Points deleted: {deleted_points_total} ({deleted_points_nan} NaN, {deleted_points_valid} valid)\n"
            f"{deleted_metric_lines}"
        )
        messagebox.showinfo("Memory Usage", stats_msg)
    
    memory_button = tk.Button(button_row2, text="Memory Info", command=show_memory_stats)
    memory_button.pack(side="left", padx=5)

    # Full Save is always enabled; folder is required for automatic offloading/snapshot.

    def _default_dialog_dir():
        """Default all file/folder dialogs to the repository root."""
        try:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        except Exception:
            return os.getcwd()

    def choose_full_save_folder():
        selected = filedialog.askdirectory(
            title="Select Full Save Folder",
            initialdir=_default_dialog_dir(),
        )
        if selected:
            full_save_dir_var.set(selected)
            try:
                refresh_metric_options()
            except Exception:
                pass

    def _ensure_full_save_folder(interactive=True):
        """Ensure a Full Save folder exists; optionally prompt user to choose one."""
        folder = full_save_dir_var.get().strip()
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
                return folder
            except Exception as e:
                if not interactive:
                    return None
                messagebox.showwarning("Full Save Folder Error", f"Cannot use selected folder:\n{e}")
        if not interactive:
            return None
        selected = filedialog.askdirectory(
            title="Select Full Save Folder",
            initialdir=_default_dialog_dir(),
        )
        if not selected:
            default_dir = os.path.abspath(os.path.join(os.getcwd(), "optimization_full_save"))
            try:
                os.makedirs(default_dir, exist_ok=True)
                full_save_dir_var.set(default_dir)
                return default_dir
            except Exception as e:
                messagebox.showerror(
                    "Full Save Folder Error",
                    f"Could not create default folder:\n{default_dir}\n\n{e}",
                )
                return None
        full_save_dir_var.set(selected)
        try:
            os.makedirs(selected, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Full Save Folder Error", f"Cannot create folder:\n{e}")
            return None
        return selected

    full_save_folder_btn = tk.Button(button_row3, text="Set Full Save Folder", command=choose_full_save_folder)
    full_save_folder_btn.pack(side="left", padx=5)

    def reanalyze_full_save_points():
        """Recompute the selected metric for in-memory and offloaded Full Save points."""
        if optimization_running:
            messagebox.showwarning("Optimization Running", "Stop optimization before reanalyzing points.")
            return
        if not param_names_list:
            messagebox.showwarning("No Data", "No parameter vectors available to reanalyze.")
            return

        def _build_bounds(param_names_for_eval):
            bounds = {}
            for pname in param_names_for_eval:
                if pname not in param_min_entries or pname not in param_max_entries:
                    raise ValueError(f"Missing bounds entries for parameter '{pname}'.")
                min_txt = param_min_entries[pname].get().strip()
                max_txt = param_max_entries[pname].get().strip()
                min_val = -np.inf if min_txt == "" else float(min_txt)
                if max_txt == "":
                    raise ValueError(f"Missing max bound for parameter '{pname}'.")
                max_val = float(max_txt)
                bounds[pname] = (min_val, max_val)
            return bounds

        def _compute_metric_from_cached_input(metric_input):
            """Compute current metric from cached simulation output (no simulation rerun)."""
            return compute_metric_from_cached_input_shared(metric_input, metric_var.get())

        def _get_cached_metric(rec):
            """Return cached metric value for current metric name if present."""
            rec = normalize_metric_input_shared(rec)
            if not isinstance(rec, dict):
                return None
            metric_map = rec.get("metric_values_by_name", None)
            if isinstance(metric_map, dict) and metric_var.get() in metric_map:
                mv = metric_map.get(metric_var.get())
                if mv is None:
                    return np.nan
                try:
                    return float(mv)
                except Exception:
                    return np.nan
            return None

        def _set_cached_metric(rec, metric_val):
            """Store metric value in per-point cache keyed by metric name."""
            rec = normalize_metric_input_shared(rec)
            if not isinstance(rec, dict):
                return
            metric_map = rec.get("metric_values_by_name", None)
            if not isinstance(metric_map, dict):
                metric_map = {}
                rec["metric_values_by_name"] = metric_map
            metric_map[metric_var.get()] = None if np.isnan(metric_val) else float(metric_val)

        try:
            # Reanalysis reads cached metric inputs only.
            seed_val = seed_entry.get().strip()
            seed = int(seed_val) if seed_val else None
            num_replicates = int(replicates_entry.get())
            if num_replicates < 1:
                raise ValueError("Replicates must be >= 1.")
        except Exception as e:
            messagebox.showerror("Reanalysis Error", f"Invalid settings for reanalysis:\n{e}")
            return

        metric_name = metric_var.get()
        full_save_dir = full_save_dir_var.get().strip()
        total_in_mem = _aligned_point_count(include_metric_inputs=False)

        def _is_batch_cached_for_metric(batch_path, manifest_obj, metric_name_to_check):
            if not isinstance(manifest_obj, dict):
                return False
            base = os.path.basename(batch_path)
            batch_index = _extract_batch_index_from_filename(batch_path)
            batch_entry = None
            for b in manifest_obj.get("batches", []):
                if not isinstance(b, dict):
                    continue
                if b.get("path") == base:
                    batch_entry = b
                    break
                if batch_entry is None and batch_index is not None and b.get("batch_index") == batch_index:
                    batch_entry = b
            if not isinstance(batch_entry, dict):
                return False
            num_records = int(batch_entry.get("num_records", 0) or 0)
            cached_counts = batch_entry.get("cached_metric_counts", {})
            if not isinstance(cached_counts, dict) or num_records <= 0:
                return False
            return int(cached_counts.get(metric_name_to_check, 0) or 0) >= num_records

        if not messagebox.askyesno(
            "Reanalyze Metrics",
            "Recompute current metric for in-memory points and Full Save offloaded batches?\n"
            "This may take a while."
        ):
            return

        # Show immediate activity feedback before any scanning/counting work starts.
        progress.stop()
        progress.configure(mode="indeterminate")
        progress.start(8)
        progress_label.config(text="Preparing reanalysis...")
        win.update()
        reanalyzed_in_mem = 0
        offloaded_reanalyzed = 0
        skipped_batches = 0

        def _ui_alive():
            try:
                return bool(win.winfo_exists()) and bool(progress.winfo_exists()) and bool(progress_label.winfo_exists())
            except Exception:
                return False

        def _safe_ui_call(fn):
            if not _ui_alive():
                return
            try:
                fn()
            except tk.TclError:
                pass

        def _safe_after(fn):
            try:
                if win.winfo_exists():
                    win.after(0, lambda: _safe_ui_call(fn))
            except Exception:
                pass

        def _ui_set_label(text):
            _safe_after(lambda t=text: progress_label.config(text=t))

        def _ui_set_determinate(maximum, value=0):
            _safe_after(lambda: (progress.stop(), progress.configure(mode="determinate", maximum=max(1, maximum), value=value)))

        def _ui_set_indeterminate(text=None):
            def _do():
                progress.stop()
                progress.configure(mode="indeterminate")
                progress.start(8)
                if text is not None:
                    progress_label.config(text=text)
            _safe_after(_do)

        def _ui_set_progress(value):
            _safe_after(lambda v=value: progress.configure(value=v))

        ui_update_interval_sec = 0.4
        last_ui_update = {"t": 0.0}

        def _throttled_ui_update(label_text=None, progress_value=None, force=False):
            now = time.time()
            if (not force) and (now - last_ui_update["t"] < ui_update_interval_sec):
                return
            last_ui_update["t"] = now
            if label_text is not None:
                _ui_set_label(label_text)
            if progress_value is not None:
                _ui_set_progress(progress_value)

        def _ui_finish():
            if not _ui_alive():
                return
            progress.stop()
            progress_label.config(text="")
            progress.configure(mode="indeterminate")
            try:
                refresh_metric_options()
            except Exception:
                pass
            messagebox.showinfo(
                "Reanalysis Complete",
                f"Reanalyzed in-memory points: {reanalyzed_in_mem}\n"
                f"Reanalyzed offloaded points: {offloaded_reanalyzed}\n"
                f"Skipped fully-cached batches: {skipped_batches}\n"
                f"Metric: {metric_name}\n"
                f"(Used cached final simulation outputs; no simulation reruns.)"
            )

        def _ui_already_cached():
            if not _ui_alive():
                return
            progress.stop()
            progress_label.config(text="")
            progress.configure(mode="indeterminate")
            try:
                refresh_metric_options()
            except Exception:
                pass
            messagebox.showinfo(
                "Reanalysis Skipped",
                f"Metric '{metric_name}' is already cached for all available Full Save points.\n"
                f"No reanalysis was needed."
            )

        def _ui_error(msg):
            if not _ui_alive():
                return
            progress.stop()
            progress_label.config(text="")
            progress.configure(mode="indeterminate")
            messagebox.showerror("Reanalysis Error", msg)

        def _worker():
            nonlocal reanalyzed_in_mem, offloaded_reanalyzed, skipped_batches, full_save_manifest_cache
            try:
                # Run the cache-completeness pre-check in worker thread so UI never blocks in "preparing".
                _ui_set_label("Preparing reanalysis (cache check)...")
                in_mem_all_cached = True
                for idx in range(total_in_mem):
                    metric_input = all_metric_inputs[idx] if idx < len(all_metric_inputs) else None
                    if metric_input is None:
                        metric_input = _lazy_load_metric_input_for_idx(idx)
                    if metric_input is None or _get_cached_metric(metric_input) is None:
                        in_mem_all_cached = False
                        break
                offloaded_all_cached = True
                batch_files_for_check = _list_offload_batch_files(full_save_dir, only_current_session=False)
                if batch_files_for_check:
                    manifest_for_check = _load_manifest()
                    for batch_path in batch_files_for_check:
                        if not _is_batch_cached_for_metric(batch_path, manifest_for_check, metric_name):
                            offloaded_all_cached = False
                            break
                if in_mem_all_cached and offloaded_all_cached:
                    win.after(0, _ui_already_cached)
                    return

                _ui_set_determinate(total_in_mem, 0)
                for idx, _ in enumerate(all_param_vectors):
                    if idx >= len(all_metric_values):
                        break
                    _throttled_ui_update(
                        label_text=f"Reanalyzing in-memory points: {idx}/{total_in_mem}",
                        progress_value=idx,
                    )
                    metric_input = all_metric_inputs[idx] if idx < len(all_metric_inputs) else None
                    if metric_input is None:
                        metric_input = _lazy_load_metric_input_for_idx(idx)
                    if metric_input is None:
                        continue
                    cached = _get_cached_metric(metric_input)
                    if cached is None:
                        metric_val = _compute_metric_from_cached_input(metric_input)
                        _set_cached_metric(metric_input, metric_val)
                    else:
                        metric_val = cached
                    replicate_info = all_replicate_info[idx] if idx < len(all_replicate_info) else (num_replicates, 0, num_replicates)
                    all_metric_values[idx] = metric_val
                    if idx < len(all_replicate_info):
                        all_replicate_info[idx] = replicate_info
                    reanalyzed_in_mem += 1
                _throttled_ui_update(progress_value=total_in_mem, force=True)

                _ui_set_indeterminate("Scanning offloaded batches...")
                manifest = _load_manifest()
                batch_files = _list_offload_batch_files(full_save_dir, only_current_session=False)
                manifest_by_path = {}
                manifest_by_index = {}
                if isinstance(manifest, dict):
                    for b in manifest.get("batches", []):
                        if not isinstance(b, dict):
                            continue
                        bp = b.get("path")
                        bi = b.get("batch_index")
                        if isinstance(bp, str) and bp:
                            manifest_by_path[bp] = b
                        if isinstance(bi, int):
                            manifest_by_index[bi] = b
                for bidx, batch_path in enumerate(batch_files):
                    base = os.path.basename(batch_path)
                    _throttled_ui_update(
                        label_text=f"Reanalyzing offloaded batch {bidx + 1}/{len(batch_files)}: {base}"
                    )

                    batch_entry = manifest_by_path.get(base)
                    batch_index = _extract_batch_index_from_filename(batch_path)
                    if batch_entry is None and batch_index is not None:
                        batch_entry = manifest_by_index.get(batch_index)
                    if isinstance(manifest, dict):
                        if isinstance(batch_entry, dict):
                            cached_counts = batch_entry.get("cached_metric_counts", {})
                            num_records = int(batch_entry.get("num_records", 0) or 0)
                            if isinstance(cached_counts, dict) and int(cached_counts.get(metric_name, 0) or 0) >= num_records and num_records > 0:
                                skipped_batches += 1
                                continue

                    try:
                        payload = _read_json_maybe_gz(batch_path)
                    except Exception:
                        continue
                    records = payload.get("records", [])
                    if not isinstance(records, list) or not records:
                        continue

                    changed = False
                    cached_count = 0
                    for rec in records:
                        metric_input = rec.get("metric_input", None)
                        if metric_input is None:
                            continue
                        metric_input_norm = normalize_metric_input_shared(metric_input)
                        if isinstance(metric_input_norm, dict):
                            metric_map = metric_input_norm.get("metric_values_by_name", None)
                            if not isinstance(metric_map, dict):
                                metric_map = {}
                                metric_input_norm["metric_values_by_name"] = metric_map
                                changed = True
                        cached = _get_cached_metric(metric_input)
                        if cached is None:
                            metric_val = _compute_metric_from_cached_input(metric_input)
                            _set_cached_metric(metric_input, metric_val)
                            changed = True
                        else:
                            metric_val = cached
                        if not np.isnan(metric_val):
                            pass
                        replicate_info = rec.get("replicate_info", [num_replicates, 0, num_replicates])
                        rec["metric_value"] = None if np.isnan(metric_val) else float(metric_val)
                        rec["replicate_info"] = list(replicate_info)
                        offloaded_reanalyzed += 1
                        changed = True
                        if _get_cached_metric(metric_input) is not None:
                            cached_count += 1
                        _throttled_ui_update(
                            label_text=f"Reanalyzing offloaded points: {offloaded_reanalyzed} processed..."
                        )

                    if changed:
                        payload["metric_name_current"] = metric_name
                        payload["reanalyzed_at_epoch"] = int(time.time())
                        try:
                            _write_json_maybe_gz(batch_path, payload)
                        except Exception:
                            pass

                    if isinstance(batch_entry, dict):
                        cached_counts = batch_entry.get("cached_metric_counts")
                        if not isinstance(cached_counts, dict):
                            cached_counts = {}
                            batch_entry["cached_metric_counts"] = cached_counts
                        prev_val = int(cached_counts.get(metric_name, 0) or 0)
                        cached_counts[metric_name] = max(prev_val, cached_count)
                        batch_entry["num_records"] = int(len(records))
                        batch_entry["updated_at_epoch"] = int(time.time())
                    elif isinstance(manifest, dict):
                        manifest.setdefault("batches", [])
                        new_entry = {
                            "batch_index": int(batch_index) if batch_index is not None else len(manifest["batches"]),
                            "path": base,
                            "num_records": int(len(records)),
                            "cached_metric_counts": {metric_name: int(cached_count)},
                            "updated_at_epoch": int(time.time()),
                        }
                        manifest["batches"].append(new_entry)
                        manifest_by_path[base] = new_entry
                        if isinstance(new_entry.get("batch_index"), int):
                            manifest_by_index[new_entry["batch_index"]] = new_entry

                full_save_manifest_cache = manifest
                _save_manifest()
                win.after(0, _ui_finish)
            except Exception as e:
                win.after(0, lambda: _ui_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    reanalyze_full_save_btn = tk.Button(button_row3, text="Reanalyze Full Save", command=reanalyze_full_save_points)
    reanalyze_full_save_btn.pack(side="left", padx=5)
    
    # Save/Load dataset buttons
    def _build_dataset_payload(
        results_data,
        runs_history_data,
        param_vectors_data,
        metric_values_data,
        replicate_info_data,
        metric_inputs_data,
        run_ids_data,
        run_indices_data,
    ):
        """Build a loadable optimization dataset payload."""
        if not results_data:
            raise ValueError("No optimization results available for dataset payload.")

        best_idx = best_run_index if best_run_index < len(results_data) else 0
        best_result_for_save = results_data[best_idx]
        best_params_for_save = best_result_for_save.get("best_params", best_result_for_save.get("final_params", {}))
        best_metric_for_save = best_result_for_save.get("best_metric", best_result_for_save.get("final_metric", np.nan))
        summary_text = build_optimization_summary_text(results_data, best_idx, best_params_for_save, best_metric_for_save)

        ui_state = {
            "metric_name": metric_var.get(),
            "visualization_metric_name": _selected_visualization_metric(),
            "visualization_metric_name_2": viz_metric2_var.get(),
            "optimization_goal": opt_goal_var.get(),
            "learning_rate": learning_rate_entry.get(),
            "max_iterations": max_iterations_entry.get(),
            "convergence_threshold": convergence_threshold_entry.get(),
            "gradient_step": gradient_step_entry.get(),
            "num_starts": num_starts_entry.get(),
            "descents_per_start": descents_per_start_entry.get(),
            "storage_sample_rate": sample_rate_entry.get(),
            "num_replicates": replicates_entry.get(),
            "seed": seed_entry.get(),
            "silent_mode": bool(silent_mode_var.get()),
            "homogeneous_population": bool(homogeneous_mode_var.get()),
            "independent_traits": bool(independent_traits_var.get()),
            "enable_m1_diffusion": bool(m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get()),
            "enable_m2_diffusion": bool(m2_diffusion_var.get()),
            "enable_m1_porin_diffusion": bool(m1_porin_diffusion_var.get()),
            "enable_diffusion_mutation": bool(diffusion_mutation_var.get()),
            "homogeneous_initial_diffusion_const": bool(homogeneous_initial_diffusion_const_var.get()),
            "enable_chemostat_flow": bool(enable_chemostat_flow_var.get()),
            "enable_initial_energy": bool(enable_initial_energy_var.get()),
            "enable_intermediate_costs": bool(enable_intermediate_costs_var.get()),
            "enable_acetate_addition": bool(enable_acetate_addition_var.get()),
            "binary_death_at_zero_energy": bool(binary_death_at_zero_energy_var.get()),
            "no_death": bool(no_death_var.get()),
            "constant_death_probability": bool(constant_death_probability_var.get()),
            "constant_duplication_probability": bool(constant_duplication_probability_var.get()),
            "full_save_folder": full_save_dir_var.get(),
        }
        if getattr(model_spec, "key", "") in {"simulation"}:
            try:
                ui_state["enable_m1_facilitated_diffusion"] = bool(m1_facilitation_var.get())
            except Exception:
                pass

        param_ui_state = {}
        for pname in param_names:
            try:
                param_ui_state[pname] = {
                    "fixed": bool(param_fix_checkboxes[pname].get()),
                    "initial": param_initial_entries[pname].get() if pname in param_initial_entries else "",
                    "min": param_min_entries[pname].get() if pname in param_min_entries else "",
                    "max": param_max_entries[pname].get() if pname in param_max_entries else "",
                }
            except Exception:
                pass

        fixed_param_values = {}
        try:
            for pname, entry in fixed_entries.items():
                fixed_param_values[pname] = entry.get()
        except Exception:
            pass

        payload = {
            'all_results': to_json_serializable(results_data),
            'all_runs_history': to_json_serializable(runs_history_data),
            'all_param_vectors': [list(pv) if isinstance(pv, np.ndarray) else pv for pv in param_vectors_data],
            'all_metric_values': [float(m) if not np.isnan(m) else None for m in metric_values_data],
            'all_replicate_info': to_json_serializable(replicate_info_data),
            'all_metric_inputs': to_json_serializable(metric_inputs_data),
            'all_run_ids': run_ids_data,
            'all_run_indices': run_indices_data,
            'best_run_index': int(best_idx),
            'param_names_list': param_names_list,
            'summary_text': summary_text,
            'ui_state': ui_state,
            'param_ui_state': param_ui_state,
            'fixed_param_values': fixed_param_values,
            'model': {
                'key': model_spec.key,
                'label': model_spec.label,
            },
            'metadata': {
                'num_runs': len(results_data),
                'num_points': len(param_vectors_data),
                'metric_name': metric_var.get(),
                'optimization_goal': opt_goal_var.get(),
                'num_starts': num_starts_entry.get(),
                'max_iterations': max_iterations_entry.get(),
                'learning_rate': learning_rate_entry.get(),
                'num_replicates': replicates_entry.get(),
            }
        }
        return payload

    def save_dataset():
        """Save the current optimization dataset to a JSON file."""
        if len(all_results) == 0:
            messagebox.showwarning("No Data", "No optimization data to save. Run an optimization first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            title="Save Optimization Dataset",
            initialdir=_default_dialog_dir(),
            **_filedialog_filetypes_kwarg_macos_safe(
                [("JSON files", "*.json"), ("All files", "*.*")]
            ),
        )
        
        if not filename:
            return
        
        try:
            # Validate data consistency before saving
            if len(all_results) != len(all_runs_history):
                messagebox.showwarning("Data Inconsistency", 
                    f"Warning: Number of results ({len(all_results)}) doesn't match number of histories ({len(all_runs_history)}). "
                    "Some data may be missing.")
            
            dataset = _build_dataset_payload(
                all_results,
                all_runs_history,
                all_param_vectors,
                all_metric_values,
                all_replicate_info,
                all_metric_inputs,
                all_run_ids,
                all_run_indices,
            )
            
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            messagebox.showinfo("Success", f"Dataset saved successfully to:\n{filename}\n\n"
                f"Saved {len(all_results)} runs with {len(all_param_vectors)} data points.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset:\n{e}")
            import traceback
            traceback.print_exc()
    
    def load_dataset():
        """Load an optimization dataset from a JSON file."""
        if optimization_running:
            messagebox.showwarning("Optimization Running", "Cannot load dataset while optimization is running. Please stop the optimization first.")
            return
        
        # macOS: omit filetypes (see _filedialog_filetypes_kwarg_macos_safe); validate below.
        while True:
            filename = filedialog.askopenfilename(
                title="Load Optimization Dataset",
                initialdir=_default_dialog_dir(),
                **_filedialog_filetypes_kwarg_macos_safe(DATASET_FILETYPES),
            )

            if not filename:
                return

            if is_supported_dataset_file(filename):
                break
            messagebox.showerror(
                "Invalid File",
                invalid_dataset_file_message()
            )
        
        try:
            if not is_supported_dataset_file(filename):
                messagebox.showerror(
                    "Invalid File",
                    invalid_dataset_file_message()
                )
                return

            # When loading a full-save consolidated snapshot file directly,
            # anchor full-save manifest discovery to the snapshot folder.
            base_name = os.path.basename(str(filename))
            if is_full_save_dataset_snapshot_basename(base_name):
                try:
                    full_save_dir_var.set(os.path.dirname(filename))
                except Exception:
                    pass

            dataset = load_dataset_payload(filename, _read_json_maybe_gz, gzip)
            
            # Restore the data
            nonlocal all_results, all_runs_history, all_param_vectors, all_metric_values
            nonlocal all_replicate_info, all_metric_inputs, all_run_ids, all_run_indices, best_run_index, param_names_list
            nonlocal offload_point_sources, offload_last_batch_path, offload_last_batch_payload, full_save_lightweight_mode, full_save_lightweight_metric_name, lightweight_metric_cache, loaded_full_save_session_id, loaded_full_save_settings_snapshot, offload_pending_records
            
            warn_if_model_mismatch(dataset, model_spec.key, model_spec.label, messagebox)

            normalized_load_state, load_alignment_warnings = prepare_decoded_aligned_dataset(
                dataset,
                np_module=np,
                goal_fallback="Maximize",
            )
            if normalized_load_state is None:
                messagebox.showwarning("Empty Dataset", "The loaded dataset contains no data.")
                return
            for warn_msg in load_alignment_warnings:
                messagebox.showwarning("Data Inconsistency", warn_msg)

            all_results = normalized_load_state["all_results"]
            all_runs_history = normalized_load_state["all_runs_history"]
            all_param_vectors = normalized_load_state["all_param_vectors"]
            all_metric_values = normalized_load_state["all_metric_values"]
            all_replicate_info = normalized_load_state["all_replicate_info"]
            all_metric_inputs = normalized_load_state["all_metric_inputs"]
            all_run_ids = normalized_load_state["all_run_ids"]
            all_run_indices = normalized_load_state["all_run_indices"]
            best_run_index = normalized_load_state["best_run_index"]
            param_names_list = normalized_load_state["param_names_list"]
            transient_state = normalized_load_state["transient_state"]
            offload_point_sources = transient_state["offload_point_sources"]
            offload_pending_records = transient_state["offload_pending_records"]
            offload_last_batch_path = transient_state["offload_last_batch_path"]
            offload_last_batch_payload = transient_state["offload_last_batch_payload"]
            full_save_lightweight_mode = transient_state["full_save_lightweight_mode"]
            full_save_lightweight_metric_name = transient_state["full_save_lightweight_metric_name"]
            lightweight_metric_cache = transient_state["lightweight_metric_cache"]
            loaded_full_save_session_id = transient_state["loaded_full_save_session_id"]
            loaded_full_save_settings_snapshot = transient_state["loaded_full_save_settings_snapshot"]
            
            # Rebuild run-level stats from loaded results (so summaries stay correct)
            run_summary_stats.update(build_run_summary_stats(all_results))

            # We can't reconstruct what was deleted by prior auto-cleanups from a saved dataset
            reset_deleted_point_stats(deleted_point_stats)

            # Restore UI settings if present (new datasets)
            restored_ui_state = restore_ui_state(
                dataset=dataset,
                entry_by_key={
                    "learning_rate": learning_rate_entry,
                    "max_iterations": max_iterations_entry,
                    "convergence_threshold": convergence_threshold_entry,
                    "gradient_step": gradient_step_entry,
                    "num_starts": num_starts_entry,
                    "descents_per_start": descents_per_start_entry,
                    "storage_sample_rate": sample_rate_entry,
                    "num_replicates": replicates_entry,
                    "seed": seed_entry,
                },
                raw_var_by_key={
                    "metric_name": metric_var,
                    "visualization_metric_name": viz_metric_var,
                    "visualization_metric_name_2": viz_metric2_var,
                    "optimization_goal": opt_goal_var,
                },
                bool_var_by_key={
                    "silent_mode": silent_mode_var,
                    "homogeneous_population": homogeneous_mode_var,
                    "independent_traits": independent_traits_var,
                    "enable_m1_diffusion": m1_diffusion_var,
                    "enable_m2_diffusion": m2_diffusion_var,
                    "enable_m1_porin_diffusion": m1_porin_diffusion_var,
                    "enable_diffusion_mutation": diffusion_mutation_var,
                    "homogeneous_initial_diffusion_const": homogeneous_initial_diffusion_const_var,
                    "enable_chemostat_flow": enable_chemostat_flow_var,
                    "enable_initial_energy": enable_initial_energy_var,
                    "enable_intermediate_costs": enable_intermediate_costs_var,
                    "enable_acetate_addition": enable_acetate_addition_var,
                    "binary_death_at_zero_energy": binary_death_at_zero_energy_var,
                    "no_death": no_death_var,
                    "constant_death_probability": constant_death_probability_var,
                    "constant_duplication_probability": constant_duplication_probability_var,
                    "enable_m1_facilitated_diffusion": m1_facilitation_var,
                },
                str_var_by_key={
                    "full_save_folder": full_save_dir_var,
                },
            )
            if restored_ui_state:
                refresh_diffusion_mutation_state()
                _refresh_constant_death_state(preserve_fix_checkbox=True)
                _refresh_constant_duplication_state(preserve_fix_checkbox=True)

            # Restore parameter checkbox/initial/min/max state if present
            restore_param_ui_state_and_fixed_values(
                dataset=dataset,
                param_fix_checkboxes=param_fix_checkboxes,
                param_initial_entries=param_initial_entries,
                param_min_entries=param_min_entries,
                param_max_entries=param_max_entries,
                fixed_entries=fixed_entries,
                refresh_optimizable_params=refresh_optimizable_params,
                refresh_fixed_params=refresh_fixed_params,
            )

            # Apply loaded fixed params to the GUI (so fields like "Number of Generations" reflect the dataset)
            try:
                loaded_fixed = extract_loaded_fixed_params(all_results, best_run_index)
                # Suspend enable-* traces: applying fixed_params toggles would otherwise force
                # linked parameters to Fixed, overwriting param_ui_state from the dataset.
                _disconnect_conditional_param_feature_write_traces()
                try:
                    applied_loaded_fixed = apply_loaded_fixed_params_to_widgets(
                        loaded_fixed=loaded_fixed,
                        fixed_entries=fixed_entries,
                        param_initial_entries=param_initial_entries,
                        seed_entry=seed_entry,
                        simulation_setting_vars={
                            "Homogeneous Population": homogeneous_mode_var,
                            "Independent Traits": independent_traits_var,
                            "Enable M1 Diffusion": m1_diffusion_var,
                            "Enable M1 Facilitated Diffusion": m1_facilitation_var,
                            "Enable M1 Porin Diffusion": m1_porin_diffusion_var,
                            "Enable M2 Diffusion": m2_diffusion_var,
                            "Enable Diffusion Mutation": diffusion_mutation_var,
                            "Homogeneous Initial Diffusion Const.": homogeneous_initial_diffusion_const_var,
                            "Enable Initial Energy": enable_initial_energy_var,
                            "Enable Chemostat Flow": enable_chemostat_flow_var,
                            "Enable Intermediate Costs": enable_intermediate_costs_var,
                            "Enable Acetate Addition": enable_acetate_addition_var,
                            "Binary Death at Zero Energy": binary_death_at_zero_energy_var,
                            NO_DEATH: no_death_var,
                            "Constant Death Probability": constant_death_probability_var,
                            "Constant Duplication Probability": constant_duplication_probability_var,
                        },
                    )
                finally:
                    _connect_conditional_param_feature_write_traces()
                # apply_loaded_fixed_params_to_widgets sets simulation_setting_vars (homogeneous,
                # diffusion, etc.) whose traces can overwrite param_ui_state; re-apply saved table.
                restore_param_ui_state_and_fixed_values(
                    dataset=dataset,
                    param_fix_checkboxes=param_fix_checkboxes,
                    param_initial_entries=param_initial_entries,
                    param_min_entries=param_min_entries,
                    param_max_entries=param_max_entries,
                    fixed_entries=fixed_entries,
                    refresh_optimizable_params=refresh_optimizable_params,
                    refresh_fixed_params=refresh_fixed_params,
                )
                if applied_loaded_fixed:
                    # Refresh displays in case loaded toggles affected widget state.
                    run_post_loaded_fixed_refreshes(
                        refresh_diffusion_mutation_state_fn=refresh_diffusion_mutation_state,
                        refresh_optimizable_params_fn=refresh_optimizable_params,
                        refresh_fixed_params_fn=refresh_fixed_params,
                        refresh_metric_options_fn=refresh_metric_options,
                        refresh_chemostat_flow_state_fn=lambda: _refresh_chemostat_flow_state(
                            preserve_fix_checkbox=True
                        ),
                        refresh_initial_energy_state_fn=lambda: _refresh_initial_energy_state(
                            preserve_fix_checkbox=True
                        ),
                        refresh_intermediate_costs_state_fn=lambda: _refresh_intermediate_costs_state(
                            preserve_fix_checkbox=True
                        ),
                        refresh_acetate_addition_state_fn=lambda: _refresh_acetate_addition_state(
                            preserve_fix_checkbox=True
                        ),
                    )
            except Exception:
                pass
            
            show_loaded_dataset_info_dialog(
                dataset=dataset,
                all_results=all_results,
                all_param_vectors=all_param_vectors,
                filename=filename,
                messagebox_module=messagebox,
            )

            # Restore / regenerate the end-of-run summary text into the results window
            restore_results_summary_text(
                results_text_widget=results_text,
                dataset=dataset,
                all_results=all_results,
                best_run_index=best_run_index,
                build_summary_text_fn=build_optimization_summary_text,
                np_module=np,
            )

            # Enable parameter heatmaps if we have results
            set_loaded_dataset_button_states(
                all_results=all_results,
                param_heatmaps_button=param_heatmaps_button,
                metric_hist_button=metric_hist_button,
            )
            # Ensure visualization metric availability is recomputed for the loaded dataset.
            try:
                refresh_metric_options()
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
            import traceback
            traceback.print_exc()

    def load_full_save_session():
        """Load Full Save session asynchronously with manifest-first fast path and cancellation."""
        nonlocal full_save_load_in_progress, full_save_load_cancel_requested, loaded_full_save_session_id
        if optimization_running:
            messagebox.showwarning("Optimization Running", "Cannot load Full Save while optimization is running. Please stop the optimization first.")
            return
        if full_save_load_in_progress:
            full_save_load_cancel_requested = True
            progress_label.config(text="Cancelling full-save load...")
            return

        # Default to the repository root for predictable navigation.
        default_picker_dir = _default_dialog_dir()
        folder = filedialog.askdirectory(
            title="Select Full Save Folder",
            initialdir=default_picker_dir,
        )
        if not folder:
            return

        try:
            session_ids = discover_full_save_session_ids(folder)
            if not session_ids:
                raise ValueError("No full-save sessions found in selected folder.")
            ordered_sessions = sorted(session_ids)
            if len(ordered_sessions) == 1:
                session_id = ordered_sessions[0]
            else:
                default_sid = ordered_sessions[-1]
                session_id = simpledialog.askstring(
                    "Choose Full Save Session",
                    "Multiple sessions found in this folder.\n"
                    "Enter session ID to load (e.g. 20260302-170133):",
                    initialvalue=default_sid,
                    parent=win,
                )
                if not session_id:
                    return
                session_id = session_id.strip()
                if session_id not in session_ids:
                    raise ValueError(f"Session '{session_id}' not found in selected folder.")
            settings_snapshot = None
            settings_path = full_save_settings_path_json(folder, session_id)
            settings_path_gz = full_save_settings_path_gz(folder, session_id)
            if os.path.exists(settings_path) or os.path.exists(settings_path_gz):
                try:
                    settings_snapshot = _read_json_maybe_gz(settings_path if os.path.exists(settings_path) else settings_path_gz)
                except Exception:
                    settings_snapshot = None
            goal_name = opt_goal_var.get()
            if isinstance(settings_snapshot, dict):
                ui_state_ss = settings_snapshot.get("ui_state", {})
                if isinstance(ui_state_ss, dict):
                    goal_name = ui_state_ss.get("optimization_goal", goal_name)
            goal_maximize = (goal_name == "Maximize")
            full_save_load_in_progress = True
            full_save_load_cancel_requested = False
            load_full_save_button.config(text="Cancel Full Save Load")
            progress_label.config(text=f"Loading Full Save {session_id}...")
            progress.start(10)

            def _set_progress(msg):
                try:
                    win.after(0, lambda m=msg: progress_label.config(text=m))
                except Exception:
                    pass

            def _worker():
                try:
                    if full_save_load_cancel_requested:
                        return {"cancelled": True}

                    dataset_path = full_save_dataset_snapshot_path(folder, session_id)
                    if os.path.exists(dataset_path):
                        _set_progress("Loading consolidated full-save snapshot...")
                        dataset = _read_json_maybe_gz(dataset_path)
                        if isinstance(dataset, dict) and isinstance(dataset.get("all_param_vectors", []), list):
                            _set_progress("Preparing consolidated snapshot for UI...")
                            state, load_warnings = prepare_decoded_aligned_dataset(
                                dataset,
                                np_module=np,
                                goal_fallback=goal_name,
                            )
                            if state is None:
                                raise ValueError("The consolidated snapshot contains no run or point data.")
                            dataset_snapshot = {
                                "ui_state": dataset.get("ui_state", {}),
                                "param_ui_state": dataset.get("param_ui_state", {}),
                                "fixed_param_values": dataset.get("fixed_param_values", {}),
                            }
                            return {
                                "mode": "dataset",
                                "session_id": session_id,
                                "folder": folder,
                                "settings_snapshot": settings_snapshot,
                                "dataset_snapshot": dataset_snapshot,
                                "goal_name": state["resolved_goal_name"],
                                "load_warnings": load_warnings,
                                "primary_metric_name": str(dataset.get("metadata", {}).get("metric_name", "") or "").strip() or None,
                                "all_results": state["all_results"],
                                "all_runs_history": state["all_runs_history"],
                                "all_param_vectors": state["all_param_vectors"],
                                "all_metric_values": state["all_metric_values"],
                                "all_replicate_info": state["all_replicate_info"],
                                "all_metric_inputs": state["all_metric_inputs"],
                                "all_run_ids": state["all_run_ids"],
                                "all_run_indices": state["all_run_indices"],
                                "best_run_index": state["best_run_index"],
                                "param_names_list": state["param_names_list"],
                                "offload_point_sources": [],
                                "full_save_lightweight_mode": False,
                                "full_save_lightweight_metric_name": None,
                            }

                    _set_progress("Resolving offload batches from manifest...")
                    manifest = load_full_save_manifest(folder, session_id, _read_json_maybe_gz)
                    manifest_batch_paths = ordered_existing_batch_paths_from_manifest(folder, manifest)
                    batch_paths = list(manifest_batch_paths)

                    fallback_scan = not batch_paths
                    if not fallback_scan and len(manifest_batch_paths) < manifest_declared_batch_path_count(manifest):
                        fallback_scan = True
                    if fallback_scan:
                        _set_progress("Manifest incomplete; scanning offload files...")
                        scanned_batch_paths = [
                            p for p in _list_offload_batch_files(folder, only_current_session=False)
                            if os.path.basename(p).startswith(f"offload_{session_id}_")
                        ]
                        batch_paths = sorted(set(batch_paths) | set(scanned_batch_paths), key=_offload_path_sort_key)
                    if not batch_paths:
                        raise ValueError("No offload batch files found for the selected manifest/session.")

                    total_batches = len(batch_paths)

                    def _on_rebuild_batch(bi, total, _batch_path):
                        if bi % 5 == 0:
                            _set_progress(f"Reading offload batches {bi + 1}/{total}...")

                    try:
                        rebuilt = reconstruct_from_offload_batches(
                            batch_paths=batch_paths,
                            read_payload=_read_json_maybe_gz,
                            optimization_goal=("Maximize" if goal_maximize else "Minimize"),
                            metric_name_fallback="Unknown",
                            include_replicate_info=True,
                            include_metric_inputs=True,
                            normalize_metric_input=normalize_metric_input_shared,
                            collect_metric_cache_by_name=True,
                            on_batch=_on_rebuild_batch,
                            should_cancel=lambda: bool(full_save_load_cancel_requested),
                        )
                    except InterruptedError:
                        return {"cancelled": True}

                    loaded_param_vectors = rebuilt["all_param_vectors"]
                    loaded_metric_values = rebuilt["all_metric_values"]
                    loaded_replicate_info = rebuilt["all_replicate_info"]
                    loaded_metric_inputs = [None] * len(loaded_param_vectors)
                    loaded_run_ids = rebuilt["all_run_ids"]
                    loaded_run_indices = rebuilt["all_run_indices"]
                    loaded_offload_sources = rebuilt["offload_point_sources"]
                    loaded_metric_cache_by_name = rebuilt["lightweight_metric_cache"]
                    resolved_param_names = rebuilt["param_names_list"]
                    loaded_results = rebuilt["all_results"]
                    loaded_primary_metric_name = rebuilt["metric_name"] or None

                    _set_progress(f"Preparing offload session for UI ({total_batches} batches)...")
                    return {
                        "mode": "offload",
                        "session_id": session_id,
                        "folder": folder,
                        "settings_snapshot": settings_snapshot,
                        "dataset_snapshot": None,
                        "goal_name": goal_name,
                        "primary_metric_name": loaded_primary_metric_name,
                        "all_results": loaded_results,
                        "all_runs_history": [[] for _ in loaded_results],
                        "all_param_vectors": loaded_param_vectors,
                        "all_metric_values": loaded_metric_values,
                        "all_replicate_info": loaded_replicate_info,
                        "all_metric_inputs": loaded_metric_inputs,
                        "all_run_ids": loaded_run_ids,
                        "all_run_indices": loaded_run_indices,
                        "offload_point_sources": loaded_offload_sources,
                        "param_names_list": resolved_param_names,
                        "best_run_index": 0,
                        "full_save_lightweight_mode": True,
                        "full_save_lightweight_metric_name": loaded_primary_metric_name,
                        "lightweight_metric_cache": loaded_metric_cache_by_name,
                        "load_warnings": [],
                    }
                except Exception as err:
                    return {"error": str(err)}

            def _finish_load(result):
                nonlocal all_results, all_runs_history, all_param_vectors, all_metric_values
                nonlocal all_replicate_info, all_metric_inputs, all_run_ids, all_run_indices, best_run_index, param_names_list
                nonlocal offload_point_sources, offload_last_batch_path, offload_last_batch_payload, full_save_lightweight_mode, full_save_lightweight_metric_name, lightweight_metric_cache, offload_pending_records
                nonlocal full_save_load_in_progress, full_save_load_cancel_requested, loaded_full_save_session_id, loaded_full_save_settings_snapshot

                progress.stop()
                progress_label.config(text="")
                full_save_load_in_progress = False
                full_save_load_cancel_requested = False
                load_full_save_button.config(text="Load Full Save Session")

                if result.get("cancelled"):
                    messagebox.showinfo("Load Cancelled", "Full Save session load was cancelled.")
                    return
                if "error" in result:
                    messagebox.showerror("Load Full Save Error", f"Failed to load full-save session:\n{result['error']}")
                    return

                for warn_msg in result.get("load_warnings") or []:
                    messagebox.showwarning("Data Inconsistency", warn_msg)

                mode = result.get("mode", "offload")
                loaded_primary_metric_name = result.get("primary_metric_name")
                loaded_full_save_session_id = result.get("session_id")
                loaded_full_save_settings_snapshot = result.get("settings_snapshot") if isinstance(result.get("settings_snapshot"), dict) else None
                loaded_goal_name = result.get("goal_name", goal_name)
                loaded_goal_maximize = (loaded_goal_name == "Maximize")
                all_results = result.get("all_results", [])
                all_runs_history = result.get("all_runs_history", [])
                all_param_vectors = result.get("all_param_vectors", [])
                all_metric_values = result.get("all_metric_values", [])
                all_replicate_info = result.get("all_replicate_info", [])
                all_metric_inputs = result.get("all_metric_inputs", [])
                all_run_ids = result.get("all_run_ids", [])
                all_run_indices = result.get("all_run_indices", [])
                best_run_index = int(result.get("best_run_index", 0) or 0)
                param_names_list = result.get("param_names_list", [])
                offload_point_sources = result.get("offload_point_sources", [])
                offload_pending_records = []
                offload_last_batch_path = None
                offload_last_batch_payload = None
                full_save_lightweight_mode = bool(result.get("full_save_lightweight_mode", False))
                full_save_lightweight_metric_name = result.get("full_save_lightweight_metric_name")
                lightweight_metric_cache = result.get("lightweight_metric_cache", {})
                if not isinstance(lightweight_metric_cache, dict):
                    lightweight_metric_cache = {}

                best_run_index = resolve_loaded_best_run_index(
                    all_results,
                    best_run_index,
                    loaded_goal_maximize,
                )

                run_summary_stats["total_runs"] = 0
                run_summary_stats["nan_runs"] = 0
                run_summary_stats["metric_stats"] = RunningStats()
                for r in all_results:
                    mv = r.get("best_metric", r.get("final_metric", np.nan))
                    run_summary_stats["total_runs"] += 1
                    if np.isnan(mv):
                        run_summary_stats["nan_runs"] += 1
                    else:
                        run_summary_stats["metric_stats"].add(mv)
                deleted_point_stats["total_points"] = 0
                deleted_point_stats["nan_points"] = 0
                deleted_point_stats["metric_stats"] = RunningStats()

                full_save_dir_var.set(folder)
                if loaded_primary_metric_name and (loaded_primary_metric_name in (model_spec.metric_names or [])):
                    try:
                        metric_var.set(loaded_primary_metric_name)
                    except Exception:
                        pass
                try:
                    viz_metric_var.set(metric_var.get())
                except Exception:
                    pass
                snapshot_to_apply = result.get("dataset_snapshot")
                if not isinstance(snapshot_to_apply, dict):
                    snapshot_to_apply = result.get("settings_snapshot")
                if isinstance(snapshot_to_apply, dict):
                    _apply_settings_snapshot(snapshot_to_apply)
                # Keep manifest discovery anchored to the folder the user just loaded.
                # Some saved snapshots may contain an empty/stale full_save_folder value.
                try:
                    full_save_dir_var.set(folder)
                except Exception:
                    pass
                if full_save_lightweight_mode:
                    try:
                        if loaded_primary_metric_name and (loaded_primary_metric_name in (model_spec.metric_names or [])):
                            metric_var.set(loaded_primary_metric_name)
                        viz_metric_var.set(metric_var.get())
                    except Exception:
                        pass
                # Recompute visualization metric availability after final folder/metric selection.
                try:
                    refresh_metric_options()
                except Exception:
                    pass

                source_msg = "consolidated dataset snapshot" if mode == "dataset" else "offload batches (lightweight)"
                results_text.delete(1.0, tk.END)
                results_text.insert(
                    tk.END,
                    f"Loaded Full Save session {session_id} from {source_msg}.\n"
                    f"Runs: {len(all_results)}\n"
                    f"Points: {len(all_param_vectors)}\n"
                    f"{'Heavy metric_input payloads remain on disk and are loaded on demand.' if full_save_lightweight_mode else 'Metric payloads loaded from consolidated snapshot.'}\n\n"
                )
                if all_results and 0 <= best_run_index < len(all_results):
                    br = all_results[best_run_index]
                    results_text.insert(
                        tk.END,
                        build_optimization_summary_text(
                            all_results,
                            best_run_index,
                            br.get("best_params", br.get("final_params", {})),
                            br.get("best_metric", br.get("final_metric", np.nan)),
                        ),
                    )
                results_text.see(tk.END)
                param_heatmaps_button.config(state=("normal" if all_results else "disabled"))
                metric_hist_button.config(state=("normal" if all_results else "disabled"))
                messagebox.showinfo(
                    "Full Save Loaded",
                    f"Session {session_id} loaded successfully from folder:\n{folder}\n"
                    f"Points: {len(all_param_vectors)}\n"
                    f"Settings restored: {'yes' if isinstance(snapshot_to_apply, dict) else 'no (snapshot not found)'}"
                )

            threading.Thread(target=lambda: win.after(0, _finish_load, _worker()), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Load Full Save Error", f"Failed to load full-save session:\n{e}")
            import traceback
            traceback.print_exc()
    
    load_button = tk.Button(button_row3, text="Load Dataset", command=load_dataset)
    load_button.pack(side="left", padx=5)
    load_full_save_button = tk.Button(button_row3, text="Load Full Save Session", command=load_full_save_session)
    load_full_save_button.pack(side="left", padx=5)
    
    # Bottom row for progress bar (takes up more space)
    progress_row = tk.Frame(button_frame)
    progress_row.pack(side="bottom", fill="x", pady=(5, 0))
    
    progress = ttk.Progressbar(progress_row, orient="horizontal", mode="indeterminate")
    progress.pack(side="left", padx=5, fill="x", expand=True)
    
    progress_label = tk.Label(progress_row, text="")
    progress_label.pack(side="left", padx=5)
    
    # === Results Panel ===
    results_frame = tk.LabelFrame(right_panel, text="Optimization Results", padx=5, pady=5)
    results_frame.pack(fill="both", expand=True)
    
    # Text widget for results
    results_text = tk.Text(results_frame, wrap=tk.WORD, font=("Courier", 10))
    results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_text.yview)
    results_text.configure(yscrollcommand=results_scrollbar.set)
    results_text.pack(side="left", fill="both", expand=True)
    results_scrollbar.pack(side="right", fill="y")
    
    # Right panel: interactive simulation diagram only.
    plot_frame = tk.LabelFrame(right_panel, text="Simulation Pathway Diagram", padx=5, pady=5)
    plot_frame.pack(fill="both", expand=True, pady=(10, 0))

    diagram_container = tk.Frame(plot_frame)
    diagram_container.pack(fill="both", expand=True, padx=4, pady=(0, 4))
    optimization_model_diagram = SimulationModelDiagram(diagram_container, width=700, height=300)
    optimization_model_diagram.canvas.pack(fill="both", expand=True)
    optimization_model_diagram.bind_to_vars(
        enable_m1_diffusion=m1_diffusion_var,
        enable_m1_facilitated_diffusion=m1_facilitation_var,
        enable_m2_diffusion=m2_diffusion_var,
        enable_m1_porin_diffusion=m1_porin_diffusion_var,
        enable_diffusion_mutation=diffusion_mutation_var,
        homogeneous_initial_diffusion_const=homogeneous_initial_diffusion_const_var,
        homogeneous_population=homogeneous_mode_var,
        independent_traits=independent_traits_var,
        enable_chemostat_flow=enable_chemostat_flow_var,
        enable_initial_energy=enable_initial_energy_var,
        enable_intermediate_costs=enable_intermediate_costs_var,
        enable_acetate_addition=enable_acetate_addition_var,
    )

    last_logged_viz_metric = {"value": _selected_visualization_metric()}

    def _on_visualization_metric_change(*_args):
        metric_name = _selected_visualization_metric()
        if metric_name != last_logged_viz_metric["value"]:
            results_text.insert(
                tk.END,
                f"[Visualization] Metric switched to: {metric_name}\n"
            )
            results_text.see(tk.END)
            last_logged_viz_metric["value"] = metric_name

    viz_metric_var.trace_add("write", _on_visualization_metric_change)

    def _on_visualization_metric2_change(*_args):
        metric_name = viz_metric2_var.get().strip()
        if not metric_name:
            return
        if full_save_lightweight_mode and len(all_param_vectors) > 0:
            try:
                _build_lightweight_metric_cache(metric_name)
            except Exception:
                pass

    viz_metric2_var.trace_add("write", _on_visualization_metric2_change)

    # Optimization state
    optimization_running = False
    optimization_paused = False
    # Stdin "pause" can set this from a background thread so the worker stops even if Tk is wedged.
    stdin_pause_hold = threading.Event()
    grad_terminal_pause_unreg = [None]  # optional unregister from register_terminal_pause_hooks
    optimization_thread = None

    def _refresh_pause_button_from_state() -> None:
        if not optimization_running:
            return
        blocked = optimization_paused or stdin_pause_hold.is_set()
        if blocked:
            pause_button.config(text="Start")
            progress.stop()
            if stdin_pause_hold.is_set() and not optimization_paused:
                progress_label.config(text="Paused (terminal)")
            else:
                progress_label.config(text="Paused")
        else:
            pause_button.config(text="Pause")
            progress.start()
            progress_label.config(text="Running...")

    def _clear_grad_terminal_pause_hooks() -> None:
        unreg = grad_terminal_pause_unreg[0]
        if unreg is not None:
            try:
                unreg()
            except Exception:
                pass
            grad_terminal_pause_unreg[0] = None
    active_run_eval_snapshot = None  # Frozen settings used for all evaluations in a running optimization.
    optimization_history = []
    all_runs_history = []  # Store history for all runs
    all_results = []  # Store results for all runs (for accessing run_index)
    best_run_index = 0  # Index of best run
    all_param_vectors = []  # Store all parameter vectors (ordering matches param_names_list)
    all_metric_values = []  # Store corresponding metric values
    all_replicate_info = []  # Store replicate information: (num_replicates, num_succeeded, num_failed)
    all_metric_inputs = []  # Store cached final simulation outputs per point for no-rerun reanalysis
    all_run_ids = []  # Store stable run IDs that don't change during cleanup
    all_run_indices = []  # Store global run index (0-based) for each data point
    offload_point_sources = []  # Optional per-point source mapping: (batch_path, record_index)
    offload_pending_records = []  # Buffered point records for batched immediate offload writes.
    offload_last_batch_path = None  # Last offload batch path used for lazy metric_input reads
    offload_last_batch_payload = None  # Cached payload of last offload batch
    full_save_lightweight_mode = False  # Keep heavy metric_input payloads on disk when True
    full_save_lightweight_metric_name = None  # Metric currently represented by all_metric_values in lightweight mode
    lightweight_metric_cache = {}  # metric_name -> list[float] for lightweight offload-loaded sessions
    param_names_list = []  # Store parameter names for consistent ordering
    run_id_counter = 0  # Counter for assigning unique run IDs
    run_start_info = {}  # Store start message info for each run (keyed by run_index)
    printed_start_messages = set()  # Track which runs have had their start message printed

    # Lifetime stats for this optimization (NOT affected by point cleanup)
    # - run_summary_stats counts at the RUN level (one entry per descent)
    # - deleted_point_stats tracks metrics of POINTS removed by auto-cleanup (for transparency)
    run_summary_stats = {
        "total_runs": 0,
        "nan_runs": 0,
        "metric_stats": RunningStats(),  # valid (non-NaN) run metrics
    }
    deleted_point_stats = {
        "total_points": 0,
        "nan_points": 0,
        "metric_stats": RunningStats(),  # valid (non-NaN) point metrics that were deleted
    }
    
    # Memory management settings
    MAX_STORED_POINTS = 10000  # Keep larger in-memory windows to reduce tiny offload batches.
    CLEANUP_THRESHOLD = 12000  # Trigger cleanup only after meaningful accumulation.
    STORAGE_SAMPLE_RATE = 5  # Store every Nth iteration (1 = all, 5 = every 5th)
    # For large runs, cleanup-driven offload is usually faster than immediate per-sampled-point offload.
    # Immediate mode can still be enabled for crash-resilience-focused workflows.
    OFFLOAD_EACH_POINT = False  # Persist primarily during cleanup/snapshot, not every sampled point.
    OFFLOAD_IMMEDIATE_FLUSH_RECORDS = 250  # Buffer size if immediate mode is enabled.
    ITERATION_LOG_FREQUENCY = 25  # Print/UI-update every N iterations to avoid callback flooding
    LOG_PER_RUN_UPDATES = False  # For large jobs, avoid per-run/per-iteration text spam.
    iteration_counter = 0  # Track iterations for sampling
    full_save_session_id = time.strftime("%Y%m%d-%H%M%S")
    full_save_batch_counter = 0
    full_save_offloaded_points = 0
    full_save_final_snapshot_written = False
    full_save_manifest_cache = None
    full_save_snapshot_in_progress = False
    full_save_load_in_progress = False
    full_save_load_cancel_requested = False
    loaded_full_save_session_id = None
    loaded_full_save_settings_snapshot = None
    cleanup_scheduled = False

    def _manifest_path():
        folder = full_save_dir_var.get().strip()
        if not folder:
            return None
        return full_save_manifest_path_json(folder, full_save_session_id)

    def _load_manifest():
        nonlocal full_save_manifest_cache
        if full_save_manifest_cache is not None:
            return full_save_manifest_cache
        folder = full_save_dir_var.get().strip()
        if not folder:
            full_save_manifest_cache = {"session_id": full_save_session_id, "batches": []}
            return full_save_manifest_cache
        full_save_manifest_cache = load_full_save_manifest(
            folder,
            full_save_session_id,
            _read_json_maybe_gz,
        )
        return full_save_manifest_cache

    def _save_manifest():
        manifest = _load_manifest()
        mp = _manifest_path()
        if not mp:
            return
        try:
            _write_json_maybe_gz(mp, manifest)
        except Exception:
            pass

    def _session_settings_path(session_id=None):
        folder = full_save_dir_var.get().strip()
        if not folder:
            return None
        sid = session_id or full_save_session_id
        if not sid:
            return None
        return full_save_settings_path_json(folder, sid)

    def _build_current_settings_snapshot():
        """Capture current GUI settings so full-save session loads can fully restore state."""
        ui_state = {
            "metric_name": metric_var.get(),
            "visualization_metric_name": _selected_visualization_metric(),
            "visualization_metric_name_2": viz_metric2_var.get(),
            "optimization_goal": opt_goal_var.get(),
            "learning_rate": learning_rate_entry.get(),
            "max_iterations": max_iterations_entry.get(),
            "convergence_threshold": convergence_threshold_entry.get(),
            "gradient_step": gradient_step_entry.get(),
            "num_starts": num_starts_entry.get(),
            "descents_per_start": descents_per_start_entry.get(),
            "storage_sample_rate": sample_rate_entry.get(),
            "num_replicates": replicates_entry.get(),
            "seed": seed_entry.get(),
            "silent_mode": bool(silent_mode_var.get()),
            "homogeneous_population": bool(homogeneous_mode_var.get()),
            "independent_traits": bool(independent_traits_var.get()),
            "enable_m1_diffusion": bool(m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get()),
            "enable_m2_diffusion": bool(m2_diffusion_var.get()),
            "enable_m1_porin_diffusion": bool(m1_porin_diffusion_var.get()),
            "enable_diffusion_mutation": bool(diffusion_mutation_var.get()),
            "homogeneous_initial_diffusion_const": bool(homogeneous_initial_diffusion_const_var.get()),
            "enable_chemostat_flow": bool(enable_chemostat_flow_var.get()),
            "enable_initial_energy": bool(enable_initial_energy_var.get()),
            "enable_intermediate_costs": bool(enable_intermediate_costs_var.get()),
            "enable_acetate_addition": bool(enable_acetate_addition_var.get()),
            "binary_death_at_zero_energy": bool(binary_death_at_zero_energy_var.get()),
            "no_death": bool(no_death_var.get()),
            "constant_death_probability": bool(constant_death_probability_var.get()),
            "constant_duplication_probability": bool(constant_duplication_probability_var.get()),
            "full_save_folder": full_save_dir_var.get(),
        }
        if getattr(model_spec, "key", "") in {"simulation"}:
            try:
                ui_state["enable_m1_facilitated_diffusion"] = bool(m1_facilitation_var.get())
            except Exception:
                pass

        param_ui_state = {}
        for pname in param_names:
            try:
                param_ui_state[pname] = {
                    "fixed": bool(param_fix_checkboxes[pname].get()),
                    "initial": param_initial_entries[pname].get() if pname in param_initial_entries else "",
                    "min": param_min_entries[pname].get() if pname in param_min_entries else "",
                    "max": param_max_entries[pname].get() if pname in param_max_entries else "",
                }
            except Exception:
                pass

        fixed_param_values = {}
        try:
            for pname, entry in fixed_entries.items():
                fixed_param_values[pname] = entry.get()
        except Exception:
            pass

        return {
            "ui_state": ui_state,
            "param_ui_state": param_ui_state,
            "fixed_param_values": fixed_param_values,
            "model": {"key": model_spec.key, "label": model_spec.label},
            "saved_at_epoch": int(time.time()),
        }

    def _save_full_save_session_settings(session_id=None):
        sp = _session_settings_path(session_id=session_id)
        if not sp:
            return False
        try:
            _write_json_maybe_gz(sp, _build_current_settings_snapshot())
            return True
        except Exception:
            return False

    def _apply_settings_snapshot(snapshot):
        """Apply ui_state/param_ui_state/fixed_param_values snapshot to current GUI."""
        if not isinstance(snapshot, dict):
            return

        ui_state = snapshot.get("ui_state", {})
        if isinstance(ui_state, dict) and ui_state:
            def _set_entry(ent, val):
                try:
                    ent.delete(0, tk.END)
                    ent.insert(0, str(val))
                except Exception:
                    pass

            # Optimization settings
            if "metric_name" in ui_state:
                try:
                    metric_var.set(ui_state["metric_name"])
                except Exception:
                    pass
            if "visualization_metric_name" in ui_state:
                try:
                    viz_metric_var.set(ui_state["visualization_metric_name"])
                except Exception:
                    pass
            if "visualization_metric_name_2" in ui_state:
                try:
                    viz_metric2_var.set(ui_state["visualization_metric_name_2"])
                except Exception:
                    pass
            if "optimization_goal" in ui_state:
                try:
                    opt_goal_var.set(ui_state["optimization_goal"])
                except Exception:
                    pass
            if "learning_rate" in ui_state:
                _set_entry(learning_rate_entry, ui_state["learning_rate"])
            if "max_iterations" in ui_state:
                _set_entry(max_iterations_entry, ui_state["max_iterations"])
            if "convergence_threshold" in ui_state:
                _set_entry(convergence_threshold_entry, ui_state["convergence_threshold"])
            if "gradient_step" in ui_state:
                _set_entry(gradient_step_entry, ui_state["gradient_step"])
            if "num_starts" in ui_state:
                _set_entry(num_starts_entry, ui_state["num_starts"])
            if "descents_per_start" in ui_state:
                _set_entry(descents_per_start_entry, ui_state["descents_per_start"])
            if "storage_sample_rate" in ui_state:
                _set_entry(sample_rate_entry, ui_state["storage_sample_rate"])
            if "num_replicates" in ui_state:
                _set_entry(replicates_entry, ui_state["num_replicates"])
            if "seed" in ui_state:
                _set_entry(seed_entry, ui_state["seed"])

            if "full_save_folder" in ui_state:
                try:
                    full_save_dir_var.set(str(ui_state["full_save_folder"]))
                except Exception:
                    pass

            # Simulation toggles
            if "silent_mode" in ui_state:
                try:
                    silent_mode_var.set(bool(ui_state["silent_mode"]))
                except Exception:
                    pass
            if "homogeneous_population" in ui_state:
                try:
                    homogeneous_mode_var.set(bool(ui_state["homogeneous_population"]))
                except Exception:
                    pass
            if "independent_traits" in ui_state:
                try:
                    independent_traits_var.set(bool(ui_state["independent_traits"]))
                except Exception:
                    pass
            if "enable_m1_diffusion" in ui_state:
                try:
                    m1_diffusion_var.set(bool(ui_state["enable_m1_diffusion"]))
                except Exception:
                    pass
            if "enable_m1_facilitated_diffusion" in ui_state:
                try:
                    m1_facilitation_var.set(bool(ui_state["enable_m1_facilitated_diffusion"]))
                except Exception:
                    pass
            if "enable_m1_porin_diffusion" in ui_state:
                try:
                    m1_porin_diffusion_var.set(bool(ui_state["enable_m1_porin_diffusion"]))
                except Exception:
                    pass
            if "enable_m2_diffusion" in ui_state:
                try:
                    m2_diffusion_var.set(bool(ui_state["enable_m2_diffusion"]))
                except Exception:
                    pass
            if "enable_diffusion_mutation" in ui_state:
                try:
                    diffusion_mutation_var.set(bool(ui_state["enable_diffusion_mutation"]))
                except Exception:
                    pass
            if "homogeneous_initial_diffusion_const" in ui_state:
                try:
                    homogeneous_initial_diffusion_const_var.set(bool(ui_state["homogeneous_initial_diffusion_const"]))
                except Exception:
                    pass
            if "enable_chemostat_flow" in ui_state:
                try:
                    enable_chemostat_flow_var.set(bool(ui_state["enable_chemostat_flow"]))
                except Exception:
                    pass
            if "enable_initial_energy" in ui_state:
                try:
                    enable_initial_energy_var.set(bool(ui_state["enable_initial_energy"]))
                except Exception:
                    pass
            if "enable_intermediate_costs" in ui_state:
                try:
                    enable_intermediate_costs_var.set(bool(ui_state["enable_intermediate_costs"]))
                except Exception:
                    pass
            if "enable_acetate_addition" in ui_state:
                try:
                    enable_acetate_addition_var.set(bool(ui_state["enable_acetate_addition"]))
                except Exception:
                    pass
            if "binary_death_at_zero_energy" in ui_state:
                try:
                    binary_death_at_zero_energy_var.set(bool(ui_state["binary_death_at_zero_energy"]))
                except Exception:
                    pass
            if "no_death" in ui_state:
                try:
                    no_death_var.set(bool(ui_state["no_death"]))
                except Exception:
                    pass
            if "constant_death_probability" in ui_state:
                try:
                    constant_death_probability_var.set(bool(ui_state["constant_death_probability"]))
                except tk.TclError:
                    pass
            if "constant_duplication_probability" in ui_state:
                try:
                    constant_duplication_probability_var.set(bool(ui_state["constant_duplication_probability"]))
                except Exception:
                    pass
            refresh_diffusion_mutation_state()

        # Parameter checkbox/initial/min/max state
        param_ui_state = snapshot.get("param_ui_state", {})
        if isinstance(param_ui_state, dict) and param_ui_state:
            for pname, st in param_ui_state.items():
                if pname not in param_fix_checkboxes:
                    continue
                if isinstance(st, dict):
                    try:
                        if "fixed" in st:
                            param_fix_checkboxes[pname].set(bool(st["fixed"]))
                    except Exception:
                        pass
                    if "initial" in st and pname in param_initial_entries:
                        try:
                            param_initial_entries[pname].delete(0, tk.END)
                            param_initial_entries[pname].insert(0, str(st["initial"]))
                        except Exception:
                            pass
                    if "min" in st and pname in param_min_entries:
                        try:
                            param_min_entries[pname].delete(0, tk.END)
                            param_min_entries[pname].insert(0, str(st["min"]))
                        except Exception:
                            pass
                    if "max" in st and pname in param_max_entries:
                        try:
                            param_max_entries[pname].delete(0, tk.END)
                            param_max_entries[pname].insert(0, str(st["max"]))
                        except Exception:
                            pass
            refresh_optimizable_params()
            refresh_fixed_params()

        fixed_param_values = snapshot.get("fixed_param_values", {})
        if isinstance(fixed_param_values, dict) and fixed_param_values:
            for pname, val in fixed_param_values.items():
                if pname in fixed_entries:
                    try:
                        fixed_entries[pname].delete(0, tk.END)
                        fixed_entries[pname].insert(0, str(val))
                    except Exception:
                        pass

        # Ensure dependent UI states are refreshed.
        try:
            refresh_metric_options()
        except Exception:
            pass
        try:
            _refresh_chemostat_flow_state(preserve_fix_checkbox=True)
        except Exception:
            pass
        try:
            _refresh_initial_energy_state(preserve_fix_checkbox=True)
        except Exception:
            pass
        try:
            _refresh_intermediate_costs_state(preserve_fix_checkbox=True)
        except Exception:
            pass
        try:
            _refresh_acetate_addition_state(preserve_fix_checkbox=True)
        except Exception:
            pass
        try:
            _refresh_death_dup_constant_params_state(preserve_fix_checkbox=True)
        except Exception:
            pass

    def _extract_batch_index_from_filename(path):
        """Parse trailing batch index from offload file path."""
        session, idx = parse_offload_filename(path)
        if not session or session != full_save_session_id:
            return None
        return idx

    def _offload_path_sort_key(path):
        """Sort offload files by parsed numeric batch index when available."""
        session, idx = parse_offload_filename(path)
        try:
            idx_int = int(idx) if idx is not None else None
        except Exception:
            idx_int = None
        return (
            0 if idx_int is not None else 1,
            idx_int if idx_int is not None else 10**12,
            str(session or ""),
            os.path.basename(str(path or "")),
        )

    def _list_offload_batch_files(folder, only_current_session=False):
        """List both .json and .json.gz offload files, optionally scoped to this session."""
        session_filter = full_save_session_id if only_current_session else None
        return list_offload_batch_files_shared(folder, session_id=session_filter, sort_by_index=True)

    def _build_offload_records(indices):
        records = []
        metric_name = metric_var.get()
        for i in indices:
            metric_val = all_metric_values[i] if i < len(all_metric_values) else np.nan
            metric_scalar = None if np.isnan(metric_val) else float(metric_val)
            metric_input_payload = all_metric_inputs[i] if i < len(all_metric_inputs) else None
            if isinstance(metric_input_payload, dict):
                metric_map = metric_input_payload.get("metric_values_by_name")
                if not isinstance(metric_map, dict):
                    metric_map = {}
                    metric_input_payload["metric_values_by_name"] = metric_map
                metric_map[metric_name] = metric_scalar
            rec = {
                "param_vector": list(all_param_vectors[i]) if i < len(all_param_vectors) else [],
                "metric_value": metric_scalar,
                "replicate_info": list(all_replicate_info[i]) if i < len(all_replicate_info) else [],
                "metric_input": metric_input_payload,
                "run_id": int(all_run_ids[i]) if i < len(all_run_ids) and all_run_ids[i] is not None else None,
                "run_index": int(all_run_indices[i]) if i < len(all_run_indices) and all_run_indices[i] is not None else None,
            }
            records.append(rec)
        return records

    def _write_offload_records_batch(records):
        """Write one offload batch and update manifest/counters."""
        nonlocal full_save_batch_counter, full_save_offloaded_points
        if not records:
            return 0, ""
        folder = full_save_dir_var.get().strip()
        if not folder:
            return 0, ""
        os.makedirs(folder, exist_ok=True)
        filename = f"offload_{full_save_session_id}_{full_save_batch_counter:05d}.json.gz"
        batch_path = os.path.join(folder, filename)
        payload = {
            "model_key": model_spec.key,
            "session_id": full_save_session_id,
            "batch_index": int(full_save_batch_counter),
            "metric_name_at_offload": metric_var.get(),
            "param_names_list": list(param_names_list),
            "records": records,
        }
        _write_json_maybe_gz(batch_path, payload)
        metric_name = metric_var.get()
        cached_points = 0
        for rec in records:
            mi = rec.get("metric_input")
            if isinstance(mi, dict):
                metric_map = mi.get("metric_values_by_name")
                if isinstance(metric_map, dict) and metric_name in metric_map:
                    cached_points += 1
        manifest = _load_manifest()
        manifest["batches"].append({
            "batch_index": int(full_save_batch_counter),
            "path": os.path.basename(batch_path),
            "num_records": len(records),
            "cached_metric_counts": {metric_name: int(cached_points)},
            "updated_at_epoch": int(time.time()),
        })
        _save_manifest()
        current_batch_index = int(full_save_batch_counter)
        full_save_batch_counter += 1
        full_save_offloaded_points += len(records)
        offload_summary = (
            f"\n[Full Save] Offloaded batch {current_batch_index + 1}: "
            f"{len(records)} points saved (total offloaded: {full_save_offloaded_points}).\n"
        )
        print(offload_summary.strip())
        try:
            win.after(0, lambda m=offload_summary: (results_text.insert(tk.END, m), results_text.see(tk.END)))
        except Exception:
            pass
        return len(records), batch_path

    def _offload_points_to_disk(indices_to_remove):
        """Persist points selected for cleanup so they can be reanalyzed later."""
        if not indices_to_remove:
            return 0, ""
        keep_sorted = sorted(indices_to_remove)
        records = _build_offload_records(keep_sorted)
        return _write_offload_records_batch(records)

    def _flush_pending_offload_records(force=False):
        """Flush buffered immediate offload records as one larger batch."""
        nonlocal offload_pending_records
        if not offload_pending_records:
            return 0
        if (not force) and len(offload_pending_records) < OFFLOAD_IMMEDIATE_FLUSH_RECORDS:
            return 0
        records = list(offload_pending_records)
        try:
            saved_count, _ = _write_offload_records_batch(records)
        except Exception:
            return 0
        offload_pending_records = []
        return int(saved_count)

    def _offload_indices_immediately(indices_to_offload):
        """Persist selected point indices now, then drop them from in-memory point arrays."""
        nonlocal all_param_vectors, all_metric_values, all_replicate_info, all_metric_inputs, all_run_ids, all_run_indices, offload_point_sources, offload_pending_records
        if not indices_to_offload:
            return 0
        keep = sorted(i for i in set(indices_to_offload) if 0 <= i < len(all_param_vectors))
        if not keep:
            return 0
        offload_pending_records.extend(_build_offload_records(keep))
        saved_count = _flush_pending_offload_records(force=False)
        keep_set = set(keep)
        remain_idx = [i for i in range(len(all_param_vectors)) if i not in keep_set]
        all_param_vectors = [all_param_vectors[i] for i in remain_idx]
        all_metric_values = [all_metric_values[i] for i in remain_idx]
        all_replicate_info = [all_replicate_info[i] for i in remain_idx]
        all_metric_inputs = [all_metric_inputs[i] for i in remain_idx]
        all_run_ids = [all_run_ids[i] for i in remain_idx]
        all_run_indices = [all_run_indices[i] for i in remain_idx]
        if len(offload_point_sources) == (len(remain_idx) + len(keep_set)):
            offload_point_sources = [offload_point_sources[i] for i in remain_idx]
        else:
            offload_point_sources = [None] * len(all_param_vectors)
        return int(saved_count if saved_count > 0 else len(keep))

    def _collect_offloaded_records_for_session():
        """Read all offloaded point batches for the current full-save session."""
        _flush_pending_offload_records(force=True)
        folder = full_save_dir_var.get().strip()
        if not folder or not os.path.isdir(folder):
            return []
        batch_files = _list_offload_batch_files(folder, only_current_session=True)
        records = []
        for path in batch_files:
            try:
                payload = _read_json_maybe_gz(path)
                recs = payload.get("records", [])
                if isinstance(recs, list):
                    records.extend(recs)
            except Exception:
                continue
        return records

    def _write_full_save_dataset_snapshot():
        """Write a loadable dataset JSON containing all points for this full-save session."""
        nonlocal full_save_final_snapshot_written
        if full_save_final_snapshot_written:
            return None
        if len(all_results) == 0:
            return None

        folder = _ensure_full_save_folder(interactive=False)
        if not folder:
            return None

        _flush_pending_offload_records(force=True)
        # Persist remaining in-memory points (the points not yet auto-cleaned away).
        if len(all_param_vectors) > 0:
            _offload_points_to_disk(set(range(len(all_param_vectors))))

        records = _collect_offloaded_records_for_session()
        if not records:
            return None

        all_param_vectors_full = []
        all_metric_values_full = []
        all_replicate_info_full = []
        all_metric_inputs_full = []
        all_run_ids_full = []
        all_run_indices_full = []

        for ridx, rec in enumerate(records):
            vec = rec.get("param_vector", [])
            mv = rec.get("metric_value", None)
            rep = rec.get("replicate_info", {})
            run_id = rec.get("run_id", None)
            run_index = rec.get("run_index", None)
            all_param_vectors_full.append(vec if isinstance(vec, list) else list(vec))
            all_metric_values_full.append(np.nan if mv is None else float(mv))
            all_replicate_info_full.append(rep)
            all_metric_inputs_full.append(rec.get("metric_input", None))
            all_run_ids_full.append(int(run_id) if run_id is not None else ridx)
            all_run_indices_full.append(int(run_index) if run_index is not None else ridx)

        dataset = _build_dataset_payload(
            all_results,
            all_runs_history,
            all_param_vectors_full,
            all_metric_values_full,
            all_replicate_info_full,
            all_metric_inputs_full,
            all_run_ids_full,
            all_run_indices_full,
        )
        dataset.setdefault("metadata", {})
        dataset["metadata"]["full_save_session_id"] = full_save_session_id
        dataset["metadata"]["full_save_batches"] = int(full_save_batch_counter)
        dataset["metadata"]["full_save_points"] = len(all_param_vectors_full)

        out_path = full_save_dataset_snapshot_path(folder, full_save_session_id)
        _write_json_maybe_gz(out_path, dataset)
        full_save_final_snapshot_written = True
        return out_path

    def _write_full_save_dataset_snapshot_async():
        """Write full-save dataset snapshot in background to keep UI responsive."""
        nonlocal full_save_snapshot_in_progress
        if full_save_snapshot_in_progress:
            return False
        full_save_snapshot_in_progress = True
        run_button.config(state="disabled")

        def _worker():
            nonlocal full_save_snapshot_in_progress
            try:
                full_dataset_path = _write_full_save_dataset_snapshot()
                if full_dataset_path:
                    win.after(
                        0,
                        lambda p=full_dataset_path: (
                            results_text.insert(tk.END, f"\n[Full Save] Wrote loadable full dataset:\n{p}\n"),
                            results_text.see(tk.END),
                        ),
                    )
            except Exception as e:
                win.after(
                    0,
                    lambda msg=str(e): (
                        results_text.insert(tk.END, f"\n[Full Save] Final dataset write failed: {msg}\n"),
                        results_text.see(tk.END),
                    ),
                )
            finally:
                full_save_snapshot_in_progress = False
                try:
                    if not optimization_running:
                        win.after(0, lambda: run_button.config(state="normal"))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()
        return True
    
    def cleanup_old_data():
        """Remove data points to manage memory usage, always removing ALL NaN points first."""
        nonlocal all_param_vectors, all_metric_values, all_replicate_info, all_metric_inputs, all_runs_history, all_run_ids, all_run_indices
        nonlocal offload_point_sources

        # These arrays are appended from the worker thread; align them defensively first
        # so cleanup can never index past the shortest list.
        n_points = _aligned_point_count(include_metric_inputs=True)
        if n_points <= 0:
            return
        if n_points != len(all_param_vectors):
            all_param_vectors = all_param_vectors[:n_points]
            all_metric_values = all_metric_values[:n_points]
            all_replicate_info = all_replicate_info[:n_points]
            all_metric_inputs = all_metric_inputs[:n_points]
            all_run_ids = all_run_ids[:n_points]
            all_run_indices = all_run_indices[:n_points]

        if n_points <= MAX_STORED_POINTS:
            return  # No cleanup needed
        
        print(f"[Memory Management] Starting cleanup: {len(all_param_vectors)} points stored (threshold: {MAX_STORED_POINTS})")
        
        # Strategy: 
        # 1. ALWAYS remove ALL NaN points first (never keep or protect NaN points)
        # 2. Never remove top 5 valid points (based on metric)
        # 3. Only remove other valid points if necessary after removing all NaN points
        
        # Determine optimization goal
        maximize = opt_goal_var.get() == "Maximize"
        
        # Find indices of top 5 VALID points (protected from deletion)
        # NaN points are NEVER protected
        valid_metrics_with_indices = [(i, metric) for i, metric in enumerate(all_metric_values) if not np.isnan(metric)]
        
        if len(valid_metrics_with_indices) >= 5:
            # Sort by metric value
            if maximize:
                sorted_valid = sorted(valid_metrics_with_indices, key=lambda x: x[1], reverse=True)
            else:
                sorted_valid = sorted(valid_metrics_with_indices, key=lambda x: x[1])
            
            top_5_indices = set([idx for idx, _ in sorted_valid[:5]])
        else:
            # If fewer than 5 valid points, protect all valid points
            top_5_indices = set([idx for idx, _ in valid_metrics_with_indices])
        
        # Find ALL NaN indices (these will ALWAYS be removed)
        nan_indices = [i for i, metric in enumerate(all_metric_values) if np.isnan(metric)]
        
        # Find valid indices (excluding top 5 protected points)
        valid_indices = []
        for i, metric in enumerate(all_metric_values):
            if not np.isnan(metric) and i not in top_5_indices:
                valid_indices.append(i)
        
        num_nan = len(nan_indices)
        removed_nan = num_nan  # Always remove ALL NaN points
        
        # After removing all NaN points, check if we still need to remove more
        points_after_nan_removal = len(all_param_vectors) - num_nan
        num_still_to_remove = max(0, points_after_nan_removal - MAX_STORED_POINTS)
        
        # Determine which valid points to remove (oldest first, excluding top 5)
        if num_still_to_remove > 0:
            # Remove oldest valid points (excluding top 5)
            valid_indices_to_remove = set(valid_indices[:num_still_to_remove])
            removed_valid = num_still_to_remove
        else:
            valid_indices_to_remove = set()
            removed_valid = 0
        
        # Combine all indices to remove (ALL NaN + some valid if needed)
        indices_to_remove = set(nan_indices) | valid_indices_to_remove

        # Track metrics of deleted points so end-of-run summaries remain accurate/transparent
        if indices_to_remove:
            removed_metrics = [all_metric_values[i] for i in indices_to_remove]
            deleted_point_stats["total_points"] += len(removed_metrics)
            for mv in removed_metrics:
                if np.isnan(mv):
                    deleted_point_stats["nan_points"] += 1
                else:
                    deleted_point_stats["metric_stats"].add(mv)

        # Optional disk offload of removed points before dropping from memory.
        offload_msg = ""
        if indices_to_remove:
            try:
                saved_count, saved_path = _offload_points_to_disk(indices_to_remove)
                if saved_count > 0:
                    offload_msg = f" Offloaded {saved_count} points to {saved_path}."
            except Exception as e:
                offload_msg = f" Full Save offload failed: {e}."
        
        indices_to_keep = [i for i in range(len(all_param_vectors)) if i not in indices_to_remove]
        
        # Keep only selected indices (including run IDs for stable numbering)
        all_param_vectors = [all_param_vectors[i] for i in indices_to_keep]
        all_metric_values = [all_metric_values[i] for i in indices_to_keep]
        all_replicate_info = [all_replicate_info[i] for i in indices_to_keep]
        all_metric_inputs = [all_metric_inputs[i] for i in indices_to_keep]
        all_run_ids = [all_run_ids[i] for i in indices_to_keep]
        all_run_indices = [all_run_indices[i] for i in indices_to_keep]
        if len(offload_point_sources) == len(indices_to_keep) + len(indices_to_remove):
            offload_point_sources = [offload_point_sources[i] for i in indices_to_keep]
        else:
            offload_point_sources = [None] * len(all_param_vectors)
        
        # Also trim run histories to avoid storing too much gradient/param data
        # Keep only the last 50 runs worth of history
        if len(all_runs_history) > 50:
            all_runs_history = all_runs_history[-50:]
        
        # Count remaining points after cleanup
        remaining_valid = len([m for m in all_metric_values if not np.isnan(m)])
        remaining_nan = len([m for m in all_metric_values if np.isnan(m)])
        
        cleanup_message = (f"\n[Memory Management] Removed {removed_nan + removed_valid} points "
            f"({removed_nan} NaN, {removed_valid} valid). Protected top 5 valid points. "
            f"Keeping {len(all_param_vectors)} points ({remaining_valid} valid, {remaining_nan} NaN)."
            f"{offload_msg}\n")
        
        # Print to terminal
        print(cleanup_message.strip())
        
        # Also insert into GUI text widget
        results_text.insert(tk.END, cleanup_message)

    def _schedule_cleanup():
        """Schedule at most one pending cleanup on the Tk thread."""
        nonlocal cleanup_scheduled
        if cleanup_scheduled:
            return
        cleanup_scheduled = True

        def _run_cleanup():
            nonlocal cleanup_scheduled
            try:
                cleanup_old_data()
            finally:
                cleanup_scheduled = False

        win.after(0, _run_cleanup)
    
    
    def go_back():
        """Return to launcher."""
        # Stop any running optimization first
        nonlocal optimization_running
        if optimization_running:
            stop_optimization()
            # Wait a moment for threads to finish
            win.after(200, lambda: _finish_close())
        else:
            _finish_close()
    
    def _finish_close():
        """Complete the window closing process."""
        root.deiconify()
        win.destroy()
    
    def on_closing():
        """Handle window close event (X button) - clean up resources."""
        nonlocal optimization_running
        if optimization_running:
            # Ask user if they want to stop
            response = messagebox.askyesno(
                "Optimization Running",
                "An optimization is currently running. Do you want to stop it and close?"
            )
            if response:
                go_back()
        else:
            go_back()
    
    # Set the window close handler for X button
    win.protocol("WM_DELETE_WINDOW", on_closing)

    def _capture_run_evaluation_snapshot():
        """Capture immutable GUI-driven settings for one optimization run."""
        fixed_params = {}
        for param_name, entry in fixed_entries.items():
            try:
                fixed_params[param_name] = float(entry.get())
            except ValueError:
                fixed_params[param_name] = entry.get()

        snap = {
            "metric_name": str(metric_var.get()),
            "optimization_goal": str(opt_goal_var.get()),
            "requested_verbose": (not bool(silent_mode_var.get())),
            "homogeneous_population": bool(homogeneous_mode_var.get()),
            "independent_traits": bool(independent_traits_var.get()),
            "enable_initial_energy": bool(enable_initial_energy_var.get()),
            "enable_chemostat_flow": bool(enable_chemostat_flow_var.get()),
            "enable_intermediate_costs": bool(enable_intermediate_costs_var.get()),
            "enable_acetate_addition": bool(enable_acetate_addition_var.get()),
            "binary_death_at_zero_energy": bool(binary_death_at_zero_energy_var.get()),
            "no_death": bool(no_death_var.get()),
            "constant_death_probability": bool(constant_death_probability_var.get()),
            "constant_duplication_probability": bool(constant_duplication_probability_var.get()),
            "fixed_params": fixed_params,
            "model_key": str(getattr(model_spec, "key", "")),
        }
        try:
            snap["initial_b_fallback"] = float(param_initial_entries["Initial B"].get())
        except Exception:
            snap["initial_b_fallback"] = float(default_params.get("Initial B", 0.5))

        if snap["model_key"] == "simulation":
            snap["enable_m1_diffusion"] = bool(m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get())
            snap["enable_m2_diffusion"] = bool(m2_diffusion_var.get())
            snap["enable_m1_facilitated_diffusion"] = bool(m1_facilitation_var.get())
            snap["enable_m1_porin_diffusion"] = bool(m1_porin_diffusion_var.get()) if snap["enable_m1_diffusion"] else False
            allow_diffusion_mutation = bool(
                m2_diffusion_var.get() or m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get()
            )
            snap["enable_diffusion_mutation"] = bool(diffusion_mutation_var.get()) if allow_diffusion_mutation else False
            snap["homogeneous_initial_diffusion_const"] = (
                bool(homogeneous_initial_diffusion_const_var.get()) if snap["enable_diffusion_mutation"] else False
            )
            snap["store_history"] = False
        return snap
    
    def evaluate_metric(params_dict, optimized_params_list, param_bounds_dict, seed=None, num_replicates=1):
        """Evaluate the metric for given parameters."""
        snapshot = active_run_eval_snapshot if (optimization_running and isinstance(active_run_eval_snapshot, dict)) else None
        
        # Build full parameter dictionary
        if snapshot:
            full_params = dict(snapshot.get("fixed_params", {}))
        else:
            full_params = {}
            # Add fixed parameters
            for param_name, entry in fixed_entries.items():
                try:
                    full_params[param_name] = float(entry.get())
                except ValueError:
                    full_params[param_name] = entry.get()
        
        # Add optimized parameters with bounds enforcement
        for param_name in optimized_params_list:
            if param_name in params_dict:
                param_value = params_dict[param_name]
                
                # Enforce bounds before using parameter
                if param_name in param_bounds_dict:
                    min_val, max_val = param_bounds_dict[param_name]
                    if min_val == -np.inf:
                        # Only enforce upper bound
                        param_value = min(param_value, max_val)
                    else:
                        # Enforce both bounds
                        param_value = np.clip(param_value, min_val, max_val)
                    param_value = float(param_value)  # Ensure Python float
                
                full_params[param_name] = param_value
        
        if getattr(model_spec, "key", "") == "simulation":
            full_params["Enable M1 Diffusion"] = bool(snapshot.get("enable_m1_diffusion")) if snapshot else bool(m1_diffusion_var.get() or m1_facilitation_var.get() or m1_porin_diffusion_var.get())
            full_params["Enable M2 Diffusion"] = bool(snapshot.get("enable_m2_diffusion")) if snapshot else bool(m2_diffusion_var.get())
            full_params["Enable M1 Facilitated Diffusion"] = bool(snapshot.get("enable_m1_facilitated_diffusion")) if snapshot else bool(m1_facilitation_var.get())
            full_params["Enable M1 Porin Diffusion"] = (
                bool(snapshot.get("enable_m1_porin_diffusion", False))
                if snapshot else bool(m1_porin_diffusion_var.get())
            )
            if full_params["Enable M1 Facilitated Diffusion"]:
                full_params["Enable M1 Porin Diffusion"] = False
            if full_params["Enable M1 Porin Diffusion"]:
                full_params["Enable M1 Diffusion"] = True
            allow_diffusion_mutation = bool(full_params["Enable M2 Diffusion"] or full_params["Enable M1 Diffusion"])
            if snapshot:
                full_params["Enable Diffusion Mutation"] = bool(snapshot.get("enable_diffusion_mutation")) if allow_diffusion_mutation else False
                full_params["Homogeneous Initial Diffusion Const."] = (
                    bool(snapshot.get("homogeneous_initial_diffusion_const")) if full_params["Enable Diffusion Mutation"] else False
                )
            else:
                full_params["Enable Diffusion Mutation"] = bool(diffusion_mutation_var.get()) if allow_diffusion_mutation else False
                full_params["Homogeneous Initial Diffusion Const."] = (
                    bool(homogeneous_initial_diffusion_const_var.get()) if full_params["Enable Diffusion Mutation"] else False
                )
        full_params["Homogeneous Population"] = bool(snapshot.get("homogeneous_population")) if snapshot else bool(homogeneous_mode_var.get())
        full_params["Independent Traits"] = bool(snapshot.get("independent_traits")) if snapshot else bool(independent_traits_var.get())
        if full_params["Independent Traits"] and full_params["Homogeneous Population"]:
            if "Initial B" not in full_params:
                if snapshot:
                    full_params["Initial B"] = float(snapshot.get("initial_b_fallback", default_params.get("Initial B", 0.5)))
                else:
                    try:
                        full_params["Initial B"] = float(param_initial_entries["Initial B"].get())
                    except Exception:
                        full_params["Initial B"] = float(default_params.get("Initial B", 0.5))
        
        # Add silent mode setting
        requested_verbose = bool(snapshot.get("requested_verbose")) if snapshot else (not bool(silent_mode_var.get()))

        # IMPORTANT:
        # - When verbose (silent off), printing from many simulations/replicates can overwhelm stdout and
        #   make the program look "stuck" even though it's just spending time printing.
        # - Additionally, parallel replicates will interleave their progress prints.
        #
        # So: if verbose is requested, force replicates to run sequentially and (by default) reduce
        # the replicate count to 1 during optimization to avoid massive output.
        if requested_verbose and optimization_running and num_replicates > 1:
            num_replicates = 1

        full_params["silent"] = not requested_verbose

        # Initial energy supply:
        # - Toggle controls whether the feature is enabled at all.
        # - Numeric value should come from the parameter system (fixed_entries / optimized params)
        #   when present, otherwise fall back to the Simulation Settings entry.
        full_params["Enable Initial Energy"] = bool(snapshot.get("enable_initial_energy")) if snapshot else bool(enable_initial_energy_var.get())
        if not full_params["Enable Initial Energy"]:
            full_params["Initial Energy"] = 0.0
        else:
            if "Initial Energy" not in full_params:
                full_params["Initial Energy"] = 0.0

        # Chemostat flow:
        # - Toggle controls whether the feature is enabled at all.
        # - Numeric value should come from the parameter system when present.
        full_params["Enable Chemostat Flow"] = bool(snapshot.get("enable_chemostat_flow")) if snapshot else bool(enable_chemostat_flow_var.get())
        if not full_params["Enable Chemostat Flow"]:
            full_params["Flow Percentage"] = 0.0
        else:
            if "Flow Percentage" not in full_params:
                full_params["Flow Percentage"] = 0.0

        # Intermediate costs:
        # - Toggle controls whether storage penalty is active.
        # - Numeric value should come from parameter system when present.
        full_params["Enable Intermediate Costs"] = bool(snapshot.get("enable_intermediate_costs")) if snapshot else bool(enable_intermediate_costs_var.get())
        if not full_params["Enable Intermediate Costs"]:
            full_params["Intermediate Costs"] = 0.0
        else:
            if "Intermediate Costs" not in full_params:
                full_params["Intermediate Costs"] = 0.0

        # Optional acetate addition each generation.
        full_params["Enable Acetate Addition"] = bool(snapshot.get("enable_acetate_addition")) if snapshot else bool(enable_acetate_addition_var.get())
        if not full_params["Enable Acetate Addition"]:
            full_params["Average In_Flow (Acetate)"] = 0.0
        else:
            if "Average In_Flow (Acetate)" not in full_params:
                full_params["Average In_Flow (Acetate)"] = 0.0

        full_params["Binary Death at Zero Energy"] = (
            bool(snapshot.get("binary_death_at_zero_energy"))
            if snapshot
            else bool(binary_death_at_zero_energy_var.get())
        )
        full_params[NO_DEATH] = (
            bool(snapshot.get("no_death"))
            if snapshot
            else bool(no_death_var.get())
        )
        full_params["Constant Death Probability"] = (
            bool(snapshot.get("constant_death_probability"))
            if snapshot
            else bool(constant_death_probability_var.get())
        )
        full_params["Constant Duplication Probability"] = (
            bool(snapshot.get("constant_duplication_probability"))
            if snapshot
            else bool(constant_duplication_probability_var.get())
        )
        
        # Simulation runs can generate enormous per-generation histories (big populations × many generations),
        # which can cause the OS to kill the process for memory usage during optimization. Expose a control:
        # default OFF for Simulation, but user-configurable.
        if getattr(model_spec, "key", "") in {"simulation"}:
            full_params["store_history"] = bool(snapshot.get("store_history", False)) if snapshot else False
            if not full_params["store_history"]:
                if "keep_optional_final_arrays" not in full_params:
                    if snapshot is not None and snapshot.get("keep_optional_final_arrays") is not None:
                        full_params["keep_optional_final_arrays"] = bool(snapshot["keep_optional_final_arrays"])
                    elif str(full_save_dir_var.get() or "").strip():
                        full_params["keep_optional_final_arrays"] = True
            full_params = normalize_simulation_params(full_params)
        
        # Initial enzyme values come from the parameter list (either fixed or optimized)
        # They're already included in full_params from the fixed_entries or optimized params_dict above
        
        # Run simulation(s) and calculate metric
        metric_name = str(snapshot.get("metric_name")) if snapshot else metric_var.get()
        
        # Helper to extract a compact, metric-relevant cache from one simulation result.
        # This allows later metric reanalysis without rerunning simulations.
        def _extract_metric_input_cache(result_obj, seed_used=None):
            if model_spec.is_failed(result_obj):
                return {"collapsed": True}
            cache = {"collapsed": bool(result_obj.get("collapsed", False))}
            keys = [
                "A_history",
                "B_history",
                "energy_history",
                "M2_export_history",
                "M2_import_history",
                "task1_performance",
                "task2_performance",
                "metabs_history",
                "storage_history",
                "change_history",
                "mutation_rate",
                "mutation_scale",
                "initial_A",
                "independent_traits",
                # Neutral T2-heavy percentile + pooled synthetic tasks (must match model_registry.compute_metric inputs)
                "investment_modifier",
                "final_metabolite_environment",
                "metabolite_environment_after_inflow_final_generation",
                "enable_m1_diffusion",
                "enable_m2_diffusion",
                "enable_intermediate_costs",
            ]
            for k in keys:
                if k in result_obj:
                    cache[k] = result_obj.get(k)
            if seed_used is not None:
                try:
                    cache["random_seed_used"] = int(seed_used)
                except Exception:
                    cache["random_seed_used"] = seed_used
            cache["metric_values_by_name"] = {}
            return to_json_serializable(cache)

        # Helper function for single replicate (for parallel execution)
        def run_single_replicate(rep_idx):
            """Run a single replicate and return (metric_value, metric_input_cache)."""
            local_params = full_params.copy()
            local_seed = None
            # Pass per-replicate seed into simulation; simulation uses a local RNG.
            if seed is not None:
                # Keep seed fixed unless multiple replicates are requested.
                local_seed = replicate_seed_for_run(seed, rep_idx, num_replicates)
                local_params["Random Seed (optional)"] = local_seed
            
            # Run simulation
            result = model_spec.run_simulation(local_params)
            
            if model_spec.is_failed(result):
                return np.nan, _extract_metric_input_cache(result, seed_used=local_seed)

            metric_val_local = model_spec.compute_metric(result, metric_name)
            metric_cache = _extract_metric_input_cache(result, seed_used=local_seed)
            if isinstance(metric_cache, dict):
                metric_map = metric_cache.get("metric_values_by_name")
                if not isinstance(metric_map, dict):
                    metric_map = {}
                    metric_cache["metric_values_by_name"] = metric_map
                metric_map[metric_name] = None if np.isnan(metric_val_local) else float(metric_val_local)
            return metric_val_local, metric_cache
        
        # Run replicates:
        # - If verbose: always sequential (avoids interleaved stdout and reduces slowdown)
        # - If silent: parallelize replicates for speed
        if num_replicates == 1:
            # Single replicate - no parallelization overhead
            try:
                metric_val, metric_input = run_single_replicate(0)
            except Exception:
                metric_val, metric_input = np.nan, {"collapsed": True}
            metric_values = [metric_val] if not np.isnan(metric_val) else []
            metric_inputs = [metric_input]
        else:
            metric_inputs = []
            if requested_verbose:
                metric_values = []
                for i in range(num_replicates):
                    try:
                        val, inp = run_single_replicate(i)
                        metric_inputs.append(inp)
                        if not np.isnan(val):
                            metric_values.append(val)
                    except Exception:
                        pass
            else:
                # Multiple replicates - parallelize for speed
                with ThreadPoolExecutor(max_workers=min(num_replicates, 4)) as executor:
                    futures = [executor.submit(run_single_replicate, i) for i in range(num_replicates)]
                    metric_values = []
                    for future in futures:
                        try:
                            val, inp = future.result()
                            metric_inputs.append(inp)
                            if not np.isnan(val):
                                metric_values.append(val)
                        except Exception:
                            # If replicate failed, skip it
                            pass
        
        # Keep one representative cache payload per point to avoid huge saves.
        selected_metric_input = None
        for mi in metric_inputs:
            if isinstance(mi, dict) and not bool(mi.get("collapsed", False)):
                selected_metric_input = mi
                break
        if selected_metric_input is None and metric_inputs:
            # Fall back to the first payload when no non-collapsed run is available.
            selected_metric_input = metric_inputs[0]

        num_succeeded = len(metric_values)
        num_failed = num_replicates - num_succeeded

        # Keep per-metric cache aligned with the primary scalar we store (mean over succeeded replicates),
        # so later reanalysis never replaces an averaged metric with a single-replicate value.
        primary_metric_value = np.nan if num_succeeded == 0 else float(np.mean(metric_values))
        if isinstance(selected_metric_input, dict):
            metric_map = selected_metric_input.get("metric_values_by_name")
            if not isinstance(metric_map, dict):
                metric_map = {}
                selected_metric_input["metric_values_by_name"] = metric_map
            metric_map[metric_name] = None if np.isnan(primary_metric_value) else float(primary_metric_value)

        if num_succeeded == 0:
            return (np.nan, (num_replicates, 0, num_failed), selected_metric_input)

        return (primary_metric_value, (num_replicates, num_succeeded, num_failed), selected_metric_input)
    
    def compute_gradient(current_params, optimized_params_list, param_bounds_dict, seed, num_replicates, step_size):
        """Compute gradient using numerical differentiation with parallel evaluation."""
        # Evaluate current point
        current_metric, _, _ = evaluate_metric(current_params, optimized_params_list, param_bounds_dict, seed, num_replicates)
        
        if np.isnan(current_metric):
            return None
        
        # Prepare all perturbed parameter sets for parallel evaluation
        perturbations = []
        for param_name in optimized_params_list:
            perturbed_params = current_params.copy()
            perturbed_params[param_name] += step_size
            perturbations.append((param_name, perturbed_params))
        
        # Evaluate all perturbations in parallel using ThreadPoolExecutor
        # Thread pool is safe here because Python GIL is released during numpy/simulation operations
        gradient = {}
        with ThreadPoolExecutor(max_workers=min(len(perturbations), 4)) as executor:
            # Submit all evaluations
            futures = {}
            for param_name, perturbed_params in perturbations:
                future = executor.submit(evaluate_metric, perturbed_params, optimized_params_list, param_bounds_dict, seed, num_replicates)
                futures[param_name] = future
            
            # Collect results
            for param_name, future in futures.items():
                try:
                    perturbed_metric, _, _ = future.result()
                    if np.isnan(perturbed_metric):
                        gradient[param_name] = 0.0
                    else:
                        gradient[param_name] = (perturbed_metric - current_metric) / step_size
                except Exception as e:
                    # If evaluation failed, set gradient to 0
                    gradient[param_name] = 0.0
        
        return gradient
    
    def start_optimization():
        """Start the gradient descent optimization."""
        nonlocal optimization_running, optimization_paused, optimization_thread, optimization_history, active_run_eval_snapshot
        nonlocal full_save_session_id, full_save_batch_counter, full_save_offloaded_points, full_save_final_snapshot_written, full_save_snapshot_in_progress
        nonlocal full_save_manifest_cache
        nonlocal all_runs_history, all_results, best_run_index, all_param_vectors, all_metric_values, all_replicate_info, all_metric_inputs, all_run_ids, all_run_indices
        nonlocal run_id_counter, param_names_list, STORAGE_SAMPLE_RATE, run_start_info, printed_start_messages
        nonlocal offload_point_sources, offload_last_batch_path, offload_last_batch_payload, full_save_lightweight_mode, full_save_lightweight_metric_name, lightweight_metric_cache, loaded_full_save_session_id, loaded_full_save_settings_snapshot, offload_pending_records
        if full_save_snapshot_in_progress:
            messagebox.showwarning(
                "Full Save Snapshot In Progress",
                "Please wait for the background Full Save snapshot write to finish before starting a new optimization."
            )
            return
        full_save_folder = _ensure_full_save_folder(interactive=True)
        if not full_save_folder:
            messagebox.showwarning(
                "Full Save Folder Required",
                "Select a Full Save folder to start optimization."
            )
            return
        
        # Get parameters to optimize (those that are NOT fixed)
        optimized_params = [p for p, var in param_fix_checkboxes.items() if (not var.get()) and (p not in hidden_params)]
        
        if len(optimized_params) == 0:
            messagebox.showerror("Error", "Please select at least one parameter to optimize (uncheck 'Fix' for parameters you want to optimize).")
            return
        
        # Get parameter bounds
        param_bounds = {}
        
        for param_name in optimized_params:
            try:
                # Min value - optional (empty means no lower bound)
                min_str = param_min_entries[param_name].get().strip()
                if min_str == "":
                    min_val = -np.inf  # No lower bound
                else:
                    min_val = float(min_str)
                
                # Max value - required
                max_str = param_max_entries[param_name].get().strip()
                if max_str == "":
                    messagebox.showerror("Error", f"Max value is required for {param_name}")
                    return
                max_val = float(max_str)
                
                if min_val >= max_val and min_val != -np.inf:
                    messagebox.showerror("Error", f"Invalid bounds for {param_name}: min must be < max")
                    return
                param_bounds[param_name] = (min_val, max_val)
            except ValueError:
                messagebox.showerror("Error", f"Invalid value for {param_name}")
                return
        
        # Get optimization settings
        try:
            learning_rate = float(learning_rate_entry.get())
            max_iterations = int(max_iterations_entry.get())
            convergence_threshold = float(convergence_threshold_entry.get())
            gradient_step = float(gradient_step_entry.get())
            seed = parse_optional_seed(seed_entry.get().strip())
            num_replicates = int(replicates_entry.get())
            num_starts = int(num_starts_entry.get())
            
            if learning_rate <= 0:
                raise ValueError("Learning rate must be positive")
            # Allow 0 iterations: "random starts only" (evaluate starts; no gradient descent steps)
            if max_iterations < 0:
                raise ValueError("Max iterations must be >= 0 (0 = no gradient descent)")
            if gradient_step <= 0:
                raise ValueError("Gradient step size must be positive")
            if num_replicates < 1:
                raise ValueError("Number of replicates must be at least 1")
            if num_starts < 1:
                raise ValueError("Number of random starts must be at least 1")
            descents_per_start_input = int(descents_per_start_entry.get())
            if descents_per_start_input < 0:
                raise ValueError("Gradient descents per start must be >= 0 (0 = random starts only)")
            random_starts_only_mode = (descents_per_start_input == 0)
            # Internally, we still execute 1 evaluation per start so the bookkeeping/progress stays consistent.
            descents_per_start = max(1, descents_per_start_input)
            if random_starts_only_mode:
                # Interpret as: no gradient descent steps (equivalent to max_iterations = 0)
                max_iterations = 0
                try:
                    max_iterations_entry.delete(0, tk.END)
                    max_iterations_entry.insert(0, "0")
                except Exception:
                    pass
            storage_sample_rate = int(sample_rate_entry.get())
            if storage_sample_rate < 1:
                raise ValueError("Storage sample rate must be at least 1")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid optimization settings: {e}")
            return

        # Freeze all GUI-driven evaluation settings for this run.
        active_run_eval_snapshot = _capture_run_evaluation_snapshot()
        frozen_goal_maximize = (str(active_run_eval_snapshot.get("optimization_goal", "Maximize")) == "Maximize")
        
        # Clear previous results
        optimization_history = []
        all_runs_history = []
        all_results = []
        all_param_vectors = []
        all_metric_values = []
        all_replicate_info = []
        all_metric_inputs = []
        all_run_ids = []
        all_run_indices = []
        offload_point_sources = []
        offload_pending_records = []
        offload_last_batch_path = None
        offload_last_batch_payload = None
        full_save_lightweight_mode = False
        full_save_lightweight_metric_name = None
        lightweight_metric_cache = {}
        loaded_full_save_session_id = None
        loaded_full_save_settings_snapshot = None
        run_id_counter = 0  # Reset run ID counter
        param_names_list = optimized_params.copy()  # Store parameter names for consistent ordering
        run_start_info = {}  # Store start message info for each run (keyed by run_index)
        printed_start_messages = set()  # Track which runs have had their start message printed
        # Disable post-run tools until we have results again
        param_heatmaps_button.config(state="disabled")
        metric_hist_button.config(state="disabled")
        # Reset run/cleanup stats for this optimization session
        run_summary_stats["total_runs"] = 0
        run_summary_stats["nan_runs"] = 0
        run_summary_stats["metric_stats"] = RunningStats()
        deleted_point_stats["total_points"] = 0
        deleted_point_stats["nan_points"] = 0
        deleted_point_stats["metric_stats"] = RunningStats()
        full_save_session_id = time.strftime("%Y%m%d-%H%M%S")
        full_save_batch_counter = 0
        full_save_offloaded_points = 0
        full_save_final_snapshot_written = False
        full_save_manifest_cache = None
        _save_full_save_session_settings(full_save_session_id)
        # Update sample rate from UI
        STORAGE_SAMPLE_RATE = storage_sample_rate
        results_text.delete(1.0, tk.END)
        if 'random_starts_only_mode' in locals() and random_starts_only_mode:
            results_text.insert(
                tk.END,
                "Note: Gradient Descents per Start = 0 → running random starts only (no gradient descent steps).\n\n",
            )
        # Update UI
        run_button.config(state="disabled")
        pause_button.config(state="normal", text="Pause")
        stop_button.config(state="normal")
        progress.start()
        progress_label.config(text="Running...")
        
        optimization_running = True
        optimization_paused = False
        stdin_pause_hold.clear()
        prev_gd_unreg = grad_terminal_pause_unreg[0]
        if prev_gd_unreg is not None:
            try:
                prev_gd_unreg()
            except Exception:
                pass
            grad_terminal_pause_unreg[0] = None

        def _gd_imm_pause() -> None:
            stdin_pause_hold.set()

        def _gd_imm_resume() -> None:
            stdin_pause_hold.clear()

        grad_terminal_pause_unreg[0] = register_terminal_pause_hooks(
            win,
            immediate_pause=_gd_imm_pause,
            immediate_resume=_gd_imm_resume,
            main_thread_after_pause=_refresh_pause_button_from_state,
            main_thread_after_resume=_refresh_pause_button_from_state,
        )

        def _wait_if_paused():
            """Block worker progress while paused; return False if stopped."""
            while optimization_running and (optimization_paused or stdin_pause_hold.is_set()):
                time.sleep(0.1)
            return optimization_running
        
        def generate_random_start(bounds):
            """Generate a random starting point within the given bounds."""
            random_params = {}
            for param_name, (min_val, max_val) in bounds.items():
                if min_val == -np.inf:
                    # If no lower bound, use a reasonable default (0 or max/2)
                    random_val = np.random.uniform(0, max_val)
                else:
                    random_val = np.random.uniform(min_val, max_val)
                random_params[param_name] = random_val
            return random_params
        
        def run_single_optimization(start_params, start_index, descent_index, total_starts, descents_per_start, descent_seed):
            """Run a single gradient descent optimization from a starting point."""
            nonlocal iteration_counter, run_id_counter
            nonlocal printed_start_messages
            
            current_params = start_params.copy()
            maximize = frozen_goal_maximize
            run_history = []
            
            # Capture the frozen run settings used by evaluate_metric (not live GUI values).
            fixed_params_at_run_start = dict((active_run_eval_snapshot or {}).get("fixed_params", {}))
            if getattr(model_spec, "key", "") == "simulation":
                fixed_params_at_run_start["Enable M1 Diffusion"] = bool((active_run_eval_snapshot or {}).get("enable_m1_diffusion", False))
                fixed_params_at_run_start["Enable M2 Diffusion"] = bool((active_run_eval_snapshot or {}).get("enable_m2_diffusion", False))
                fixed_params_at_run_start["Enable M1 Facilitated Diffusion"] = bool((active_run_eval_snapshot or {}).get("enable_m1_facilitated_diffusion", False))
                fixed_params_at_run_start["Enable M1 Porin Diffusion"] = bool(
                    (active_run_eval_snapshot or {}).get(
                        "enable_m1_porin_diffusion",
                        False,
                    )
                )
                allow_diffusion_mutation = bool(fixed_params_at_run_start["Enable M2 Diffusion"] or fixed_params_at_run_start["Enable M1 Diffusion"])
                fixed_params_at_run_start["Enable Diffusion Mutation"] = bool((active_run_eval_snapshot or {}).get("enable_diffusion_mutation", False)) if allow_diffusion_mutation else False
                fixed_params_at_run_start["Homogeneous Initial Diffusion Const."] = (
                    bool((active_run_eval_snapshot or {}).get("homogeneous_initial_diffusion_const", False))
                    if fixed_params_at_run_start["Enable Diffusion Mutation"] else False
                )
            fixed_params_at_run_start["Enable Initial Energy"] = bool((active_run_eval_snapshot or {}).get("enable_initial_energy", False))
            if not fixed_params_at_run_start["Enable Initial Energy"]:
                fixed_params_at_run_start["Initial Energy"] = 0.0
            else:
                if "Initial Energy" not in fixed_params_at_run_start:
                    fixed_params_at_run_start["Initial Energy"] = 0.0
            fixed_params_at_run_start["Enable Chemostat Flow"] = bool((active_run_eval_snapshot or {}).get("enable_chemostat_flow", False))
            if not fixed_params_at_run_start["Enable Chemostat Flow"]:
                fixed_params_at_run_start["Flow Percentage"] = 0.0
            else:
                if "Flow Percentage" not in fixed_params_at_run_start:
                    fixed_params_at_run_start["Flow Percentage"] = 0.0
            fixed_params_at_run_start["Enable Intermediate Costs"] = bool((active_run_eval_snapshot or {}).get("enable_intermediate_costs", False))
            if not fixed_params_at_run_start["Enable Intermediate Costs"]:
                fixed_params_at_run_start["Intermediate Costs"] = 0.0
            else:
                if "Intermediate Costs" not in fixed_params_at_run_start:
                    fixed_params_at_run_start["Intermediate Costs"] = 0.0
            fixed_params_at_run_start["Enable Acetate Addition"] = bool((active_run_eval_snapshot or {}).get("enable_acetate_addition", False))
            if not fixed_params_at_run_start["Enable Acetate Addition"]:
                fixed_params_at_run_start["Average In_Flow (Acetate)"] = 0.0
            else:
                if "Average In_Flow (Acetate)" not in fixed_params_at_run_start:
                    fixed_params_at_run_start["Average In_Flow (Acetate)"] = 0.0
            fixed_params_at_run_start["Binary Death at Zero Energy"] = bool(
                (active_run_eval_snapshot or {}).get("binary_death_at_zero_energy", False)
            )
            fixed_params_at_run_start[NO_DEATH] = bool(
                (active_run_eval_snapshot or {}).get("no_death", False)
            )
            fixed_params_at_run_start["Constant Death Probability"] = bool(
                (active_run_eval_snapshot or {}).get("constant_death_probability", False)
            )
            fixed_params_at_run_start["Constant Duplication Probability"] = bool(
                (active_run_eval_snapshot or {}).get("constant_duplication_probability", False)
            )
            if getattr(model_spec, "key", "") in {"simulation"}:
                fixed_params_at_run_start = normalize_simulation_params(dict(fixed_params_at_run_start))
            fixed_params_at_run_start["Homogeneous Population"] = bool((active_run_eval_snapshot or {}).get("homogeneous_population", False))
            fixed_params_at_run_start["Independent Traits"] = bool((active_run_eval_snapshot or {}).get("independent_traits", False))
            if getattr(model_spec, "key", "") in {"simulation"}:
                fixed_params_at_run_start["store_history"] = bool((active_run_eval_snapshot or {}).get("store_history", False))
            
            # Capture the exact per-run seed used for this descent.
            # This makes point/run relaunch reproducible in Individual GUI.
            fixed_params_at_run_start["Random Seed (optional)"] = (
                str(int(descent_seed)) if descent_seed is not None else ""
            )
            
            # Track best metric and parameters seen during the run
            best_metric_seen = None
            best_params_seen = None
            
            # Calculate global run index
            global_run_index = start_index * descents_per_start + descent_index
            total_runs = total_starts * descents_per_start
            
            # Store start message info (will be printed when first non-NaN result appears)
            nonlocal run_start_info
            run_start_info[global_run_index] = {
                "start_index": start_index,
                "descent_index": descent_index,
                "total_starts": total_starts,
                "descents_per_start": descents_per_start,
                "start_params": start_params.copy(),
                "total_runs": total_runs
            }
            
            # Store starting parameters immediately (even if run fails before first iteration)
            # This ensures all runs appear in the t-SNE plot
            start_param_vector = [start_params.get(pname, 0.0) for pname in param_names_list]
            all_param_vectors.append(start_param_vector)
            
            # Try to evaluate starting metric
            start_metric, start_replicate_info, start_metric_inputs = evaluate_metric(
                start_params, optimized_params, param_bounds, descent_seed, num_replicates
            )
            all_metric_values.append(start_metric)
            all_replicate_info.append(start_replicate_info)
            all_metric_inputs.append(start_metric_inputs)
            
            # Assign stable run ID and increment counter
            current_run_id = run_id_counter
            run_id_counter += 1
            all_run_ids.append(current_run_id)
            all_run_indices.append(global_run_index)  # Store global run index for this point
            if OFFLOAD_EACH_POINT:
                _offload_indices_immediately({len(all_param_vectors) - 1})
            
            # Reset iteration counter for this run
            nonlocal iteration_counter
            iteration_counter = 0
            
            # Initialize best metric with starting metric if valid
            if not np.isnan(start_metric):
                best_metric_seen = start_metric
                best_params_seen = start_params.copy()
            
            # Track if any iteration actually completed (for determining if we should store final point)
            # Only store final point if at least one iteration completed (not just started)
            iterations_completed = False
            
            # If starting metric is NaN, the run has already failed - still continue to try
            # but we've already captured the starting point
            
            for iteration in range(max_iterations):
                if not _wait_if_paused():
                    break
                
                # Evaluate current metric (use descent_seed for this specific descent)
                current_metric, replicate_info, metric_inputs = evaluate_metric(
                    current_params, optimized_params, param_bounds, descent_seed, num_replicates
                )
                
                if np.isnan(current_metric):
                    # Silently break on NaN - don't print anything
                    break
                
                # If we get here, the metric is valid, so mark that an iteration completed
                iterations_completed = True
                
                # Update best metric seen during this run
                if not np.isnan(current_metric):
                    if best_metric_seen is None:
                        best_metric_seen = current_metric
                        best_params_seen = current_params.copy()
                    else:
                        if maximize:
                            if current_metric > best_metric_seen:
                                best_metric_seen = current_metric
                                best_params_seen = current_params.copy()
                        else:
                            if current_metric < best_metric_seen:
                                best_metric_seen = current_metric
                                best_params_seen = current_params.copy()
                
                # Compute gradient (use descent_seed for this specific descent)
                if not _wait_if_paused():
                    break
                gradient = compute_gradient(current_params, optimized_params, param_bounds, descent_seed, num_replicates, gradient_step)
                
                if gradient is None:
                    # Silently break on gradient failure - don't print anything
                    break
                
                # Update parameters using gradient descent
                new_params = {}
                for param_name in optimized_params:
                    grad = gradient[param_name]
                    # Flip sign if minimizing
                    if not maximize:
                        grad = -grad
                    
                    # Update parameter
                    new_val = current_params[param_name] + learning_rate * grad
                    
                    # Apply bounds - ensure parameter is in bounds dictionary
                    if param_name not in param_bounds:
                        # If bounds not found, skip this parameter (shouldn't happen, but safety check)
                        new_params[param_name] = current_params[param_name]
                        continue
                    
                    min_val, max_val = param_bounds[param_name]
                    if min_val == -np.inf:
                        # Only clip upper bound
                        new_val = min(new_val, max_val)
                    else:
                        # Clip both bounds - use np.clip for robust handling
                        new_val = np.clip(new_val, min_val, max_val)
                    
                    # Double-check bounds are enforced (handle edge cases with floating point precision)
                    if min_val != -np.inf:
                        new_val = max(new_val, min_val)
                    new_val = min(new_val, max_val)
                    
                    new_params[param_name] = float(new_val)  # Ensure it's a Python float, not numpy type
                
                # Store history for this run (only keep essential data)
                history_entry = {
                    "iteration": iteration,
                    "params": current_params.copy(),
                    "metric": current_metric,
                    "gradient": gradient.copy()
                }
                run_history.append(history_entry)
                
                # Store parameter vector and metric for dimension reduction (BEFORE update)
                # Use sampling to reduce memory usage - only store every Nth iteration
                iteration_counter += 1
                should_store = (iteration_counter % STORAGE_SAMPLE_RATE == 0) or (iteration == 0) or (iteration == max_iterations - 1)
                
                if should_store:
                    # Create a vector in the same order as param_names_list
                    param_vector = [current_params.get(pname, 0.0) for pname in param_names_list]
                    all_param_vectors.append(param_vector)
                    all_metric_values.append(current_metric)
                    all_replicate_info.append(replicate_info)
                    all_metric_inputs.append(metric_inputs)
                    all_run_ids.append(current_run_id)  # Use same run ID for all iterations in this run
                    all_run_indices.append(global_run_index)  # Store global run index for this point
                    if OFFLOAD_EACH_POINT:
                        _offload_indices_immediately({len(all_param_vectors) - 1})
                    
                    # Periodic memory cleanup
                    if len(all_param_vectors) >= CLEANUP_THRESHOLD:
                        print(f"[Memory Management] Triggering cleanup: {len(all_param_vectors)} points >= {CLEANUP_THRESHOLD} threshold")
                        _schedule_cleanup()
                
                # Update display (show current run progress) - only if metric is not NaN
                should_log_iteration = (
                    iteration == 0
                    or iteration == (max_iterations - 1)
                    or (iteration % max(1, ITERATION_LOG_FREQUENCY) == 0)
                )
                if should_log_iteration and not np.isnan(current_metric):
                    # Schedule update on main thread - capture all variables explicitly
                    def update_display():
                        update_optimization_display(iteration, current_params.copy(), current_metric, gradient.copy(), 
                                                    global_run_index, total_runs)
                    win.after(0, update_display)
                
                # Check convergence
                if iteration > 0:
                    prev_metric = run_history[-2]["metric"]
                    metric_change = abs(current_metric - prev_metric)
                    if metric_change < convergence_threshold:
                        if LOG_PER_RUN_UPDATES:
                            win.after(
                                0,
                                lambda gidx=global_run_index, it=iteration, mc=metric_change:
                                    results_text.insert(tk.END, f"Run {gidx + 1}: Converged at iteration {it} (change: {mc:.6f})\n")
                            )
                        break
                
                # Final bounds check before updating current_params
                # This ensures parameters never escape bounds even if there's a bug elsewhere
                for param_name in optimized_params:
                    if param_name in param_bounds and param_name in new_params:
                        min_val, max_val = param_bounds[param_name]
                        if min_val == -np.inf:
                            new_params[param_name] = min(new_params[param_name], max_val)
                        else:
                            new_params[param_name] = np.clip(new_params[param_name], min_val, max_val)
                        new_params[param_name] = float(new_params[param_name])  # Ensure Python float
                
                current_params = new_params
                
                # Skip storing "after" vectors to reduce memory usage
                # The next iteration will capture the updated state anyway
            
            # Final evaluation (use descent_seed for this specific descent)
            # Only evaluate and store final point if at least one iteration completed successfully
            # If the run failed immediately (starting metric was NaN and loop broke immediately),
            # we've already stored the starting point, so don't store a duplicate final point
            if iterations_completed:
                final_metric, final_replicate_info, final_metric_inputs = evaluate_metric(
                    current_params, optimized_params, param_bounds, descent_seed, num_replicates
                )

                # Store final parameter vector and metric (important for visualization)
                param_vector = [current_params.get(pname, 0.0) for pname in param_names_list]
                all_param_vectors.append(param_vector)
                all_metric_values.append(final_metric)
                all_replicate_info.append(final_replicate_info)
                all_metric_inputs.append(final_metric_inputs)
                all_run_ids.append(current_run_id)  # Use same run ID for final point
                all_run_indices.append(global_run_index)  # Store global run index for final point
                if OFFLOAD_EACH_POINT:
                    _offload_indices_immediately({len(all_param_vectors) - 1})
            else:
                # No iterations completed, use starting metric as final metric
                final_metric = start_metric
                final_replicate_info = start_replicate_info
            
            # Check for cleanup after storing final point
            if len(all_param_vectors) >= CLEANUP_THRESHOLD:
                print(f"[Memory Management] Triggering cleanup after final point: {len(all_param_vectors)} points >= {CLEANUP_THRESHOLD} threshold")
                _schedule_cleanup()
            
            # Update best metric if final metric is better
            if not np.isnan(final_metric):
                if best_metric_seen is None:
                    best_metric_seen = final_metric
                    best_params_seen = current_params.copy()
                else:
                    if maximize:
                        if final_metric > best_metric_seen:
                            best_metric_seen = final_metric
                            best_params_seen = current_params.copy()
                    else:
                        if final_metric < best_metric_seen:
                            best_metric_seen = final_metric
                            best_params_seen = current_params.copy()
            
            # Use best metric and params if we found any, otherwise use final
            if best_metric_seen is not None:
                best_metric = best_metric_seen
                best_params = best_params_seen.copy()
            else:
                best_metric = final_metric
                best_params = current_params.copy()

            # If no iterations were printed (e.g., max_iterations == 0), still print a concise per-run summary
            # for any successful (non-NaN) run so the user can see metrics/params for random starts.
            if (max_iterations == 0 or not iterations_completed) and (not np.isnan(best_metric)):
                if LOG_PER_RUN_UPDATES:
                    def print_run_summary():
                        # Print the standard start header if it wasn't printed yet
                        if global_run_index not in printed_start_messages and global_run_index in run_start_info:
                            info = run_start_info[global_run_index]
                            results_text.insert(tk.END, f"\n{'='*50}\n")
                            results_text.insert(
                                tk.END,
                                f"Start {info['start_index'] + 1}/{info['total_starts']}, "
                                f"Descent {info['descent_index'] + 1}/{info['descents_per_start']} "
                                f"(Run {global_run_index + 1}/{info['total_runs']})\n"
                            )
                            results_text.insert(tk.END, f"Starting parameters: {info['start_params']}\n")
                            printed_start_messages.add(global_run_index)

                        results_text.insert(
                            tk.END,
                            f"Run {global_run_index + 1}/{total_runs}: Best Metric = {best_metric:.6f}\n"
                        )
                        results_text.insert(tk.END, "Best Parameters:\n")
                        for p_name, p_val in best_params.items():
                            try:
                                results_text.insert(tk.END, f"  {p_name}: {float(p_val):.6f}\n")
                            except Exception:
                                results_text.insert(tk.END, f"  {p_name}: {p_val}\n")
                        results_text.see(tk.END)

                    win.after(0, print_run_summary)
            
            return {
                "start_index": start_index,
                "descent_index": descent_index,
                "run_index": global_run_index,
                "descent_seed": (int(descent_seed) if descent_seed is not None else None),
                "final_params": current_params.copy(),
                "final_metric": final_metric,
                "best_params": best_params,
                "best_metric": best_metric,
                "history": run_history,
                "fixed_params": fixed_params_at_run_start.copy()  # Store fixed parameters used during this run
            }
        
        def optimization_worker():
            """Worker function for multi-start optimization."""
            nonlocal all_runs_history, all_results, best_run_index, optimization_history
            
            maximize = frozen_goal_maximize
            all_results = []
            
            # Generate random starting points for every run.
            random_starts = []
            # Set seed for reproducible random starts (if seed provided).
            if seed is not None:
                np.random.seed(seed)
            for _ in range(num_starts):
                random_start = generate_random_start(param_bounds)
                random_starts.append(random_start)
            
            # Run optimization from each starting point
            # For each starting point, run multiple gradient descents
            for start_idx, start_params in enumerate(random_starts):
                if not _wait_if_paused():
                    break
                
                # Run multiple gradient descents from this starting point
                for descent_idx in range(descents_per_start):
                    if not _wait_if_paused():
                        break
                    
                    # Keep per-descent seed fixed for fair comparison across runs.
                    # Replicate-level offsets (when num_replicates > 1) are applied inside evaluate_metric.
                    descent_seed = descent_seed_for_run(seed, start_idx, descent_idx)
                    
                    result = run_single_optimization(start_params, start_idx, descent_idx, num_starts, descents_per_start, descent_seed)

                    # Track run-level stats independent of any point cleanup
                    run_summary_stats["total_runs"] += 1
                    result_metric_for_stats = result.get("best_metric", result.get("final_metric", np.nan))
                    if np.isnan(result_metric_for_stats):
                        run_summary_stats["nan_runs"] += 1
                    else:
                        run_summary_stats["metric_stats"].add(result_metric_for_stats)

                    all_results.append(result)
                    all_runs_history.append(result["history"])
                    
                    # Update best run (only consider runs with valid metrics)
                    # Use best_metric instead of final_metric to find the best run
                    # IMPORTANT: Use position in all_results list, not run_index
                    result_position = len(all_results) - 1
                    result_metric = result.get("best_metric", result["final_metric"])
                    if not np.isnan(result_metric):
                        if len(all_results) == 1:
                            # First valid run becomes the best
                            best_run_index = result_position
                        else:
                            # Compare with current best (only if current best is valid)
                            if best_run_index < len(all_results):
                                current_best_result = all_results[best_run_index]
                                current_best_metric = current_best_result.get("best_metric", current_best_result["final_metric"])
                                if not np.isnan(current_best_metric):
                                    # Both are valid, compare normally
                                    if maximize:
                                        if result_metric > current_best_metric:
                                            best_run_index = result_position
                                    else:
                                        if result_metric < current_best_metric:
                                            best_run_index = result_position
                                else:
                                    # Current best is NaN but this one is valid, so this becomes best
                                    best_run_index = result_position
                            else:
                                # Current best_run_index is out of bounds, use this one
                                best_run_index = result_position
                    
                    # Update progress
                    total_runs = num_starts * descents_per_start
                    win.after(0, lambda gidx=result["run_index"], total=total_runs: 
                        progress_label.config(text=f"Run {gidx + 1}/{total} complete..."))
            
            # Update optimization_history to show best run
            # Find the best valid run if current best is invalid
            if len(all_runs_history) > 0:
                # Check if current best_run_index is valid (bound check)
                if best_run_index >= len(all_results):
                    best_run_index = len(all_results) - 1
                
                best_result_metric = all_results[best_run_index].get("best_metric", all_results[best_run_index]["final_metric"])
                
                if np.isnan(best_result_metric):
                    # Find the best valid run using best_metric
                    valid_results = [(i, r) for i, r in enumerate(all_results) 
                                   if not np.isnan(r.get("best_metric", r["final_metric"]))]
                    if len(valid_results) > 0:
                        if maximize:
                            best_valid = max(valid_results, key=lambda x: x[1].get("best_metric", x[1]["final_metric"]))
                        else:
                            best_valid = min(valid_results, key=lambda x: x[1].get("best_metric", x[1]["final_metric"]))
                        # Use the enumeration index (position in all_results), not run_index
                        best_run_index = best_valid[0]
                    else:
                        # No valid runs, just use the first one
                        best_run_index = 0
                
                # Ensure best_run_index is within bounds of all_runs_history
                if best_run_index < len(all_runs_history):
                    optimization_history = all_runs_history[best_run_index]
                else:
                    optimization_history = []
            
            # Finalize with best result (use best_params and best_metric)
            # Ensure best_run_index is within bounds
            if best_run_index < len(all_results):
                best_result = all_results[best_run_index]
                best_params = best_result.get("best_params", best_result["final_params"])
                best_metric = best_result.get("best_metric", best_result["final_metric"])
            else:
                # Fallback if index is out of bounds
                best_result = all_results[0] if len(all_results) > 0 else {}
                best_params = best_result.get("best_params", best_result.get("final_params", {}))
                best_metric = best_result.get("best_metric", best_result.get("final_metric", np.nan))
            win.after(0, lambda: finalize_optimization(best_params, best_metric, all_results))
        
        def _handle_worker_crash(msg):
            nonlocal optimization_running, optimization_paused, active_run_eval_snapshot
            optimization_running = False
            optimization_paused = False
            stdin_pause_hold.clear()
            active_run_eval_snapshot = None
            _clear_grad_terminal_pause_hooks()
            run_button.config(state="normal")
            pause_button.config(state="disabled", text="Pause")
            stop_button.config(state="disabled")
            progress.stop()
            progress_label.config(text="Error")
            results_text.insert(tk.END, f"\n[Error] Optimization worker crashed: {msg}\n")
            results_text.see(tk.END)
            messagebox.showerror(
                "Optimization Error",
                f"Optimization stopped due to an internal error:\n{msg}\n\n"
                "Try lowering Number of Generations / starts, or tightening parameter bounds."
            )

        def _safe_optimization_worker():
            try:
                optimization_worker()
            except Exception as e:
                import traceback
                traceback.print_exc()
                win.after(0, lambda msg=str(e): _handle_worker_crash(msg))

        optimization_thread = threading.Thread(target=_safe_optimization_worker, daemon=True)
        optimization_thread.start()
    
    def update_optimization_display(iteration, params, metric, gradient, run_index=None, total_runs=None):
        """Update the display with current optimization state."""
        if not LOG_PER_RUN_UPDATES:
            return
        # Only print if metric is not NaN
        if np.isnan(metric):
            return
        
        nonlocal run_start_info, printed_start_messages
        
        # Print start message for this run if we haven't already (only on first non-NaN result)
        if run_index is not None and run_index not in printed_start_messages:
            if run_index in run_start_info:
                info = run_start_info[run_index]
                results_text.insert(tk.END, f"\n{'='*50}\n")
                results_text.insert(tk.END, f"Start {info['start_index'] + 1}/{info['total_starts']}, Descent {info['descent_index'] + 1}/{info['descents_per_start']} (Run {run_index + 1}/{info['total_runs']})\n")
                results_text.insert(tk.END, f"Starting parameters: {info['start_params']}\n")
                printed_start_messages.add(run_index)
        
        run_prefix = f"Run {run_index + 1}/{total_runs}: " if run_index is not None and total_runs is not None else ""
        results_text.insert(tk.END, f"\n{run_prefix}Iteration {iteration}:\n")
        results_text.insert(tk.END, f"  Metric: {metric:.6f}\n")
        results_text.insert(tk.END, f"  Parameters:\n")
        for param_name, value in params.items():
            grad_val = gradient.get(param_name, 0.0)
            results_text.insert(tk.END, f"    {param_name}: {value:.6f} (grad: {grad_val:.6f})\n")
        results_text.see(tk.END)
    
    def finalize_optimization(final_params, final_metric, all_results=None):
        """Finalize optimization and update UI."""
        nonlocal optimization_running, optimization_paused, full_save_snapshot_in_progress, active_run_eval_snapshot
        
        optimization_running = False
        optimization_paused = False
        stdin_pause_hold.clear()
        active_run_eval_snapshot = None
        _clear_grad_terminal_pause_hooks()
        run_button.config(state="normal")
        pause_button.config(state="disabled", text="Pause")
        stop_button.config(state="disabled")
        progress.stop()
        progress_label.config(text="")
        
        # Ensure every completed run persists its in-memory points as offload batches,
        # even when memory cleanup thresholds were never reached.
        final_flush_count = 0
        final_offload_count = 0
        if all_results:
            try:
                _ensure_full_save_folder(interactive=False)
                final_flush_count = int(_flush_pending_offload_records(force=True) or 0)
                if len(all_param_vectors) > 0:
                    final_offload_count, _ = _offload_points_to_disk(set(range(len(all_param_vectors))))
                    final_offload_count = int(final_offload_count or 0)
            except Exception as e:
                try:
                    results_text.insert(tk.END, f"\n[Full Save] Final offload flush failed: {e}\n")
                except Exception:
                    pass
        
        # Print the standard summary block
        summary_text = build_optimization_summary_text(all_results or [], best_run_index, final_params, final_metric)
        results_text.insert(tk.END, summary_text)
        # Persist offload batches directly (no combined one-file packing).
        if all_results:
            results_text.insert(
                tk.END,
                (
                    "\n[Full Save] Session complete. Data is stored in offload batches "
                    "(no combined dataset file).\n"
                    f"[Full Save] Final flush: pending={final_flush_count}, "
                    f"in-memory offloaded={final_offload_count}.\n"
                )
            )
        results_text.see(tk.END)

        # Draw parameter heatmaps when optimization results exist.
        param_heatmaps_button.config(state=("normal" if all_results else "disabled"))
        metric_hist_button.config(state=("normal" if all_results else "disabled"))
    
    def toggle_pause_optimization():
        """Pause/resume a running optimization."""
        nonlocal optimization_paused
        if not optimization_running:
            return

        if optimization_paused:
            optimization_paused = False
            stdin_pause_hold.clear()
            results_text.insert(tk.END, "\nOptimization resumed.\n")
        else:
            optimization_paused = True
            stdin_pause_hold.clear()
            results_text.insert(tk.END, "\nOptimization paused.\n")
        results_text.see(tk.END)
        _refresh_pause_button_from_state()

    def stop_optimization():
        """Stop the running optimization."""
        nonlocal optimization_running, optimization_paused, active_run_eval_snapshot
        optimization_running = False
        optimization_paused = False
        stdin_pause_hold.clear()
        active_run_eval_snapshot = None
        _clear_grad_terminal_pause_hooks()
        pause_button.config(state="disabled", text="Pause")
        progress.stop()
        progress_label.config(text="Stopping...")
        results_text.insert(tk.END, "\nOptimization stopped by user.\n")
        results_text.see(tk.END)

