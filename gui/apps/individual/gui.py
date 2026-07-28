import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import queue
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from simulation.helpers import investment_func
from simulation.change_history import parse_change_history_rows
from simulation.core import _effective_chemostat_volume
from gui.common.colors import OKABE_ITO, get_series_color
from gui.common.data_utils import downsample_for_plot
from gui.common.model_diagram import SimulationModelDiagram
from gui.metrics import describe_metric, filter_metric_options_for_simulation_settings
from gui.common.seed_policy import parse_optional_seed, replicate_seed_for_run
from gui.common.death_dup_probabilities import (
    compute_death_probabilities,
    compute_duplication_probabilities,
    effective_duplication_probabilities_for_display,
)
from gui.common.simulation_settings import (
    AVERAGE_INFLOW_ACETATE,
    BINARY_DEATH_AT_ZERO_ENERGY,
    CONSTANT_DEATH_PROBABILITY,
    CONSTANT_DUPLICATION_PROBABILITY,
    CONSTANT_PROBABILITY,
    ENABLE_ACETATE_ADDITION,
    ENABLE_CHEMOSTAT_FLOW,
    ENABLE_DIFFUSION_MUTATION,
    ENABLE_INITIAL_ENERGY,
    ENABLE_INTERMEDIATE_COSTS,
    ENABLE_M1_DIFFUSION,
    ENABLE_M1_FACILITATED_DIFFUSION,
    ENABLE_M1_PORIN_DIFFUSION,
    ENABLE_M2_DIFFUSION,
    FLOW_PERCENTAGE,
    HOMOGENEOUS_INITIAL_DIFFUSION_CONST,
    HOMOGENEOUS_POPULATION,
    INDEPENDENT_TRAITS,
    INITIAL_ENERGY,
    INTERMEDIATE_COSTS,
    NO_DEATH,
    normalize_simulation_params,
    any_constant_probability_mode,
    apply_death_dup_flow_toggle_ui,
    resolve_constant_probability,
)
from gui.common.tooltips import (
    PARAMETER_TOOLTIPS,
    GRADIENT_DESCENT_TOOLTIPS,
    SIMULATION_SETTINGS_TOOLTIPS,
    HOMOGENEOUS_TOOLTIP,
)
from gui.common.widgets import CreateToolTip


def _mean_std_series(history):
    """
    history: list where each element is an array-like of per-organism values for that generation (or None).
    Returns (means, stds, sizes) arrays of length len(history).
    """
    n = len(history)
    means = np.zeros(n, dtype=float)
    stds = np.zeros(n, dtype=float)
    sizes = np.zeros(n, dtype=int)

    for i, arr in enumerate(history):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        sizes[i] = int(a.size)
        means[i] = float(np.mean(a))
        stds[i] = float(np.std(a))
    return means, stds, sizes


def _mean_series(history):
    """Return mean per generation for a list of per-organism arrays (or None)."""
    n = len(history)
    means = np.zeros(n, dtype=float)
    for i, arr in enumerate(history):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        means[i] = float(np.mean(a))
    return means


def _sum_series(history):
    """Return sum per generation for a list of per-organism arrays (or None)."""
    n = len(history)
    sums = np.zeros(n, dtype=float)
    for i, arr in enumerate(history):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        sums[i] = float(np.sum(a))
    return sums


def individual_gui(win, root, model_spec, preset_params=None):
    """
    Individual GUI for the Simulation model.
    This is intentionally lightweight and model-specific.
    """
    if preset_params is None:
        preset_params = {}

    win.title(f"Individual Simulation — {getattr(model_spec, 'label', 'Simulation')}")

    # Layout: left controls, middle results, right diagnostics
    container = tk.Frame(win)
    container.pack(fill="both", expand=True)

    left = tk.Frame(container, padx=10, pady=10)
    left.pack(side="left", fill="y")

    diagnostics = tk.Frame(container, padx=10, pady=10)
    diagnostics.pack(side="right", fill="y")

    results = tk.Frame(container, padx=10, pady=10)
    results.pack(side="right", fill="both", expand=True)

    # Controls
    tk.Label(left, text="Simulation — Individual Run", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 8))

    def _create_scrollable_section(parent, title, height=220, *, expand=False):
        section = tk.LabelFrame(parent, text=title, padx=8, pady=8)
        if expand:
            # Grow with the window: extra vertical space in the left column goes here.
            section.pack(fill="both", expand=True, pady=(0, 8))
            canvas = tk.Canvas(section, highlightthickness=0)
        else:
            section.pack(fill="x", pady=(0, 8))
            canvas = tk.Canvas(section, height=height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(section, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def _sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        scrollable.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable, canvas

    params_frame, _params_canvas = _create_scrollable_section(left, "Parameters", height=250)

    # Use model defaults as the source of truth for which parameters exist
    default_params = dict(getattr(model_spec, "default_params", {}) or {})
    # Allow preset overrides, but only for recognized model parameters
    preset_params = dict(preset_params or {})
    for k, v in preset_params.items():
        if k in default_params:
            default_params[k] = v
    default_params.setdefault(
        "Initial Energy",
        float(preset_params.get("Initial Energy", default_params.get("Initial Energy", 0.0))),
    )

    entries = {}
    entry_vars = {}
    labels = {}
    entry_state = {}
    enable_initial_energy_var = tk.BooleanVar(
        value=bool(preset_params.get("Enable Initial Energy", False))
    )
    param_tooltip_aliases = {
        "Average In_Flow": "Average In-Flow",
        "Average In-Flow": "Average In-Flow",
    }

    def _tooltip_for_param(name):
        key = param_tooltip_aliases.get(name, name)
        return PARAMETER_TOOLTIPS.get(
            key,
            f"{name}\n\nModel parameter used in this simulation."
        )

    def add_entry(row, name, default):
        lbl = tk.Label(params_frame, text=f"{name}:")
        lbl.grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=str(default))
        e = tk.Entry(params_frame, width=14, textvariable=var)
        e.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        entries[name] = e
        entry_vars[name] = var
        labels[name] = lbl
        tt = _tooltip_for_param(name)
        CreateToolTip(lbl, tt)
        CreateToolTip(e, tt)

    def _set_param_row_visible(name: str, visible: bool) -> None:
        """Show or hide a Parameters label+entry pair (grid_remove hides completely)."""
        wl = labels.get(name)
        en = entries.get(name)
        if visible:
            if wl is not None:
                wl.grid()
            if en is not None:
                en.grid()
        else:
            if wl is not None:
                wl.grid_remove()
            if en is not None:
                en.grid_remove()

    def snapshot_entries():
        for name, var in entry_vars.items():
            entry_state[name] = var.get()

    def restore_entries():
        for name, value in entry_state.items():
            if name in entry_vars:
                entry_vars[name].set(value)

    # Render numeric entries (excluding Homogeneous Population and Independent Traits; controlled by checkboxes)
    param_order = list(default_params.keys())
    row = 0
    for p in param_order:
        if p in {"Homogeneous Population", "Independent Traits"}:
            continue
        add_entry(row, p, default_params[p])
        row += 1

    # Additional toggles
    toggles_frame, _toggles_canvas = _create_scrollable_section(left, "Settings", height=250, expand=True)

    homogeneous_var = tk.BooleanVar(value=bool(preset_params.get("Homogeneous Population", default_params.get("Homogeneous Population", False))))
    homogeneous_cb = tk.Checkbutton(
        toggles_frame,
        text="Homogeneous Population (use Initial A/B)",
        variable=homogeneous_var,
    )
    homogeneous_cb.grid(row=0, column=0, sticky="w")
    CreateToolTip(homogeneous_cb, HOMOGENEOUS_TOOLTIP)

    independent_traits_var = tk.BooleanVar(
        value=bool(preset_params.get("Independent Traits", default_params.get("Independent Traits", False)))
    )
    independent_traits_cb = tk.Checkbutton(
        toggles_frame,
        text="Independent Traits A/B (no 1-A constraint)",
        variable=independent_traits_var,
    )
    independent_traits_cb.grid(row=1, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(independent_traits_cb, GRADIENT_DESCENT_TOOLTIPS["Independent A/B Traits"])

    # Silent mode toggle (suppress simulation progress printing)
    silent_var = tk.BooleanVar(value=bool(preset_params.get("silent", True)))
    silent_cb = tk.Checkbutton(
        toggles_frame,
        text="Silent Mode (suppress progress)",
        variable=silent_var,
    )
    silent_cb.grid(row=2, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(silent_cb, SIMULATION_SETTINGS_TOOLTIPS["Silent Mode"])

    enable_m1_diffusion_var = tk.BooleanVar(value=bool(preset_params.get("Enable M1 Diffusion", False)))
    enable_m2_diffusion_var = tk.BooleanVar(value=bool(preset_params.get("Enable M2 Diffusion", True)))
    enable_diffusion_mutation_var = tk.BooleanVar(value=bool(preset_params.get("Enable Diffusion Mutation", False)))
    homogeneous_initial_diffusion_const_var = tk.BooleanVar(value=bool(preset_params.get("Homogeneous Initial Diffusion Const.", False)))
    enable_m1_facilitated_diffusion_var = tk.BooleanVar(value=bool(preset_params.get("Enable M1 Facilitated Diffusion", False)))
    m1_porin_diffusion_var = tk.BooleanVar(value=bool(preset_params.get("Enable M1 Porin Diffusion", False)))
    enable_chemostat_flow_var = tk.BooleanVar(value=bool(preset_params.get("Enable Chemostat Flow", False)))
    m1_diffusion_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable M1 Diffusion (simple)",
        variable=enable_m1_diffusion_var,
    )
    m1_diffusion_cb.grid(row=3, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(m1_diffusion_cb, GRADIENT_DESCENT_TOOLTIPS["Enable M1 Diffusion (simple)"])
    m2_diffusion_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable M2 Diffusion",
        variable=enable_m2_diffusion_var,
    )
    m2_diffusion_cb.grid(row=4, column=0, sticky="w")
    CreateToolTip(m2_diffusion_cb, GRADIENT_DESCENT_TOOLTIPS["Enable M2 Diffusion"])
    diffusion_mutation_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable Diffusion Mutation",
        variable=enable_diffusion_mutation_var,
    )
    diffusion_mutation_cb.grid(row=5, column=0, sticky="w")
    CreateToolTip(diffusion_mutation_cb, GRADIENT_DESCENT_TOOLTIPS["Enable Diffusion Mutation"])
    homogeneous_initial_diffusion_const_cb = tk.Checkbutton(
        toggles_frame,
        text="Homogeneous Initial Diffusion Const.",
        variable=homogeneous_initial_diffusion_const_var,
    )
    homogeneous_initial_diffusion_const_cb.grid(row=6, column=0, sticky="w")
    CreateToolTip(
        homogeneous_initial_diffusion_const_cb,
        GRADIENT_DESCENT_TOOLTIPS["Homogeneous Initial Diffusion Const."]
    )
    m1_facilitation_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable M1 Facilitated Diffusion",
        variable=enable_m1_facilitated_diffusion_var,
    )
    m1_facilitation_cb.grid(row=7, column=0, sticky="w")
    CreateToolTip(m1_facilitation_cb, GRADIENT_DESCENT_TOOLTIPS["Enable M1 Facilitated Diffusion"])

    m1_porin_diffusion_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable M1 Porin Diffusion (import-only)",
        variable=m1_porin_diffusion_var,
    )
    m1_porin_diffusion_cb.grid(row=8, column=0, sticky="w")
    CreateToolTip(m1_porin_diffusion_cb, GRADIENT_DESCENT_TOOLTIPS["Enable M1 Porin Diffusion"])

    chemostat_flow_cb = tk.Checkbutton(
        toggles_frame,
        text="Chemostat Flow",
        variable=enable_chemostat_flow_var,
    )
    chemostat_flow_cb.grid(row=9, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(chemostat_flow_cb, SIMULATION_SETTINGS_TOOLTIPS["Enable Chemostat Flow"])

    # Initial energy: toggle in Settings; numeric "Initial Energy" is in the Parameters section.
    enable_initial_energy_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable Initial Energy",
        variable=enable_initial_energy_var,
    )
    enable_initial_energy_cb.grid(row=10, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(enable_initial_energy_cb, SIMULATION_SETTINGS_TOOLTIPS["Enable Initial Energy"])

    def _refresh_initial_energy_state():
        if "Initial Energy" not in entries:
            return
        on = bool(enable_initial_energy_var.get())
        _set_param_row_visible("Initial Energy", on)
        if not on and "Initial Energy" in entry_vars:
            entry_vars["Initial Energy"].set("0.0")

    enable_initial_energy_var.trace_add("write", lambda *_: _refresh_initial_energy_state())
    _refresh_initial_energy_state()

    # Intermediate storage energetic penalty toggle (uses "Intermediate Costs" parameter).
    enable_intermediate_costs_var = tk.BooleanVar(value=bool(preset_params.get("Enable Intermediate Costs", False)))
    enable_intermediate_costs_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable Intermediate Costs",
        variable=enable_intermediate_costs_var,
    )
    enable_intermediate_costs_cb.grid(row=11, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(enable_intermediate_costs_cb, SIMULATION_SETTINGS_TOOLTIPS["Enable Intermediate Costs"])

    def _refresh_intermediate_costs_state():
        if enable_m2_diffusion_var.get():
            enable_intermediate_costs_cb.configure(state="normal")
        else:
            enable_intermediate_costs_var.set(False)
            enable_intermediate_costs_cb.configure(state="disabled")
        if "Intermediate Costs" not in entries:
            return
        show = bool(enable_intermediate_costs_var.get())
        _set_param_row_visible("Intermediate Costs", show)
        if not show and "Intermediate Costs" in entry_vars:
            entry_vars["Intermediate Costs"].set("0.0")

    enable_intermediate_costs_var.trace_add("write", lambda *_: _refresh_intermediate_costs_state())
    enable_m2_diffusion_var.trace_add("write", lambda *_: _refresh_intermediate_costs_state())
    _refresh_intermediate_costs_state()

    # Optional acetate addition each generation.
    enable_acetate_addition_var = tk.BooleanVar(value=bool(preset_params.get("Enable Acetate Addition", False)))
    enable_acetate_addition_cb = tk.Checkbutton(
        toggles_frame,
        text="Enable Acetate Addition",
        variable=enable_acetate_addition_var,
    )
    enable_acetate_addition_cb.grid(row=12, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(enable_acetate_addition_cb, SIMULATION_SETTINGS_TOOLTIPS["Enable Acetate Addition"])

    def _refresh_acetate_addition_state():
        if "Average In_Flow (Acetate)" not in entries:
            return
        show = bool(enable_acetate_addition_var.get())
        _set_param_row_visible("Average In_Flow (Acetate)", show)
        if not show and "Average In_Flow (Acetate)" in entry_vars:
            entry_vars["Average In_Flow (Acetate)"].set("0.0")

    enable_acetate_addition_var.trace_add("write", lambda *_: _refresh_acetate_addition_state())
    _refresh_acetate_addition_state()

    binary_death_at_zero_energy_var = tk.BooleanVar(
        value=bool(preset_params.get(BINARY_DEATH_AT_ZERO_ENERGY, False))
    )
    binary_death_at_zero_energy_cb = tk.Checkbutton(
        toggles_frame,
        text="Binary Death at Zero Energy",
        variable=binary_death_at_zero_energy_var,
    )
    binary_death_at_zero_energy_cb.grid(row=13, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(binary_death_at_zero_energy_cb, SIMULATION_SETTINGS_TOOLTIPS["Binary Death at Zero Energy"])

    no_death_var = tk.BooleanVar(value=bool(preset_params.get(NO_DEATH, False)))
    no_death_cb = tk.Checkbutton(
        toggles_frame,
        text=NO_DEATH,
        variable=no_death_var,
    )
    no_death_cb.grid(row=14, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(no_death_cb, SIMULATION_SETTINGS_TOOLTIPS[NO_DEATH])

    constant_death_probability_var = tk.BooleanVar(
        value=bool(preset_params.get(CONSTANT_DEATH_PROBABILITY, False))
    )
    constant_death_probability_cb = tk.Checkbutton(
        toggles_frame,
        text="Constant Death Probability",
        variable=constant_death_probability_var,
    )
    constant_death_probability_cb.grid(row=15, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(
        constant_death_probability_cb,
        SIMULATION_SETTINGS_TOOLTIPS["Constant Death Probability"],
    )

    constant_duplication_probability_var = tk.BooleanVar(
        value=bool(preset_params.get(CONSTANT_DUPLICATION_PROBABILITY, False))
    )
    constant_duplication_probability_cb = tk.Checkbutton(
        toggles_frame,
        text="Constant Duplication Probability",
        variable=constant_duplication_probability_var,
    )
    constant_duplication_probability_cb.grid(row=16, column=0, sticky="w", pady=(6, 0))
    CreateToolTip(
        constant_duplication_probability_cb,
        SIMULATION_SETTINGS_TOOLTIPS["Constant Duplication Probability"],
    )

    def _constant_probability_mode_active() -> bool:
        return any_constant_probability_mode(
            {
                NO_DEATH: bool(no_death_var.get()),
                CONSTANT_DEATH_PROBABILITY: bool(constant_death_probability_var.get()),
                CONSTANT_DUPLICATION_PROBABILITY: bool(constant_duplication_probability_var.get()),
            }
        )

    _death_dup_flow_reconciling = {"active": False}

    def _refresh_death_dup_rate_visibility(*, prefer: str = ""):
        """Apply order-independent death/dup/flow constraints, then sync param visibility."""
        if _death_dup_flow_reconciling["active"]:
            return
        _death_dup_flow_reconciling["active"] = True
        try:
            visibility = apply_death_dup_flow_toggle_ui(
                no_death_var=no_death_var,
                constant_death_probability_var=constant_death_probability_var,
                constant_duplication_probability_var=constant_duplication_probability_var,
                enable_chemostat_flow_var=enable_chemostat_flow_var,
                no_death_checkbox=no_death_cb,
                constant_death_checkbox=constant_death_probability_cb,
                constant_duplication_checkbox=constant_duplication_probability_cb,
                binary_death_at_zero_energy=bool(binary_death_at_zero_energy_var.get()),
                prefer=prefer,
            )
            if "Death Decay Rate" in entries:
                _set_param_row_visible("Death Decay Rate", not visibility.hide_death_decay_rate)
            for name in ("Duplication Sigmoid Midpoint", "Duplication Sigmoid Intensity"):
                if name in entries:
                    _set_param_row_visible(name, not visibility.hide_duplication_sigmoid)
            if CONSTANT_PROBABILITY in entries:
                _set_param_row_visible(CONSTANT_PROBABILITY, visibility.show_constant_probability)
        finally:
            _death_dup_flow_reconciling["active"] = False

    _refresh_death_dup_rate_visibility()

    _preset_toggles = {
        NO_DEATH: bool(no_death_var.get()),
        CONSTANT_DEATH_PROBABILITY: bool(constant_death_probability_var.get()),
        CONSTANT_DUPLICATION_PROBABILITY: bool(constant_duplication_probability_var.get()),
    }
    if any_constant_probability_mode(_preset_toggles):
        resolved = resolve_constant_probability({**preset_params, **_preset_toggles})
        if resolved is not None and CONSTANT_PROBABILITY in entry_vars:
            entry_vars[CONSTANT_PROBABILITY].set(str(resolved))

    def _refresh_flow_percentage_state():
        if "Flow Percentage" not in entries:
            return
        show = bool(enable_chemostat_flow_var.get())
        _set_param_row_visible("Flow Percentage", show)
        if not show and "Flow Percentage" in entry_vars:
            entry_vars["Flow Percentage"].set("0.0")
        _refresh_death_dup_rate_visibility()
        # Called during early UI init before update_probability_plot is defined.
        try:
            update_probability_plot()
        except NameError:
            pass
    enable_chemostat_flow_var.trace_add("write", lambda *_: _refresh_flow_percentage_state())
    _refresh_flow_percentage_state()

    updating_m1_mode = False

    def refresh_m1_diffusion_mode():
        nonlocal updating_m1_mode
        if updating_m1_mode:
            return
        updating_m1_mode = True
        # Default: all three mode toggles are selectable.
        m1_diffusion_cb.configure(state="normal")
        m1_facilitation_cb.configure(state="normal")
        m1_porin_diffusion_cb.configure(state="normal")
        if enable_m1_facilitated_diffusion_var.get():
            enable_m1_diffusion_var.set(False)
            m1_porin_diffusion_var.set(False)
            m1_diffusion_cb.configure(state="disabled")
            m1_porin_diffusion_cb.configure(state="disabled")
        elif m1_porin_diffusion_var.get():
            enable_m1_diffusion_var.set(False)
            enable_m1_facilitated_diffusion_var.set(False)
            m1_diffusion_cb.configure(state="disabled")
            m1_facilitation_cb.configure(state="disabled")
        elif enable_m1_diffusion_var.get():
            enable_m1_facilitated_diffusion_var.set(False)
            m1_porin_diffusion_var.set(False)
            m1_facilitation_cb.configure(state="disabled")
            m1_porin_diffusion_cb.configure(state="disabled")
        updating_m1_mode = False

    enable_m1_diffusion_var.trace_add("write", lambda *_: refresh_m1_diffusion_mode())
    enable_m1_facilitated_diffusion_var.trace_add("write", lambda *_: refresh_m1_diffusion_mode())
    m1_porin_diffusion_var.trace_add("write", lambda *_: refresh_m1_diffusion_mode())
    refresh_m1_diffusion_mode()

    def refresh_diffusion_mutation_state():
        m1_effective_on = bool(
            enable_m1_diffusion_var.get() or enable_m1_facilitated_diffusion_var.get() or m1_porin_diffusion_var.get()
        )
        if enable_m2_diffusion_var.get() or m1_effective_on:
            diffusion_mutation_cb.configure(state="normal")
        else:
            enable_diffusion_mutation_var.set(False)
            diffusion_mutation_cb.configure(state="disabled")
        if enable_diffusion_mutation_var.get():
            homogeneous_initial_diffusion_const_cb.configure(state="normal")
        else:
            homogeneous_initial_diffusion_const_var.set(False)
            homogeneous_initial_diffusion_const_cb.configure(state="disabled")

    enable_m1_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    enable_m1_facilitated_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    m1_porin_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    enable_m2_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    enable_diffusion_mutation_var.trace_add("write", lambda *_: refresh_diffusion_mutation_state())
    refresh_diffusion_mutation_state()

    def _diffusion_transport_on() -> bool:
        return bool(
            enable_m1_diffusion_var.get()
            or enable_m1_facilitated_diffusion_var.get()
            or m1_porin_diffusion_var.get()
            or enable_m2_diffusion_var.get()
        )

    def refresh_diffusion_related_param_visibility():
        snapshot_entries()
        diffusion_on = _diffusion_transport_on()
        for param_name, show in (
            ("Diffusion Constant", diffusion_on),
            ("Chemostat Volume", diffusion_on),
        ):
            if param_name not in entries:
                continue
            w_label = labels.get(param_name)
            w_entry = entries[param_name]
            if show:
                if w_label is not None:
                    w_label.grid()
                w_entry.grid()
            else:
                if w_label is not None:
                    w_label.grid_remove()
                w_entry.grid_remove()
        restore_entries()

    enable_m1_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_related_param_visibility())
    enable_m1_facilitated_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_related_param_visibility())
    m1_porin_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_related_param_visibility())
    enable_m2_diffusion_var.trace_add("write", lambda *_: refresh_diffusion_related_param_visibility())
    refresh_diffusion_related_param_visibility()

    def refresh_facilitation_param_visibility():
        snapshot_entries()
        show_trait = enable_m1_facilitated_diffusion_var.get() and homogeneous_var.get()
        show_cost = enable_m1_facilitated_diffusion_var.get()

        if "Initial Facilitation" in entries:
            w_label = labels.get("Initial Facilitation")
            w_entry = entries["Initial Facilitation"]
            if show_trait:
                if w_label is not None:
                    w_label.grid()
                w_entry.grid()
            else:
                if w_label is not None:
                    w_label.grid_remove()
                w_entry.grid_remove()

        if "Cost of Transport" in entries:
            w_label = labels.get("Cost of Transport")
            w_entry = entries["Cost of Transport"]
            if show_cost:
                if w_label is not None:
                    w_label.grid()
                w_entry.grid()
            else:
                if w_label is not None:
                    w_label.grid_remove()
                w_entry.grid_remove()
        restore_entries()

    enable_m1_facilitated_diffusion_var.trace_add("write", lambda *_: refresh_facilitation_param_visibility())
    refresh_facilitation_param_visibility()

    preset_seed = preset_params.get("Random Seed (optional)", "")
    seed_var = tk.StringVar(value=str(preset_seed) if preset_seed is not None else "")
    seed_label = tk.Label(toggles_frame, text="Random Seed (optional):")
    seed_label.grid(row=17, column=0, sticky="w", pady=(8, 0))
    seed_entry = tk.Entry(toggles_frame, width=14, textvariable=seed_var)
    seed_entry.grid(row=18, column=0, sticky="w")
    CreateToolTip(seed_label, PARAMETER_TOOLTIPS["Random Seed (optional)"])
    CreateToolTip(seed_entry, PARAMETER_TOOLTIPS["Random Seed (optional)"])

    # Initial A/B should be hidden when not homogeneous
    def refresh_initial_a_visibility():
        snapshot_entries()
        if "Initial A" not in entries:
            return
        w_label = labels.get("Initial A")
        w_entry = entries["Initial A"]
        if homogeneous_var.get():
            if w_label is not None:
                w_label.grid()
            w_entry.grid()
        else:
            if w_label is not None:
                w_label.grid_remove()
            w_entry.grid_remove()
        if "Initial B" in entries:
            w_label_b = labels.get("Initial B")
            w_entry_b = entries["Initial B"]
            show_b = homogeneous_var.get() and independent_traits_var.get()
            if show_b:
                if w_label_b is not None:
                    w_label_b.grid()
                w_entry_b.grid()
            else:
                if w_label_b is not None:
                    w_label_b.grid_remove()
                w_entry_b.grid_remove()
        restore_entries()
        refresh_facilitation_param_visibility()

    homogeneous_var.trace_add("write", lambda *_: refresh_initial_a_visibility())
    independent_traits_var.trace_add("write", lambda *_: refresh_initial_a_visibility())

    # === Results plot area (middle column): tabs like the main individual GUI ===
    notebook = ttk.Notebook(results)
    notebook.pack(fill="both", expand=True)

    tab_main = tk.Frame(notebook)
    tab_m1 = tk.Frame(notebook)
    tab_m2 = tk.Frame(notebook)
    tab_budgets = tk.Frame(notebook)
    tab_budgets2 = tk.Frame(notebook)
    tab_metrics = tk.Frame(notebook)

    notebook.add(tab_main, text="Main Plots")
    notebook.add(tab_m1, text="M1 Metabolite")
    notebook.add(tab_m2, text="M2 Metabolite")
    notebook.add(tab_budgets, text="Budgets")
    notebook.add(tab_budgets2, text="Budget 2")
    notebook.add(tab_metrics, text="Metrics")

    # --- Main Plots tab (heat strips + 4 panels) ---
    fig = plt.Figure(figsize=(9.5, 6.2), dpi=100)
    # IMPORTANT: dedicate a fixed column for the colorbar so the heat-strip axes
    # never "shrink" across repeated runs (colorbars won't accumulate width).
    gs = fig.add_gridspec(5, 3, height_ratios=[0.22, 0.22, 0.22, 1, 1], width_ratios=[1, 1, 0.08])
    axAB = fig.add_subplot(gs[0, 0:2])  # 1D heatmap strip (Trait A)
    axBstrip = fig.add_subplot(gs[1, 0:2])  # 1D heatmap strip (Trait B; independent mode)
    axTstrip = fig.add_subplot(gs[2, 0:2])  # 1D heatmap strip (Facilitated diffusion genotype T; optional)
    axCbar = fig.add_subplot(gs[0:3, 2])  # shared colorbar axis for heat strips
    axSpacer = fig.add_subplot(gs[3:5, 2])  # unused spacer column (keeps layout stable)
    axSpacer.axis("off")
    axCbar.axis("off")

    axN = fig.add_subplot(gs[3, 0])
    axFlux = fig.add_subplot(gs[3, 1])
    axEnergy = fig.add_subplot(gs[4, 0])
    axChanges = fig.add_subplot(gs[4, 1])
    canvas = FigureCanvasTkAgg(fig, master=tab_main)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    ab_cbar = None

    # --- M1 Metabolite tab ---
    m1_fig = plt.Figure(figsize=(9.0, 6.2), dpi=100)
    m1_gs = m1_fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    m1_ax1 = m1_fig.add_subplot(m1_gs[0, 0])  # M1 env
    m1_ax2 = m1_fig.add_subplot(m1_gs[0, 1])  # M1 internal
    m1_ax3 = m1_fig.add_subplot(m1_gs[1, 0])  # total M1 (stored + env)
    m1_ax4 = m1_fig.add_subplot(m1_gs[1, 1])  # global M1 diffusion
    m1_ax_hist = m1_fig.add_subplot(m1_gs[2, :])  # Internal M1 histogram (current gen)
    m1_canvas = FigureCanvasTkAgg(m1_fig, master=tab_m1)
    m1_canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- M2 Metabolite tab ---
    m2_fig = plt.Figure(figsize=(9.0, 6.2), dpi=100)
    m2_gs = m2_fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
    m2_ax1 = m2_fig.add_subplot(m2_gs[0, 0])  # M2 env
    m2_ax2 = m2_fig.add_subplot(m2_gs[0, 1])  # M2 internal
    m2_ax3 = m2_fig.add_subplot(m2_gs[1, 0])  # total M2 (stored + env)
    m2_ax4 = m2_fig.add_subplot(m2_gs[1, 1])  # global M2 diffusion
    m2_ax_hist = m2_fig.add_subplot(m2_gs[2, :])  # Internal M2 histogram (current gen)
    m2_canvas = FigureCanvasTkAgg(m2_fig, master=tab_m2)
    m2_canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Budgets tab ---
    bud_fig = plt.Figure(figsize=(9.0, 6.2), dpi=100)
    bud_gs = bud_fig.add_gridspec(2, 2)
    bud_ax1 = bud_fig.add_subplot(bud_gs[0, 0])  # Task budgets (mean)
    bud_ax2 = bud_fig.add_subplot(bud_gs[0, 1])  # Import/export (mean)
    bud_ax3 = bud_fig.add_subplot(bud_gs[1, 0])  # Totals + M2 budget
    bud_ax4 = bud_fig.add_subplot(bud_gs[1, 1])  # Net flux mean
    bud_canvas = FigureCanvasTkAgg(bud_fig, master=tab_budgets)
    bud_canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Budget 2 tab (M1/M2 budget breakdowns only) ---
    bud2_fig = plt.Figure(figsize=(9.0, 4.6), dpi=100)
    bud2_gs = bud2_fig.add_gridspec(1, 2)
    bud2_ax_m2 = bud2_fig.add_subplot(bud2_gs[0, 0])  # M2 budget breakdown
    bud2_ax_m1 = bud2_fig.add_subplot(bud2_gs[0, 1])  # M1 budget breakdown
    bud2_canvas = FigureCanvasTkAgg(bud2_fig, master=tab_budgets2)
    bud2_canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Metrics tab ---
    metrics_controls = tk.Frame(tab_metrics, padx=10, pady=10)
    metrics_controls.pack(side="top", fill="x")
    metrics_controls_row1 = tk.Frame(metrics_controls)
    metrics_controls_row1.pack(side="top", fill="x")
    metrics_controls_row2 = tk.Frame(metrics_controls)
    metrics_controls_row2.pack(side="top", fill="x", pady=(6, 0))
    tk.Label(metrics_controls_row1, text="Metric:").pack(side="left")
    def _available_metric_options():
        metrics = list(getattr(model_spec, "metric_names", []) or [])
        if getattr(model_spec, "key", "") == "simulation":
            m1_on = bool(
                enable_m1_diffusion_var.get()
                or enable_m1_facilitated_diffusion_var.get()
                or m1_porin_diffusion_var.get()
            )
            metrics = filter_metric_options_for_simulation_settings(
                metrics,
                enable_m2_diffusion=bool(enable_m2_diffusion_var.get()),
                enable_m1_diffusion=m1_on,
                enable_intermediate_costs=bool(enable_intermediate_costs_var.get()),
            )
        return metrics

    metric_options = _available_metric_options()
    selected_metric_var = tk.StringVar(value=(metric_options[0] if metric_options else ""))
    metrics_menu = ttk.Combobox(
        metrics_controls_row1,
        textvariable=selected_metric_var,
        values=metric_options,
        state="readonly",
        width=42,
    )
    metrics_menu.pack(side="left", padx=(6, 8), fill="x", expand=True)
    run_metric_btn = tk.Button(metrics_controls_row1, text="Run")
    run_metric_btn.pack(side="left")
    tk.Label(metrics_controls_row2, text="Seed Sweep N:").pack(side="left")
    seed_sweep_count_var = tk.StringVar(value="100")
    seed_sweep_count_entry = tk.Entry(metrics_controls_row2, textvariable=seed_sweep_count_var, width=6)
    seed_sweep_count_entry.pack(side="left", padx=(0, 6))
    run_seed_sweep_btn = tk.Button(metrics_controls_row2, text="Sweep Seeds")
    run_seed_sweep_btn.pack(side="left")
    metric_description_var = tk.StringVar(value="")
    metric_description_lbl = tk.Label(
        tab_metrics,
        textvariable=metric_description_var,
        anchor="nw",
        justify="left",
        wraplength=400,
        fg="#444444",
        padx=10,
        pady=0,
    )
    metric_description_lbl.pack(side="top", fill="x", pady=(0, 8))

    metrics_result_var = tk.StringVar(value="Run a simulation, then select a metric and click Run.")
    metrics_result_lbl = tk.Label(
        tab_metrics,
        textvariable=metrics_result_var,
        anchor="nw",
        justify="left",
        wraplength=400,
        padx=10,
        pady=10,
    )
    metrics_result_lbl.pack(side="top", fill="both", expand=True)

    seed_sweep_panel = tk.LabelFrame(tab_metrics, text="Seed Sweep", padx=10, pady=8)
    seed_sweep_panel.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

    seed_sweep_progress_var = tk.DoubleVar(value=0.0)
    seed_sweep_progress_text_var = tk.StringVar(value="0 / 0 (0.0%)")
    seed_sweep_summary_var = tk.StringVar(value="Run a seed sweep to view distribution here.")

    seed_sweep_progress_bar = ttk.Progressbar(
        seed_sweep_panel,
        mode="determinate",
        maximum=1.0,
        variable=seed_sweep_progress_var,
    )
    seed_sweep_progress_bar.pack(fill="x", pady=(0, 6))
    seed_sweep_progress_text_lbl = tk.Label(
        seed_sweep_panel,
        textvariable=seed_sweep_progress_text_var,
        anchor="w",
        justify="left",
        wraplength=400,
    )
    seed_sweep_progress_text_lbl.pack(fill="x", pady=(0, 6))

    seed_sweep_fig = plt.Figure(figsize=(8.8, 4.2), dpi=100)
    seed_sweep_ax = seed_sweep_fig.add_subplot(111)
    seed_sweep_canvas = FigureCanvasTkAgg(seed_sweep_fig, master=seed_sweep_panel)
    seed_sweep_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(0, 6))
    seed_sweep_summary_lbl = tk.Label(
        seed_sweep_panel,
        textvariable=seed_sweep_summary_var,
        anchor="w",
        justify="left",
        wraplength=400,
    )
    seed_sweep_summary_lbl.pack(fill="x")

    def _sync_metrics_tab_wrapping(_event=None):
        """Keep label wraplength in sync with tab width (fixed wraplength clips when tab is narrow)."""
        try:
            tw = int(tab_metrics.winfo_width())
            pw = int(seed_sweep_panel.winfo_width())
        except tk.TclError:
            return
        if tw < 40:
            return
        inner = max(120, tw - 28)
        metric_description_lbl.configure(wraplength=inner)
        metrics_result_lbl.configure(wraplength=inner)
        if pw >= 40:
            pwrap = max(120, pw - 24)
            seed_sweep_progress_text_lbl.configure(wraplength=pwrap)
            seed_sweep_summary_lbl.configure(wraplength=pwrap)

    tab_metrics.bind("<Configure>", lambda _e: _sync_metrics_tab_wrapping())
    win.after_idle(_sync_metrics_tab_wrapping)

    _seed_sweep_fig_px = [0, 0]

    def _sync_seed_sweep_fig_geometry(event):
        if event.widget is not seed_sweep_canvas.get_tk_widget():
            return
        w, h = int(event.width), int(event.height)
        if w < 24 or h < 24:
            return
        if _seed_sweep_fig_px[0] == w and _seed_sweep_fig_px[1] == h:
            return
        _seed_sweep_fig_px[0] = w
        _seed_sweep_fig_px[1] = h
        dpi = float(seed_sweep_fig.get_dpi())
        try:
            seed_sweep_fig.set_size_inches(w / dpi, h / dpi, forward=False)
        except Exception:
            return
        seed_sweep_canvas.draw_idle()

    seed_sweep_canvas.get_tk_widget().bind("<Configure>", _sync_seed_sweep_fig_geometry)

    def _reset_seed_sweep_plot(title_text):
        seed_sweep_ax.clear()
        seed_sweep_ax.set_title(title_text)
        seed_sweep_ax.text(
            0.5,
            0.5,
            "Histogram will appear here after sweep completes.",
            ha="center",
            va="center",
            transform=seed_sweep_ax.transAxes,
        )
        seed_sweep_ax.set_xticks([])
        seed_sweep_ax.set_yticks([])
        seed_sweep_canvas.draw_idle()

    _reset_seed_sweep_plot("Seed Sweep Distribution")

    source_metric_name = str(preset_params.get("_source_metric_name", "") or "").strip()
    source_metric_value_raw = preset_params.get("_source_metric_value", np.nan)
    try:
        source_metric_value = float(source_metric_value_raw)
    except Exception:
        source_metric_value = np.nan
    source_replicate_info_raw = preset_params.get("_source_replicate_info", None)
    source_num_replicates = 1
    source_num_succeeded = None
    source_num_failed = None
    if isinstance(source_replicate_info_raw, (list, tuple)) and len(source_replicate_info_raw) >= 1:
        try:
            source_num_replicates = max(1, int(source_replicate_info_raw[0]))
        except Exception:
            source_num_replicates = 1
        if len(source_replicate_info_raw) >= 2:
            try:
                source_num_succeeded = int(source_replicate_info_raw[1])
            except Exception:
                source_num_succeeded = None
        if len(source_replicate_info_raw) >= 3:
            try:
                source_num_failed = int(source_replicate_info_raw[2])
            except Exception:
                source_num_failed = None

    def _refresh_metric_description(*_args):
        metric_name = selected_metric_var.get().strip()
        if not metric_name:
            metric_description_var.set("Select a metric to see its definition.")
            return
        metric_description_var.set(f"Metric Definition: {describe_metric(metric_name)}")

    selected_metric_var.trace_add("write", _refresh_metric_description)
    _refresh_metric_description()

    def _refresh_metric_options():
        options = _available_metric_options()
        metrics_menu["values"] = options
        current = selected_metric_var.get().strip()
        if current not in options:
            selected_metric_var.set(options[0] if options else "")

    enable_m2_diffusion_var.trace_add("write", lambda *_: _refresh_metric_options())
    enable_m1_diffusion_var.trace_add("write", lambda *_: _refresh_metric_options())
    enable_m1_facilitated_diffusion_var.trace_add("write", lambda *_: _refresh_metric_options())
    m1_porin_diffusion_var.trace_add("write", lambda *_: _refresh_metric_options())
    enable_intermediate_costs_var.trace_add("write", lambda *_: _refresh_metric_options())

    def _show_initial_message(ax, title):
        ax.clear()
        ax.set_title(title)
        ax.text(0.5, 0.5, "Run a simulation to populate this tab.", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    # === Diagnostics (right column) ===
    prob_frame = tk.LabelFrame(diagnostics, text="Death/Duplication Prob", padx=6, pady=6)
    prob_frame.pack(fill="x", pady=(0, 10))

    prob_warning_var = tk.StringVar(value="")
    prob_warning_lbl = tk.Label(
        prob_frame,
        textvariable=prob_warning_var,
        anchor="w",
        justify="left",
        wraplength=320,
        fg="crimson",
        bg="#fff3f3",
        relief="solid",
        borderwidth=1,
        padx=6,
        pady=4,
    )

    prob_fig = plt.Figure(figsize=(3.2, 2.3), dpi=100)
    prob_ax = prob_fig.add_subplot(1, 1, 1)
    prob_canvas = FigureCanvasTkAgg(prob_fig, master=prob_frame)
    prob_canvas.get_tk_widget().pack(fill="both", expand=True)

    inv_frame = tk.LabelFrame(diagnostics, text="Investment Function", padx=6, pady=6)
    inv_frame.pack(fill="x", pady=(0, 10))

    inv_fig = plt.Figure(figsize=(3.2, 2.3), dpi=100)
    inv_ax = inv_fig.add_subplot(1, 1, 1)
    inv_canvas = FigureCanvasTkAgg(inv_fig, master=inv_frame)
    inv_canvas.get_tk_widget().pack(fill="both", expand=True)

    status = tk.StringVar(value="Ready.")
    status_lbl = tk.Label(left, textvariable=status, anchor="w", justify="left", wraplength=260)
    status_lbl.pack(fill="x", pady=(6, 0))

    progress = ttk.Progressbar(left, mode="indeterminate")
    progress.pack(fill="x", pady=(6, 0))
    progress.stop()

    # Keep last results for tab redraws if needed
    last_result = {}

    def _run_selected_metric():
        metric_name = selected_metric_var.get().strip()
        if not metric_name:
            messagebox.showwarning("Metric Required", "Please select a metric first.")
            return
        if not last_result:
            messagebox.showwarning("No Simulation Result", "Run a simulation first, then calculate a metric.")
            return
        compute_metric_fn = getattr(model_spec, "compute_metric", None)
        if not callable(compute_metric_fn):
            messagebox.showerror("Metric Unavailable", "This model does not define metric computation.")
            return
        run_model_fn = getattr(model_spec, "run_simulation", None)
        cmp_params_snapshot = None
        cmp_base_seed = None
        if (
            source_metric_name
            and metric_name == source_metric_name
            and np.isfinite(source_metric_value)
            and source_num_replicates > 1
            and callable(run_model_fn)
        ):
            try:
                cmp_params_snapshot = parse_params()
                cmp_base_seed = cmp_params_snapshot.pop("_seed", None)
                if cmp_base_seed is None:
                    cmp_base_seed = parse_optional_seed(seed_var.get().strip())
            except Exception:
                cmp_params_snapshot = None
                cmp_base_seed = None

        progress.start(10)
        status.set(f"Computing metric: {metric_name}...")
        run_result_snapshot = dict(last_result)

        def worker():
            try:
                metric_value = float(compute_metric_fn(run_result_snapshot, metric_name))
            except Exception as e:
                def _show_err():
                    progress.stop()
                    status.set("Metric computation failed.")
                    messagebox.showerror("Metric Error", f"Failed to compute metric '{metric_name}':\n{e}")
                win.after(0, _show_err)
                return

            run_gens = len(run_result_snapshot.get("A_history", []) or [])
            lines = [f"Metric: {metric_name}"]
            if np.isnan(metric_value):
                lines.append("Value: NaN")
                lines.append("Status: unavailable for the current result (collapsed run or missing required data).")
            else:
                lines.append(f"Value: {metric_value:.10g}")
                lines.append(f"Generations in result: {run_gens}")

            # If launched from a GD/UMAP point, show source-point metric context.
            if source_metric_name:
                if metric_name == source_metric_name and np.isfinite(source_metric_value):
                    if source_num_succeeded is not None and source_num_failed is not None:
                        lines.append(
                            f"Source point value (GD/UMAP): {source_metric_value:.10g} "
                            f"[replicates {source_num_succeeded}/{source_num_replicates} succeeded, {source_num_failed} failed]"
                        )
                    else:
                        lines.append(
                            f"Source point value (GD/UMAP): {source_metric_value:.10g} "
                            f"[replicates: {source_num_replicates}]"
                        )

                    # Compute a comparable replicate-averaged metric when possible.
                    if cmp_params_snapshot is not None and cmp_base_seed is not None and callable(run_model_fn):
                        try:
                            vals = []
                            for rep_i in range(source_num_replicates):
                                local_params = dict(cmp_params_snapshot)
                                local_params["Random Seed (optional)"] = replicate_seed_for_run(
                                    int(cmp_base_seed), rep_i, source_num_replicates
                                )
                                local_params["silent"] = True
                                rep_res = run_model_fn(local_params)
                                rep_val = float(compute_metric_fn(rep_res, metric_name))
                                if not np.isnan(rep_val):
                                    vals.append(rep_val)
                            if vals:
                                lines.append(
                                    f"Comparable replicate mean (this GUI): {float(np.mean(vals)):.10g} "
                                    f"[{len(vals)}/{source_num_replicates} succeeded]"
                                )
                        except Exception:
                            pass
                elif metric_name != source_metric_name:
                    lines.append(
                        f"Source point tracked metric: {source_metric_name}"
                    )

            def finish():
                progress.stop()
                metrics_result_var.set("\n".join(lines))
                status.set(f"Metric computed: {metric_name}")

            win.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    run_metric_btn.configure(command=_run_selected_metric)

    def _run_metric_seed_sweep():
        metric_name = selected_metric_var.get().strip()
        if not metric_name:
            messagebox.showwarning("Metric Required", "Please select a metric first.")
            return
        run_model_fn = getattr(model_spec, "run_simulation", None)
        compute_metric_fn = getattr(model_spec, "compute_metric", None)
        if not callable(run_model_fn) or not callable(compute_metric_fn):
            messagebox.showerror("Unavailable", "This model does not support seed-sweep metric analysis.")
            return
        try:
            n_sweeps = int(float(seed_sweep_count_var.get() or 0))
        except Exception:
            n_sweeps = 100
        n_sweeps = max(2, min(n_sweeps, 2000))
        seed_sweep_count_var.set(str(n_sweeps))

        try:
            base_params = parse_params()
        except Exception as e:
            messagebox.showerror("Parameter Error", f"Failed to parse current parameters:\n{e}")
            return

        base_seed = base_params.pop("_seed", None)
        if base_seed is None:
            base_seed = parse_optional_seed(seed_var.get().strip())
        rng = np.random.default_rng(base_seed if base_seed is not None else None)
        sampled_seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=n_sweeps)]

        progress.start(10)
        status.set(f"Running seed sweep for '{metric_name}' ({n_sweeps} seeds)...")
        win.update_idletasks()
        run_seed_sweep_btn.configure(state="disabled")
        seed_sweep_progress_bar.configure(maximum=float(n_sweeps))
        seed_sweep_progress_var.set(0.0)
        seed_sweep_progress_text_var.set(f"0 / {n_sweeps} (0.0%)")
        seed_sweep_summary_var.set(f"Running '{metric_name}' across {n_sweeps} random seeds...")
        _reset_seed_sweep_plot(f"Seed Sweep Distribution: {metric_name}")
        worker_queue = queue.Queue()

        def worker():
            vals_local = []
            failed_local = 0
            for i, local_seed in enumerate(sampled_seeds, start=1):
                local_params = dict(base_params)
                local_params["Random Seed (optional)"] = local_seed
                local_params["silent"] = True
                try:
                    res = run_model_fn(local_params)
                    mv = float(compute_metric_fn(res, metric_name))
                    if np.isnan(mv):
                        failed_local += 1
                    else:
                        vals_local.append(mv)
                except Exception:
                    failed_local += 1
                worker_queue.put(("progress", i, len(vals_local), failed_local))
            worker_queue.put(("done", vals_local, failed_local))

        def poll_worker_queue():
            finished = False
            while True:
                try:
                    event = worker_queue.get_nowait()
                except queue.Empty:
                    break
                etype = event[0]
                if etype == "progress":
                    i, valid_n, failed_n = event[1], event[2], event[3]
                    seed_sweep_progress_var.set(float(i))
                    seed_sweep_progress_text_var.set(
                        f"{i} / {n_sweeps} ({(100.0 * i / n_sweeps):.1f}%)"
                    )
                    status.set(
                        f"Seed sweep '{metric_name}': {i}/{n_sweeps} "
                        f"(valid={valid_n}, failed={failed_n})"
                    )
                elif etype == "done":
                    vals, failed = event[1], event[2]
                    progress.stop()
                    run_seed_sweep_btn.configure(state="normal")

                    if not vals:
                        seed_sweep_summary_var.set(
                            f"No valid metric values were produced for '{metric_name}' "
                            f"across {n_sweeps} random seeds. Failed/NaN: {failed}."
                        )
                        _reset_seed_sweep_plot(f"Seed Sweep Distribution: {metric_name}")
                        status.set("Seed sweep finished: no valid values.")
                        finished = True
                        continue

                    arr = np.asarray(vals, dtype=float)
                    mean_v = float(np.mean(arr))
                    std_v = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

                    current_val = np.nan
                    if last_result:
                        try:
                            current_val = float(compute_metric_fn(last_result, metric_name))
                        except Exception:
                            current_val = np.nan

                    percentile = np.nan
                    if np.isfinite(current_val):
                        percentile = float(100.0 * np.mean(arr < current_val))

                    bins = min(80, max(20, int(np.sqrt(arr.size) * 2)))
                    seed_sweep_ax.clear()
                    seed_sweep_ax.hist(arr, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
                    if np.isfinite(current_val):
                        seed_sweep_ax.axvline(
                            current_val, color="#E45756", linestyle="--", linewidth=2, label="Current run"
                        )
                    seed_sweep_ax.axvline(mean_v, color="#54A24B", linestyle="-", linewidth=2, label="Sweep mean")
                    seed_sweep_ax.set_title(f"Metric over random seeds: {metric_name}")
                    seed_sweep_ax.set_xlabel("Metric value")
                    seed_sweep_ax.set_ylabel("Count")
                    seed_sweep_ax.grid(alpha=0.25)
                    seed_sweep_ax.legend(loc="best")
                    seed_sweep_canvas.draw_idle()

                    summary_lines = [
                        f"Seeds sampled: {n_sweeps}",
                        f"Valid runs: {arr.size}",
                        f"Failed/NaN: {failed}",
                        f"Mean: {mean_v:.8g}",
                        f"Std (sample): {std_v:.8g}",
                    ]
                    if np.isfinite(current_val):
                        summary_lines.append(f"Current run value: {current_val:.8g}")
                        summary_lines.append(f"Current run percentile in sweep: {percentile:.2f}")
                    else:
                        summary_lines.append("Current run value: NaN / unavailable")
                    seed_sweep_summary_var.set(" | ".join(summary_lines))

                    status.set(
                        f"Seed sweep complete for '{metric_name}': valid={arr.size}, "
                        f"failed={failed}, mean={mean_v:.6g}, std={std_v:.6g}"
                    )
                    finished = True

            if not finished:
                win.after(100, poll_worker_queue)

        threading.Thread(target=worker, daemon=True).start()
        win.after(100, poll_worker_queue)

    run_seed_sweep_btn.configure(command=_run_metric_seed_sweep)

    def _get_gen_array(history, g):
        """Get per-organism array for generation g, falling back to nearest previous non-empty."""
        if not history:
            return np.array([], dtype=float)
        try:
            g = int(g)
        except Exception:
            g = len(history) - 1
        g = max(0, min(g, len(history) - 1))
        for idx in range(g, -1, -1):
            arr = history[idx]
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float)
            if a.size > 0:
                return a
        return np.array([], dtype=float)

    def update_internal_histograms():
        """Update internal M1/M2 histograms for the latest available generation."""
        res = last_result or {}
        A_hist = res.get("A_history", []) or []
        storage_m1_hist = res.get("storage_M1_history", []) or []
        storage_m2_hist = res.get("storage_history", []) or []

        # Choose a conservative max generation based on what we actually have stored.
        max_gen = max(len(A_hist), len(storage_m1_hist), len(storage_m2_hist)) - 1
        if max_gen < 0:
            max_gen = 0

        g = max_gen

        m1_ax_hist.clear()
        m2_ax_hist.clear()

        m1_vals = _get_gen_array(storage_m1_hist, g)
        m2_vals = _get_gen_array(storage_m2_hist, g)

        # Filter out non-finite values
        m1_vals = m1_vals[np.isfinite(m1_vals)] if m1_vals.size else m1_vals
        m2_vals = m2_vals[np.isfinite(m2_vals)] if m2_vals.size else m2_vals

        bins = 50
        if m1_vals.size > 0:
            m1_ax_hist.hist(m1_vals, bins=bins, color=get_series_color("m1_env"), alpha=0.85, edgecolor="none")
            m1_ax_hist.set_title(f"Internal M1 histogram (gen {g}; N={m1_vals.size})", fontsize=10)
            m1_ax_hist.set_xlabel("Internal M1 (amount per organism)")
            m1_ax_hist.set_ylabel("Count")
        else:
            _show_initial_message(m1_ax_hist, "Internal M1 histogram (no data)")

        if m2_vals.size > 0:
            m2_ax_hist.hist(m2_vals, bins=bins, color=get_series_color("m2_internal"), alpha=0.85, edgecolor="none")
            m2_ax_hist.set_title(f"Internal M2 histogram (gen {g}; N={m2_vals.size})", fontsize=10)
            m2_ax_hist.set_xlabel("Internal M2 (amount per organism)")
            m2_ax_hist.set_ylabel("Count")
        else:
            _show_initial_message(m2_ax_hist, "Internal M2 histogram (no data)")

        m1_fig.tight_layout()
        m2_fig.tight_layout()
        m1_canvas.draw()
        m2_canvas.draw()

    def update_probability_plot():
        """Update the death/duplication probability plot based on current parameters."""
        try:
            no_death = bool(no_death_var.get())
            binary_death = bool(binary_death_at_zero_energy_var.get())
            constant_death = bool(constant_death_probability_var.get())
            constant_dup = bool(constant_duplication_probability_var.get())
            binary_death_effective = binary_death and (not no_death)
            constant_death_effective = constant_death and (not no_death)
            dupx0_val = float(entries.get("Duplication Sigmoid Midpoint").get() or 2.5)
            dupk_val = float(entries.get("Duplication Sigmoid Intensity").get() or 5.0)
            death_decay_val = float(entries.get("Death Decay Rate").get() or 10.0)
            flat_prob_raw = float(entries.get(CONSTANT_PROBABILITY).get() or 0.5) if CONSTANT_PROBABILITY in entries else 0.5
            # Match simulation/core.py: constant_probability is clamped before use.
            flat_prob_val = float(np.clip(flat_prob_raw, 0.0, 1.0))
            enable_flow = bool(enable_chemostat_flow_var.get())
            flow_pct_val = float(entries.get("Flow Percentage").get() or 0.0) if "Flow Percentage" in entries else 0.0
            flow_prob_val = float(np.clip(flow_pct_val / 100.0, 0.0, 1.0)) if enable_flow else 0.0
            death_prob_val = 0.0 if no_death else (flat_prob_val if constant_death_effective else 0.5)
            dup_prob_val = flat_prob_val if constant_dup else 0.5
            correction_active = False
            correction_note = ""
            invalid_mode = False
            invalid_reasons = []
            if no_death and constant_death:
                invalid_mode = True
                invalid_reasons.append("No Death and Constant Death cannot both be enabled.")
            if no_death and constant_dup and (not enable_flow):
                invalid_mode = True
                invalid_reasons.append("No Death + Constant Duplication requires Chemostat Flow enabled.")
            if no_death and constant_dup and enable_flow and flow_pct_val > 50.0:
                invalid_mode = True
                invalid_reasons.append("No Death + Constant Duplication requires Flow Percentage <= 50.")
            if constant_dup and no_death:
                flow_frac = flow_prob_val
                denom = max(1e-12, 1.0 - flow_frac)
                dup_prob_val = float(np.clip(flow_frac / denom, 0.0, 1.0))
                correction_active = True
                correction_note = f"No Death mode: dup p={dup_prob_val:.4g} from flow={flow_pct_val:.4g}%"
            elif constant_dup and constant_death_effective and (not binary_death_effective):
                # Match core exactly: duplication correction uses p/(1-p) and
                # this mode is valid only for base p<=0.5.
                if flat_prob_val <= 0.5:
                    denom = max(1e-12, 1.0 - dup_prob_val)
                    dup_prob_val = dup_prob_val / denom
                    dup_prob_val = float(np.clip(dup_prob_val, 0.0, 1.0))
                    correction_active = True
                    correction_note = f"Corrected dup p={dup_prob_val:.4g} from base p={flat_prob_val:.4g}"
                else:
                    invalid_mode = True
                    invalid_reasons.append("Coupled constant death+dup requires Constant Probability <= 0.5.")

            # Choose energy range: show sub-zero behavior, then up to near-saturation of dup. sigmoid
            E_min = -1.0
            target_p = 0.995
            if constant_dup:
                E_max = 5.0
            elif dupk_val > 0:
                logit = np.log(target_p / (1.0 - target_p))
                E_sat = dupx0_val + (logit / dupk_val)
                E_max = max(1.0, min(50.0, E_sat + 0.5))
            else:
                E_max = 5.0

            E = np.linspace(E_min, E_max, 250)

            if invalid_mode:
                p_death = np.full_like(E, np.nan, dtype=float)
                p_dup = np.full_like(E, np.nan, dtype=float)
            elif no_death:
                p_death = np.zeros_like(E, dtype=float)
                p_dup = effective_duplication_probabilities_for_display(
                    E,
                    compute_duplication_probabilities(
                        E,
                        constant_duplication_probability=constant_dup,
                        duplication_probability=dup_prob_val,
                        dupk=dupk_val,
                        dupx0=dupx0_val,
                    ),
                    binary_death_at_zero_energy=binary_death_effective,
                )
            else:
                p_death = compute_death_probabilities(
                    E,
                    binary_death_at_zero_energy=binary_death_effective,
                    constant_death_probability=constant_death_effective,
                    death_probability=death_prob_val,
                    death_decay_rate=death_decay_val,
                )
                p_dup = effective_duplication_probabilities_for_display(
                    E,
                    compute_duplication_probabilities(
                        E,
                        constant_duplication_probability=constant_dup,
                        duplication_probability=dup_prob_val,
                        dupk=dupk_val,
                        dupx0=dupx0_val,
                    ),
                    binary_death_at_zero_energy=binary_death_effective,
                )

            prob_ax.clear()
            prob_ax.plot(E, p_death, color=get_series_color("death"), lw=1.2, label="Death")
            dup_label = "Duplicate (corrected)" if correction_active else "Duplicate"
            prob_ax.plot(E, p_dup, color=get_series_color("duplication"), lw=1.2, label=dup_label)
            p_flow = np.full_like(E, flow_prob_val, dtype=float)
            flow_label = "Chemostat Overflow" if enable_flow else "Chemostat Overflow (off)"
            prob_ax.plot(E, p_flow, color="black", lw=1.2, ls="--", label=flow_label)
            prob_ax.set_ylim(0, 1)
            prob_ax.set_xlim(E_min, E_max)
            prob_ax.set_title("Death/Duplication Prob", fontsize=9)
            prob_ax.set_xlabel("Energy", fontsize=8)
            prob_ax.set_ylabel("Probability", fontsize=8)
            prob_ax.legend(fontsize=7, loc="upper right")
            prob_ax.tick_params(labelsize=7)
            if invalid_reasons:
                prob_warning_var.set("Invalid settings:\n" + "\n".join(invalid_reasons))
                if not prob_warning_lbl.winfo_ismapped():
                    prob_warning_lbl.pack(fill="x", pady=(0, 6))
                prob_warning_lbl.configure(fg="crimson", bg="#fff3f3")
            else:
                if correction_note:
                    prob_warning_var.set(correction_note)
                    if not prob_warning_lbl.winfo_ismapped():
                        prob_warning_lbl.pack(fill="x", pady=(0, 6))
                    prob_warning_lbl.configure(fg="dimgray", bg="#f5f5f5")
                else:
                    prob_warning_var.set("")
                    if prob_warning_lbl.winfo_ismapped():
                        prob_warning_lbl.pack_forget()

            prob_fig.tight_layout()
            prob_canvas.draw()
        except (ValueError, TypeError, AttributeError):
            # Leave it as-is if params invalid/missing
            pass

    def _refresh_death_dup_rate_ui(*, prefer: str = ""):
        _refresh_death_dup_rate_visibility(prefer=prefer)
        update_probability_plot()

    binary_death_at_zero_energy_var.trace_add(
        "write", lambda *_: _refresh_death_dup_rate_ui()
    )
    no_death_var.trace_add("write", lambda *_: _refresh_death_dup_rate_ui(prefer="no_death"))
    constant_death_probability_var.trace_add(
        "write", lambda *_: _refresh_death_dup_rate_ui(prefer="constant_death")
    )
    constant_duplication_probability_var.trace_add(
        "write", lambda *_: _refresh_death_dup_rate_ui()
    )

    def update_investment_plot():
        """Plot the current investment function y=f(x) on x in [0,1]."""
        if independent_traits_var.get():
            inv_ax.clear()
            inv_ax.text(
                0.5,
                0.5,
                "Investment function disabled\n(independent traits)",
                ha="center",
                va="center",
                fontsize=9,
            )
            inv_ax.set_xticks([])
            inv_ax.set_yticks([])
            inv_fig.tight_layout()
            inv_canvas.draw()
            return
        try:
            inv_mod_val = float(entries.get("Investment Modifier").get() or 1.0)
        except Exception:
            return

        x = np.linspace(0.0, 1.0, 200)
        try:
            # This model currently uses the Exponential investment function.
            y = investment_func(inv_mod_val, x)
            y = np.asarray(y, dtype=float)
        except Exception:
            return

        inv_ax.clear()
        inv_ax.plot(x, y, color=get_series_color("m1_env"), lw=1.4)
        inv_ax.set_xlim(0, 1)
        inv_ax.set_ylim(0, 1)
        inv_ax.set_title(f"Exponential (modifier={inv_mod_val:g})", fontsize=9)
        inv_ax.set_xlabel("Trait value (x)", fontsize=8)
        inv_ax.set_ylabel("Investment f(x)", fontsize=8)
        inv_ax.tick_params(labelsize=7)
        inv_fig.tight_layout()
        inv_canvas.draw()

    def refresh_independent_traits_state():
        # Coupled traits only: hide Investment Modifier when using independent A/B.
        _set_param_row_visible("Investment Modifier", not bool(independent_traits_var.get()))
        update_investment_plot()

    def parse_params():
        params = {}
        for name, e in entries.items():
            # Skip Initial A when not homogeneous
            if name == "Initial A" and not homogeneous_var.get():
                continue
            if name == "Initial B" and (not homogeneous_var.get() or not independent_traits_var.get()):
                continue
            if name == "Initial Facilitation" and (not homogeneous_var.get() or not enable_m1_facilitated_diffusion_var.get()):
                continue
            if name == "Cost of Transport" and not enable_m1_facilitated_diffusion_var.get():
                continue
            if name == "Intermediate Costs" and not enable_intermediate_costs_var.get():
                continue
            if name == "Initial Energy":
                continue
            if name == "Average In_Flow (Acetate)" and not enable_acetate_addition_var.get():
                continue
            if name == "Flow Percentage" and not enable_chemostat_flow_var.get():
                continue
            if name == "Death Decay Rate" and (
                no_death_var.get() or binary_death_at_zero_energy_var.get() or constant_death_probability_var.get()
            ):
                continue
            if name in ("Duplication Sigmoid Midpoint", "Duplication Sigmoid Intensity") and constant_duplication_probability_var.get():
                continue
            if name == CONSTANT_PROBABILITY and not _constant_probability_mode_active():
                continue
            if name == "Investment Modifier" and independent_traits_var.get():
                continue
            val = e.get().strip()
            try:
                if name == "Number of Generations" or name == "Initial Organism Count":
                    params[name] = int(float(val))
                else:
                    params[name] = float(val)
            except Exception:
                raise ValueError(f"Invalid value for '{name}': {val!r}")

        params[HOMOGENEOUS_POPULATION] = bool(homogeneous_var.get())
        params[INDEPENDENT_TRAITS] = bool(independent_traits_var.get())
        params["silent"] = bool(silent_var.get())
        params[ENABLE_M1_DIFFUSION] = bool(enable_m1_diffusion_var.get())
        params[ENABLE_M2_DIFFUSION] = bool(enable_m2_diffusion_var.get())
        params[ENABLE_M1_FACILITATED_DIFFUSION] = bool(enable_m1_facilitated_diffusion_var.get())
        params[ENABLE_M1_PORIN_DIFFUSION] = bool(m1_porin_diffusion_var.get())
        params[ENABLE_CHEMOSTAT_FLOW] = bool(enable_chemostat_flow_var.get())
        if params[ENABLE_M1_FACILITATED_DIFFUSION] or params[ENABLE_M1_PORIN_DIFFUSION]:
            params[ENABLE_M1_DIFFUSION] = True
        if params[ENABLE_M1_FACILITATED_DIFFUSION]:
            params[ENABLE_M1_PORIN_DIFFUSION] = False
        params[ENABLE_INITIAL_ENERGY] = bool(enable_initial_energy_var.get())
        if params[ENABLE_INITIAL_ENERGY]:
            if "Initial Energy" in entries:
                ie_txt = entries["Initial Energy"].get().strip()
                try:
                    params[INITIAL_ENERGY] = float(ie_txt) if ie_txt else 0.0
                except Exception:
                    params[INITIAL_ENERGY] = 0.0
            else:
                params[INITIAL_ENERGY] = 0.0
        else:
            params[INITIAL_ENERGY] = 0.0
        params[ENABLE_INTERMEDIATE_COSTS] = bool(enable_intermediate_costs_var.get())
        if not params[ENABLE_INTERMEDIATE_COSTS]:
            params[INTERMEDIATE_COSTS] = 0.0
        elif INTERMEDIATE_COSTS not in params:
            params[INTERMEDIATE_COSTS] = 0.0
        params[ENABLE_ACETATE_ADDITION] = bool(enable_acetate_addition_var.get())
        if not params[ENABLE_ACETATE_ADDITION]:
            params[AVERAGE_INFLOW_ACETATE] = 0.0
        elif AVERAGE_INFLOW_ACETATE not in params:
            params[AVERAGE_INFLOW_ACETATE] = 0.0
        params[BINARY_DEATH_AT_ZERO_ENERGY] = bool(binary_death_at_zero_energy_var.get())
        params[NO_DEATH] = bool(no_death_var.get())
        params[CONSTANT_DEATH_PROBABILITY] = bool(constant_death_probability_var.get())
        params[CONSTANT_DUPLICATION_PROBABILITY] = bool(constant_duplication_probability_var.get())
        if _constant_probability_mode_active() and CONSTANT_PROBABILITY not in params:
            params[CONSTANT_PROBABILITY] = 0.5
        if not params[ENABLE_CHEMOSTAT_FLOW]:
            params[FLOW_PERCENTAGE] = 0.0
        allow_diffusion_mutation = params[ENABLE_M2_DIFFUSION] or params[ENABLE_M1_DIFFUSION]
        params[ENABLE_DIFFUSION_MUTATION] = bool(enable_diffusion_mutation_var.get()) if allow_diffusion_mutation else False
        params[HOMOGENEOUS_INITIAL_DIFFUSION_CONST] = (
            bool(homogeneous_initial_diffusion_const_var.get()) if params[ENABLE_DIFFUSION_MUTATION] else False
        )

        seed_txt = seed_var.get().strip()
        if seed_txt != "":
            try:
                params["_seed"] = int(seed_txt)
            except Exception:
                raise ValueError(f"Invalid seed: {seed_txt!r}")

        return normalize_simulation_params(params)

    def plot_results(result: dict):
        nonlocal last_result
        last_result = result or {}

        nonlocal ab_cbar, axCbar
        axAB.clear()
        axTstrip.clear()
        axN.clear()
        axFlux.clear()
        axEnergy.clear()
        axChanges.clear()
        # IMPORTANT: if we created the colorbar using a fixed cax (`axCbar`), calling
        # `ab_cbar.remove()` will also remove that axes from the figure, breaking future updates.
        if ab_cbar is not None:
            try:
                if getattr(ab_cbar, "ax", None) is axCbar:
                    axCbar.cla()
                    axCbar.axis("off")
                else:
                    ab_cbar.remove()
            except Exception:
                pass
            ab_cbar = None

        # Self-heal if a previous colorbar removal detached axCbar from the figure
        try:
            if axCbar is None or axCbar.get_figure() is None:
                # Keep the rebuilt colorbar axis span consistent with initial layout.
                axCbar = fig.add_subplot(gs[0:3, 2])
                axCbar.axis("off")
        except Exception:
            pass
        axCbar.cla()
        axCbar.axis("off")

        if not result:
            axAB.set_title("Run collapsed / failed")
            axTstrip.set_title("Facilitated diffusion genotype (T) — N/A")
            axTstrip.axis("off")
            canvas.draw()
            # Also clear other tabs
            for a, t in [
                (m1_ax1, "M1 (env concentration)"),
                (m1_ax2, "M1 (internal concentration, mean ± std)"),
                (m1_ax3, "Total Stored vs Env M1"),
                (m1_ax4, "Global M1 Diffusion (Import/Export)"),
                (m2_ax1, "M2 (env concentration)"),
                (m2_ax2, "M2 (internal concentration, mean ± std)"),
                (m2_ax3, "Total Stored vs Env M2"),
                (m2_ax4, "Global M2 Diffusion (Import/Export)"),
                (axChanges, "Deaths / Duplications"),
                (bud_ax1, "Task budgets"),
                (bud_ax2, "M1 Transport Behavior Distribution"),
                (bud_ax3, "Totals / budget"),
                (bud_ax4, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
                (bud2_ax_m2, "M2 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
                (bud2_ax_m1, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
            ]:
                _show_initial_message(a, t)
            m1_fig.tight_layout()
            m2_fig.tight_layout()
            bud_fig.tight_layout()
            bud2_fig.tight_layout()
            m1_canvas.draw()
            m2_canvas.draw()
            bud_canvas.draw()
            bud2_canvas.draw()
            return

        collapsed = bool(result.get("collapsed", False))
        if collapsed:
            status.set("Collapsed (showing last available data).")

        A_hist = result.get("A_history", [])
        B_hist = result.get("B_history", [])
        T_hist = result.get("T_history", [])
        exp_hist = result.get("M2_export_history", [])
        imp_hist = result.get("M2_import_history", [])
        m1_exp_hist = result.get("M1_export_history", [])
        m1_imp_hist = result.get("M1_import_history", [])
        task1_hist = result.get("task1_performance", [])
        task2_hist = result.get("task2_performance", [])
        metabs_hist = result.get("metabs_history", [])
        energy_hist = result.get("energy_history", [])
        storage_hist = result.get("storage_history", [])
        cell_volume = 1.0
        run_params = result.get("_run_params", {}) if isinstance(result, dict) else {}
        m1_diffusion_active = bool(
            run_params.get("Enable M1 Diffusion", enable_m1_diffusion_var.get())
            or run_params.get("Enable M1 Facilitated Diffusion", enable_m1_facilitated_diffusion_var.get())
            or run_params.get("Enable M1 Porin Diffusion", m1_porin_diffusion_var.get())
        )
        m2_diffusion_active = bool(run_params.get("Enable M2 Diffusion", enable_m2_diffusion_var.get()))
        diffusion_density_plots = m1_diffusion_active or m2_diffusion_active

        if not A_hist:
            axAB.set_title("No history returned")
            canvas.draw()
            return

        A_mean, A_std, pop = _mean_std_series(A_hist)
        independent_traits = bool(run_params.get("Independent Traits", False))
        if B_hist:
            B_mean, B_std, _ = _mean_std_series(B_hist)
        else:
            # Fallback to coupled traits when independent mode is off
            if independent_traits:
                B_mean = np.zeros_like(A_mean)
                B_std = np.zeros_like(A_std)
            else:
                B_mean = 1.0 - A_mean
                B_std = A_std

        t = np.arange(len(A_mean))

        # Heat line of Trait A for the current (final) generation
        t_last = len(A_hist) - 1
        gens_hist = np.arange(1, t_last + 2)
        A_last = A_hist[-1]
        if A_last is None:
            for v in reversed(A_hist):
                if v is not None:
                    A_last = v
                    break
        if A_last is None:
            axAB.set_title("Trait A (no final values)")
        else:
            A_last = np.asarray(A_last, dtype=float)
            if B_hist and B_hist[-1] is not None:
                B_last = np.asarray(B_hist[-1], dtype=float)
            else:
                B_last = 1.0 - A_last if not independent_traits else np.zeros_like(A_last)

            n = int(min(len(A_last), len(B_last)))
            A_last = A_last[:n]
            B_last = B_last[:n]

            bins = 60
            counts, edges = np.histogram(A_last, bins=bins, range=(0.0, 1.0))
            counts = counts.astype(float)
            counts = counts / counts.max() if counts.max() > 0 else counts
            heat = counts[np.newaxis, :]
            im = axAB.imshow(
                heat,
                origin="lower",
                aspect="auto",
                extent=[0.0, 1.0, 0.0, 1.0],
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )
            axAB.set_yticks([])
            axAB.set_ylim(0, 1)
            axAB.set_title(f"Trait A (gen {t_last}; N={n})", fontsize=10)
            axAB.set_xlabel("Trait A", fontsize=9)
            # Use fixed colorbar axis (prevents cumulative shrinking)
            axCbar.axis("on")
            ab_cbar = fig.colorbar(im, cax=axCbar)
            ab_cbar.set_label("Normalized density", fontsize=8)
            ab_cbar.ax.tick_params(labelsize=7)

        # Heat line of Trait B (only when independent traits are enabled)
        if independent_traits and B_hist:
            B_last = B_hist[-1]
            if B_last is None:
                for v in reversed(B_hist):
                    if v is not None:
                        B_last = v
                        break
            if B_last is None:
                axBstrip.set_title("Trait B (no final values)", fontsize=10)
                axBstrip.axis("off")
            else:
                B_last = np.asarray(B_last, dtype=float)
                bins = 60
                counts, _ = np.histogram(B_last, bins=bins, range=(0.0, 1.0))
                counts = counts.astype(float)
                counts = counts / counts.max() if counts.max() > 0 else counts
                heat = counts[np.newaxis, :]
                axBstrip.imshow(
                    heat,
                    origin="lower",
                    aspect="auto",
                    extent=[0.0, 1.0, 0.0, 1.0],
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                )
                axBstrip.set_yticks([])
                axBstrip.set_ylim(0, 1)
                axBstrip.set_title(f"Trait B (gen {t_last})", fontsize=10)
                axBstrip.set_xlabel("Trait B", fontsize=9)
        else:
            axBstrip.set_title("Trait B — disabled", fontsize=10)
            axBstrip.axis("off")

        # Second heat strip: facilitated diffusion genotype (T), only when enabled and available
        if enable_m1_facilitated_diffusion_var.get() and T_hist:
            T_last = T_hist[-1]
            if T_last is None:
                for v in reversed(T_hist):
                    if v is not None:
                        T_last = v
                        break
            if T_last is None:
                axTstrip.set_title("Facilitated diffusion genotype T (no final values)", fontsize=10)
                axTstrip.axis("off")
            else:
                T_last = np.asarray(T_last, dtype=float)
                if T_last.size > 0:
                    bins = 60
                    counts, _ = np.histogram(T_last, bins=bins, range=(0.0, 1.0))
                    counts = counts.astype(float)
                    counts = counts / counts.max() if counts.max() > 0 else counts
                    heat = counts[np.newaxis, :]
                    axTstrip.imshow(
                        heat,
                        origin="lower",
                        aspect="auto",
                        extent=[0.0, 1.0, 0.0, 1.0],
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    axTstrip.set_yticks([])
                    axTstrip.set_ylim(0, 1)
                    axTstrip.set_title(f"Facilitated diffusion genotype (T) (gen {t_last}; N={int(T_last.size)})", fontsize=10)
                    axTstrip.set_xlabel("Trait T (facilitation)", fontsize=9)
                else:
                    axTstrip.set_title("Facilitated diffusion genotype (T) (empty)", fontsize=10)
                    axTstrip.axis("off")
        else:
            # Keep layout stable but hide when not enabled
            axTstrip.set_title("Facilitated diffusion genotype (T) — disabled", fontsize=10)
            axTstrip.axis("off")

        axN.set_title("Population size")
        axN.plot(t, pop, color=get_series_color("population"), lw=1.6)
        axN.set_xlabel("Generation")
        axN.set_ylabel("N")

        # Trait A mean ± std (main plots)
        if A_hist:
            A_mean, A_std, _ = _mean_std_series(A_hist)
            axFlux.set_title("Trait A (mean ± std)")
            axFlux.plot(t[: len(A_mean)], A_mean, color=get_series_color("m1_env"), lw=1.6)
            axFlux.fill_between(t[: len(A_mean)], A_mean - A_std, A_mean + A_std, color=get_series_color("m1_env"), alpha=0.2)
            axFlux.set_xlabel("Generation")
            axFlux.set_ylabel("Trait A")
        else:
            axFlux.set_title("Trait A (N/A)")
            axFlux.axis("off")

        if energy_hist:
            e_mean, e_std, _ = _mean_std_series(energy_hist)
            t_e = np.arange(len(e_mean))
            axEnergy.plot(t_e, e_mean, lw=1.6, color=get_series_color("energy"))
            axEnergy.fill_between(
                t_e,
                e_mean - e_std,
                e_mean + e_std,
                color=get_series_color("energy"),
                alpha=0.2,
            )
            axEnergy.set_title("Energy (mean ± 1 std)")
            axEnergy.set_xlabel("Generation")
            axEnergy.set_ylabel("Mean energy")
        else:
            _show_initial_message(axEnergy, "Energy (mean ± 1 std)")

        change_hist = result.get("change_history", [])
        if change_hist:
            parsed_rows = parse_change_history_rows(change_hist)
            if parsed_rows:
                deaths = np.asarray([row[0] for row in parsed_rows], dtype=float)
                dups = np.asarray([row[1] for row in parsed_rows], dtype=float)
                flow_removed = np.asarray([row[3] for row in parsed_rows], dtype=float)
                t_c = np.arange(len(deaths))
                axChanges.plot(t_c, deaths, lw=1.4, color=get_series_color("death"), label="Deaths")
                axChanges.plot(t_c, dups, lw=1.4, color=get_series_color("duplication"), label="Duplications")
                axChanges.plot(t_c, flow_removed, lw=1.4, color=get_series_color("net"), label="Flow Removals")
                axChanges.set_title("Deaths / Duplications / Flow Removals")
                axChanges.set_xlabel("Generation")
                axChanges.set_ylabel("Count")
                axChanges.legend(fontsize=7, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.08))
                axChanges.tick_params(labelsize=8)
                axChanges.grid(alpha=0.3, zorder=0)
            else:
                _show_initial_message(axChanges, "Deaths / Duplications")
        else:
            _show_initial_message(axChanges, "Deaths / Duplications")

        fig.tight_layout()
        canvas.draw()

        # === Metabolite tabs ===
        m1_ax1.clear()
        m1_ax2.clear()
        m1_ax3.clear()
        m1_ax4.clear()
        m1_ax_hist.clear()
        m2_ax1.clear()
        m2_ax2.clear()
        m2_ax3.clear()
        m2_ax4.clear()
        m2_ax_hist.clear()
        # Precompute net flux mean for M2 tab
        net_mean = None
        if exp_hist and imp_hist and len(exp_hist) == len(imp_hist):
            net_mean = np.zeros(len(exp_hist), dtype=float)
            for i in range(len(exp_hist)):
                ex = exp_hist[i]
                im = imp_hist[i]
                if ex is None or im is None:
                    continue
                ex = np.asarray(ex, dtype=float)
                im = np.asarray(im, dtype=float)
                if ex.size == 0 or im.size == 0:
                    continue
                net_mean[i] = float(np.mean(ex - im))

        if metabs_hist:
            met_arr = np.asarray(metabs_hist, dtype=float)
            t_met = np.arange(len(met_arr))
            if met_arr.ndim == 2 and met_arr.shape[1] >= 2:
                n = min(len(met_arr), len(pop)) if len(pop) else len(met_arr)
                met_arr = met_arr[:n]
                t_met = t_met[:n]
                if diffusion_density_plots:
                    try:
                        chemostat_volume = float(entries.get("Chemostat Volume").get())
                    except Exception:
                        chemostat_volume = 5000.0
                    if chemostat_volume <= 0:
                        chemostat_volume = 1.0
                    pop_arr = pop[:n] if len(pop) else np.full(n, np.nan)
                    effective_vol = np.asarray(
                        [
                            _effective_chemostat_volume(chemostat_volume, int(p), cell_volume)
                            if np.isfinite(p)
                            else chemostat_volume
                            for p in pop_arr
                        ],
                        dtype=float,
                    )
                    m1_y = met_arr[:, 0] / effective_vol
                    m2_y = met_arr[:, 1] / effective_vol
                    y_label = "Concentration"
                    m1_title = "M1 (env concentration)"
                    m2_title = "M2 (env concentration)"
                else:
                    m1_y = met_arr[:, 0]
                    m2_y = met_arr[:, 1]
                    y_label = "Amount"
                    m1_title = "M1 (env amount)"
                    m2_title = "M2 (env amount)"

                m1_ax1.plot(t_met, m1_y, lw=1.6, color=get_series_color("m1_env"))
                m1_ax1.set_title(m1_title)
                m1_ax1.set_xlabel("Generation")
                m1_ax1.set_ylabel(y_label)

                m2_ax1.plot(t_met, m2_y, lw=1.6, color=get_series_color("m2_env"))
                m2_ax1.set_title(m2_title)
                m2_ax1.set_xlabel("Generation")
                m2_ax1.set_ylabel(y_label)
            else:
                _show_initial_message(m1_ax1, "M1 (env pool)")
                _show_initial_message(m2_ax1, "M2 (env pool)")
        else:
            _show_initial_message(m1_ax1, "M1 (env pool)")
            _show_initial_message(m2_ax1, "M2 (env pool)")

        storage_m1_hist = result.get("storage_M1_history", None)
        if m1_diffusion_active and storage_m1_hist:
            m1_mean, m1_std, _ = _mean_std_series(storage_m1_hist)
            t_m1 = np.arange(len(m1_mean))
            m1_mean_conc = m1_mean / cell_volume
            m1_std_conc = m1_std / cell_volume
            # Mean + 1σ and 2σ bands
            c = get_series_color("m1_env")
            m1_ax2.plot(t_m1, m1_mean_conc, lw=1.8, color=c, label="Mean")

            # Shaded bands (2σ behind 1σ)
            m1_ax2.fill_between(
                t_m1,
                m1_mean_conc - 2.0 * m1_std_conc,
                m1_mean_conc + 2.0 * m1_std_conc,
                color=c,
                alpha=0.10,
                label="±2 std",
            )
            m1_ax2.fill_between(
                t_m1,
                m1_mean_conc - m1_std_conc,
                m1_mean_conc + m1_std_conc,
                color=c,
                alpha=0.20,
                label="±1 std",
            )

            # Outlier points: per-organism internal M1 values outside ±2σ at each generation
            out_x = []
            out_y = []
            for g in range(len(m1_mean)):
                arr = storage_m1_hist[g] if g < len(storage_m1_hist) else None
                if arr is None:
                    continue
                a = np.asarray(arr, dtype=float)
                if a.size == 0:
                    continue
                mu = float(m1_mean[g])
                sd = float(m1_std[g])
                if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
                    continue
                mask = np.abs(a - mu) > (2.0 * sd)
                if not np.any(mask):
                    continue
                vals = a[mask] / cell_volume
                # Downsample per generation if extremely many outliers
                if vals.size > 200:
                    vals = downsample_for_plot(vals, max_points=200)
                out_x.extend([g] * int(vals.size))
                out_y.extend(vals.tolist())

            if len(out_x) > 0:
                if len(out_x) > 2000:
                    idx = np.linspace(0, len(out_x) - 1, 2000, dtype=int)
                    out_x_plot = np.asarray(out_x, dtype=int)[idx]
                    out_y_plot = np.asarray(out_y, dtype=float)[idx]
                else:
                    out_x_plot = np.asarray(out_x, dtype=int)
                    out_y_plot = np.asarray(out_y, dtype=float)

                m1_ax2.scatter(
                    out_x_plot,
                    out_y_plot,
                    s=8,
                    alpha=0.35,
                    color="black",
                    linewidths=0,
                    label="Outliers (>2 std)",
                    zorder=5,
                )

            m1_ax2.set_title("M1 (internal concentration, mean ± 1/2 std + outliers)")
            m1_ax2.set_xlabel("Generation")
            m1_ax2.set_ylabel("Concentration")
            m1_ax2.grid(alpha=0.25)
            m1_ax2.legend(fontsize=7, loc="upper right", frameon=True)
        else:
            msg = "M1 (internal concentration, mean ± std)"
            if not m1_diffusion_active:
                msg = "M1 internal concentration (requires M1 diffusion)"
            _show_initial_message(m1_ax2, msg)

        if storage_m1_hist and metabs_hist:
            storage_m1_sum = _sum_series(storage_m1_hist)
            met_arr = np.asarray(metabs_hist, dtype=float)
            m1_env = met_arr[:, 0] if met_arr.ndim == 2 and met_arr.shape[1] >= 1 else None
            if m1_env is not None and len(m1_env) > 0:
                n_tot = min(len(storage_m1_sum), len(m1_env))
                t_tot = np.arange(n_tot)
                m1_ax3.plot(t_tot, storage_m1_sum[:n_tot], lw=1.6, color=get_series_color("m1_env"), label="Stored M1 (total)")
                m1_ax3.plot(t_tot, m1_env[:n_tot], lw=1.6, color=get_series_color("m1_env"), alpha=0.6, label="Env M1 (total)")
                m1_ax3.set_title("Total Stored vs Env M1")
                m1_ax3.set_xlabel("Generation")
                m1_ax3.set_ylabel("Total amount")
                m1_ax3.legend(fontsize=7, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08))
                m1_ax3.tick_params(labelsize=8)
                m1_ax3.grid(alpha=0.3, zorder=0)
            else:
                _show_initial_message(m1_ax3, "Total Stored vs Env M1")
        else:
            _show_initial_message(m1_ax3, "Total Stored vs Env M1")

        if m1_diffusion_active and m1_exp_hist and m1_imp_hist and len(m1_exp_hist) == len(m1_imp_hist):
            total_import = np.zeros(t_last + 1, dtype=float)
            total_export = np.zeros(t_last + 1, dtype=float)
            for g in range(t_last + 1):
                ex = m1_exp_hist[g] if g < len(m1_exp_hist) else None
                im = m1_imp_hist[g] if g < len(m1_imp_hist) else None
                if im is not None and len(im) > 0:
                    total_import[g] = float(np.sum(np.asarray(im, dtype=float)))
                if ex is not None and len(ex) > 0:
                    total_export[g] = float(np.sum(np.asarray(ex, dtype=float)))

            m1_ax4.stackplot(
                gens_hist,
                total_import,
                total_export,
                labels=["Total Import", "Total Export"],
                colors=[get_series_color("import"), get_series_color("export")],
                alpha=0.8,
            )
            net = total_export - total_import
            m1_ax4.plot(gens_hist, net, color=get_series_color("net"), lw=1.2, alpha=0.7, label="Net (Export-Import)")
            m1_ax4.set_title("Global M1 Diffusion (Import/Export)", fontsize=10)
            m1_ax4.set_xlabel("Generation")
            m1_ax4.set_ylabel("Total amount", fontsize=9)
            m1_ax4.legend(fontsize=7, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08))
            m1_ax4.tick_params(labelsize=8)
            m1_ax4.grid(alpha=0.3, zorder=0)
        else:
            if m1_diffusion_active:
                _show_initial_message(m1_ax4, "Global M1 Diffusion (Import/Export)")
            else:
                _show_initial_message(m1_ax4, "Global M1 Diffusion (Import/Export)")
                m1_ax4.text(0.5, 0.38, "Diffusion disabled", ha="center", va="center", transform=m1_ax4.transAxes)

        if m2_diffusion_active and storage_hist:
            m2_mean, m2_std, _ = _mean_std_series(storage_hist)
            t_s = np.arange(len(m2_mean))
            m2_mean_conc = m2_mean / cell_volume
            m2_std_conc = m2_std / cell_volume
            m2_ax2.plot(t_s, m2_mean_conc, lw=1.6, color=get_series_color("m2_internal"))
            m2_ax2.fill_between(t_s, m2_mean_conc - m2_std_conc, m2_mean_conc + m2_std_conc, color=get_series_color("m2_internal"), alpha=0.2)
            m2_ax2.set_title("M2 (internal concentration, mean ± std)")
            m2_ax2.set_xlabel("Generation")
            m2_ax2.set_ylabel("Concentration")
        else:
            msg = "M2 (internal concentration, mean ± std)"
            if not m2_diffusion_active:
                msg = "M2 internal concentration (requires M2 diffusion)"
            _show_initial_message(m2_ax2, msg)

        if m2_diffusion_active and net_mean is not None:
            total_import = np.zeros(t_last + 1, dtype=float)
            total_export = np.zeros(t_last + 1, dtype=float)
            for g in range(t_last + 1):
                ex = exp_hist[g] if g < len(exp_hist) else None
                im = imp_hist[g] if g < len(imp_hist) else None
                if im is not None and len(im) > 0:
                    total_import[g] = float(np.sum(np.asarray(im, dtype=float)))
                if ex is not None and len(ex) > 0:
                    total_export[g] = float(np.sum(np.asarray(ex, dtype=float)))

            m2_ax4.stackplot(
                gens_hist,
                total_import,
                total_export,
                labels=["Total Import", "Total Export"],
                colors=[get_series_color("import"), get_series_color("export")],
                alpha=0.8,
            )
            net = total_export - total_import
            m2_ax4.plot(gens_hist, net, color=get_series_color("net"), lw=1.2, alpha=0.7, label="Net (Export-Import)")
            m2_ax4.set_title("Global M2 Diffusion (Import/Export)", fontsize=10)
            m2_ax4.set_xlabel("Generation")
            m2_ax4.set_ylabel("Total amount", fontsize=9)
            m2_ax4.legend(fontsize=7, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08))
            m2_ax4.tick_params(labelsize=8)
            m2_ax4.grid(alpha=0.3, zorder=0)
        else:
            if m2_diffusion_active:
                _show_initial_message(m2_ax4, "Global M2 Diffusion (Import/Export)")
            else:
                _show_initial_message(m2_ax4, "Global M2 Diffusion (Import/Export)")
                m2_ax4.text(0.5, 0.38, "Diffusion disabled", ha="center", va="center", transform=m2_ax4.transAxes)

        if storage_hist and metabs_hist:
            storage_sum = _sum_series(storage_hist)
            met_arr = np.asarray(metabs_hist, dtype=float)
            m2_env = met_arr[:, 1] if met_arr.ndim == 2 and met_arr.shape[1] >= 2 else None
            if m2_env is not None and len(m2_env) > 0:
                n_tot = min(len(storage_sum), len(m2_env))
                t_tot = np.arange(n_tot)
                m2_ax3.plot(t_tot, storage_sum[:n_tot], lw=1.6, color=get_series_color("m2_internal"), label="Stored M2 (total)")
                m2_ax3.plot(t_tot, m2_env[:n_tot], lw=1.6, color=get_series_color("m2_env"), label="Env M2 (total)")
                m2_ax3.set_title("Total Stored vs Env M2")
                m2_ax3.set_xlabel("Generation")
                m2_ax3.set_ylabel("Total amount")
                m2_ax3.legend(fontsize=7, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08))
                m2_ax3.tick_params(labelsize=8)
                m2_ax3.grid(alpha=0.3, zorder=0)
            else:
                _show_initial_message(m2_ax3, "Total Stored vs Env M2")
        else:
            _show_initial_message(m2_ax3, "Total Stored vs Env M2")

        m1_fig.tight_layout()
        m2_fig.tight_layout()
        m1_canvas.draw()
        m2_canvas.draw()

        # Update histograms (latest generation)
        update_internal_histograms()

        # === Budgets tab (stacked area charts like the main individual GUI) ===
        bud_ax1.clear()
        bud_ax2.clear()
        bud_ax3.clear()
        bud_ax4.clear()
        bud2_ax_m2.clear()
        bud2_ax_m1.clear()

        # Safety check
        number_gen = int(result.get("number_gen", 0)) if isinstance(result.get("number_gen", 0), (int, float)) else 0
        # Use the *available* history length for plotting (important when results are “minimal history”).
        available_gens = max(
            len(A_hist) if isinstance(A_hist, list) else 0,
            len(exp_hist) if isinstance(exp_hist, list) else 0,
            len(imp_hist) if isinstance(imp_hist, list) else 0,
            len(m1_exp_hist) if isinstance(m1_exp_hist, list) else 0,
            len(m1_imp_hist) if isinstance(m1_imp_hist, list) else 0,
            len(task1_hist) if isinstance(task1_hist, list) else 0,
            len(task2_hist) if isinstance(task2_hist, list) else 0,
        )
        if available_gens > 0:
            number_gen = min(number_gen, available_gens) if number_gen > 0 else available_gens
        if number_gen <= 0:
            for a, t_title in [
                (bud_ax1, "M2 Transport Behavior Distribution"),
                (bud_ax2, "M1 Transport Behavior Distribution"),
                (bud_ax3, "M2 Import Intensity Distribution"),
                (bud_ax4, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
            ]:
                _show_initial_message(a, t_title)
            bud_fig.tight_layout()
            bud_canvas.draw()
            for a, t_title in [
                (bud2_ax_m2, "M2 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
                (bud2_ax_m1, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
            ]:
                _show_initial_message(a, t_title)
            bud2_fig.tight_layout()
            bud2_canvas.draw()

        if collapsed:
            for ax in (axAB, axTstrip, m1_ax1, m1_ax3, m1_ax4, bud_ax1, bud2_ax_m2, bud2_ax_m1):
                ax.text(
                    0.02,
                    0.98,
                    "Collapsed",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="red",
                    fontsize=9,
                    alpha=0.8,
                )
            canvas.draw()
            m1_canvas.draw()
            bud_canvas.draw()
            bud2_canvas.draw()
            return

        t_last = number_gen - 1
        gens_hist = np.arange(1, t_last + 2)

        # --- Plot 1: M2 transport behavior distribution (net export-import bins) ---
        bins = [-np.inf, -0.2, -0.05, -0.001, 0.001, 0.05, 0.2, np.inf]
        bin_names = [
            "Strong Import",
            "Moderate Import",
            "Weak Import",
            "Non-Transport",
            "Weak Export",
            "Moderate Export",
            "Strong Export",
        ]
        bin_colors = [
            OKABE_ITO["blue"],
            OKABE_ITO["sky"],
            OKABE_ITO["green"],
            OKABE_ITO["light_gray"],
            OKABE_ITO["yellow"],
            OKABE_ITO["orange"],
            OKABE_ITO["vermillion"],
        ]

        bin_labels = []
        for i, name in enumerate(bin_names):
            lower = bins[i]
            upper = bins[i + 1]
            if lower == -np.inf:
                label = f"{name} (< {upper})"
            elif upper == np.inf:
                label = f"{name} (> {lower})"
            else:
                label = f"{name} ({lower} to {upper})"
            bin_labels.append(label)

        if m2_diffusion_active:
            fractions_by_bin = [[] for _ in range(len(bin_labels))]
            for g in range(t_last + 1):
                ex = exp_hist[g] if g < len(exp_hist) else None
                im = imp_hist[g] if g < len(imp_hist) else None
                if ex is not None and im is not None and len(ex) > 0 and len(im) > 0:
                    ex = np.asarray(ex, dtype=float)
                    im = np.asarray(im, dtype=float)
                    n = min(len(ex), len(im))
                    flux_arr = ex[:n] - im[:n]
                    total = len(flux_arr)
                    for i in range(len(bin_labels)):
                        count = np.sum((flux_arr >= bins[i]) & (flux_arr < bins[i + 1]))
                        fractions_by_bin[i].append(count / total if total else 0.0)
                else:
                    # No organisms => all non-transport
                    for i in range(len(bin_labels)):
                        fractions_by_bin[i].append(1.0 if i == 3 else 0.0)

            fractions_by_bin = [np.asarray(frac, dtype=float) for frac in fractions_by_bin]
            bud_ax1.stackplot(gens_hist, *fractions_by_bin, labels=bin_labels, colors=bin_colors, alpha=0.8)
            bud_ax1.set_title("M2 Transport Behavior Distribution", fontsize=10)
            bud_ax1.set_xlabel("Generation", fontsize=9)
            bud_ax1.set_ylabel("Fraction of Population", fontsize=9)
            bud_ax1.set_ylim(0, 1)
            bud_ax1.legend(
                fontsize=6.5,
                loc="upper center",
                ncol=2,
                frameon=True,
                bbox_to_anchor=(0.5, -0.08),
                fancybox=True,
                framealpha=0.95,
            )
            bud_ax1.tick_params(labelsize=8)
            bud_ax1.grid(alpha=0.3, zorder=0)
        else:
            _show_initial_message(bud_ax1, "M2 Transport Behavior Distribution")

        # --- Plot 2: M1 transport behavior distribution (net export-import bins) ---
        if m1_diffusion_active and m1_exp_hist and m1_imp_hist:
            fractions_by_bin = [[] for _ in bin_labels]
            for g in range(t_last + 1):
                ex = m1_exp_hist[g] if g < len(m1_exp_hist) else None
                im = m1_imp_hist[g] if g < len(m1_imp_hist) else None
                if ex is None or im is None:
                    for i in range(len(bin_labels)):
                        fractions_by_bin[i].append(0.0)
                    continue
                ex = np.asarray(ex, dtype=float)
                im = np.asarray(im, dtype=float)
                if ex.size == 0 or im.size == 0:
                    for i in range(len(bin_labels)):
                        fractions_by_bin[i].append(0.0)
                    continue
                net = ex - im
                total = float(len(net))
                for i in range(len(bin_labels)):
                    lower = bins[i]
                    upper = bins[i + 1]
                    count = float(np.sum((net >= lower) & (net < upper)))
                    fractions_by_bin[i].append(count / total if total > 0 else 0.0)
            fractions_by_bin = [np.asarray(frac, dtype=float) for frac in fractions_by_bin]
            bud_ax2.stackplot(gens_hist, *fractions_by_bin, labels=bin_labels, colors=bin_colors, alpha=0.8)
            bud_ax2.set_title("M1 Transport Behavior Distribution", fontsize=10)
            bud_ax2.set_xlabel("Generation", fontsize=9)
            bud_ax2.set_ylabel("Fraction of Population", fontsize=9)
            bud_ax2.set_ylim(0, 1)
            bud_ax2.legend(fontsize=6.5, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.08))
            bud_ax2.tick_params(labelsize=8)
            bud_ax2.grid(alpha=0.3, zorder=0)
        else:
            _show_initial_message(bud_ax2, "M1 Transport Behavior Distribution")

        # --- Plot 3: Per-organism M2 budget breakdown (matches main individual GUI style) ---
        # Each organism: Production + Imports - Consumption - Exports = ~ΔStorage
        run_params = result.get("_run_params", {}) if isinstance(result, dict) else {}
        m2_diffusion_used = bool(run_params.get("Enable M2 Diffusion", enable_m2_diffusion_var.get()))
        if (
            task1_hist
            and task2_hist
            and exp_hist
            and imp_hist
            and t_last < len(task1_hist)
            and t_last < len(task2_hist)
            and t_last < len(exp_hist)
            and t_last < len(imp_hist)
            and task1_hist[t_last] is not None
            and task2_hist[t_last] is not None
            and exp_hist[t_last] is not None
            and imp_hist[t_last] is not None
            and len(task1_hist[t_last]) > 0
            and len(task2_hist[t_last]) > 0
        ):
            m2_produced = np.asarray(task1_hist[t_last], dtype=float)
            m2_consumed = np.asarray(task2_hist[t_last], dtype=float)
            imports = np.asarray(imp_hist[t_last], dtype=float)
            exports = np.asarray(exp_hist[t_last], dtype=float)

            # Align lengths defensively
            n_orgs = int(min(len(m2_produced), len(m2_consumed), len(imports), len(exports)))
            m2_produced = m2_produced[:n_orgs]
            m2_consumed = m2_consumed[:n_orgs]
            imports = imports[:n_orgs]
            exports = exports[:n_orgs]

            # Downsample if too many organisms
            max_bars = 2000
            if n_orgs > max_bars:
                idx = np.linspace(0, n_orgs - 1, max_bars, dtype=int)
                prod_plot = m2_produced[idx]
                cons_plot = m2_consumed[idx]
                imp_plot = imports[idx]
                exp_plot = exports[idx]
                n_plot = max_bars
            else:
                prod_plot = m2_produced
                cons_plot = m2_consumed
                imp_plot = imports
                exp_plot = exports
                n_plot = n_orgs

            # Sort organisms for easier interpretation (consumers -> producers)
            net_production = prod_plot - cons_plot
            sort_idx = np.argsort(net_production)
            prod_sorted = prod_plot[sort_idx]
            cons_sorted = cons_plot[sort_idx]
            imp_sorted = imp_plot[sort_idx]
            exp_sorted = exp_plot[sort_idx]

            x_positions = np.arange(n_plot)

            # Positive side: sources
            bud_ax3.bar(
                x_positions,
                prod_sorted,
                width=1.0,
                color=get_series_color("production"),
                alpha=0.8,
                edgecolor="none",
                label="Production (Task 1)",
            )

            # Negative side: sinks
            bud_ax3.bar(
                x_positions,
                -cons_sorted,
                width=1.0,
                color=get_series_color("consumption"),
                alpha=0.8,
                edgecolor="none",
                label="Consumption (Task 2)",
            )

            if m2_diffusion_used:
                bud_ax3.bar(
                    x_positions,
                    imp_sorted,
                    width=1.0,
                    bottom=prod_sorted,
                    color=get_series_color("import"),
                    alpha=0.8,
                    edgecolor="none",
                    label="Import (Diffusion in)",
                )
                bud_ax3.bar(
                    x_positions,
                    -exp_sorted,
                    width=1.0,
                    bottom=-cons_sorted,
                    # Use a strong, color-blind friendly export color (avoid gray vs purple ambiguity)
                    color=get_series_color("export"),
                    alpha=0.8,
                    edgecolor="none",
                    label="Export (Diffusion out)",
                )
                net_balance = prod_sorted + imp_sorted - cons_sorted - exp_sorted
            else:
                net_balance = prod_sorted - cons_sorted
            bud_ax3.scatter(
                x_positions,
                net_balance,
                c=get_series_color("net"),
                s=3,
                alpha=0.8,
                zorder=5,
                label="Net Balance",
            )

            bud_ax3.axhline(0, color=OKABE_ITO["black"], lw=2, alpha=0.8, zorder=1)
            bud_ax3.set_xlabel(f"Organisms (sorted by net production, N = {n_orgs})", fontsize=9)
            bud_ax3.set_ylabel("M2 Flow per Generation", fontsize=9)
            bud_ax3.set_title("M2 Budget Breakdown: Sources (>0) vs Sinks (<0)", fontsize=10, pad=20)
            bud_ax3.tick_params(labelsize=8)
            bud_ax3.grid(axis="y", alpha=0.3, zorder=0)
            bud_ax3.legend(
                fontsize=7,
                loc="upper center",
                ncol=2,
                frameon=True,
                bbox_to_anchor=(0.5, -0.18),
                fancybox=True,
                framealpha=0.95,
            )
        else:
            _show_initial_message(bud_ax3, "M2 Budget Breakdown: Sources (>0) vs Sinks (<0)")

        # --- Plot 4: Per-organism M1 budget breakdown ---
        if (
            m1_diffusion_active
            and task1_hist
            and m1_exp_hist
            and m1_imp_hist
            and t_last < len(task1_hist)
            and t_last < len(m1_exp_hist)
            and t_last < len(m1_imp_hist)
            and task1_hist[t_last] is not None
            and m1_exp_hist[t_last] is not None
            and m1_imp_hist[t_last] is not None
            and len(task1_hist[t_last]) > 0
        ):
            m1_consumed = np.asarray(task1_hist[t_last], dtype=float)
            m1_imports = np.asarray(m1_imp_hist[t_last], dtype=float)
            m1_exports = np.asarray(m1_exp_hist[t_last], dtype=float)

            n_orgs = int(min(len(m1_consumed), len(m1_imports), len(m1_exports)))
            m1_consumed = m1_consumed[:n_orgs]
            m1_imports = m1_imports[:n_orgs]
            m1_exports = m1_exports[:n_orgs]

            max_bars = 2000
            if n_orgs > max_bars:
                idx = np.linspace(0, n_orgs - 1, max_bars, dtype=int)
                cons_plot = m1_consumed[idx]
                imp_plot = m1_imports[idx]
                exp_plot = m1_exports[idx]
                n_plot = max_bars
            else:
                cons_plot = m1_consumed
                imp_plot = m1_imports
                exp_plot = m1_exports
                n_plot = n_orgs

            net_balance = imp_plot - cons_plot - exp_plot
            sort_idx = np.argsort(net_balance)
            cons_sorted = cons_plot[sort_idx]
            imp_sorted = imp_plot[sort_idx]
            exp_sorted = exp_plot[sort_idx]

            x_positions = np.arange(n_plot)

            bud_ax4.bar(
                x_positions,
                imp_sorted,
                width=1.0,
                color=get_series_color("import"),
                alpha=0.8,
                edgecolor="none",
                label="Import (Diffusion in)",
            )
            bud_ax4.bar(
                x_positions,
                -cons_sorted,
                width=1.0,
                color=get_series_color("consumption"),
                alpha=0.8,
                edgecolor="none",
                label="Consumption (Task 1)",
            )
            bud_ax4.bar(
                x_positions,
                -exp_sorted,
                width=1.0,
                bottom=-cons_sorted,
                # Use a strong, color-blind friendly export color (avoid gray vs purple ambiguity)
                color=get_series_color("export"),
                alpha=0.8,
                edgecolor="none",
                label="Export (Diffusion out)",
            )

            bud_ax4.scatter(
                x_positions,
                net_balance,
                c=get_series_color("net"),
                s=3,
                alpha=0.8,
                zorder=5,
                label="Net Balance",
            )

            bud_ax4.axhline(0, color=OKABE_ITO["black"], lw=2, alpha=0.8, zorder=1)
            bud_ax4.set_xlabel(f"Organisms (sorted by net balance, N = {n_orgs})", fontsize=9)
            bud_ax4.set_ylabel("M1 Flow per Generation", fontsize=9)
            bud_ax4.set_title("M1 Budget Breakdown: Sources (>0) vs Sinks (<0)", fontsize=10, pad=20)
            bud_ax4.tick_params(labelsize=8)
            bud_ax4.grid(axis="y", alpha=0.3, zorder=0)
            bud_ax4.legend(
                fontsize=7,
                loc="upper center",
                ncol=2,
                frameon=True,
                bbox_to_anchor=(0.5, -0.18),
                fancybox=True,
                framealpha=0.95,
            )
        else:
            if m1_diffusion_active:
                _show_initial_message(bud_ax4, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)")
            else:
                _show_initial_message(bud_ax4, "M1 Budget Breakdown (disabled)")

        # === Budget 2 tab (M2/M1 budget breakdowns only, sorted by import -> export) ===
        # --- M2 budget breakdown (sorted by net import) ---
        if (
            task1_hist
            and task2_hist
            and exp_hist
            and imp_hist
            and t_last < len(task1_hist)
            and t_last < len(task2_hist)
            and t_last < len(exp_hist)
            and t_last < len(imp_hist)
            and task1_hist[t_last] is not None
            and task2_hist[t_last] is not None
            and exp_hist[t_last] is not None
            and imp_hist[t_last] is not None
            and len(task1_hist[t_last]) > 0
            and len(task2_hist[t_last]) > 0
        ):
            m2_produced = np.asarray(task1_hist[t_last], dtype=float)
            m2_consumed = np.asarray(task2_hist[t_last], dtype=float)
            imports = np.asarray(imp_hist[t_last], dtype=float)
            exports = np.asarray(exp_hist[t_last], dtype=float)

            n_orgs = int(min(len(m2_produced), len(m2_consumed), len(imports), len(exports)))
            m2_produced = m2_produced[:n_orgs]
            m2_consumed = m2_consumed[:n_orgs]
            imports = imports[:n_orgs]
            exports = exports[:n_orgs]

            max_bars = 2000
            if n_orgs > max_bars:
                idx = np.linspace(0, n_orgs - 1, max_bars, dtype=int)
                prod_plot = m2_produced[idx]
                cons_plot = m2_consumed[idx]
                imp_plot = imports[idx]
                exp_plot = exports[idx]
                n_plot = max_bars
            else:
                prod_plot = m2_produced
                cons_plot = m2_consumed
                imp_plot = imports
                exp_plot = exports
                n_plot = n_orgs

            net_transport = imp_plot - exp_plot
            sort_idx = np.argsort(-net_transport)
            prod_sorted = prod_plot[sort_idx]
            cons_sorted = cons_plot[sort_idx]
            imp_sorted = imp_plot[sort_idx]
            exp_sorted = exp_plot[sort_idx]

            x_positions = np.arange(n_plot)

            bud2_ax_m2.bar(
                x_positions,
                prod_sorted,
                width=1.0,
                color=get_series_color("production"),
                alpha=0.8,
                edgecolor="none",
                label="Production (Task 1)",
            )
            bud2_ax_m2.bar(
                x_positions,
                -cons_sorted,
                width=1.0,
                color=get_series_color("consumption"),
                alpha=0.8,
                edgecolor="none",
                label="Consumption (Task 2)",
            )

            if m2_diffusion_used:
                bud2_ax_m2.bar(
                    x_positions,
                    imp_sorted,
                    width=1.0,
                    bottom=prod_sorted,
                    color=get_series_color("import"),
                    alpha=0.8,
                    edgecolor="none",
                    label="Import (Diffusion in)",
                )
                bud2_ax_m2.bar(
                    x_positions,
                    -exp_sorted,
                    width=1.0,
                    bottom=-cons_sorted,
                    color=get_series_color("export"),
                    alpha=0.8,
                    edgecolor="none",
                    label="Export (Diffusion out)",
                )
                net_balance = prod_sorted + imp_sorted - cons_sorted - exp_sorted
            else:
                net_balance = prod_sorted - cons_sorted

            bud2_ax_m2.scatter(
                x_positions,
                net_balance,
                c=get_series_color("net"),
                s=3,
                alpha=0.8,
                zorder=5,
                label="Net Balance",
            )
            bud2_ax_m2.axhline(0, color=OKABE_ITO["black"], lw=2, alpha=0.8, zorder=1)
            bud2_ax_m2.set_xlabel(f"Organisms (sorted by net import, N = {n_orgs})", fontsize=9)
            bud2_ax_m2.set_ylabel("M2 Flow per Generation", fontsize=9)
            bud2_ax_m2.set_title("M2 Budget Breakdown (Import → Export)", fontsize=10, pad=20)
            bud2_ax_m2.tick_params(labelsize=8)
            bud2_ax_m2.grid(axis="y", alpha=0.3, zorder=0)
            bud2_ax_m2.legend(fontsize=7, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.18))
        else:
            _show_initial_message(bud2_ax_m2, "M2 Budget Breakdown: Sources (>0) vs Sinks (<0)")

        # --- M1 budget breakdown (sorted by net import) ---
        if (
            m1_diffusion_active
            and task1_hist
            and m1_exp_hist
            and m1_imp_hist
            and t_last < len(task1_hist)
            and t_last < len(m1_exp_hist)
            and t_last < len(m1_imp_hist)
            and task1_hist[t_last] is not None
            and m1_exp_hist[t_last] is not None
            and m1_imp_hist[t_last] is not None
            and len(task1_hist[t_last]) > 0
        ):
            m1_consumed = np.asarray(task1_hist[t_last], dtype=float)
            m1_imports = np.asarray(m1_imp_hist[t_last], dtype=float)
            m1_exports = np.asarray(m1_exp_hist[t_last], dtype=float)

            n_orgs = int(min(len(m1_consumed), len(m1_imports), len(m1_exports)))
            m1_consumed = m1_consumed[:n_orgs]
            m1_imports = m1_imports[:n_orgs]
            m1_exports = m1_exports[:n_orgs]

            max_bars = 2000
            if n_orgs > max_bars:
                idx = np.linspace(0, n_orgs - 1, max_bars, dtype=int)
                cons_plot = m1_consumed[idx]
                imp_plot = m1_imports[idx]
                exp_plot = m1_exports[idx]
                n_plot = max_bars
            else:
                cons_plot = m1_consumed
                imp_plot = m1_imports
                exp_plot = m1_exports
                n_plot = n_orgs

            net_transport = imp_plot - exp_plot
            sort_idx = np.argsort(-net_transport)
            cons_sorted = cons_plot[sort_idx]
            imp_sorted = imp_plot[sort_idx]
            exp_sorted = exp_plot[sort_idx]

            x_positions = np.arange(n_plot)

            bud2_ax_m1.bar(
                x_positions,
                imp_sorted,
                width=1.0,
                color=get_series_color("import"),
                alpha=0.8,
                edgecolor="none",
                label="Import (Diffusion in)",
            )
            bud2_ax_m1.bar(
                x_positions,
                -cons_sorted,
                width=1.0,
                color=get_series_color("consumption"),
                alpha=0.8,
                edgecolor="none",
                label="Consumption (Task 1)",
            )
            bud2_ax_m1.bar(
                x_positions,
                -exp_sorted,
                width=1.0,
                bottom=-cons_sorted,
                color=get_series_color("export"),
                alpha=0.8,
                edgecolor="none",
                label="Export (Diffusion out)",
            )

            net_balance = imp_sorted - cons_sorted - exp_sorted
            bud2_ax_m1.scatter(
                x_positions,
                net_balance,
                c=get_series_color("net"),
                s=3,
                alpha=0.8,
                zorder=5,
                label="Net Balance",
            )
            bud2_ax_m1.axhline(0, color=OKABE_ITO["black"], lw=2, alpha=0.8, zorder=1)
            bud2_ax_m1.set_xlabel(f"Organisms (sorted by net import, N = {n_orgs})", fontsize=9)
            bud2_ax_m1.set_ylabel("M1 Flow per Generation", fontsize=9)
            bud2_ax_m1.set_title("M1 Budget Breakdown (Import → Export)", fontsize=10, pad=20)
            bud2_ax_m1.tick_params(labelsize=8)
            bud2_ax_m1.grid(axis="y", alpha=0.3, zorder=0)
            bud2_ax_m1.legend(fontsize=7, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.18))
        else:
            if m1_diffusion_active:
                _show_initial_message(bud2_ax_m1, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)")
            else:
                _show_initial_message(bud2_ax_m1, "M1 Budget Breakdown (disabled)")

        # Space for legends below axes
        # Extra space for legends below top plots + bar-chart legend below bottom-left plot
        bud_fig.subplots_adjust(top=0.95, bottom=0.18, left=0.10, right=0.95, hspace=2.2, wspace=0.25)
        bud_canvas.draw()
        bud2_fig.tight_layout()
        bud2_canvas.draw()

    def run_clicked():
        try:
            params = parse_params()
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        status.set("Running...")
        progress.start(10)

        def worker():
            try:
                # Keep a copy of the actual run settings so plots reflect what was run,
                # even if the user toggles checkboxes after the run finishes.
                params_for_record = dict(params)
                seed = params.pop("_seed", None)
                if seed is not None:
                    np.random.seed(seed)
                    params["Random Seed (optional)"] = seed

                res = model_spec.run_simulation(params)
            except ValueError as e:
                err_msg = str(e)
                win.after(
                    0,
                    lambda msg=err_msg: messagebox.showerror(
                        "Invalid simulation settings",
                        msg,
                    ),
                )
                res = None
            except Exception as e:
                err_msg = str(e)
                win.after(0, lambda msg=err_msg: messagebox.showerror("Run failed", msg))
                res = None
            finally:
                def finish():
                    progress.stop()
                    status.set("Done." if res else "Failed.")
                    if res:
                        if isinstance(res, dict):
                            res["_run_params"] = params_for_record
                        plot_results(res)
                win.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    button_frame = tk.Frame(left)
    button_frame.pack(fill="x", pady=(10, 0))

    tk.Button(button_frame, text="Run Simulation", command=run_clicked).pack(side="left")
    tk.Button(button_frame, text="← Back", command=lambda: (root.deiconify(), win.destroy())).pack(side="left", padx=(10, 0))

    # === Simulation pathway diagram (bottom of left column) ===
    diagram_frame = tk.LabelFrame(diagnostics, text="Simulation Pathway Diagram", padx=6, pady=6)
    diagram_frame.pack(fill="x", pady=(10, 0))
    model_diagram = SimulationModelDiagram(diagram_frame, width=320, height=200)
    model_diagram.canvas.pack(fill="x")
    model_diagram.bind_to_vars(
        enable_m1_diffusion=enable_m1_diffusion_var,
        enable_m1_facilitated_diffusion=enable_m1_facilitated_diffusion_var,
        enable_m2_diffusion=enable_m2_diffusion_var,
        enable_m1_porin_diffusion=m1_porin_diffusion_var,
        enable_diffusion_mutation=enable_diffusion_mutation_var,
        homogeneous_initial_diffusion_const=homogeneous_initial_diffusion_const_var,
        homogeneous_population=homogeneous_var,
        independent_traits=independent_traits_var,
        enable_chemostat_flow=enable_chemostat_flow_var,
        enable_initial_energy=enable_initial_energy_var,
        enable_intermediate_costs=enable_intermediate_costs_var,
        enable_acetate_addition=enable_acetate_addition_var,
    )

    refresh_initial_a_visibility()
    _refresh_death_dup_rate_ui()
    refresh_independent_traits_state()

    # Initialize empty tabs
    _show_initial_message(axAB, "Trait A (heatmap)")
    _show_initial_message(axBstrip, "Trait B (heatmap)")
    _show_initial_message(axTstrip, "Facilitated diffusion genotype (T) (heatmap)")
    _show_initial_message(axN, "Population size")
    _show_initial_message(axFlux, "Mean net M2 flux")
    _show_initial_message(axEnergy, "Energy (mean ± 1 std)")
    _show_initial_message(axChanges, "Deaths / Duplications")
    fig.tight_layout()
    canvas.draw()

    for a, t in [
        (m1_ax1, "M1 (env concentration)"),
        (m1_ax2, "M1 (internal concentration, mean ± std)"),
        (m1_ax3, "Total Stored vs Env M1"),
        (m1_ax4, "Global M1 Diffusion (Import/Export)"),
        (m1_ax_hist, "Internal M1 histogram"),
        (m2_ax1, "M2 (env concentration)"),
        (m2_ax2, "M2 (internal concentration, mean ± std)"),
        (m2_ax3, "Total Stored vs Env M2"),
        (m2_ax4, "Global M2 Diffusion (Import/Export)"),
        (m2_ax_hist, "Internal M2 histogram"),
    ]:
        _show_initial_message(a, t)
    m1_fig.tight_layout()
    m2_fig.tight_layout()
    m1_canvas.draw()
    m2_canvas.draw()

    for a, t in [
                (bud_ax1, "M2 Transport Behavior Distribution"),
                (bud_ax2, "M1 Transport Behavior Distribution"),
                (bud_ax3, "M2 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
                (bud_ax4, "M1 Budget Breakdown: Sources (>0) vs Sinks (<0)"),
    ]:
        _show_initial_message(a, t)
    bud_fig.tight_layout()
    bud_canvas.draw()

    # Live-update probability plot when relevant parameters change
    for _p in (
        "Duplication Sigmoid Midpoint",
        "Duplication Sigmoid Intensity",
        "Death Decay Rate",
        "Flow Percentage",
        CONSTANT_PROBABILITY,
    ):
        if _p in entries:
            entries[_p].bind("<KeyRelease>", lambda _e: update_probability_plot())
            entries[_p].bind("<FocusOut>", lambda _e: update_probability_plot())

    # Live-update investment function plot when modifier changes
    if "Investment Modifier" in entries:
        entries["Investment Modifier"].bind("<KeyRelease>", lambda _e: update_investment_plot())
        entries["Investment Modifier"].bind("<FocusOut>", lambda _e: update_investment_plot())
    independent_traits_var.trace_add("write", lambda *_: refresh_independent_traits_state())

    # No generation slider; histograms always reflect the latest generation.
