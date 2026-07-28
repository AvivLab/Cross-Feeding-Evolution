"""
Gradient-descent-style parameter + simulation settings panel for Monte Carlo tools.

Mirrors the Simulation model layout in ``gradient_descent_gui`` (optimizable table,
fixed parameters, diffusion / homogeneous / chemostat toggles) without optimization-only controls.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from gui.common.simulation_settings import (
    CONSTANT_PROBABILITY,
    NO_DEATH,
    apply_death_dup_flow_toggle_ui,
    normalize_simulation_params,
    sync_death_dup_hidden_params,
    sync_monte_carlo_panel_settings,
    sync_optional_param_row_visibility,
)
from gui.metrics import filter_metric_options_for_simulation_settings
from gui.models.registry import OptimizationModelSpec
from gui.common.tooltips import (
    PARAMETER_TOOLTIPS,
    GRADIENT_DESCENT_TOOLTIPS,
    SIMULATION_SETTINGS_TOOLTIPS,
    HOMOGENEOUS_TOOLTIP,
)
from gui.common.widgets import CreateToolTip


def _bind_vertical_mousewheel(canvas: tk.Canvas) -> None:
    """Bind mouse wheel / trackpad to scroll a canvas vertically (Windows, macOS, Linux)."""

    def _on_mousewheel(event: tk.Event) -> Optional[str]:
        d = getattr(event, "delta", 0) or 0
        if d:
            steps = int(-1 * (d / 120)) if abs(d) > 10 else (-1 if d > 0 else 1)
            canvas.yview_scroll(steps, "units")
        return "break"

    def _on_linux_up(_event: tk.Event) -> None:
        canvas.yview_scroll(-3, "units")

    def _on_linux_down(_event: tk.Event) -> None:
        canvas.yview_scroll(3, "units")

    canvas.bind("<MouseWheel>", _on_mousewheel, add="+")
    canvas.bind("<Button-4>", _on_linux_up, add="+")
    canvas.bind("<Button-5>", _on_linux_down, add="+")


def _bind_vertical_mousewheel_descendants(canvas: tk.Canvas, root: tk.Misc) -> None:
    """Forward wheel events from nested widgets (drawn above the canvas) to ``canvas``."""

    def _on_mousewheel(event: tk.Event) -> Optional[str]:
        d = getattr(event, "delta", 0) or 0
        if d:
            steps = int(-1 * (d / 120)) if abs(d) > 10 else (-1 if d > 0 else 1)
            canvas.yview_scroll(steps, "units")
        return "break"

    def _on_linux_up(_event: tk.Event) -> None:
        canvas.yview_scroll(-3, "units")

    def _on_linux_down(_event: tk.Event) -> None:
        canvas.yview_scroll(3, "units")

    def _walk(w: tk.Misc) -> None:
        try:
            w.bind("<MouseWheel>", _on_mousewheel)
            w.bind("<Button-4>", _on_linux_up)
            w.bind("<Button-5>", _on_linux_down)
        except tk.TclError:
            return
        for c in w.winfo_children():
            _walk(c)

    _walk(root)


@dataclass
class MonteCarloGdPanel:
    """Widget bundle + readers for one primary/neutral configuration column."""

    outer: tk.Widget
    model_spec: OptimizationModelSpec
    default_params: Dict[str, Any]
    param_names: List[str]
    hidden_params: Set[str]
    param_fix_checkboxes: Dict[str, tk.BooleanVar]
    param_min_entries: Dict[str, tk.Entry]
    param_max_entries: Dict[str, tk.Entry]
    param_initial_entries: Dict[str, tk.Entry]
    fixed_entries: Dict[str, tk.Entry]
    param_widgets: Dict[str, List[tk.Widget]]
    homogeneous_mode_var: tk.BooleanVar
    independent_traits_var: tk.BooleanVar
    m1_diffusion_var: tk.BooleanVar
    m2_diffusion_var: tk.BooleanVar
    diffusion_mutation_var: tk.BooleanVar
    homogeneous_initial_diffusion_const_var: tk.BooleanVar
    m1_facilitation_var: tk.BooleanVar
    m1_porin_diffusion_var: tk.BooleanVar
    enable_initial_energy_var: tk.BooleanVar
    enable_chemostat_flow_var: tk.BooleanVar
    enable_intermediate_costs_var: tk.BooleanVar
    enable_acetate_addition_var: tk.BooleanVar
    binary_death_at_zero_energy_var: tk.BooleanVar
    no_death_var: tk.BooleanVar
    constant_death_probability_var: tk.BooleanVar
    constant_duplication_probability_var: tk.BooleanVar
    silent_mode_var: tk.BooleanVar
    refresh_optimizable_params: Callable[[], None]
    refresh_fixed_params: Callable[[], None]
    settings_outer: tk.Misc
    _on_settings_change: Optional[Callable[[], None]] = None

    def optimizable_param_names(self) -> List[str]:
        return [
            p
            for p in self.param_names
            if (p not in self.hidden_params) and (not self.param_fix_checkboxes[p].get())
        ]

    def read_optimizable_bounds(self) -> Dict[str, Tuple[float, float]]:
        names = self.optimizable_param_names()
        if not names:
            raise ValueError("Select at least one optimizable (unfixed) parameter with min/max bounds.")
        bounds: Dict[str, Tuple[float, float]] = {}
        for pname in names:
            if pname not in self.param_min_entries or pname not in self.param_max_entries:
                raise RuntimeError(f"No min/max UI entries for parameter {pname!r}.")
            min_txt = self.param_min_entries[pname].get().strip()
            max_txt = self.param_max_entries[pname].get().strip()
            if max_txt == "":
                raise ValueError(f"Max value is required for {pname}.")
            lo = float("-inf") if min_txt == "" else float(min_txt)
            hi = float(max_txt)
            if lo != float("-inf") and lo >= hi:
                raise ValueError(f"Invalid bounds for {pname}: min must be < max.")
            bounds[pname] = (lo, hi)
        return bounds

    def filtered_metric_names(self, names: List[str]) -> List[str]:
        """Metric names compatible with current diffusion / intermediate / independent-trait toggles."""
        m1_any = bool(
            self.m1_diffusion_var.get()
            or self.m1_facilitation_var.get()
            or self.m1_porin_diffusion_var.get()
        )
        out = filter_metric_options_for_simulation_settings(
            list(names),
            enable_m2_diffusion=bool(self.m2_diffusion_var.get()),
            enable_m1_diffusion=m1_any,
            enable_intermediate_costs=bool(self.enable_intermediate_costs_var.get()),
        )
        if self.independent_traits_var.get():
            blocked = {
                "Trait Std Dev (Coupled)",
                "Trait Std Dev (Neutral Perc.)",
                "Trait Entropy (Neutral Perc.)",
            }
            out = [m for m in out if m not in blocked]
        return out

    def read_numeric_and_toggles(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        numeric = dict(self.default_params)
        for pname, entry in self.fixed_entries.items():
            try:
                numeric[pname] = float(entry.get())
            except (ValueError, TypeError):
                pass

        m1_any = bool(
            self.m1_diffusion_var.get()
            or self.m1_facilitation_var.get()
            or self.m1_porin_diffusion_var.get()
        )
        toggles: Dict[str, Any] = {
            "Homogeneous Population": bool(self.homogeneous_mode_var.get()),
            "Independent Traits": bool(self.independent_traits_var.get()),
            "Enable Initial Energy": bool(self.enable_initial_energy_var.get()),
            "Enable Chemostat Flow": bool(self.enable_chemostat_flow_var.get()),
            "Enable Intermediate Costs": bool(self.enable_intermediate_costs_var.get()),
            "Enable Acetate Addition": bool(self.enable_acetate_addition_var.get()),
            "Enable M1 Diffusion": m1_any,
            "Enable M2 Diffusion": bool(self.m2_diffusion_var.get()),
            "Enable M1 Facilitated Diffusion": bool(self.m1_facilitation_var.get()),
            "Enable M1 Porin Diffusion": bool(self.m1_porin_diffusion_var.get()),
        }
        if toggles["Enable M1 Facilitated Diffusion"]:
            toggles["Enable M1 Porin Diffusion"] = False
        if toggles["Enable M1 Porin Diffusion"]:
            toggles["Enable M1 Diffusion"] = True
        allow_dm = bool(toggles["Enable M2 Diffusion"] or toggles["Enable M1 Diffusion"])
        toggles["Enable Diffusion Mutation"] = bool(self.diffusion_mutation_var.get()) if allow_dm else False
        toggles["Homogeneous Initial Diffusion Const."] = (
            bool(self.homogeneous_initial_diffusion_const_var.get())
            if toggles["Enable Diffusion Mutation"]
            else False
        )

        if not toggles["Enable Initial Energy"]:
            numeric["Initial Energy"] = 0.0
        if not toggles["Enable Chemostat Flow"]:
            numeric["Flow Percentage"] = 0.0
        if not toggles["Enable Intermediate Costs"]:
            numeric["Intermediate Costs"] = 0.0
        if not toggles["Enable Acetate Addition"]:
            numeric["Average In_Flow (Acetate)"] = 0.0

        toggles["Binary Death at Zero Energy"] = bool(self.binary_death_at_zero_energy_var.get())
        toggles[NO_DEATH] = bool(self.no_death_var.get())
        toggles["Constant Death Probability"] = bool(self.constant_death_probability_var.get())
        toggles["Constant Duplication Probability"] = bool(self.constant_duplication_probability_var.get())

        merged = normalize_simulation_params({**numeric, **toggles})
        numeric = {k: merged[k] for k in self.param_names if k in merged}
        # Return reconciled toggle flags (same rules as core / job JSON), not raw checkbox state.
        out_toggles = {key: merged[key] for key in toggles if key in merged}
        return numeric, out_toggles

    def sync_simulation_settings_from(self, src: "MonteCarloGdPanel") -> None:
        """Copy simulation toggles from another panel (primary → neutral)."""
        sync_monte_carlo_panel_settings(self, src)

    def set_simulation_settings_visible(self, visible: bool) -> None:
        """Show or hide the whole Simulation Settings block (LabelFrame + scroll area)."""
        try:
            rc = self.settings_outer.master
        except tk.TclError:
            return
        if visible:
            self.settings_outer.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
            rc.grid_rowconfigure(1, weight=1)
        else:
            self.settings_outer.grid_remove()
            rc.grid_rowconfigure(1, weight=0)

    def _notify_settings(self) -> None:
        if self._on_settings_change is not None:
            try:
                self._on_settings_change()
            except Exception:
                pass


def build_monte_carlo_gd_panel(
    parent: tk.Widget,
    model_spec: OptimizationModelSpec,
    *,
    title: str,
    on_settings_change: Optional[Callable[[], None]] = None,
) -> MonteCarloGdPanel:
    """Build a scrollable GD-like column (parameters + fixed + simulation settings)."""
    default_params = dict(model_spec.default_params or {})
    default_params.setdefault("Initial Energy", 0.0)
    default_params.setdefault("Intermediate Costs", 0.0)
    default_params.setdefault("Average In_Flow (Acetate)", 0.0)

    param_names = list(default_params.keys())

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

    def get_display_name(param_name: str) -> str:
        return param_short_names.get(param_name, param_name)

    def get_default_bounds(param_name: str, default_val: Any) -> Tuple[str, str]:
        if param_name == "Flow Percentage":
            return ("0.0", "50.0")
        if isinstance(default_val, (int, float)):
            if default_val == 0:
                return ("0.0", "10.0")
            if default_val < 1:
                return (str(max(0, default_val * 0.1)), str(default_val * 10))
            if default_val < 10:
                return (str(max(0, default_val * 0.5)), str(default_val * 2))
            return (str(max(0, default_val * 0.5)), str(default_val * 2))
        return ("0.0", "10.0")

    outer = tk.LabelFrame(parent, text=title, padx=6, pady=6)
    outer.pack(fill="both", expand=True, padx=2, pady=2)
    outer.grid_columnconfigure(0, weight=1)
    outer.grid_columnconfigure(1, weight=1)
    outer.grid_rowconfigure(0, weight=1)

    left_col = tk.Frame(outer)
    left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
    left_col.grid_rowconfigure(0, weight=1)
    left_col.grid_columnconfigure(0, weight=1)

    right_col = tk.Frame(outer)
    right_col.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
    right_col.grid_rowconfigure(0, weight=1)
    right_col.grid_rowconfigure(1, weight=1)
    right_col.grid_columnconfigure(0, weight=1)

    param_opt_frame = tk.LabelFrame(left_col, text="Parameters to Optimize", padx=5, pady=5)
    param_opt_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
    param_opt_frame.grid_rowconfigure(0, weight=1)
    param_opt_frame.grid_columnconfigure(0, weight=1)

    param_canvas = tk.Canvas(param_opt_frame, highlightthickness=0)
    param_scrollbar = ttk.Scrollbar(param_opt_frame, orient="vertical", command=param_canvas.yview)
    param_scrollable_frame = tk.Frame(param_canvas)

    def _on_param_configure(_event=None):
        param_canvas.configure(scrollregion=param_canvas.bbox("all"))

    param_scrollable_frame.bind("<Configure>", _on_param_configure)
    _param_inner_win = param_canvas.create_window((0, 0), window=param_scrollable_frame, anchor="nw")

    def _on_canvas_configure(event):
        try:
            param_canvas.itemconfig(_param_inner_win, width=max(1, int(event.width)))
        except tk.TclError:
            pass

    param_canvas.bind("<Configure>", _on_canvas_configure)
    param_canvas.configure(yscrollcommand=param_scrollbar.set)
    param_canvas.grid(row=0, column=0, sticky="nsew")
    param_scrollbar.grid(row=0, column=1, sticky="ns")
    _bind_vertical_mousewheel(param_canvas)

    tk.Label(param_scrollable_frame, text="Fix", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=2, pady=2)
    tk.Label(param_scrollable_frame, text="Parameter", font=("Arial", 9, "bold")).grid(
        row=0, column=1, sticky="w", padx=2, pady=2
    )
    tk.Label(param_scrollable_frame, text="Min (optional)", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=2, pady=2)
    tk.Label(param_scrollable_frame, text="Max", font=("Arial", 9, "bold")).grid(row=0, column=3, padx=2, pady=2)

    param_fix_checkboxes: Dict[str, tk.BooleanVar] = {}
    param_initial_entries: Dict[str, tk.Entry] = {}
    param_min_entries: Dict[str, tk.Entry] = {}
    param_max_entries: Dict[str, tk.Entry] = {}
    param_widgets: Dict[str, List[tk.Widget]] = {}
    hidden_params: Set[str] = set()
    hidden_params.update(
        {
            "Initial Energy",
            "Intermediate Costs",
            "Average In_Flow (Acetate)",
            "Flow Percentage",
            "Independent Traits",
            "Initial B",
            "Constant Probability",
        }
    )

    def _tooltip_for_param_name(param_name: str) -> str:
        alias = {"Average In_Flow": "Average In-Flow", "Average In-Flow": "Average In-Flow"}
        key = alias.get(param_name, param_name)
        return PARAMETER_TOOLTIPS.get(key, f"{param_name}\n\nModel parameter.")

    def refresh_optimizable_params() -> None:
        for widgets in param_widgets.values():
            for widget in widgets:
                widget.grid_remove()
        fixed_params = [p for p, var in param_fix_checkboxes.items() if var.get() and (p not in hidden_params)]
        optimizable_params = [p for p in param_names if (p not in fixed_params) and (p not in hidden_params)]
        for i, param_name in enumerate(optimizable_params):
            row = i + 1
            widgets = param_widgets[param_name]
            widgets[4].grid(row=row, column=0, padx=2, pady=1, sticky="w")
            widgets[0].grid(row=row, column=1, sticky="w", padx=2, pady=1)
            widgets[2].grid(row=row, column=2, padx=2, pady=1)
            widgets[3].grid(row=row, column=3, padx=2, pady=1)
        param_canvas.update_idletasks()
        try:
            param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        except tk.TclError:
            pass

    fixed_entries: Dict[str, tk.Entry] = {}

    def refresh_fixed_params() -> None:
        fixed_values = {name: entry.get() for name, entry in fixed_entries.items()}
        for widget in fixed_scrollable_frame.winfo_children():
            widget.destroy()
        fixed_entries.clear()
        fixed_params = [p for p, var in param_fix_checkboxes.items() if var.get() and (p not in hidden_params)]
        if fixed_params:
            tk.Label(fixed_scrollable_frame, text="Unfix", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=2, pady=2)
            tk.Label(fixed_scrollable_frame, text="Parameter", font=("Arial", 9, "bold")).grid(
                row=0, column=1, padx=2, pady=2
            )
            tk.Label(fixed_scrollable_frame, text="Value", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=2, pady=2)
        for i, param_name in enumerate(fixed_params):
            row = i + 1
            unfix_var = param_fix_checkboxes[param_name]
            unfix_checkbox = tk.Checkbutton(
                fixed_scrollable_frame,
                text="",
                variable=unfix_var,
                command=lambda pn=param_name: on_fix_toggle(pn),
            )
            unfix_checkbox.grid(row=row, column=0, padx=2, pady=1, sticky="w")
            CreateToolTip(unfix_checkbox, f"{GRADIENT_DESCENT_TOOLTIPS['Unfix Parameter']}\n\nParameter: {param_name}")
            param_label = tk.Label(fixed_scrollable_frame, text=get_display_name(param_name) + ":")
            param_label.grid(row=row, column=1, sticky="e", padx=2, pady=1)
            entry = tk.Entry(fixed_scrollable_frame, width=10)
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
        fixed_canvas.update_idletasks()
        try:
            fixed_canvas.configure(scrollregion=fixed_canvas.bbox("all"))
        except tk.TclError:
            pass
        _bind_vertical_mousewheel_descendants(fixed_canvas, fixed_scrollable_frame)

    def on_fix_toggle(_param_name: str) -> None:
        refresh_optimizable_params()
        refresh_fixed_params()

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
        row = i + 1
        should_fix = param_name in fixed_by_default
        var = tk.BooleanVar(value=should_fix)
        param_fix_checkboxes[param_name] = var
        fix_checkbox = tk.Checkbutton(
            param_scrollable_frame,
            text="",
            variable=var,
            command=lambda pn=param_name: on_fix_toggle(pn),
            font=("Arial", 8),
        )
        fix_checkbox.grid(row=row, column=0, padx=2, pady=1, sticky="w")
        CreateToolTip(fix_checkbox, f"{GRADIENT_DESCENT_TOOLTIPS['Fix Parameter']}\n\nParameter: {param_name}")
        label = tk.Label(param_scrollable_frame, text=get_display_name(param_name))
        label.grid(row=row, column=1, sticky="w", padx=2, pady=1)
        initial_entry = tk.Entry(param_scrollable_frame, width=8)
        initial_entry.insert(0, str(default_params[param_name]))
        param_initial_entries[param_name] = initial_entry
        default_val = default_params[param_name]
        _min_def, max_default = get_default_bounds(param_name, default_val)
        min_entry = tk.Entry(param_scrollable_frame, width=8)
        if param_name == "Chemostat Volume":
            min_entry.insert(0, "10000")
        elif param_name == "Average In_Flow":
            min_entry.insert(0, "200.0")
        elif param_name == "Average In_Flow (Acetate)":
            min_entry.insert(0, "0.0")
        elif param_name == "Cost of Life":
            min_entry.insert(0, "0.001" if getattr(model_spec, "key", "") == "simulation" else "0.0")
        elif param_name == "Intermediate Costs":
            min_entry.insert(0, "0.0")
        elif param_name == "Investment Modifier":
            min_entry.insert(0, "0.2")
        elif param_name == "Acetate Ratio":
            min_entry.insert(0, "0.1")
        elif param_name == "Duplication Sigmoid Intensity":
            min_entry.insert(0, "0.0")
        elif param_name == CONSTANT_PROBABILITY:
            min_entry.insert(0, "0.0")
        else:
            min_entry.insert(0, "0.0")
        min_entry.grid(row=row, column=2, padx=2, pady=1)
        param_min_entries[param_name] = min_entry
        max_entry = tk.Entry(param_scrollable_frame, width=8)
        if param_name == "Cost of Life":
            max_entry.insert(0, "0.2")
        elif param_name == "Degradation Rate":
            max_entry.insert(0, "1.0")
        elif param_name in ["Initial A", "Initial B", "Initial Facilitation"]:
            max_entry.insert(0, "1.0")
        elif param_name == "Mutation Rate":
            max_entry.insert(0, "0.05")
        elif param_name == "Average In_Flow":
            max_entry.insert(0, "400.0")
        elif param_name == "Average In_Flow (Acetate)":
            max_entry.insert(0, "400.0")
        elif param_name == "Investment Modifier":
            max_entry.insert(0, "5.0")
        elif param_name == "Diffusion Constant":
            max_entry.insert(0, "1.0")
        elif param_name == "Cost of Transport":
            max_entry.insert(0, "1.0")
        elif param_name == "Intermediate Costs":
            max_entry.insert(0, "1.0")
        elif param_name == "Acetate Ratio":
            max_entry.insert(0, "10.0")
        elif param_name == "Initial Organism Count":
            max_entry.insert(0, "1000")
        elif param_name == "Chemostat Volume":
            max_entry.insert(0, "20000")
        elif param_name == "Duplication Sigmoid Intensity":
            max_entry.insert(0, "10.0")
        elif param_name == CONSTANT_PROBABILITY:
            max_entry.insert(0, "1.0")
        else:
            max_entry.insert(0, max_default)
        max_entry.grid(row=row, column=3, padx=2, pady=1)
        param_max_entries[param_name] = max_entry
        param_widgets[param_name] = [label, initial_entry, min_entry, max_entry, fix_checkbox]
        tt = _tooltip_for_param_name(param_name)
        CreateToolTip(label, tt)
        CreateToolTip(initial_entry, tt)
        CreateToolTip(min_entry, tt)
        CreateToolTip(max_entry, tt)

    refresh_optimizable_params()
    _bind_vertical_mousewheel_descendants(param_canvas, param_scrollable_frame)
    param_canvas.update_idletasks()
    try:
        param_canvas.configure(scrollregion=param_canvas.bbox("all"))
    except tk.TclError:
        pass

    fixed_frame = tk.LabelFrame(right_col, text="Fixed Parameters", padx=5, pady=5)
    fixed_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
    fixed_frame.grid_rowconfigure(0, weight=1)
    fixed_frame.grid_columnconfigure(0, weight=1)

    fixed_canvas = tk.Canvas(fixed_frame, highlightthickness=0)
    fixed_scrollbar = ttk.Scrollbar(fixed_frame, orient="vertical", command=fixed_canvas.yview)
    fixed_scrollable_frame = tk.Frame(fixed_canvas)

    def _on_fixed_inner_configure(_event=None):
        fixed_canvas.configure(scrollregion=fixed_canvas.bbox("all"))

    fixed_scrollable_frame.bind("<Configure>", _on_fixed_inner_configure)
    _fixed_inner_win = fixed_canvas.create_window((0, 0), window=fixed_scrollable_frame, anchor="nw")

    def _on_fixed_canvas_configure(event):
        try:
            fixed_canvas.itemconfig(_fixed_inner_win, width=max(1, int(event.width)))
        except tk.TclError:
            pass

    fixed_canvas.bind("<Configure>", _on_fixed_canvas_configure)
    fixed_canvas.configure(yscrollcommand=fixed_scrollbar.set)
    fixed_canvas.grid(row=0, column=0, sticky="nsew")
    fixed_scrollbar.grid(row=0, column=1, sticky="ns")
    _bind_vertical_mousewheel(fixed_canvas)
    refresh_fixed_params()

    settings_outer = tk.LabelFrame(right_col, text="Simulation Settings", padx=5, pady=5)
    settings_outer.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
    settings_outer.grid_rowconfigure(0, weight=1)
    settings_outer.grid_columnconfigure(0, weight=1)

    settings_canvas = tk.Canvas(settings_outer, highlightthickness=0)
    settings_scrollbar = ttk.Scrollbar(settings_outer, orient="vertical", command=settings_canvas.yview)
    other_frame = tk.Frame(settings_canvas)

    def _on_settings_inner_configure(_event=None):
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

    other_frame.bind("<Configure>", _on_settings_inner_configure)
    _settings_inner_win = settings_canvas.create_window((0, 0), window=other_frame, anchor="nw")

    def _on_settings_canvas_configure(event):
        try:
            settings_canvas.itemconfig(_settings_inner_win, width=max(1, int(event.width)))
        except tk.TclError:
            pass

    settings_canvas.bind("<Configure>", _on_settings_canvas_configure)
    settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
    settings_canvas.grid(row=0, column=0, sticky="nsew")
    settings_scrollbar.grid(row=0, column=1, sticky="ns")
    _bind_vertical_mousewheel(settings_canvas)

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
    for w, tip_key in (
        (m1_diffusion_check, "Enable M1 Diffusion (simple)"),
        (m2_diffusion_check, "Enable M2 Diffusion"),
        (diffusion_mutation_check, "Enable Diffusion Mutation"),
        (homogeneous_initial_diffusion_const_check, "Homogeneous Initial Diffusion Const."),
        (m1_facilitation_check, "Enable M1 Facilitated Diffusion"),
        (m1_porin_diffusion_check, "Enable M1 Porin Diffusion"),
    ):
        CreateToolTip(w, GRADIENT_DESCENT_TOOLTIPS[tip_key])

    def refresh_diffusion_mutation_state() -> None:
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

    updating_m1_mode = False

    def refresh_m1_diffusion_mode() -> None:
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

    def refresh_diffusion_constant_visibility() -> None:
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

    def refresh_m1_diffusion_ui() -> None:
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
        for w in (
            m1_diffusion_check,
            m2_diffusion_check,
            diffusion_mutation_check,
            homogeneous_initial_diffusion_const_check,
            m1_facilitation_check,
            m1_porin_diffusion_check,
        ):
            w.grid_remove()

    homogeneous_mode_var = tk.BooleanVar(value=False)
    homogeneous_checkbox = tk.Checkbutton(
        other_frame,
        text="Use Homogeneous Initial Genotype",
        variable=homogeneous_mode_var,
    )
    homogeneous_checkbox.grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
    CreateToolTip(homogeneous_checkbox, HOMOGENEOUS_TOOLTIP)

    independent_traits_var = tk.BooleanVar(value=bool(default_params.get("Independent Traits", False)))
    independent_traits_checkbox = tk.Checkbutton(
        other_frame,
        text="Independent A/B Traits",
        variable=independent_traits_var,
    )
    independent_traits_checkbox.grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
    CreateToolTip(independent_traits_checkbox, GRADIENT_DESCENT_TOOLTIPS["Independent A/B Traits"])

    silent_mode_var = tk.BooleanVar(value=True)
    silent_checkbox = tk.Checkbutton(
        other_frame,
        text="Silent Mode (suppress simulation progress)",
        variable=silent_mode_var,
    )
    silent_checkbox.grid(row=8, column=0, columnspan=2, sticky="w", pady=2)
    CreateToolTip(silent_checkbox, SIMULATION_SETTINGS_TOOLTIPS["Silent Mode"])

    enable_initial_energy_var = tk.BooleanVar(value=False)
    enable_initial_energy_check = tk.Checkbutton(
        other_frame,
        text="Enable Initial Energy",
        variable=enable_initial_energy_var,
    )
    enable_initial_energy_check.grid(row=9, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_initial_energy_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Initial Energy"])

    def _refresh_initial_energy_state(
        *,
        preserve_fix_checkbox: bool = False,
        rebuild_param_tables: bool = True,
    ) -> None:
        sync_optional_param_row_visibility(
            hidden_params,
            "Initial Energy",
            visible=bool(enable_initial_energy_var.get()),
            param_fix_checkboxes=param_fix_checkboxes,
            force_fix=not preserve_fix_checkbox,
        )
        if enable_initial_energy_var.get():
            if "Initial Energy" in param_initial_entries and not param_initial_entries["Initial Energy"].get().strip():
                param_initial_entries["Initial Energy"].insert(0, "0.0")
            if "Initial Energy" in fixed_entries and not fixed_entries["Initial Energy"].get().strip():
                fixed_entries["Initial Energy"].insert(0, "0.0")
        if rebuild_param_tables:
            refresh_optimizable_params()
            refresh_fixed_params()

    enable_initial_energy_var.trace_add("write", lambda *_: _refresh_initial_energy_state())
    _refresh_initial_energy_state()

    enable_chemostat_flow_var = tk.BooleanVar(value=False)
    enable_chemostat_flow_check = tk.Checkbutton(
        other_frame,
        text="Enable Chemostat Flow",
        variable=enable_chemostat_flow_var,
    )
    enable_chemostat_flow_check.grid(row=10, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_chemostat_flow_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Chemostat Flow"])

    def _refresh_chemostat_flow_state(
        *,
        preserve_fix_checkbox: bool = False,
        rebuild_param_tables: bool = True,
    ) -> None:
        sync_optional_param_row_visibility(
            hidden_params,
            "Flow Percentage",
            visible=bool(enable_chemostat_flow_var.get()),
            param_fix_checkboxes=param_fix_checkboxes,
            force_fix=not preserve_fix_checkbox,
        )
        if enable_chemostat_flow_var.get():
            if "Flow Percentage" in param_initial_entries and not param_initial_entries["Flow Percentage"].get().strip():
                param_initial_entries["Flow Percentage"].insert(0, "0.0")
            if "Flow Percentage" in fixed_entries and not fixed_entries["Flow Percentage"].get().strip():
                fixed_entries["Flow Percentage"].insert(0, "0.0")
        # Death/dup may change hidden_params when flow toggles; rebuild once at the end.
        try:
            _refresh_death_dup_constant_params_state(
                preserve_fix_checkbox=preserve_fix_checkbox,
                prefer="",
                rebuild_param_tables=False,
            )
        except NameError:
            # Early init: death/dup refresh helpers are defined later.
            pass
        if rebuild_param_tables:
            refresh_optimizable_params()
            refresh_fixed_params()

    enable_chemostat_flow_var.trace_add("write", lambda *_: _refresh_chemostat_flow_state())
    _refresh_chemostat_flow_state()

    enable_intermediate_costs_var = tk.BooleanVar(value=False)
    enable_intermediate_costs_check = tk.Checkbutton(
        other_frame,
        text="Enable Intermediate Costs",
        variable=enable_intermediate_costs_var,
    )
    enable_intermediate_costs_check.grid(row=11, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_intermediate_costs_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Intermediate Costs"])

    def _refresh_intermediate_costs_state(
        *,
        preserve_fix_checkbox: bool = False,
        rebuild_param_tables: bool = True,
    ) -> None:
        if m2_diffusion_var.get():
            enable_intermediate_costs_check.configure(state="normal")
        else:
            enable_intermediate_costs_var.set(False)
            enable_intermediate_costs_check.configure(state="disabled")
        sync_optional_param_row_visibility(
            hidden_params,
            "Intermediate Costs",
            visible=bool(enable_intermediate_costs_var.get()),
            param_fix_checkboxes=param_fix_checkboxes,
            force_fix=not preserve_fix_checkbox,
        )
        if enable_intermediate_costs_var.get():
            if "Intermediate Costs" in param_initial_entries and not param_initial_entries["Intermediate Costs"].get().strip():
                param_initial_entries["Intermediate Costs"].insert(0, "0.0")
            if "Intermediate Costs" in fixed_entries and not fixed_entries["Intermediate Costs"].get().strip():
                fixed_entries["Intermediate Costs"].insert(0, "0.0")
        if rebuild_param_tables:
            refresh_optimizable_params()
            refresh_fixed_params()

    enable_intermediate_costs_var.trace_add("write", lambda *_: _refresh_intermediate_costs_state())
    m2_diffusion_var.trace_add("write", lambda *_: _refresh_intermediate_costs_state())
    _refresh_intermediate_costs_state()

    enable_acetate_addition_var = tk.BooleanVar(value=False)
    enable_acetate_addition_check = tk.Checkbutton(
        other_frame,
        text="Enable Acetate Addition",
        variable=enable_acetate_addition_var,
    )
    enable_acetate_addition_check.grid(row=12, column=0, sticky="w", pady=(4, 0))
    CreateToolTip(enable_acetate_addition_check, SIMULATION_SETTINGS_TOOLTIPS["Enable Acetate Addition"])

    def _refresh_acetate_addition_state(
        *,
        preserve_fix_checkbox: bool = False,
        rebuild_param_tables: bool = True,
    ) -> None:
        sync_optional_param_row_visibility(
            hidden_params,
            "Average In_Flow (Acetate)",
            visible=bool(enable_acetate_addition_var.get()),
            param_fix_checkboxes=param_fix_checkboxes,
            force_fix=not preserve_fix_checkbox,
        )
        if enable_acetate_addition_var.get():
            if (
                "Average In_Flow (Acetate)" in param_initial_entries
                and not param_initial_entries["Average In_Flow (Acetate)"].get().strip()
            ):
                param_initial_entries["Average In_Flow (Acetate)"].insert(0, "0.0")
            if "Average In_Flow (Acetate)" in fixed_entries and not fixed_entries["Average In_Flow (Acetate)"].get().strip():
                fixed_entries["Average In_Flow (Acetate)"].insert(0, "0.0")
        if rebuild_param_tables:
            refresh_optimizable_params()
            refresh_fixed_params()

    enable_acetate_addition_var.trace_add("write", lambda *_: _refresh_acetate_addition_state())
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

    _death_dup_flow_reconciling = {"active": False}

    def _refresh_death_dup_constant_params_state(
        *,
        preserve_fix_checkbox: bool = False,
        prefer: str = "",
        rebuild_param_tables: bool = True,
    ) -> None:
        if _death_dup_flow_reconciling["active"]:
            return
        _death_dup_flow_reconciling["active"] = True
        try:
            visibility = apply_death_dup_flow_toggle_ui(
                no_death_var=no_death_var,
                constant_death_probability_var=constant_death_probability_var,
                constant_duplication_probability_var=constant_duplication_probability_var,
                enable_chemostat_flow_var=enable_chemostat_flow_var,
                no_death_checkbox=no_death_check,
                constant_death_checkbox=constant_death_probability_check,
                constant_duplication_checkbox=constant_duplication_probability_check,
                binary_death_at_zero_energy=bool(binary_death_at_zero_energy_var.get()),
                prefer=prefer,
            )
            sync_death_dup_hidden_params(
                hidden_params,
                visibility,
                param_fix_checkboxes=param_fix_checkboxes,
                force_fix_constant_probability=not preserve_fix_checkbox,
            )
            if rebuild_param_tables:
                refresh_optimizable_params()
                refresh_fixed_params()
        finally:
            _death_dup_flow_reconciling["active"] = False

    def _refresh_binary_death_state(*, preserve_fix_checkbox: bool = False) -> None:
        _refresh_death_dup_constant_params_state(preserve_fix_checkbox=preserve_fix_checkbox)

    def _refresh_constant_death_state(*, preserve_fix_checkbox: bool = False, prefer: str = "constant_death") -> None:
        _refresh_death_dup_constant_params_state(
            preserve_fix_checkbox=preserve_fix_checkbox,
            prefer=prefer,
        )

    def _refresh_constant_duplication_state(*, preserve_fix_checkbox: bool = False) -> None:
        _refresh_death_dup_constant_params_state(preserve_fix_checkbox=preserve_fix_checkbox)

    binary_death_at_zero_energy_var.trace_add("write", lambda *_: _refresh_binary_death_state())
    no_death_var.trace_add(
        "write",
        lambda *_: _refresh_constant_death_state(prefer="no_death"),
    )
    constant_death_probability_var.trace_add(
        "write",
        lambda *_: _refresh_constant_death_state(prefer="constant_death"),
    )
    constant_duplication_probability_var.trace_add("write", lambda *_: _refresh_constant_duplication_state())
    _refresh_death_dup_constant_params_state()

    def toggle_homogeneous_params() -> None:
        homogeneous_enabled = homogeneous_mode_var.get()
        initial_enzyme_params = list(getattr(model_spec, "homogeneous_param_names", []) or [])
        if homogeneous_enabled:
            for param in initial_enzyme_params:
                hidden_params.discard(param)
            for param in initial_enzyme_params:
                if param in param_fix_checkboxes:
                    param_fix_checkboxes[param].set(False)
        else:
            for param in initial_enzyme_params:
                hidden_params.add(param)
            for param in initial_enzyme_params:
                if param in param_fix_checkboxes:
                    param_fix_checkboxes[param].set(True)
        refresh_optimizable_params()
        refresh_fixed_params()
        refresh_facilitation_param_visibility()
        refresh_independent_traits_visibility()

    def refresh_independent_traits_visibility() -> None:
        independent_enabled = independent_traits_var.get()
        if "Independent Traits" in param_fix_checkboxes:
            param_fix_checkboxes["Independent Traits"].set(True)
        if "Investment Modifier" in param_names:
            if independent_enabled:
                hidden_params.add("Investment Modifier")
                if "Investment Modifier" in param_fix_checkboxes:
                    param_fix_checkboxes["Investment Modifier"].set(True)
            else:
                hidden_params.discard("Investment Modifier")
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

    def refresh_facilitation_param_visibility() -> None:
        if getattr(model_spec, "key", "") != "simulation":
            return
        show = m1_facilitation_var.get()
        if "Cost of Transport" in param_names:
            if show:
                hidden_params.discard("Cost of Transport")
                if "Cost of Transport" in param_fix_checkboxes:
                    param_fix_checkboxes["Cost of Transport"].set(False)
            else:
                hidden_params.add("Cost of Transport")
                if "Cost of Transport" in param_fix_checkboxes:
                    param_fix_checkboxes["Cost of Transport"].set(True)
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

    homogeneous_mode_var.trace_add("write", lambda *_: toggle_homogeneous_params())
    independent_traits_var.trace_add("write", lambda *_: refresh_independent_traits_visibility())
    m1_facilitation_var.trace_add("write", lambda *_: refresh_facilitation_param_visibility())
    toggle_homogeneous_params()
    refresh_facilitation_param_visibility()
    refresh_independent_traits_visibility()

    other_frame.grid_columnconfigure(0, weight=1)
    _bind_vertical_mousewheel_descendants(settings_canvas, other_frame)
    settings_canvas.update_idletasks()
    try:
        settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
    except tk.TclError:
        pass

    panel = MonteCarloGdPanel(
        outer=outer,
        model_spec=model_spec,
        default_params=default_params,
        param_names=param_names,
        hidden_params=hidden_params,
        param_fix_checkboxes=param_fix_checkboxes,
        param_min_entries=param_min_entries,
        param_max_entries=param_max_entries,
        param_initial_entries=param_initial_entries,
        fixed_entries=fixed_entries,
        param_widgets=param_widgets,
        homogeneous_mode_var=homogeneous_mode_var,
        independent_traits_var=independent_traits_var,
        m1_diffusion_var=m1_diffusion_var,
        m2_diffusion_var=m2_diffusion_var,
        diffusion_mutation_var=diffusion_mutation_var,
        homogeneous_initial_diffusion_const_var=homogeneous_initial_diffusion_const_var,
        m1_facilitation_var=m1_facilitation_var,
        m1_porin_diffusion_var=m1_porin_diffusion_var,
        enable_initial_energy_var=enable_initial_energy_var,
        enable_chemostat_flow_var=enable_chemostat_flow_var,
        enable_intermediate_costs_var=enable_intermediate_costs_var,
        enable_acetate_addition_var=enable_acetate_addition_var,
        binary_death_at_zero_energy_var=binary_death_at_zero_energy_var,
        no_death_var=no_death_var,
        constant_death_probability_var=constant_death_probability_var,
        constant_duplication_probability_var=constant_duplication_probability_var,
        silent_mode_var=silent_mode_var,
        refresh_optimizable_params=refresh_optimizable_params,
        refresh_fixed_params=refresh_fixed_params,
        settings_outer=settings_outer,
        _on_settings_change=on_settings_change,
    )

    def _fire(*_args: Any) -> None:
        panel._notify_settings()

    for _v in (
        m1_diffusion_var,
        m2_diffusion_var,
        m1_facilitation_var,
        m1_porin_diffusion_var,
        enable_intermediate_costs_var,
        independent_traits_var,
        homogeneous_mode_var,
    ):
        _v.trace_add("write", _fire)

    return panel
