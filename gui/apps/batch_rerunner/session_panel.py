"""Read-only loaded-session summary and pathway diagram for Batch Re-Runner."""

from __future__ import annotations

import statistics
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from gui.common.model_diagram import SimulationModelDiagram
from gui.common.simulation_settings import (
    MONTE_CARLO_PANEL_TOGGLE_BINDINGS,
    expected_monte_carlo_exported_toggle_keys,
    normalize_simulation_params,
    prune_irrelevant_numeric_parameters_for_export,
)
from headless.hpc_common import prune_irrelevant_bounds_for_toggles
from headless.neutral_comparison import _bounds_from_save_json
from headless.primary_batch_session_meta import load_campaign_summary
from headless.primary_hit_rescreen import SessionRescreenContext

_DIAGRAM_UI_KEYS: Tuple[str, ...] = (
    "enable_m1_diffusion",
    "enable_m1_facilitated_diffusion",
    "enable_m2_diffusion",
    "enable_m1_porin_diffusion",
    "enable_diffusion_mutation",
    "homogeneous_initial_diffusion_const",
    "homogeneous_population",
    "independent_traits",
    "enable_chemostat_flow",
    "enable_initial_energy",
    "enable_intermediate_costs",
    "enable_acetate_addition",
)


def _fmt_num(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _fmt_stat(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    rounded = round(value, 2)
    text = f"{rounded:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _format_primary_hit_count_summary(hit_counts: Sequence[Any]) -> Optional[str]:
    """Summarize per-batch primary hit counts as mean ± std (bounded length)."""
    vals: list[float] = []
    for raw in hit_counts:
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    mean = statistics.fmean(vals)
    if len(vals) == 1:
        return f"Primary hits per batch: {_fmt_stat(mean)} (1 batch)"
    stdev = statistics.stdev(vals)
    return (
        f"Primary hits per batch: {_fmt_stat(mean)} ± {_fmt_stat(stdev)} "
        f"({len(vals)} batches)"
    )


def _resolve_primary_bounds(
    campaign: Mapping[str, Any],
    toggles: Mapping[str, Any],
) -> Dict[str, Tuple[float, float]]:
    bounds_raw = campaign.get("primary_bounds")
    if bounds_raw is None:
        return {}
    bounds = _bounds_from_save_json(bounds_raw, label="primary_bounds")
    return prune_irrelevant_bounds_for_toggles(bounds, dict(toggles))


def _ordered_sampled_param_names(
    param_names_list: Sequence[str],
    bounds: Mapping[str, Tuple[float, float]],
) -> list[str]:
    """Sampled = primary_bounds keys, preferring offload vector order."""
    bound_keys = set(bounds.keys())
    ordered = [str(p) for p in param_names_list if str(p) in bound_keys]
    if ordered:
        return ordered
    return sorted(str(k) for k in bound_keys)


_TOGGLE_PARAM_KEYS = frozenset(expected_monte_carlo_exported_toggle_keys())


def _ordered_fixed_param_names(
    numeric_base: Mapping[str, Any],
    toggles: Mapping[str, Any],
    sampled_keys: set[str],
) -> list[str]:
    """Fixed = relevant numeric scalars not covered by primary_bounds or toggles."""
    pruned = prune_irrelevant_numeric_parameters_for_export(dict(numeric_base), dict(toggles))
    toggle_keys = set(_TOGGLE_PARAM_KEYS)
    toggle_keys.update(str(k) for k in toggles.keys())
    fixed: list[str] = []
    for key in sorted(pruned.keys()):
        if key in sampled_keys or key in toggle_keys:
            continue
        if pruned[key] is None:
            continue
        fixed.append(key)
    return fixed


def _diagram_state_from_toggles(
    toggles: Mapping[str, Any],
    numeric: Mapping[str, Any],
) -> Dict[str, bool]:
    merged = normalize_simulation_params({**dict(numeric), **dict(toggles)})
    out: Dict[str, bool] = {}
    for binding in MONTE_CARLO_PANEL_TOGGLE_BINDINGS:
        key = binding.ui_state_key
        if key and key in _DIAGRAM_UI_KEYS:
            out[key] = bool(merged.get(binding.param_key, False))
    for key in _DIAGRAM_UI_KEYS:
        out.setdefault(key, False)
    return out


def format_rescreen_session_summary(
    ctx: SessionRescreenContext,
    campaign: Optional[Mapping[str, Any]] = None,
) -> str:
    """Human-readable summary of the primary campaign settings used for re-screen."""
    camp = dict(campaign) if isinstance(campaign, Mapping) else {}
    lines: list[str] = []

    title = str(ctx.plot_label or ctx.session_id).strip()
    lines.append(title)
    lines.append(f"Session id: {ctx.session_id}")
    if ctx.config_key:
        lines.append(f"Configuration key: {ctx.config_key}")
    if ctx.suite_tag:
        lines.append(f"Suite tag: {ctx.suite_tag}")

    if camp:
        n_runs = camp.get("n_runs")
        n_batches = camp.get("n_primary_batches")
        base_seed = camp.get("base_seed")
        if n_runs is not None and n_batches is not None:
            lines.append(f"Primary batch: {n_runs} runs × {n_batches} batch(es)")
        if base_seed is not None:
            lines.append(f"Primary base seed: {base_seed}")
        hit_summary = _format_primary_hit_count_summary(
            camp.get("primary_hit_counts") if isinstance(camp.get("primary_hit_counts"), list) else []
        )
        if hit_summary:
            lines.append(hit_summary)

    lines.append("")
    lines.append("Metric filters:")
    if ctx.metric_checks:
        for metric, op, thr in ctx.metric_checks:
            lines.append(f"  • {metric} {op} {_fmt_num(thr)}")
    else:
        lines.append("  (none)")

    merged = normalize_simulation_params({**dict(ctx.numeric_base), **dict(ctx.toggles)})
    lines.append("")
    lines.append("Simulation toggles:")
    for binding in MONTE_CARLO_PANEL_TOGGLE_BINDINGS:
        val = bool(merged.get(binding.param_key, False))
        lines.append(f"  • {binding.param_key}: {'on' if val else 'off'}")

    bounds = _resolve_primary_bounds(camp, ctx.toggles) if camp else {}
    sampled = _ordered_sampled_param_names(ctx.param_names_list, bounds)
    sampled_keys = set(sampled)
    lines.append("")
    lines.append(f"Sampled parameters ({len(sampled)}):")
    if sampled:
        for name in sampled:
            lo, hi = bounds[name]
            lines.append(f"  • {name}: [{_fmt_num(lo)}, {_fmt_num(hi)}]")
    else:
        lines.append("  (none)")

    fixed_keys = _ordered_fixed_param_names(ctx.numeric_base, ctx.toggles, sampled_keys)
    lines.append("")
    lines.append(f"Fixed parameters ({len(fixed_keys)}):")
    if fixed_keys:
        for key in fixed_keys:
            lines.append(f"  • {key}: {_fmt_num(ctx.numeric_base.get(key))}")
    else:
        lines.append("  (none)")

    lines.append("")
    if ctx.sim_light_used:
        metrics = ", ".join(ctx.sim_light_canon) if ctx.sim_light_canon else "(default)"
        lines.append(f"Light tracking: on ({metrics})")
    else:
        lines.append("Light tracking: off")

    if ctx.settings_path:
        lines.append("")
        lines.append(f"Settings: {ctx.settings_path}")

    return "\n".join(lines)


class RerunnerSessionPanel:
    """Scrollable settings summary plus pathway diagram for a loaded campaign."""

    def __init__(self, parent: tk.Widget) -> None:
        self.outer = ttk.LabelFrame(parent, text="Loaded session", padding=6)
        self.outer.grid_rowconfigure(0, weight=1)
        self.outer.grid_columnconfigure(0, weight=1)

        text_frame = ttk.Frame(self.outer)
        text_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        scroll = ttk.Scrollbar(text_frame, orient="vertical")
        scroll.grid(row=0, column=1, sticky="ns")
        self._text = tk.Text(
            text_frame,
            height=14,
            wrap=tk.WORD,
            yscrollcommand=scroll.set,
            font=("Menlo", 10),
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self._text.grid(row=0, column=0, sticky="nsew")
        scroll.config(command=self._text.yview)

        diagram_frame = tk.LabelFrame(self.outer, text="Simulation Pathway Diagram", padx=4, pady=4)
        diagram_frame.grid(row=1, column=0, sticky="nsew")
        diagram_frame.grid_rowconfigure(0, weight=1)
        diagram_frame.grid_columnconfigure(0, weight=1)

        self._diagram_vars: Dict[str, tk.BooleanVar] = {
            key: tk.BooleanVar(value=False) for key in _DIAGRAM_UI_KEYS
        }
        self._diagram = SimulationModelDiagram(diagram_frame, width=420, height=220)
        self._diagram.canvas.grid(row=0, column=0, sticky="nsew")
        self._diagram.bind_to_vars(**self._diagram_vars)

        self.clear()

    def clear(self) -> None:
        self._set_text("Load a finished Batch Runner campaign to view its settings and pathway diagram.")
        for var in self._diagram_vars.values():
            var.set(False)

    def set_context(self, ctx: SessionRescreenContext) -> None:
        campaign = load_campaign_summary(ctx.session_dir)
        self._set_text(format_rescreen_session_summary(ctx, campaign))
        state = _diagram_state_from_toggles(ctx.toggles, ctx.numeric_base)
        for key, var in self._diagram_vars.items():
            var.set(bool(state.get(key, False)))

    def _set_text(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.insert("1.0", text)
        self._text.configure(state="disabled")
