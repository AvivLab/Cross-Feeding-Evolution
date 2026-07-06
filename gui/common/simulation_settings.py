"""Shared simulation setting keys and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


# Canonical simulation parameter keys used across GUIs and handoff payloads.
ENABLE_M1_DIFFUSION = "Enable M1 Diffusion"
ENABLE_M1_FACILITATED_DIFFUSION = "Enable M1 Facilitated Diffusion"
ENABLE_M1_PORIN_DIFFUSION = "Enable M1 Porin Diffusion"
ENABLE_M2_DIFFUSION = "Enable M2 Diffusion"
ENABLE_DIFFUSION_MUTATION = "Enable Diffusion Mutation"
HOMOGENEOUS_INITIAL_DIFFUSION_CONST = "Homogeneous Initial Diffusion Const."
ENABLE_CHEMOSTAT_FLOW = "Enable Chemostat Flow"
FLOW_PERCENTAGE = "Flow Percentage"
ENABLE_INITIAL_ENERGY = "Enable Initial Energy"
INITIAL_ENERGY = "Initial Energy"
ENABLE_INTERMEDIATE_COSTS = "Enable Intermediate Costs"
INTERMEDIATE_COSTS = "Intermediate Costs"
ENABLE_ACETATE_ADDITION = "Enable Acetate Addition"
AVERAGE_INFLOW_ACETATE = "Average In_Flow (Acetate)"
HOMOGENEOUS_POPULATION = "Homogeneous Population"
INDEPENDENT_TRAITS = "Independent Traits"
RANDOM_SEED_OPTIONAL = "Random Seed (optional)"
BINARY_DEATH_AT_ZERO_ENERGY = "Binary Death at Zero Energy"
NO_DEATH = "No Death"
CONSTANT_DEATH_PROBABILITY = "Constant Death Probability"
CONSTANT_DUPLICATION_PROBABILITY = "Constant Duplication Probability"
CONSTANT_PROBABILITY = "Constant Probability"

# Background-only keys for ``run_simulation_wrapper`` (not GUI parameter panels).
INTERNAL_SIMULATION_RUNTIME_PARAM_KEYS: Tuple[str, ...] = (
    "keep_optional_final_arrays",
    "store_history",
    "minimal_tracking",
    "tracking_metric_names",
    "tracking_metric_name",
    "silent",
)


def strip_internal_simulation_runtime_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop internal simulation runtime keys before human-facing JSON export."""
    out = dict(params)
    for key in INTERNAL_SIMULATION_RUNTIME_PARAM_KEYS:
        out.pop(key, None)
    return out


@dataclass(frozen=True)
class MonteCarloPanelToggleBinding:
    """Maps a simulation toggle param key ↔ Monte Carlo panel BooleanVar attribute."""

    param_key: str
    panel_attr: str
    ui_state_key: Optional[str] = None


# Panel vars copied primary→neutral and restored from JSON ``*_toggles`` objects.
MONTE_CARLO_PANEL_TOGGLE_BINDINGS: Tuple[MonteCarloPanelToggleBinding, ...] = (
    MonteCarloPanelToggleBinding(HOMOGENEOUS_POPULATION, "homogeneous_mode_var", "homogeneous_population"),
    MonteCarloPanelToggleBinding(INDEPENDENT_TRAITS, "independent_traits_var", "independent_traits"),
    MonteCarloPanelToggleBinding(ENABLE_INITIAL_ENERGY, "enable_initial_energy_var", "enable_initial_energy"),
    MonteCarloPanelToggleBinding(ENABLE_CHEMOSTAT_FLOW, "enable_chemostat_flow_var", "enable_chemostat_flow"),
    MonteCarloPanelToggleBinding(ENABLE_INTERMEDIATE_COSTS, "enable_intermediate_costs_var", "enable_intermediate_costs"),
    MonteCarloPanelToggleBinding(ENABLE_ACETATE_ADDITION, "enable_acetate_addition_var", "enable_acetate_addition"),
    MonteCarloPanelToggleBinding(ENABLE_M2_DIFFUSION, "m2_diffusion_var", "enable_m2_diffusion"),
    MonteCarloPanelToggleBinding(ENABLE_M1_FACILITATED_DIFFUSION, "m1_facilitation_var", "enable_m1_facilitated_diffusion"),
    MonteCarloPanelToggleBinding(ENABLE_M1_PORIN_DIFFUSION, "m1_porin_diffusion_var", "enable_m1_porin_diffusion"),
    MonteCarloPanelToggleBinding(ENABLE_M1_DIFFUSION, "m1_diffusion_var", "enable_m1_diffusion"),
    MonteCarloPanelToggleBinding(ENABLE_DIFFUSION_MUTATION, "diffusion_mutation_var", "enable_diffusion_mutation"),
    MonteCarloPanelToggleBinding(
        HOMOGENEOUS_INITIAL_DIFFUSION_CONST,
        "homogeneous_initial_diffusion_const_var",
        "homogeneous_initial_diffusion_const",
    ),
    MonteCarloPanelToggleBinding(
        BINARY_DEATH_AT_ZERO_ENERGY,
        "binary_death_at_zero_energy_var",
        "binary_death_at_zero_energy",
    ),
    MonteCarloPanelToggleBinding(
        NO_DEATH,
        "no_death_var",
        "no_death",
    ),
    MonteCarloPanelToggleBinding(
        CONSTANT_DEATH_PROBABILITY,
        "constant_death_probability_var",
        "constant_death_probability",
    ),
    MonteCarloPanelToggleBinding(
        CONSTANT_DUPLICATION_PROBABILITY,
        "constant_duplication_probability_var",
        "constant_duplication_probability",
    ),
)


def expected_monte_carlo_exported_toggle_keys() -> Tuple[str, ...]:
    """Toggle keys written by ``MonteCarloGdPanel.read_numeric_and_toggles`` / job JSON."""
    return tuple(b.param_key for b in MONTE_CARLO_PANEL_TOGGLE_BINDINGS)


def apply_toggles_to_monte_carlo_panel(panel: Any, toggles: Mapping[str, Any]) -> None:
    """Restore simulation toggles from a ``primary_toggles`` / ``neutral_toggles`` dict."""
    for binding in MONTE_CARLO_PANEL_TOGGLE_BINDINGS:
        if binding.param_key in toggles:
            getattr(panel, binding.panel_attr).set(_coerce_bool(toggles[binding.param_key], False))


def any_constant_probability_mode(toggles: Mapping[str, Any]) -> bool:
    """True when flat death and/or flat duplication uses ``Constant Probability``."""
    if _coerce_bool(toggles.get(NO_DEATH, False)) and _coerce_bool(
        toggles.get(CONSTANT_DUPLICATION_PROBABILITY, False)
    ):
        # In No Death + Constant Duplication mode, duplication is derived
        # from chemostat flow and Constant Probability is intentionally ignored.
        return False
    return _coerce_bool(toggles.get(CONSTANT_DEATH_PROBABILITY, False)) or _coerce_bool(
        toggles.get(CONSTANT_DUPLICATION_PROBABILITY, False)
    )


def both_constant_death_and_duplication(toggles: Mapping[str, Any]) -> bool:
    """True when flat death and flat duplication share one constant rate."""
    return _coerce_bool(toggles.get(CONSTANT_DEATH_PROBABILITY, False)) and _coerce_bool(
        toggles.get(CONSTANT_DUPLICATION_PROBABILITY, False)
    )


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_constant_probability(params: Mapping[str, Any]) -> Optional[float]:
    """Return the flat rate when either constant death or constant duplication is on."""
    no_death = _coerce_bool(params.get(NO_DEATH, False))
    constant_death = _coerce_bool(params.get(CONSTANT_DEATH_PROBABILITY, False))
    constant_dup = _coerce_bool(params.get(CONSTANT_DUPLICATION_PROBABILITY, False))
    if no_death and constant_dup:
        return None
    if not (constant_death or constant_dup):
        return None
    if CONSTANT_PROBABILITY in params:
        return _clamp_unit_interval(float(params[CONSTANT_PROBABILITY]))
    return 0.5


def sync_monte_carlo_panel_settings(dst: Any, src: Any) -> None:
    """Copy simulation settings BooleanVars from one Monte Carlo panel to another."""
    dst.silent_mode_var.set(src.silent_mode_var.get())
    for binding in MONTE_CARLO_PANEL_TOGGLE_BINDINGS:
        dst_var = getattr(dst, binding.panel_attr, None)
        src_var = getattr(src, binding.panel_attr, None)
        if dst_var is None or src_var is None:
            continue
        try:
            dst_var.set(src_var.get())
        except Exception:
            continue
    src_toggles = {
        NO_DEATH: bool(getattr(src, "no_death_var", None) and src.no_death_var.get()),
        CONSTANT_DEATH_PROBABILITY: bool(getattr(src, "constant_death_probability_var", None) and src.constant_death_probability_var.get()),
        CONSTANT_DUPLICATION_PROBABILITY: bool(
            getattr(src, "constant_duplication_probability_var", None)
            and src.constant_duplication_probability_var.get()
        ),
    }
    if any_constant_probability_mode(src_toggles):
        _copy_panel_fixed_param(dst, src, CONSTANT_PROBABILITY)


def _copy_panel_fixed_param(dst: Any, src: Any, param_key: str) -> None:
    """Copy one fixed-parameter entry value between Monte Carlo panels."""
    src_entries = getattr(src, "fixed_entries", None)
    dst_entries = getattr(dst, "fixed_entries", None)
    if not src_entries or not dst_entries:
        return
    src_entry = src_entries.get(param_key)
    dst_entry = dst_entries.get(param_key)
    if src_entry is None or dst_entry is None:
        return
    try:
        val = src_entry.get()
    except Exception:
        return
    try:
        dst_entry.delete(0, "end")
        dst_entry.insert(0, val)
    except Exception:
        pass


def simulation_toggles_from_ui_state(ui_state: Mapping[str, Any]) -> Dict[str, bool]:
    """Map Gradient Descent / full-save ``ui_state`` snake_case keys to simulation param keys."""
    out: Dict[str, bool] = {}
    for binding in MONTE_CARLO_PANEL_TOGGLE_BINDINGS:
        if binding.ui_state_key and binding.ui_state_key in ui_state:
            out[binding.param_key] = _coerce_bool(ui_state[binding.ui_state_key], False)
    return out


def monte_carlo_panel_toggle_watch_vars(panel: Any) -> Tuple[Any, ...]:
    """BooleanVars that should trigger primary→neutral sync when ``match_neutral_to_primary`` is on."""
    return tuple(getattr(panel, b.panel_attr) for b in MONTE_CARLO_PANEL_TOGGLE_BINDINGS) + (panel.silent_mode_var,)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce common bool-like values to bool, with a fallback default."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"true", "1", "yes", "on", "y", "t"}:
            return True
        if txt in {"false", "0", "no", "off", "n", "f"}:
            return False
    return bool(default)


def normalize_simulation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a normalized copy of simulation setting params.

    Normalization rules:
    - M1 modes are mutually constrained:
      facilitated -> M1 on, porin off
      porin -> M1 on
    - Diffusion mutation requires (M2 on OR M1 on)
    - Homogeneous initial D requires diffusion mutation on
    - Intermediate costs require M2 diffusion on (internal M2 storage)
    - Feature value keys are clamped to 0 when their feature toggle is off
    """
    out = dict(params)

    m1 = _coerce_bool(out.get(ENABLE_M1_DIFFUSION, False))
    m1_fac = _coerce_bool(out.get(ENABLE_M1_FACILITATED_DIFFUSION, False))
    m1_porin = _coerce_bool(out.get(ENABLE_M1_PORIN_DIFFUSION, False))
    m2 = _coerce_bool(out.get(ENABLE_M2_DIFFUSION, False))

    if m1_fac:
        m1 = True
        m1_porin = False
    elif m1_porin:
        m1 = True

    out[ENABLE_M1_DIFFUSION] = m1
    out[ENABLE_M1_FACILITATED_DIFFUSION] = m1_fac
    out[ENABLE_M1_PORIN_DIFFUSION] = m1_porin
    out[ENABLE_M2_DIFFUSION] = m2

    allow_diff_mutation = bool(m1 or m2)
    diff_mut = _coerce_bool(out.get(ENABLE_DIFFUSION_MUTATION, False)) if allow_diff_mutation else False
    out[ENABLE_DIFFUSION_MUTATION] = diff_mut
    out[HOMOGENEOUS_INITIAL_DIFFUSION_CONST] = (
        _coerce_bool(out.get(HOMOGENEOUS_INITIAL_DIFFUSION_CONST, False)) if diff_mut else False
    )

    enable_initial_energy = _coerce_bool(out.get(ENABLE_INITIAL_ENERGY, False))
    out[ENABLE_INITIAL_ENERGY] = enable_initial_energy
    if not enable_initial_energy:
        out[INITIAL_ENERGY] = 0.0

    enable_flow = _coerce_bool(out.get(ENABLE_CHEMOSTAT_FLOW, False))
    out[ENABLE_CHEMOSTAT_FLOW] = enable_flow
    if not enable_flow:
        out[FLOW_PERCENTAGE] = 0.0

    enable_intermediate = _coerce_bool(out.get(ENABLE_INTERMEDIATE_COSTS, False)) if m2 else False
    out[ENABLE_INTERMEDIATE_COSTS] = enable_intermediate
    if not enable_intermediate:
        out[INTERMEDIATE_COSTS] = 0.0

    enable_acetate = _coerce_bool(out.get(ENABLE_ACETATE_ADDITION, False))
    out[ENABLE_ACETATE_ADDITION] = enable_acetate
    if not enable_acetate:
        out[AVERAGE_INFLOW_ACETATE] = 0.0

    out[HOMOGENEOUS_POPULATION] = _coerce_bool(out.get(HOMOGENEOUS_POPULATION, False))
    out[INDEPENDENT_TRAITS] = _coerce_bool(out.get(INDEPENDENT_TRAITS, False))
    out[BINARY_DEATH_AT_ZERO_ENERGY] = _coerce_bool(out.get(BINARY_DEATH_AT_ZERO_ENERGY, False))
    out[NO_DEATH] = _coerce_bool(out.get(NO_DEATH, False))
    out[CONSTANT_DEATH_PROBABILITY] = _coerce_bool(out.get(CONSTANT_DEATH_PROBABILITY, False))
    if out[NO_DEATH]:
        # No Death and Constant Death are mutually exclusive; No Death takes precedence.
        out[CONSTANT_DEATH_PROBABILITY] = False
    out[CONSTANT_DUPLICATION_PROBABILITY] = _coerce_bool(out.get(CONSTANT_DUPLICATION_PROBABILITY, False))
    if out[NO_DEATH] and out[CONSTANT_DUPLICATION_PROBABILITY] and (not out[ENABLE_CHEMOSTAT_FLOW]):
        # No Death + Constant Duplication is only meaningful when flow is enabled
        # (duplication is derived from chemostat outflow in this mode).
        out[CONSTANT_DUPLICATION_PROBABILITY] = False
    if any_constant_probability_mode(out):
        shared = resolve_constant_probability(out)
        assert shared is not None
        out[CONSTANT_PROBABILITY] = shared
    else:
        out.pop(CONSTANT_PROBABILITY, None)
    return out


# Scalar GUI names (not every knob has a module-level constant).
_DIFFUSION_CONSTANT = "Diffusion Constant"
_CHEMOSTAT_VOLUME = "Chemostat Volume"
_COST_OF_TRANSPORT = "Cost of Transport"
_INITIAL_FACILITATION = "Initial Facilitation"
_INITIAL_B = "Initial B"


def diffusion_transport_active_from_toggles(toggles: Mapping[str, Any]) -> bool:
    """True when M1 and/or M2 concentration-based diffusion is enabled."""
    m1 = _coerce_bool(toggles.get(ENABLE_M1_DIFFUSION, False))
    m2 = _coerce_bool(toggles.get(ENABLE_M2_DIFFUSION, False))
    m1_fac = _coerce_bool(toggles.get(ENABLE_M1_FACILITATED_DIFFUSION, False))
    m1_porin = _coerce_bool(toggles.get(ENABLE_M1_PORIN_DIFFUSION, False))
    return m1 or m2 or m1_fac or m1_porin


def prune_irrelevant_numeric_parameters_for_export(
    numeric: Mapping[str, Any],
    toggles: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Drop scalar knobs that ``simulation.core`` does not use under the given toggle regime.

    Intended for human-facing JSON (Neutral Set Comparison saves, headless summary exports,
    HPC job files): smaller payloads and less confusion. Omitted keys are still filled from
    defaults inside ``normalize_simulation_params`` / ``run_simulation_wrapper`` when absent.
    """
    out = dict(numeric)
    diffusion_on = diffusion_transport_active_from_toggles(toggles)
    m1_fac = _coerce_bool(toggles.get(ENABLE_M1_FACILITATED_DIFFUSION, False))
    if not diffusion_on:
        out.pop(_DIFFUSION_CONSTANT, None)
        out.pop(_CHEMOSTAT_VOLUME, None)
    if not m1_fac:
        out.pop(_INITIAL_FACILITATION, None)
        out.pop(_COST_OF_TRANSPORT, None)
    if not _coerce_bool(toggles.get(ENABLE_INTERMEDIATE_COSTS, False)):
        out.pop(INTERMEDIATE_COSTS, None)
    if not _coerce_bool(toggles.get(ENABLE_ACETATE_ADDITION, False)):
        out.pop(AVERAGE_INFLOW_ACETATE, None)
    if not _coerce_bool(toggles.get(INDEPENDENT_TRAITS, False)):
        out.pop(_INITIAL_B, None)
    if not _coerce_bool(toggles.get(ENABLE_INITIAL_ENERGY, False)):
        out.pop(INITIAL_ENERGY, None)
    if not _coerce_bool(toggles.get(ENABLE_CHEMOSTAT_FLOW, False)):
        out.pop(FLOW_PERCENTAGE, None)
    if (
        _coerce_bool(toggles.get(NO_DEATH, False))
        or _coerce_bool(toggles.get(BINARY_DEATH_AT_ZERO_ENERGY, False))
        or _coerce_bool(
        toggles.get(CONSTANT_DEATH_PROBABILITY, False)
    )):
        out.pop("Death Decay Rate", None)
    if not any_constant_probability_mode(toggles):
        out.pop(CONSTANT_PROBABILITY, None)
    if _coerce_bool(toggles.get(CONSTANT_DUPLICATION_PROBABILITY, False)):
        out.pop("Duplication Sigmoid Midpoint", None)
        out.pop("Duplication Sigmoid Intensity", None)
    return out
