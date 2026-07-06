from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np

from simulation.helpers import investment_func, truncated_uniform_trait_mutation
from simulation.change_history import build_change_history_row
from gui.metrics import normalize_metric_name
from gui.common.simulation_settings import resolve_constant_probability

# Metrics that still need final-generation task / transport / energy / change / storage
# arrays when running with store_history=False and minimal_tracking=True (minimal_payload_mode).
_MINIMAL_TRACKING_NEEDS_TASK_HIST = frozenset(
    {
        "Task2 Share Weighted Prob. Mean",
        "Task2 Share Weighted Prob. Mean (100 bins)",
        "Task Specialization Index (|share-0.5|)",
        "Crossfeeding % of Acetate",
        "Crossfeeding % of Acetate (neutral perc.)",
        "Exchanged M2 Percent",
    }
)
_MINIMAL_TRACKING_NEEDS_CHANGE_HISTORY = frozenset(
    {
        "Trait Std Dev (Neutral Perc.)",
        "Trait Entropy (Neutral Perc.)",
        "Crossfeeding % of Acetate (neutral perc.)",
        "Exchanged M2 Percent (neutral perc.)",
    }
)


def _active_tracking_metric_names(
    tracking_metric_name: str,
    tracking_metric_names: Optional[Sequence[str]],
) -> frozenset[str]:
    """Canonical metric names whose minimal-tracking retention rules are OR-merged."""
    names: set[str] = set()
    if tracking_metric_name:
        names.add(tracking_metric_name)
    if tracking_metric_names:
        for raw in tracking_metric_names:
            c = normalize_metric_name(str(raw or ""))
            if c:
                names.add(c)
    return frozenset(names)


def _coerce_bool_param(params: dict, key: str, default: bool) -> bool:
    """Parse a boolean-like parameter value into a strict bool."""
    raw = params.get(key, default)
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if raw in (0, 0.0):
            return False
        if raw in (1, 1.0):
            return True
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in {"true", "1", "yes", "on", "y", "t"}:
            return True
        if val in {"false", "0", "no", "off", "n", "f"}:
            return False
    raise ValueError(f"Parameter '{key}' must be boolean-like; got {raw!r} ({type(raw).__name__})")


def _effective_chemostat_volume(chemostat_volume: float, pop_size: int, cell_volume: float) -> float:
    """Effective env fluid volume V_eff for env concentrations (diffusion regime only)."""
    v = float(chemostat_volume) - (float(pop_size) * float(cell_volume))
    return v if v >= float(cell_volume) else float(cell_volume)


def _uses_chemostat_volume(enable_m1_diffusion: bool, enable_m2_diffusion: bool) -> bool:
    """True when chemostat_volume is used (env concentrations + optional duplication volume cap)."""
    return bool(enable_m1_diffusion) or bool(enable_m2_diffusion)

def _diffusion_import_scale_capped(env_amount: float, total_import_desired: float) -> tuple[float, float]:
    """
    Diffusion import helper:
    - imports up to total_import_desired, but cannot exceed env_amount
    - if env_amount is insufficient, scales imports down proportionally
    Returns (scale, env_after).
    """
    if env_amount <= 0.0 or total_import_desired <= 0.0:
        return 0.0, env_amount
    if total_import_desired <= env_amount:
        return 1.0, (env_amount - total_import_desired)
    # Not enough in env: scale down so total import equals env_amount
    return (env_amount / total_import_desired), 0.0


def run_simulation(
    cost_of_life: float,
    mutation_rate: float,
    mutation_scale: float,
    number_gen: int,
    organism_count: int,
    chemostat_volume: float,
    sin_average: float,
    inv_mod: float,
    dupx0: float,
    dupk: float,
    diffusion_const: float,
    death_decay_rate: float,
    binary_death_at_zero_energy: bool = False,
    constant_death_probability: bool = False,
    constant_duplication_probability: bool = False,
    constant_probability: float = 0.5,
    acetate_average: float = 0.0,
    acetate_ratio: float = 1.0,
    independent_traits: bool = False,
    homogeneous_population: bool = False,
    initial_A: float = 0.5,
    initial_B: float = 0.5,
    enable_m1_facilitated_diffusion: bool = False,
    cost_of_transport: float = 0.0,
    initial_facilitation: float = 0.5,
    silent: bool = True,
    store_history: bool = True,
    enable_m1_diffusion: bool = False,
    m1_porin_simple_diffusion: bool = False,
    enable_m2_diffusion: bool = True,
    enable_diffusion_mutation: bool = False,
    homogeneous_initial_diffusion_const: bool = False,
    enable_initial_energy: bool = False,
    initial_energy: float = 0.0,
    enable_intermediate_costs: bool = False,
    intermediate_costs: float = 0.0,
    enable_chemostat_flow: bool = False,
    flow_percentage: float = 0.0,
    enable_acetate_addition: bool = False,
    minimal_tracking: bool = False,
    tracking_metric_name: str = "",
    tracking_metric_names: Optional[Tuple[str, ...]] = None,
    keep_optional_final_arrays: Optional[bool] = None,
    seed=None,
    no_death: bool = False,
):
    """Run the enzyme-population simulation for one parameterization.

    This is the canonical model execution entry point used by both GUI and
    optimization workflows. It applies duplication/death events (with optional
    per-duplicate truncated-uniform trait mutation on ``[0, 1]``),
    diffusion (M1/M2; simple/facilitated/porin modes), optional chemostat flow,
    and optional acetate/energy/intermediate-cost toggles across generations.

    Returns a result dictionary containing time-series histories used by metrics
    and plotting. The `change_history` field uses the canonical positional
    layout from `simulation.change_history`.
    """
    seed_int = None
    if seed not in (None, "", "None"):
        try:
            seed_int = int(seed)
        except Exception:
            try:
                seed_int = int(float(seed))
            except Exception:
                seed_int = None
    # Per-run RNG avoids global RNG contention under threaded replicates.
    rng_np = np.random.default_rng(seed_int)
    """
    A lightweight, import-safe core for the Simulation model.

    Returns a dict with:
    - 'A_history': list[ndarray] (trait values each generation)
    - 'B_history': list[ndarray] (trait values each generation; may be independent)
    - 'metabs_history': list[[metab1_env, metab2_env]]
    - 'energy_history': list[ndarray]
    - 'storage_history': list[ndarray] (stored M2)
    - 'task1_performance': list[ndarray] (per-organism Task 1 amounts each generation)
    - 'task2_performance': list[ndarray] (per-organism Task 2 amounts each generation)
    - 'M2_export_history': list[ndarray] (per-organism M2 exported each generation)
    - 'M2_import_history': list[ndarray] (per-organism M2 imported from environment each generation)
    - 'change_history': list[[deaths, dups, accepted_mutations, flow_removed, accepted_aux_mutations]]
      where accepted_mutations tracks A/B trait mutations and accepted_aux_mutations tracks T/D mutations.
      Access via helpers in simulation.change_history to avoid positional index drift.
    - 'investment_modifier': float (coupled exponential investment exponent)
    - 'final_metabolite_environment': [M1_env, M2_env] after the last generation completes (deaths/flow/etc.)
    - 'metabolite_environment_after_inflow_final_generation': [M1_env, M2_env] on the last scheduled
      generation immediately after M1 inflow and optional acetate M2 inflow (for neutral metrics).
    - 'enable_m1_diffusion', 'enable_m2_diffusion', 'enable_intermediate_costs': bool flags for metrics
      that only apply in the pooled-env (no diffusion / no intermediate storage penalty) regime.
    - 'number_gen', 'collapsed'

    When ``minimal_tracking`` is True and ``store_history`` is False, optional
    ``tracking_metric_names`` (plus ``tracking_metric_name``) lists canonical metric names;
    per-generation retention is the OR of each metric's requirements so multi-metric filters stay valid.

    ``keep_optional_final_arrays``: when ``None`` (default), final-generation energy/storage/flux
    snapshots are omitted only in ``minimal_payload_mode``. Pass ``True`` to retain them under light
    tracking (one vector per array at the last generation, not full time series).
    """
    cost_of_life = float(cost_of_life)
    if cost_of_life <= 0.0:
        raise ValueError("Simulation model requires Cost of Life > 0 to avoid runaway growth.")

    diffusion_const = float(diffusion_const)
    acetate_ratio = float(acetate_ratio)
    death_decay_rate = float(death_decay_rate)
    binary_death_at_zero_energy = bool(binary_death_at_zero_energy)
    no_death = bool(no_death)
    constant_death_probability = bool(constant_death_probability)
    constant_duplication_probability = bool(constant_duplication_probability)
    flat_probability = float(constant_probability)
    if flat_probability < 0.0:
        flat_probability = 0.0
    elif flat_probability > 1.0:
        flat_probability = 1.0
    if (
        constant_death_probability
        and constant_duplication_probability
        and (not binary_death_at_zero_energy)
        and (not no_death)
        and flat_probability > 0.5
    ):
        raise ValueError(
            "Constant Probability must be <= 0.5 when Constant Death Probability and "
            "Constant Duplication Probability are both enabled with Binary Death at Zero Energy disabled."
        )
    death_probability = flat_probability if constant_death_probability else 0.0
    if no_death:
        death_probability = 0.0
    duplication_probability = flat_probability if constant_duplication_probability else 0.0
    enable_chemostat_flow = bool(enable_chemostat_flow)
    flow_percentage = float(flow_percentage)
    if (
        no_death
        and constant_duplication_probability
        and enable_chemostat_flow
        and flow_percentage > 50.0
    ):
        raise ValueError(
            "Flow Percentage must be <= 50 when No Death and Constant Duplication Probability are both enabled."
        )
    cost_of_transport = float(cost_of_transport)
    enable_m1_diffusion = bool(enable_m1_diffusion)
    enable_m2_diffusion = bool(enable_m2_diffusion)
    m1_porin_simple_diffusion = bool(m1_porin_simple_diffusion)
    enable_diffusion_mutation = bool(enable_diffusion_mutation)
    homogeneous_initial_diffusion_const = bool(homogeneous_initial_diffusion_const)
    enable_m1_facilitated_diffusion = bool(enable_m1_facilitated_diffusion)
    enable_acetate_addition = bool(enable_acetate_addition)
    minimal_tracking = bool(minimal_tracking)
    tracking_metric_name = normalize_metric_name(str(tracking_metric_name or ""))
    extra_tm_tuple: Tuple[str, ...] = tuple()
    if tracking_metric_names:
        extra_tm_tuple = tuple(
            normalize_metric_name(str(x or ""))
            for x in tracking_metric_names
            if normalize_metric_name(str(x or ""))
        )
    active_tracking = _active_tracking_metric_names(tracking_metric_name, extra_tm_tuple)
    enable_intermediate_costs = bool(enable_intermediate_costs)
    intermediate_costs = float(intermediate_costs)
    if intermediate_costs < 0.0:
        intermediate_costs = 0.0
    if enable_m1_facilitated_diffusion:
        enable_m1_diffusion = True

    independent_traits = bool(independent_traits)
    minimal_payload_mode = minimal_tracking and (not store_history)

    metric_requires_task_hist = bool(active_tracking & _MINIMAL_TRACKING_NEEDS_TASK_HIST)
    metric_requires_change_history = bool(active_tracking & _MINIMAL_TRACKING_NEEDS_CHANGE_HISTORY)

    keep_task_hist = (not minimal_payload_mode) or metric_requires_task_hist
    keep_change_history = (not minimal_payload_mode) or metric_requires_change_history
    if keep_optional_final_arrays is None:
        keep_optional_final_arrays = not minimal_payload_mode
    else:
        keep_optional_final_arrays = bool(keep_optional_final_arrays)

    # Initial trait distribution
    if homogeneous_population:
        # Homogeneous initial population
        Anumbers = np.full(organism_count, float(initial_A), dtype=float)
        Bnumbers = np.full(organism_count, float(initial_B), dtype=float) if independent_traits else None
    else:
        Anumbers = rng_np.random(organism_count)
        Bnumbers = rng_np.random(organism_count) if independent_traits else None
    initial_A_values = Anumbers.copy()
    np.clip(Anumbers, 0.0, 1.0, out=Anumbers)
    if independent_traits and Bnumbers is not None:
        np.clip(Bnumbers, 0.0, 1.0, out=Bnumbers)

    # Optional M1 facilitation trait (0-1), used when enabled.
    Tnumbers = None
    if enable_m1_facilitated_diffusion:
        if homogeneous_population:
            Tnumbers = np.full(organism_count, float(initial_facilitation), dtype=float)
        else:
            Tnumbers = rng_np.random(organism_count)
        np.clip(Tnumbers, 0.0, 1.0, out=Tnumbers)

    # Optional per-organism diffusion trait (D). When mutation is disabled, D is constant.
    Dnumbers = None
    if enable_m2_diffusion and enable_diffusion_mutation:
        if homogeneous_initial_diffusion_const:
            Dnumbers = np.full(organism_count, diffusion_const, dtype=float)
        else:
            Dnumbers = rng_np.random(organism_count)

    e0 = float(initial_energy)
    if e0 < 0.0:
        e0 = 0.0
    storedEnergy = np.full(organism_count, e0, dtype=float) if bool(enable_initial_energy) else np.zeros(organism_count, dtype=float)
    storedMetab1 = np.zeros(organism_count, dtype=float)
    storedMetab2 = np.zeros(organism_count, dtype=float)

    def _extend_inplace(base: np.ndarray, tail: np.ndarray) -> np.ndarray:
        """Append tail to base with lower churn than repeated concatenate."""
        if tail is None:
            return base
        tail = np.asarray(tail, dtype=base.dtype)
        n_add = int(tail.size)
        if n_add <= 0:
            return base
        old_n = int(base.size)
        try:
            # Boolean-mask filtering above yields owning arrays; resize can often extend in-place.
            base.resize(old_n + n_add, refcheck=False)
            base[old_n:] = tail
            return base
        except Exception:
            return np.concatenate((base, tail))

    metab1env = 0.0
    metab2env = 0.0

    inflow_amount = float(sin_average)
    acetate_inflow_amount = float(acetate_average)

    # When running inside optimization loops, storing full per-generation histories can blow up memory
    # (population can be large; arrays each generation are huge). In that case, store only the final
    # generation arrays in 1-element lists so metrics can read a final-generation history.
    if store_history:
        A_history = [None] * number_gen
        B_history = [None] * number_gen
        T_history = [None] * number_gen  # Facilitated diffusion genotype (optional)
        task1_performance_history = [None] * number_gen
        task2_performance_history = [None] * number_gen
        M1_export_history = [None] * number_gen
        M1_import_history = [None] * number_gen
        M2_export_history = [None] * number_gen
        M2_import_history = [None] * number_gen
        metabs_history = []
        energy_history = []
        storage_history = []
        storage_M1_history = []
    else:
        A_history = []
        B_history = []
        T_history = []
        task1_performance_history = []
        task2_performance_history = []
        M1_export_history = []
        M1_import_history = []
        M2_export_history = []
        M2_import_history = []
        metabs_history = []
        energy_history = []
        storage_history = []
        storage_M1_history = []
    change_history = []

    start_time = time.time()
    cell_volume = 1.0
    # [M1_env, M2_env] after M1 inflow (+ optional acetate M2 inflow) on the last scheduled generation only.
    metabolite_environment_after_inflow_final_generation = None

    for nG in range(number_gen):
        flow_removed_num = 0
        if len(Anumbers) == 0:
            if not silent:
                print(
                    f"Simulation: Collapsed at generation {nG} (population extinct).",
                    flush=True,
                )
            return {
                "A_history": A_history[:nG] if store_history else [],
                "B_history": B_history[:nG] if store_history else [],
                "T_history": T_history[:nG] if store_history else [],
                "number_gen": nG,
                "independent_traits": independent_traits,
                "mutation_rate": float(mutation_rate),
                "mutation_scale": float(mutation_scale),
                "initial_A": initial_A_values.copy(),
                "metabs_history": metabs_history if store_history else [],
                "energy_history": energy_history if store_history else [],
                "storage_history": storage_history if store_history else [],
                "storage_M1_history": storage_M1_history[:nG] if store_history else [],
                "task1_performance": task1_performance_history[:nG] if store_history else [],
                "task2_performance": task2_performance_history[:nG] if store_history else [],
                "M2_export_history": M2_export_history[:nG] if store_history else [],
                "M2_import_history": M2_import_history[:nG] if store_history else [],
                "M1_export_history": M1_export_history[:nG] if store_history else [],
                "M1_import_history": M1_import_history[:nG] if store_history else [],
                "change_history": change_history,
                "collapsed": True,
                "investment_modifier": float(inv_mod),
                "final_metabolite_environment": [float(metab1env), float(metab2env)],
                "metabolite_environment_after_inflow_final_generation": None,
                "enable_m1_diffusion": bool(enable_m1_diffusion),
                "enable_m2_diffusion": bool(enable_m2_diffusion),
                "enable_intermediate_costs": bool(enable_intermediate_costs),
            }

        # Progress indicator with ETA (skip if silent)
        if (not silent) and (nG % max(1, number_gen // 20) == 0) and (nG > 0):
            elapsed = time.time() - start_time
            generations_per_sec = nG / elapsed if elapsed > 0 else 0.0
            remaining_gens = number_gen - nG
            eta = (remaining_gens / generations_per_sec) if generations_per_sec > 0 else 0.0
            print(
                f"Simulation: Generation {nG}/{number_gen} ({100*nG/number_gen:.1f}%), "
                f"Pop: {len(Anumbers)}, ETA: {eta:.1f}s",
                flush=True,
            )

        # Traits stay in [0, 1] via initialization and bounded kernels; avoid per-generation full-array clipping.
        if independent_traits:
            # Independent traits: use raw values (no investment function).
            Afunc = Anumbers
            Bfunc = Bnumbers
        else:
            # Coupled traits: B = 1 - A
            Bnumbers = 1.0 - Anumbers
            # Investment function type is fixed internally
            Afunc = investment_func(inv_mod, Anumbers)
            Bfunc = investment_func(inv_mod, Bnumbers)

        # Add the amount of M1 available to the environment
        metab1env += inflow_amount
        if metab1env < 0:
            metab1env = 0.0

        # Optional acetate inflow: add M2 to the environment each generation.
        if enable_acetate_addition:
            metab2env += acetate_inflow_amount
            if metab2env < 0:
                metab2env = 0.0

        if nG == number_gen - 1:
            metabolite_environment_after_inflow_final_generation = [float(metab1env), float(metab2env)]

        # Handle M1 diffusion (optional). When enabled, M1 diffuses into internal pools before Task A conversion.
        if enable_m1_diffusion:
            effective_chemostat_volume = _effective_chemostat_volume(chemostat_volume, len(storedMetab1), cell_volume)
            env_concentration = float(metab1env) / float(effective_chemostat_volume) if effective_chemostat_volume > 0.0 else 0.0
            internal_concentration = storedMetab1 / cell_volume
            if enable_m1_facilitated_diffusion:
                diffusion_flux = Tnumbers * (env_concentration - internal_concentration) * cell_volume
            else:
                diffusion_flux = diffusion_const * (env_concentration - internal_concentration) * cell_volume
            import_desired = np.maximum(diffusion_flux, 0.0)
            m1_import_only_active = bool(enable_m1_facilitated_diffusion or m1_porin_simple_diffusion)
            export_flux_m1 = np.zeros_like(import_desired) if m1_import_only_active else np.maximum(-diffusion_flux, 0.0)

            total_import_desired = float(np.sum(import_desired))
            if total_import_desired > 0.0 and metab1env > 0.0:
                # Diffusion import: import desired unless env is limiting, then scale down.
                import_scale, metab1env = _diffusion_import_scale_capped(float(metab1env), float(total_import_desired))
                import_flux_m1 = import_desired * import_scale
            else:
                import_flux_m1 = np.zeros_like(import_desired)
            if metab1env < 0.0:
                metab1env = 0.0

            # Mass-balanced update: cap export by internal M1 available when export is computed.
            # (Import/export are elementwise mutually exclusive here, but this ordering is clearer.)
            export_actual = np.minimum(export_flux_m1, np.maximum(storedMetab1, 0.0))
            storedMetab1 = (storedMetab1 - export_actual) + import_flux_m1
            export_flux_m1 = export_actual
            metab1env += float(np.sum(export_flux_m1))
            if metab1env < 0.0:
                metab1env = 0.0

            # Task A: conversion is capacity-limited by Afunc and availability-limited by stored M1.
            A_share = np.minimum(Afunc, storedMetab1)
            storedMetab1 -= A_share
        else:
            # The amount of M1 that can be converted to M2 determined by the investment function and the amount of M1 available
            change1 = min(float(np.sum(Afunc)), float(metab1env))

            # Subtract the amount of M1 that was converted to M2 from the environment
            metab1env -= change1
            if metab1env < 0:
                metab1env = 0.0

            # Per-capita shares for Task A conversion
            sumA = float(np.sum(Afunc))
            A_share = (Afunc / sumA) * change1 if sumA else np.zeros_like(Afunc)

        # If M2 diffusion is enabled, perform the diffusion process
        if enable_m2_diffusion:
            # Store produced M2 before diffusion/consumption
            storedMetab2 += A_share

            # Task B competes with diffusion (bidirectional)
            # Consumption is capacity-limited by Bfunc and availability-limited by stored M2.
            desired_consumption = np.minimum(Bfunc, storedMetab2)
            effective_chemostat_volume = _effective_chemostat_volume(chemostat_volume, len(storedMetab2), cell_volume)
            env_concentration = float(metab2env) / float(effective_chemostat_volume) if effective_chemostat_volume > 0.0 else 0.0
            internal_concentration = storedMetab2 / cell_volume
            if enable_diffusion_mutation and Dnumbers is not None:
                diffusion_flux = Dnumbers * (internal_concentration - env_concentration) * cell_volume
            else:
                diffusion_flux = diffusion_const * (internal_concentration - env_concentration) * cell_volume
            export_flux = np.maximum(diffusion_flux, 0.0)
            import_desired = np.maximum(-diffusion_flux, 0.0)

            total_import_desired = float(np.sum(import_desired))
            if total_import_desired > 0.0 and metab2env > 0.0:
                # Diffusion import: import desired unless env is limiting, then scale down.
                import_scale, metab2env = _diffusion_import_scale_capped(float(metab2env), float(total_import_desired))
                import_flux = import_desired * import_scale
            else:
                import_flux = np.zeros_like(import_desired)
            if metab2env < 0.0:
                metab2env = 0.0

            total_outflow = desired_consumption + export_flux
            excess_mask = total_outflow > storedMetab2
            if np.any(excess_mask):
                scale = storedMetab2[excess_mask] / total_outflow[excess_mask]
                desired_consumption[excess_mask] *= scale
                export_flux[excess_mask] *= scale

            storedMetab2 -= (desired_consumption + export_flux)
            storedMetab2 += import_flux
            metab2env += float(np.sum(export_flux))
            if metab2env < 0.0:
                metab2env = 0.0

            B_share = desired_consumption
        # If M2 diffusion is not enabled, do not perform the diffusion process
        else:
            # Export all produced M2; Task B draws from the environment
            export_flux = A_share.copy()
            metab2env += float(np.sum(export_flux))
            if metab2env < 0.0:
                metab2env = 0.0
            storedMetab2[:] = 0.0

            total_demand = float(np.sum(Bfunc))
            if total_demand > 0.0 and metab2env > 0.0:
                scale = min(1.0, metab2env / total_demand)
                B_share = Bfunc * scale
                metab2env -= total_demand * scale
            else:
                B_share = np.zeros_like(Bfunc)
            import_flux = B_share.copy()

        if not enable_m1_diffusion:
            import_flux_m1 = np.zeros_like(A_share)
            export_flux_m1 = np.zeros_like(A_share)

        # Track per-organism task performance and transport (per generation)
        # Task 1: how much M1 was converted to M2 per organism (A_share)
        # Task 2: how much M2 was consumed per organism (B_share)
        # M1 Export/Import: per-organism M1 diffusion out/in
        # Export: how much M2 diffused out per organism (export_flux)
        # Import: how much M2 diffused in per organism (import_flux)
        if store_history:
            task1_performance_history[nG] = A_share.copy()
            task2_performance_history[nG] = B_share.copy()
            M1_export_history[nG] = export_flux_m1.copy()
            M1_import_history[nG] = import_flux_m1.copy()
            M2_export_history[nG] = export_flux.copy()
            M2_import_history[nG] = import_flux.copy()

        # Energy update
        life_cost = cost_of_life * (Anumbers + Bnumbers) if independent_traits else cost_of_life
        # Optional energetic penalty proportional to internal intermediate storage (stored M2).
        # This discourages hoarding by reducing energy as storage increases.
        intermediate_storage_penalty = (intermediate_costs * storedMetab2) if enable_intermediate_costs else 0.0
        if enable_m1_facilitated_diffusion:
            storedEnergy += (A_share + (acetate_ratio * B_share)) - life_cost - (cost_of_transport * Tnumbers) - intermediate_storage_penalty
        else:
            storedEnergy += (A_share + (acetate_ratio * B_share)) - life_cost - intermediate_storage_penalty

        # Death / Survival:
        # - Constant death: flat death_probability everywhere; if binary death is also on,
        #   E <= 0 is forced to P(death)=1.
        # - Binary only: P(death)=1 if E <= 0 else 0.
        # - Default: P(death)=1 if E <= 0 else exp(-death_decay_rate * E).
        if no_death:
            death_probs = np.zeros_like(storedEnergy, dtype=float)
        elif constant_death_probability:
            death_probs = np.full_like(storedEnergy, death_probability, dtype=float)
            if binary_death_at_zero_energy:
                death_probs[storedEnergy <= 0] = 1.0
        elif binary_death_at_zero_energy:
            death_probs = np.ones_like(storedEnergy, dtype=float)
            death_probs[storedEnergy > 0] = 0.0
        else:
            death_probs = np.ones_like(storedEnergy, dtype=float)
            positive_mask = storedEnergy > 0
            death_probs[positive_mask] = np.exp(-death_decay_rate * storedEnergy[positive_mask])
        death_mask = rng_np.random(len(storedEnergy)) < death_probs
        survival_mask = ~death_mask

        deathsNum = int(np.sum(death_mask))
        # Dying organisms release stored M2 back to the environment
        if deathsNum > 0:
            metab2env += float(np.sum(storedMetab2[death_mask]))
            if metab2env < 0:
                metab2env = 0.0
            if enable_m1_diffusion:
                metab1env += float(np.sum(storedMetab1[death_mask]))
                if metab1env < 0:
                    metab1env = 0.0
        Anumbers = Anumbers[survival_mask]
        if independent_traits and Bnumbers is not None:
            Bnumbers = Bnumbers[survival_mask]
        storedEnergy = storedEnergy[survival_mask]
        storedMetab1 = storedMetab1[survival_mask]
        storedMetab2 = storedMetab2[survival_mask]
        if enable_m1_facilitated_diffusion and Tnumbers is not None:
            Tnumbers = Tnumbers[survival_mask]
        if enable_m2_diffusion and enable_diffusion_mutation and Dnumbers is not None:
            Dnumbers = Dnumbers[survival_mask]

        # Duplication probability:
        # - Constant mode: fixed P(dup) at all energies
        # - Sigmoid mode: P(dup)=0 at E <= 0, logistic in energy for E > 0
        if constant_duplication_probability:
            dup_p = float(duplication_probability)
            if no_death:
                # No Death + Constant Duplication:
                # use chemostat outflow (phi) as the balancing reference and convert
                # to per-step duplication p = phi / (1 - phi), clipped to [0,1].
                # Same correction form as constant death mode.
                try:
                    flow_frac = float(flow_percentage) / 100.0 if enable_chemostat_flow else 0.0
                except Exception:
                    flow_frac = 0.0
                flow_frac = float(np.clip(flow_frac, 0.0, 1.0))
                denom = max(1e-12, 1.0 - flow_frac)
                dup_p = flow_frac / denom
            elif constant_death_probability and (not binary_death_at_zero_energy):
                # Correction for coupled constant death+duplication updates:
                # if death uses probability p each step, choosing duplication as
                # p + p^2/(1-p) (equivalently p/(1-p)) balances expected losses so
                # death and duplication cancel on average, keeping population stable.
                # This only yields a valid Bernoulli duplication probability when
                # p <= 0.5, so larger p is rejected during parameter validation.
                denom = max(1e-12, 1.0 - dup_p)
                dup_p = dup_p / denom
            dup_p = float(np.clip(dup_p, 0.0, 1.0))
            dup_probabilities = np.full_like(storedEnergy, dup_p, dtype=float)
        else:
            dup_probabilities = np.zeros_like(storedEnergy, dtype=float)
            above_zero = storedEnergy > 0
            dup_probabilities[above_zero] = 1.0 / (
                1.0 + np.exp(-dupk * (storedEnergy[above_zero] - dupx0))
            )

        will_duplicate = rng_np.random(len(storedEnergy)) < dup_probabilities

        # Volume-limited reproduction (diffusion regime only; pooled env has no V_eff cap).
        if _uses_chemostat_volume(enable_m1_diffusion, enable_m2_diffusion):
            max_allowed_volume = chemostat_volume - cell_volume
            current_volume = len(storedEnergy) * cell_volume
            volume_available = max_allowed_volume - current_volume
            volume_available = max(0.0, volume_available)
            if cell_volume <= 0:
                max_new_organisms = 0
            else:
                max_new_organisms = max(0, int(volume_available / cell_volume))

            if np.sum(will_duplicate) > max_new_organisms:
                duplicate_indices = np.where(will_duplicate)[0]
                if max_new_organisms > 0:
                    allowed_indices = rng_np.choice(duplicate_indices, size=max_new_organisms, replace=False)
                    will_duplicate = np.zeros(len(storedEnergy), dtype=bool)
                    will_duplicate[allowed_indices] = True
                else:
                    will_duplicate = np.zeros(len(storedEnergy), dtype=bool)

        dupsNum = int(np.sum(will_duplicate))

        num_accepted_mutations = 0  # A/B accepted mutations (used by neutral metrics assumptions)
        num_accepted_aux_mutations = 0  # Accepted mutations in auxiliary traits (T/D)
        if np.any(will_duplicate):
            duplicates = Anumbers[will_duplicate].copy()
            dup_B = Bnumbers[will_duplicate].copy() if independent_traits else None
            dup_T = Tnumbers[will_duplicate].copy() if (enable_m1_facilitated_diffusion and Tnumbers is not None) else None
            dup_energy = storedEnergy[will_duplicate] / 2.0
            dup_metab1 = storedMetab1[will_duplicate] / 2.0
            dup_metab = storedMetab2[will_duplicate] / 2.0
            storedEnergy[will_duplicate] /= 2.0
            storedMetab1[will_duplicate] /= 2.0
            storedMetab2[will_duplicate] /= 2.0

            # Mutations (A): truncated-uniform kernel on [0,1] (same as sweep script).
            mut_mask = rng_np.random(len(duplicates)) < mutation_rate
            mutated = truncated_uniform_trait_mutation(duplicates, mut_mask, mutation_scale, rng_np)
            accept_mask = np.abs(mutated - duplicates) > 1e-12
            duplicates[accept_mask] = mutated[accept_mask]
            num_accepted_mutations = int(np.sum(accept_mask))

            # Mutations (B) when independent
            if independent_traits and dup_B is not None:
                mut_mask_b = rng_np.random(len(dup_B)) < mutation_rate
                mutated_b = truncated_uniform_trait_mutation(dup_B, mut_mask_b, mutation_scale, rng_np)
                accept_mask_b = np.abs(mutated_b - dup_B) > 1e-12
                dup_B[accept_mask_b] = mutated_b[accept_mask_b]
                num_accepted_mutations += int(np.sum(accept_mask_b))

            if enable_m1_facilitated_diffusion and dup_T is not None:
                mut_T = rng_np.random(len(dup_T)) < mutation_rate
                if np.any(mut_T):
                    mutated_T = truncated_uniform_trait_mutation(dup_T, mut_T, mutation_scale, rng_np)
                    accept_T = np.abs(mutated_T - dup_T) > 1e-12
                    dup_T[accept_T] = mutated_T[accept_T]
                    num_accepted_aux_mutations += int(np.sum(accept_T))

            Anumbers = _extend_inplace(Anumbers, duplicates)
            if independent_traits and dup_B is not None:
                Bnumbers = _extend_inplace(Bnumbers, dup_B)
            storedEnergy = _extend_inplace(storedEnergy, dup_energy)
            storedMetab1 = _extend_inplace(storedMetab1, dup_metab1)
            storedMetab2 = _extend_inplace(storedMetab2, dup_metab)
            if enable_m1_facilitated_diffusion and dup_T is not None:
                Tnumbers = _extend_inplace(Tnumbers, dup_T)

            if enable_m2_diffusion and enable_diffusion_mutation and Dnumbers is not None:
                dup_D = Dnumbers[will_duplicate].copy()
                mut_D = rng_np.random(len(dup_D)) < mutation_rate
                if np.any(mut_D):
                    mutated_D = truncated_uniform_trait_mutation(dup_D, mut_D, mutation_scale, rng_np)
                    accept_D = np.abs(mutated_D - dup_D) > 1e-12
                    dup_D[accept_D] = mutated_D[accept_D]
                    num_accepted_aux_mutations += int(np.sum(accept_D))
                Dnumbers = _extend_inplace(Dnumbers, dup_D)

        # Optional chemostat flow: Bernoulli organism removal; env pools scaled by (1 - phi).
        if enable_chemostat_flow:
            try:
                flow_frac = float(flow_percentage) / 100.0
            except Exception:
                flow_frac = 0.0
            if flow_frac > 0.0:
                flow_frac = min(max(flow_frac, 0.0), 1.0)
                pop_size = len(Anumbers)

                # Bernoulli per-organism flow removal:
                # each organism has independent probability = flow_frac of being removed.
                # This avoids floor/round artifacts at small population sizes
                # (e.g., a single organism can still flow out).
                if pop_size > 0:
                    remove_mask = rng_np.random(pop_size) < flow_frac
                    n_remove = int(np.sum(remove_mask))
                    if n_remove > 0:
                        keep_mask = ~remove_mask
                        flow_removed_num = int(n_remove)
                        Anumbers = Anumbers[keep_mask]
                        if independent_traits and Bnumbers is not None:
                            Bnumbers = Bnumbers[keep_mask]
                        storedEnergy = storedEnergy[keep_mask]
                        storedMetab1 = storedMetab1[keep_mask]
                        storedMetab2 = storedMetab2[keep_mask]
                        if enable_m1_facilitated_diffusion and Tnumbers is not None:
                            Tnumbers = Tnumbers[keep_mask]
                        if enable_m2_diffusion and enable_diffusion_mutation and Dnumbers is not None:
                            Dnumbers = Dnumbers[keep_mask]

                metab1env *= (1.0 - flow_frac)
                metab2env *= (1.0 - flow_frac)

        # Record end-of-generation state (post death/dup/mutation), so “final” metrics use the actual final state.
        if store_history:
            A_history[nG] = Anumbers.copy()
            B_history[nG] = Bnumbers.copy()
            if enable_m1_facilitated_diffusion and Tnumbers is not None:
                T_history[nG] = Tnumbers.copy()
            metabs_history.append([float(metab1env), float(metab2env)])
            energy_history.append(storedEnergy.copy())
            storage_history.append(storedMetab2.copy())
            storage_M1_history.append(storedMetab1.copy())

        if keep_change_history:
            change_history.append(
                build_change_history_row(
                    deaths=deathsNum,
                    dups=dupsNum,
                    accepted_mutations=num_accepted_mutations,
                    flow_removed=flow_removed_num,
                    accepted_aux_mutations=num_accepted_aux_mutations,
                )
            )

    if not store_history:
        # Minimal, metric-compatible outputs: keep only last-generation arrays.
        A_history = [Anumbers.copy()] if len(Anumbers) else []
        if independent_traits:
            B_history = [Bnumbers.copy()] if len(Anumbers) else []
        else:
            B_history = [(1.0 - Anumbers).copy()] if len(Anumbers) else []
        T_history = [Tnumbers.copy()] if (keep_optional_final_arrays and enable_m1_facilitated_diffusion and Tnumbers is not None and len(Anumbers)) else []
        task1_performance_history = [A_share.copy()] if (keep_task_hist and "A_share" in locals()) else []
        task2_performance_history = [B_share.copy()] if (keep_task_hist and "B_share" in locals()) else []
        M1_export_history = [export_flux_m1.copy()] if (keep_optional_final_arrays and "export_flux_m1" in locals()) else []
        M1_import_history = [import_flux_m1.copy()] if (keep_optional_final_arrays and "import_flux_m1" in locals()) else []
        M2_export_history = [export_flux.copy()] if (keep_optional_final_arrays and "export_flux" in locals()) else []
        M2_import_history = [import_flux.copy()] if (keep_optional_final_arrays and "import_flux" in locals()) else []
        metabs_history = [[float(metab1env), float(metab2env)]] if keep_optional_final_arrays else []
        energy_history = [storedEnergy.copy()] if keep_optional_final_arrays else []
        storage_history = [storedMetab2.copy()] if keep_optional_final_arrays else []
        storage_M1_history = [storedMetab1.copy()] if keep_optional_final_arrays else []

    if not silent:
        mean_A = float(np.mean(Anumbers)) if len(Anumbers) else 0.0
        if independent_traits and Bnumbers is not None and len(Bnumbers):
            mean_B = float(np.mean(Bnumbers))
        else:
            mean_B = float(np.mean(1.0 - Anumbers)) if len(Anumbers) else 0.0
        mean_E = float(np.mean(storedEnergy)) if len(storedEnergy) else 0.0
        print(
            f"Simulation: Completed {number_gen} generations. "
            f"Final pop: {len(Anumbers)} | Mean A: {mean_A:.4f} | Mean B: {mean_B:.4f} | Mean Energy: {mean_E:.4f}",
            flush=True,
        )

    return {
        "A_history": A_history,
        "B_history": B_history,
        "T_history": T_history,
        "number_gen": number_gen,
        "independent_traits": independent_traits,
        "mutation_rate": float(mutation_rate),
        "mutation_scale": float(mutation_scale),
        "initial_A": initial_A_values.copy(),
        "metabs_history": metabs_history,
        "energy_history": energy_history,
        "storage_history": storage_history,
        "storage_M1_history": storage_M1_history,
        "task1_performance": task1_performance_history,
        "task2_performance": task2_performance_history,
        "M1_export_history": M1_export_history,
        "M1_import_history": M1_import_history,
        "M2_export_history": M2_export_history,
        "M2_import_history": M2_import_history,
        "change_history": change_history,
        "collapsed": False,
        "investment_modifier": float(inv_mod),
        "final_metabolite_environment": [float(metab1env), float(metab2env)],
        "metabolite_environment_after_inflow_final_generation": metabolite_environment_after_inflow_final_generation,
        "enable_m1_diffusion": bool(enable_m1_diffusion),
        "enable_m2_diffusion": bool(enable_m2_diffusion),
        "enable_intermediate_costs": bool(enable_intermediate_costs),
    }


def run_simulation_wrapper(params: dict) -> dict:
    """
    Wrapper matching the optimization GUI interface: accepts GUI-style params dict.
    Only numeric params are exposed by default; categorical options are fixed here.
    """
    seed = params.get("Random Seed (optional)")
    enable_m1_diffusion = _coerce_bool_param(params, "Enable M1 Diffusion", False)
    enable_m2_diffusion = _coerce_bool_param(params, "Enable M2 Diffusion", True)
    enable_m1_facilitated_diffusion = _coerce_bool_param(params, "Enable M1 Facilitated Diffusion", False)
    m1_porin_simple_diffusion = _coerce_bool_param(
        params,
        "Enable M1 Porin Diffusion",
        False,
    )
    if enable_m1_facilitated_diffusion or m1_porin_simple_diffusion:
        enable_m1_diffusion = True
    allow_diffusion_mutation = enable_m2_diffusion or enable_m1_diffusion
    enable_diffusion_mutation = _coerce_bool_param(params, "Enable Diffusion Mutation", False) if allow_diffusion_mutation else False
    homogeneous_initial_diffusion_const = _coerce_bool_param(params, "Homogeneous Initial Diffusion Const.", False)
    enable_chemostat_flow = _coerce_bool_param(params, "Enable Chemostat Flow", False)
    enable_acetate_addition = _coerce_bool_param(params, "Enable Acetate Addition", False)
    enable_intermediate_costs = _coerce_bool_param(params, "Enable Intermediate Costs", False) if enable_m2_diffusion else False
    minimal_tracking = _coerce_bool_param(params, "minimal_tracking", False)
    tracking_metric_name = normalize_metric_name(str(params.get("tracking_metric_name", "") or ""))
    raw_tmns = params.get("tracking_metric_names")
    tracking_metric_names_arg: Optional[Tuple[str, ...]] = None
    if isinstance(raw_tmns, (list, tuple)) and raw_tmns:
        tracking_metric_names_arg = tuple(
            normalize_metric_name(str(x or "")) for x in raw_tmns if normalize_metric_name(str(x or ""))
        )
        if not tracking_metric_names_arg:
            tracking_metric_names_arg = None
    # Required shared params
    # Keep Cost of Life strictly positive (model invariant).
    # Use the model default when missing and clamp malformed/non-positive inputs.
    try:
        cost_of_life = float(params.get("Cost of Life", 0.1))
    except Exception:
        cost_of_life = 0.1
    if cost_of_life <= 0.0:
        cost_of_life = 0.001
    mutation_rate = float(params.get("Mutation Rate", 0.01))
    mutation_scale = float(params.get("Mutation Scale", 0.1))
    number_gen = int(params.get("Number of Generations", 1000))
    organism_count = int(params.get("Initial Organism Count", 100))
    chemostat_volume = float(params.get("Chemostat Volume", 10000.0))
    flow_percentage = float(params.get("Flow Percentage", 0.0))

    # Model-specific numeric params
    sin_average = float(params.get("Average In_Flow", 100.0))
    acetate_average = float(params.get("Average In_Flow (Acetate)", 0.0))
    inv_mod = float(params.get("Investment Modifier", 1.0))
    dupx0 = float(params.get("Duplication Sigmoid Midpoint", 2.5))
    dupk = float(params.get("Duplication Sigmoid Intensity", 5.0))
    diffusion_const = float(params.get("Diffusion Constant", 0.01))
    acetate_ratio = float(params.get("Acetate Ratio", 1.0))
    death_decay_rate = float(params.get("Death Decay Rate", 10.0))
    binary_death_at_zero_energy = _coerce_bool_param(params, "Binary Death at Zero Energy", False)
    no_death = _coerce_bool_param(params, "No Death", False)
    constant_death_probability = _coerce_bool_param(params, "Constant Death Probability", False)
    constant_duplication_probability = _coerce_bool_param(params, "Constant Duplication Probability", False)
    constant_probability = resolve_constant_probability(params)
    if constant_probability is None:
        constant_probability = 0.5
    independent_traits = _coerce_bool_param(params, "Independent Traits", False)
    homogeneous_population = _coerce_bool_param(params, "Homogeneous Population", False)
    initial_A = float(params.get("Initial A", 0.5))
    initial_B = float(params.get("Initial B", 0.5))
    cost_of_transport = float(params.get("Cost of Transport", 0.0))
    initial_facilitation = float(params.get("Initial Facilitation", 0.5))
    silent = _coerce_bool_param(params, "silent", _coerce_bool_param(params, "Silent Mode", True))
    store_history = _coerce_bool_param(params, "store_history", True)
    keep_optional_final_arrays_arg: Optional[bool] = None
    if "keep_optional_final_arrays" in params:
        keep_optional_final_arrays_arg = _coerce_bool_param(
            params, "keep_optional_final_arrays", False
        )
    enable_initial_energy = _coerce_bool_param(params, "Enable Initial Energy", False)
    initial_energy = float(params.get("Initial Energy", 0.0))
    intermediate_costs = float(params.get("Intermediate Costs", 0.0))

    result = run_simulation(
        cost_of_life=cost_of_life,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        number_gen=number_gen,
        organism_count=organism_count,
        chemostat_volume=chemostat_volume,
        acetate_average=(acetate_average if enable_acetate_addition else 0.0),
        enable_chemostat_flow=enable_chemostat_flow,
        flow_percentage=flow_percentage,
        enable_acetate_addition=enable_acetate_addition,
        minimal_tracking=minimal_tracking,
        tracking_metric_name=tracking_metric_name,
        tracking_metric_names=tracking_metric_names_arg,
        keep_optional_final_arrays=keep_optional_final_arrays_arg,
        sin_average=sin_average,
        inv_mod=inv_mod,
        dupx0=dupx0,
        dupk=dupk,
        diffusion_const=diffusion_const,
        acetate_ratio=acetate_ratio,
        death_decay_rate=death_decay_rate,
        binary_death_at_zero_energy=binary_death_at_zero_energy,
        no_death=no_death,
        constant_death_probability=constant_death_probability,
        constant_duplication_probability=constant_duplication_probability,
        constant_probability=constant_probability,
        independent_traits=independent_traits,
        homogeneous_population=homogeneous_population,
        initial_A=initial_A,
        initial_B=initial_B,
        enable_m1_facilitated_diffusion=enable_m1_facilitated_diffusion,
        cost_of_transport=cost_of_transport,
        initial_facilitation=initial_facilitation,
        silent=silent,
        store_history=store_history,
        enable_m1_diffusion=enable_m1_diffusion,
        m1_porin_simple_diffusion=(
            m1_porin_simple_diffusion if (enable_m1_diffusion or enable_m1_facilitated_diffusion or m1_porin_simple_diffusion) else False
        ),
        enable_m2_diffusion=enable_m2_diffusion,
        enable_diffusion_mutation=enable_diffusion_mutation,
        homogeneous_initial_diffusion_const=homogeneous_initial_diffusion_const,
        enable_initial_energy=enable_initial_energy,
        initial_energy=initial_energy,
        enable_intermediate_costs=enable_intermediate_costs,
        intermediate_costs=(intermediate_costs if enable_intermediate_costs else 0.0),
        seed=seed,
    )
    return result