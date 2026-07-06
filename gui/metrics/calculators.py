"""Per-metric calculators and dispatch for ``calculate_sweep_metric``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from gui.metrics.definitions import (
    TASK_RATIO_BINS,
    TASK_RATIO_BINS_FINE,
    T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS,
    T2_HEAVY_POOL_MODEL_METRICS,
    normalize_metric_name,
)
from gui.metrics.neutral_replay import (
    neutral_percentile_pair,
    neutral_percentile_t2heavy_task2_share,
    parse_neutral_events,
    shannon_entropy_from_traits,
)

_EPS = 1e-10
_EPS_T2 = 1e-12


def _any_none(*values) -> bool:
    """True if any argument is None (safe with numpy arrays; unlike ``None in tuple``)."""
    return any(v is None for v in values)


@dataclass
class SweepMetricInputs:
    """Fields used by active simulation metrics (see ``definitions._METRIC_CATALOG``)."""

    enzyme_A_final: Any
    task1_final: Any
    task2_final: Any
    transport_final: Any
    metabs_history: Optional[List] = None
    chemostat_volume: Optional[float] = None
    cell_volume: Optional[float] = None
    independent_traits: Optional[bool] = None
    change_history: Optional[List] = None
    mutation_rate: Optional[float] = None
    mutation_scale: Optional[float] = None
    initial_traits: Any = None
    investment_modifier: Optional[float] = None
    metabolite_environment_after_inflow_final_generation: Any = None


def t2_heavy_pool_model_result_compatible(result) -> bool:
    if not isinstance(result, dict) or bool(result.get("collapsed", False)):
        return False
    req = ("enable_m1_diffusion", "enable_m2_diffusion", "enable_intermediate_costs")
    if not all(k in result for k in req):
        return False
    return (
        not bool(result["enable_m1_diffusion"])
        and not bool(result["enable_m2_diffusion"])
        and not bool(result["enable_intermediate_costs"])
    )


def t2_percent_of_total_task2_from_t1_t2_ratio_subset(task1_final, task2_final) -> float:
    t1 = np.maximum(np.asarray(task1_final, dtype=float), 0.0)
    t2 = np.maximum(np.asarray(task2_final, dtype=float), 0.0)
    n = int(min(t1.size, t2.size))
    if n <= 0:
        return float("nan")
    t1, t2 = t1[:n], t2[:n]
    x = float(T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS)
    if not np.isfinite(x) or x <= 0.0:
        return float("nan")
    total_t2 = float(np.sum(t2))
    if total_t2 <= _EPS_T2:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(t2 > _EPS_T2, t1 / t2, np.inf)
    mask = (t2 > _EPS_T2) & np.isfinite(ratio) & (ratio < x)
    return float(100.0 * float(np.sum(t2[mask])) / total_t2)


def exchanged_m2_percent_from_t1_t2(task1_final, task2_final) -> float:
    """100 × Σ max(0, task2−task1) / Σ task2 on final-generation task arrays."""
    t1 = np.maximum(np.asarray(task1_final, dtype=float), 0.0)
    t2 = np.maximum(np.asarray(task2_final, dtype=float), 0.0)
    n = int(min(t1.size, t2.size))
    if n <= 0:
        return float("nan")
    t1, t2 = t1[:n], t2[:n]
    total_t2 = float(np.sum(t2))
    if total_t2 <= _EPS_T2:
        return float("nan")
    excess = np.maximum(0.0, t2 - t1)
    return float(100.0 * float(np.sum(excess)) / total_t2)


def _pop_size(inp: SweepMetricInputs) -> int:
    return len(inp.enzyme_A_final) if inp.enzyme_A_final is not None else 0


def calc_final_population_size(inp: SweepMetricInputs) -> float:
    return float(_pop_size(inp))


def calc_task2_share_weighted_prob_mean(inp: SweepMetricInputs, *, fine_bins: bool) -> float:
    if inp.task1_final is None or inp.task2_final is None:
        return np.nan
    task1 = np.asarray(inp.task1_final, dtype=float)
    task2 = np.asarray(inp.task2_final, dtype=float)
    n = int(task1.size)
    if n == 0 or task2.size != n:
        return np.nan
    denom = task1 + task2
    valid = np.isfinite(task1) & np.isfinite(task2) & np.isfinite(denom) & (denom > _EPS)
    shares = task2[valid] / denom[valid]
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return 0.0
    r_min, r_max = float(np.min(shares)), float(np.max(shares))
    if r_max <= r_min:
        p_single = float(shares.size) / float(n)
        return float((1.0 / float(n)) * np.sum(shares * p_single))
    bins = TASK_RATIO_BINS_FINE if fine_bins else TASK_RATIO_BINS
    counts, edges = np.histogram(shares, bins=bins, range=(r_min, r_max))
    probs = counts.astype(float) / float(n)
    bin_idx = np.clip(np.searchsorted(edges, shares, side="right") - 1, 0, len(probs) - 1)
    return float((1.0 / float(n)) * np.sum(shares * probs[bin_idx]))


def calc_task_specialization_index(inp: SweepMetricInputs) -> float:
    if inp.task1_final is None or inp.task2_final is None:
        return np.nan
    task1 = np.asarray(inp.task1_final, dtype=float)
    task2 = np.asarray(inp.task2_final, dtype=float)
    if task1.size == 0 or task2.size != task1.size:
        return np.nan
    denom = task1 + task2
    valid = np.isfinite(task1) & np.isfinite(task2) & np.isfinite(denom) & (denom > _EPS)
    if not np.any(valid):
        return 0.0
    shares = task2[valid] / denom[valid]
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return 0.0
    return float(2.0 * np.mean(np.abs(shares - 0.5)))


def calc_trait_std_dev_coupled(inp: SweepMetricInputs) -> float:
    if inp.independent_traits is None or bool(inp.independent_traits):
        return np.nan
    return float(np.std(np.asarray(inp.enzyme_A_final, dtype=float)))


def calc_trait_shannon_entropy_final(inp: SweepMetricInputs) -> float:
    return shannon_entropy_from_traits(np.asarray(inp.enzyme_A_final, dtype=float))


def calc_trait_std_dev_neutral_perc(inp: SweepMetricInputs) -> float:
    if inp.independent_traits is None or bool(inp.independent_traits):
        return np.nan
    if _any_none(inp.change_history, inp.mutation_rate, inp.mutation_scale, inp.initial_traits):
        return np.nan
    events = parse_neutral_events(inp.change_history)
    std_p, _ = neutral_percentile_pair(
        inp.enzyme_A_final,
        inp.initial_traits,
        events,
        inp.mutation_rate,
        inp.mutation_scale,
        events_cache_key=id(inp.change_history),
        which="std",
    )
    return float(std_p)


def calc_trait_entropy_neutral_perc(inp: SweepMetricInputs) -> float:
    if inp.independent_traits is None or bool(inp.independent_traits):
        return np.nan
    if _any_none(inp.change_history, inp.mutation_rate, inp.mutation_scale, inp.initial_traits):
        return np.nan
    events = parse_neutral_events(inp.change_history)
    _, ent_p = neutral_percentile_pair(
        inp.enzyme_A_final,
        inp.initial_traits,
        events,
        inp.mutation_rate,
        inp.mutation_scale,
        events_cache_key=id(inp.change_history),
        which="entropy",
    )
    return float(ent_p)


def _parse_metab_after_inflow(mpair) -> Optional[tuple]:
    if mpair is None:
        return None
    if not isinstance(mpair, (list, tuple)) or len(mpair) < 2:
        return None
    try:
        m1e, m2e = float(mpair[0]), float(mpair[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(m1e) or not np.isfinite(m2e):
        return None
    return m1e, m2e


def _neutral_perc_task2_pool_metric(
    inp: SweepMetricInputs,
    *,
    t2_percent_fn,
    error_label: str,
) -> float:
    if bool(inp.independent_traits):
        return np.nan
    if _any_none(
        inp.change_history,
        inp.mutation_rate,
        inp.mutation_scale,
        inp.initial_traits,
        inp.task1_final,
        inp.task2_final,
    ):
        return np.nan
    mpair = _parse_metab_after_inflow(inp.metabolite_environment_after_inflow_final_generation)
    if mpair is None:
        print(
            f"[metrics] ERROR: {error_label} — metabolite_environment_after_inflow_final_generation "
            "missing or invalid. Returning NaN.",
            file=sys.stderr,
            flush=True,
        )
        return np.nan
    m1e, m2e = mpair
    try:
        inv = float(inp.investment_modifier) if inp.investment_modifier is not None else float("nan")
    except Exception:
        inv = float("nan")
    if not np.isfinite(inv):
        return np.nan
    events = parse_neutral_events(inp.change_history)
    return float(
        neutral_percentile_t2heavy_task2_share(
            inp.enzyme_A_final,
            inp.task1_final,
            inp.task2_final,
            inp.initial_traits,
            events,
            inp.mutation_rate,
            inp.mutation_scale,
            m1e,
            m2e,
            inv,
            events_cache_key=id(inp.change_history),
            t2_percent_fn=t2_percent_fn,
        )
    )


def calc_crossfeeding_acetate_neutral_perc(inp: SweepMetricInputs) -> float:
    return _neutral_perc_task2_pool_metric(
        inp,
        t2_percent_fn=t2_percent_of_total_task2_from_t1_t2_ratio_subset,
        error_label="Crossfeeding % of Acetate (neutral perc.)",
    )


def calc_exchanged_m2_percent_neutral_perc(inp: SweepMetricInputs) -> float:
    return _neutral_perc_task2_pool_metric(
        inp,
        t2_percent_fn=exchanged_m2_percent_from_t1_t2,
        error_label="Exchanged M2 Percent (neutral perc.)",
    )


def calc_crossfeeding_acetate(inp: SweepMetricInputs) -> float:
    if inp.task1_final is None or inp.task2_final is None:
        return np.nan
    t1 = np.asarray(inp.task1_final, dtype=float)
    t2 = np.asarray(inp.task2_final, dtype=float)
    if t1.size == 0 or t2.size == 0:
        return np.nan
    return t2_percent_of_total_task2_from_t1_t2_ratio_subset(t1, t2)


def calc_exchanged_m2_percent(inp: SweepMetricInputs) -> float:
    """Share of total task-2 work that exceeds own task-1 (pooled cross-feeding proxy)."""
    if inp.task1_final is None or inp.task2_final is None:
        return np.nan
    return exchanged_m2_percent_from_t1_t2(inp.task1_final, inp.task2_final)


_METRIC_CALCULATORS: Dict[str, Callable[[SweepMetricInputs], float]] = {
    "Final Population Size": calc_final_population_size,
    "Task2 Share Weighted Prob. Mean": lambda inp: calc_task2_share_weighted_prob_mean(inp, fine_bins=False),
    "Task2 Share Weighted Prob. Mean (100 bins)": lambda inp: calc_task2_share_weighted_prob_mean(inp, fine_bins=True),
    "Task Specialization Index (|share-0.5|)": calc_task_specialization_index,
    "Trait Std Dev (Coupled)": calc_trait_std_dev_coupled,
    "Trait Shannon Entropy (Final)": calc_trait_shannon_entropy_final,
    "Trait Std Dev (Neutral Perc.)": calc_trait_std_dev_neutral_perc,
    "Trait Entropy (Neutral Perc.)": calc_trait_entropy_neutral_perc,
    "Crossfeeding % of Acetate (neutral perc.)": calc_crossfeeding_acetate_neutral_perc,
    "Crossfeeding % of Acetate": calc_crossfeeding_acetate,
    "Exchanged M2 Percent": calc_exchanged_m2_percent,
    "Exchanged M2 Percent (neutral perc.)": calc_exchanged_m2_percent_neutral_perc,
}


def _last_nonempty_generation_array(history) -> Optional[np.ndarray]:
    """Return the last non-None generation array from a simulation history list."""
    if not history:
        return None
    last = history[-1]
    if last is not None:
        return np.asarray(last, dtype=float)
    for value in reversed(history):
        if value is not None:
            return np.asarray(value, dtype=float)
    return None


def _log_pool_crossfeeding_metric_error(metric_name: str) -> None:
    print(
        f"[metrics] ERROR: {metric_name} applies only when M1 diffusion, M2 diffusion, "
        "and intermediate costs are all off (pooled-env regime), and requires enable_* "
        "flags on the simulation result. Returning NaN.",
        file=sys.stderr,
        flush=True,
    )


def compute_metric_from_simulation_result(result: Any, metric_name: str) -> float:
    """
    Compute a catalog metric from a ``run_simulation`` / ``run_simulation_wrapper`` result dict.

    Single entry point used by ``gui.models.registry`` and cached metric-input replay.
    """
    metric_key = normalize_metric_name(metric_name)
    if not isinstance(result, dict) or bool(result.get("collapsed", False)):
        return float("nan")

    a_last = _last_nonempty_generation_array(result.get("A_history"))
    if a_last is None or a_last.size == 0:
        return float("nan")

    if metric_key in T2_HEAVY_POOL_MODEL_METRICS and not t2_heavy_pool_model_result_compatible(result):
        _log_pool_crossfeeding_metric_error(metric_key)
        return float("nan")

    t1 = _last_nonempty_generation_array(result.get("task1_performance"))
    t2 = _last_nonempty_generation_array(result.get("task2_performance"))

    return calculate_sweep_metric(
        enzyme_A_final=a_last,
        task1_final=t1,
        task2_final=t2,
        metric=metric_key,
        metabs_history=result.get("metabs_history"),
        independent_traits=result.get("independent_traits"),
        change_history=result.get("change_history"),
        mutation_rate=result.get("mutation_rate"),
        mutation_scale=result.get("mutation_scale"),
        initial_traits=result.get("initial_A"),
        investment_modifier=result.get("investment_modifier"),
        metabolite_environment_after_inflow_final_generation=result.get(
            "metabolite_environment_after_inflow_final_generation"
        ),
    )


def calculate_sweep_metric(
    enzyme_A_final,
    enzyme_B_final=None,
    enzyme_T_final=None,
    task1_final=None,
    task2_final=None,
    transport_final=None,
    metric=None,
    metabs_history=None,
    storage_M2_final=None,
    chemostat_volume=None,
    cell_volume=None,
    independent_traits=None,
    change_history=None,
    mutation_rate=None,
    mutation_scale=None,
    initial_traits=None,
    investment_modifier=None,
    final_metabolite_environment=None,
    metabolite_environment_after_inflow_final_generation=None,
    **_,
) -> float:
    """
    Calculate a metric from final-generation simulation arrays.

    See ``gui.metrics.definitions`` for formulas. Unused keyword args (e.g. ``enzyme_B_final``,
    ``storage_M2_final``) are accepted for caller compatibility but ignored.
    """
    _ = (enzyme_B_final, enzyme_T_final, storage_M2_final, final_metabolite_environment)
    metric_key = normalize_metric_name(metric)
    if enzyme_A_final is None or len(enzyme_A_final) == 0:
        return np.nan

    inp = SweepMetricInputs(
        enzyme_A_final=enzyme_A_final,
        task1_final=task1_final,
        task2_final=task2_final,
        transport_final=transport_final,
        metabs_history=metabs_history,
        chemostat_volume=chemostat_volume,
        cell_volume=cell_volume,
        independent_traits=independent_traits,
        change_history=change_history,
        mutation_rate=mutation_rate,
        mutation_scale=mutation_scale,
        initial_traits=initial_traits,
        investment_modifier=investment_modifier,
        metabolite_environment_after_inflow_final_generation=metabolite_environment_after_inflow_final_generation,
    )
    fn = _METRIC_CALCULATORS.get(metric_key)
    if fn is None:
        return np.nan
    return float(fn(inp))


def filter_metric_options_for_simulation_settings(
    metric_names,
    *,
    enable_m1_diffusion=False,
    enable_m2_diffusion=False,
    enable_intermediate_costs=False,
):
    from gui.metrics.definitions import T2_HEAVY_POOL_MODEL_METRICS

    names = list(metric_names or [])
    pool_only = (not bool(enable_m1_diffusion)) and (not bool(enable_m2_diffusion)) and (
        not bool(enable_intermediate_costs)
    )
    if not pool_only:
        names = [n for n in names if normalize_metric_name(n) not in T2_HEAVY_POOL_MODEL_METRICS]
    return names
