"""
Centralized metric calculations for heatmaps across all GUIs.

**Read first:** ``gui.metrics.definitions`` — catalog of every metric name, formula, range,
and tooltip.

**Implementation:**
- ``gui.metrics.calculators`` — direct formulas and ``calculate_sweep_metric`` dispatch
- ``gui.metrics.neutral_replay`` — neutral drift replay and percentile Monte Carlo

Import from this package: ``from gui.metrics import …``
"""

from gui.metrics.calculators import (
    SweepMetricInputs,
    calculate_sweep_metric,
    compute_metric_from_simulation_result,
    filter_metric_options_for_simulation_settings,
    t2_heavy_pool_model_result_compatible,
    t2_percent_of_total_task2_from_t1_t2_ratio_subset,
)
from gui.metrics.definitions import (
    NEUTRAL_PERCENTILE_MAX_SIMS,
    NEUTRAL_PERCENTILE_MIN_SIMS,
    NEUTRAL_PERCENTILE_TARGET_CI_PCT,
    SIMULATION_METRIC_NAMES,
    T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS,
    T2_HEAVY_POOL_MODEL_METRICS,
    TASK_RATIO_BINS,
    TASK_RATIO_BINS_FINE,
    TRAIT_ENTROPY_BINS,
    MetricDefinition,
    describe_metric,
    format_metric_math_reference,
    get_metric_definition,
    metric_compute_cost_rank,
    metric_supports_seed_sweep_light_mode,
    normalize_metric_name,
)

__all__ = [
    "MetricDefinition",
    "NEUTRAL_PERCENTILE_MAX_SIMS",
    "NEUTRAL_PERCENTILE_MIN_SIMS",
    "NEUTRAL_PERCENTILE_TARGET_CI_PCT",
    "SIMULATION_METRIC_NAMES",
    "SweepMetricInputs",
    "T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS",
    "T2_HEAVY_POOL_MODEL_METRICS",
    "TASK_RATIO_BINS",
    "TASK_RATIO_BINS_FINE",
    "TRAIT_ENTROPY_BINS",
    "calculate_sweep_metric",
    "compute_metric_from_simulation_result",
    "describe_metric",
    "filter_metric_options_for_simulation_settings",
    "format_metric_math_reference",
    "get_metric_definition",
    "metric_compute_cost_rank",
    "metric_supports_seed_sweep_light_mode",
    "normalize_metric_name",
    "t2_heavy_pool_model_result_compatible",
    "t2_percent_of_total_task2_from_t1_t2_ratio_subset",
]
