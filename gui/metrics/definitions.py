"""
Canonical catalog of simulation metrics: names, formulas, tooltips, and registry flags.

Read this file first when adding or interpreting a metric. Implementation lives in
``gui.metrics.calculators`` (direct formulas) and ``gui.metrics.neutral_replay`` (Monte Carlo
neutral replays). The public import path is ``gui.metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Shared constants (referenced in formulas below)
# ---------------------------------------------------------------------------

NEUTRAL_PERCENTILE_MAX_SIMS = 60
NEUTRAL_PERCENTILE_MIN_SIMS = 20
NEUTRAL_PERCENTILE_TARGET_CI_PCT = 3.0

TRAIT_ENTROPY_BINS = 20
TASK_RATIO_BINS = 30
TASK_RATIO_BINS_FINE = 100

# Task1/Task2 < x defines the acetate-heavy subset (code constant, not a GUI knob).
T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS = 1.0

_NEUTRAL_PERC_MONTE_CARLO_BLURB = (
    f"Runs many neutral replays (up to {NEUTRAL_PERCENTILE_MAX_SIMS}, at least "
    f"{NEUTRAL_PERCENTILE_MIN_SIMS} before stopping early when the estimate is stable within "
    f"about {NEUTRAL_PERCENTILE_TARGET_CI_PCT}%)."
)


@dataclass(frozen=True)
class MetricDefinition:
    """One row in the metric catalog."""

    name: str
    math: str  # Plain-text formula summary
    range: str
    tooltip: str
    compute_cost: int = 500
    in_simulation_list: bool = True
    pool_only_regime: bool = False


def normalize_metric_name(metric_name: str) -> str:
    """Return stripped metric name (exact catalog names only; no aliases)."""
    return str(metric_name or "").strip()


# =============================================================================
# METRIC CATALOG (simulation metrics exposed in GUIs / HPC job JSON)
# =============================================================================

_METRIC_CATALOG: Tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="Final Population Size",
        math="Count how many organisms are alive at the last generation.",
        range="Zero or more (a whole number).",
        tooltip="Final number of organisms at the end of the simulation.",
        compute_cost=0,
    ),
    MetricDefinition(
        name="Task2 Share Weighted Prob. Mean",
        math=(
            "For each organism, task-2 share = task2 / (task1 + task2).\n"
            "Put all shares into 30 histogram bins between the smallest and largest share seen.\n"
            "The metric is the average share, but each organism is weighted by how crowded "
            "its bin is — not the same as a simple average of shares.\n"
            "Organisms with missing or invalid task data count as zero but still count toward N."
        ),
        range="Between 0 and 1.",
        tooltip=(
            "Histogram-weighted mean of per-organism task-2 share (30 bins)."
        ),
        compute_cost=60,
    ),
    MetricDefinition(
        name="Task2 Share Weighted Prob. Mean (100 bins)",
        math="Same as Task2 Share Weighted Prob. Mean, but uses 100 histogram bins instead of 30.",
        range="Between 0 and 1.",
        tooltip="Same as Task2 Share Weighted Prob. Mean, with 100 histogram bins.",
        compute_cost=66,
    ),
    MetricDefinition(
        name="Task Specialization Index (|share-0.5|)",
        math=(
            "For each organism, task-2 share = task2 / (task1 + task2).\n"
            "Measure how far each share is from 50/50, average that distance, then double it.\n"
            "0 means everyone is balanced between tasks; 1 means everyone is strongly one-sided."
        ),
        range="Between 0 and 1.",
        tooltip=(
            "How specialized the population is toward one task (0 = balanced, 1 = extreme)."
        ),
        compute_cost=56,
    ),
    MetricDefinition(
        name="Trait Std Dev (Coupled)",
        math=(
            "Standard deviation of trait A in the final population (coupled model only: B = 1 − A).\n"
            "Every trait lies between 0 and 1, so spread is always between 0 and 0.5.\n"
            "0 means everyone has the same trait; 0.5 is the most spread possible (half at 0, half at 1)."
        ),
        range="Between 0 and 0.5. Not defined for independent A/B traits. One organism gives 0.",
        tooltip=(
            "How spread out final A traits are on [0, 1] (coupled model only)."
        ),
        compute_cost=8,
    ),
    MetricDefinition(
        name="Trait Shannon Entropy (Final)",
        math=(
            "Split final trait A values into 20 bins from 0 to 1.\n"
            "Shannon entropy measures how evenly organisms are spread across those bins.\n"
            "Low = most organisms look alike; high = many different trait values."
        ),
        range="Zero or higher (natural-log units).",
        tooltip="Diversity of final A traits on a 0–1 histogram (20 bins).",
        compute_cost=72,
    ),
    MetricDefinition(
        name="Trait Std Dev (Neutral Perc.)",
        math=(
            "Compute trait spread (std dev of A) in the real run.\n"
            "Replay the same births, deaths, flow removals, and mutation counts each generation, "
            "but with random neutral trait dynamics, many times.\n"
            "Percentile = percent of replays with spread at or below the real run.\n"
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB}"
        ),
        range="0 to 100. Not defined for independent A/B or without change history.",
        tooltip=(
            "Percentile of real trait std dev under neutral replay schedule. "
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB} NaN if independent_traits."
        ),
        compute_cost=200,
    ),
    MetricDefinition(
        name="Trait Entropy (Neutral Perc.)",
        math=(
            "Compute trait diversity (Shannon entropy of A, 20 bins) in the real run.\n"
            "Same neutral replay idea as Trait Std Dev (Neutral Perc.): replay demographics and "
            "mutation counts, randomize traits each time.\n"
            "Percentile = percent of replays with diversity at or below the real run."
        ),
        range="0 to 100. Not defined for independent A/B or without change history.",
        tooltip=(
            "Percentile of real trait entropy under the same neutral replay schedule. "
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB} NaN if independent_traits."
        ),
        compute_cost=200,
    ),
    MetricDefinition(
        name="Crossfeeding % of Acetate",
        math=(
            "Add up all task-2 work in the final population.\n"
            "Among organisms doing task 2, take those that do relatively little task 1 compared "
            f"to task 2 (task1/task2 below {T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS}).\n"
            "Metric = 100 × (their task-2 work) / (total task-2 work).\n"
            "Only when nutrients are pooled (M1 diffusion, M2 diffusion, and intermediate costs all off)."
        ),
        range="0 to 100 percent. Not defined if total task-2 work is zero or settings disagree.",
        tooltip=(
            "Share of total Task-2 work from organisms with Task1/Task2 below the code cutoff. "
            "Only in pooled-nutrient regime (diffusion and intermediate costs off)."
        ),
        compute_cost=32,
        pool_only_regime=True,
    ),
    MetricDefinition(
        name="Exchanged M2 Percent",
        math=(
            "Per organism, excess task-2 = max(0, task2 − task1) (final-generation performance).\n"
            "Interpret excess as acetate uptake not covered by that organism's own task-1 production "
            "in the pooled pool.\n"
            "Metric = 100 × (sum of excess task-2) / (sum of all task-2).\n"
            "Only when M1 diffusion, M2 diffusion, and intermediate costs are all off."
        ),
        range="0 to 100 percent. Not defined if total task-2 work is zero or settings disagree.",
        tooltip=(
            "Fraction of total Task-2 work that exceeds own Task-1 (cross-fed acetate proxy). "
            "Pooled-nutrient regime only (all diffusion and intermediate costs off)."
        ),
        compute_cost=28,
        pool_only_regime=True,
    ),
    MetricDefinition(
        name="Exchanged M2 Percent (neutral perc.)",
        math=(
            "Compute Exchanged M2 Percent in the real run.\n"
            "Each neutral replay keeps the same trait history, then rebuilds task 1 and task 2 "
            "from nutrient pools after inflow, and applies the same deficit formula "
            "(100 × Σ max(0, task2−task1) / Σ task2).\n"
            "Percentile = percent of replays at or below the real run.\n"
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB}"
        ),
        range=(
            "0 to 100. Needs pooled-nutrient settings, coupled traits, and final M1/M2 snapshot."
        ),
        tooltip=(
            "Where the real Exchanged M2 Percent sits vs neutral trait replays (0 = most replays higher). "
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB} "
            "Needs metabolite_environment_after_inflow_final_generation."
        ),
        compute_cost=240,
        pool_only_regime=True,
    ),
    MetricDefinition(
        name="Crossfeeding % of Acetate (neutral perc.)",
        math=(
            "Compute Crossfeeding % of Acetate in the real run.\n"
            "Each neutral replay keeps the same trait history, then rebuilds task 1 and task 2 "
            "from nutrient pools after inflow, and applies the same acetate formula.\n"
            "Percentile = percent of replays at or below the real run.\n"
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB}"
        ),
        range=(
            "0 to 100. Needs pooled-nutrient settings, coupled traits, and final M1/M2 snapshot."
        ),
        tooltip=(
            "Where the real Crossfeeding % of Acetate sits vs neutral trait replays (0 = most replays higher). "
            f"{_NEUTRAL_PERC_MONTE_CARLO_BLURB} "
            "Needs metabolite_environment_after_inflow_final_generation."
        ),
        compute_cost=240,
        pool_only_regime=True,
    ),
)

_ALL_DEFINITIONS: Tuple[MetricDefinition, ...] = _METRIC_CATALOG

# ---------------------------------------------------------------------------
# Derived registries (single source of truth)
# ---------------------------------------------------------------------------

SIMULATION_METRIC_NAMES: List[str] = [
    d.name for d in _METRIC_CATALOG if d.in_simulation_list
]

_METRIC_COMPUTE_COST_RANK: Dict[str, int] = {
    d.name: d.compute_cost for d in _ALL_DEFINITIONS
}

T2_HEAVY_POOL_MODEL_METRICS: FrozenSet[str] = frozenset(
    d.name for d in _ALL_DEFINITIONS if d.pool_only_regime
)


def metric_compute_cost_rank(metric_name: str) -> int:
    return int(_METRIC_COMPUTE_COST_RANK.get(normalize_metric_name(metric_name), 500))


def metric_supports_seed_sweep_light_mode(metric_name: str) -> bool:
    return normalize_metric_name(metric_name) in set(SIMULATION_METRIC_NAMES)


def get_metric_definition(metric_name: str) -> Optional[MetricDefinition]:
    key = normalize_metric_name(metric_name)
    for d in _ALL_DEFINITIONS:
        if d.name == key:
            return d
    return None


def describe_metric(metric_name: str) -> str:
    """Tooltip text; falls back to generic message if name is not in the catalog."""
    key = normalize_metric_name(metric_name)
    for d in _ALL_DEFINITIONS:
        if d.name == key:
            return d.tooltip
    return f"{key}\n\nNo detailed description is registered yet for this metric."


def format_metric_math_reference(metric_name: str) -> str:
    """Full math + range block for documentation or debug UI."""
    key = normalize_metric_name(metric_name)
    d = get_metric_definition(key)
    if d is None:
        return f"{key}: (no definition registered)"
    return f"{d.name}\n  Math: {d.math}\n  Range: {d.range}"

