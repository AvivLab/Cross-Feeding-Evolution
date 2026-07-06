from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from gui.metrics import (
    SIMULATION_METRIC_NAMES,
    compute_metric_from_simulation_result,
)


@dataclass(frozen=True)
class OptimizationModelSpec:
    """
    Defines how Gradient Descent should interact with a particular model:
    - which parameters exist (default_params)
    - which metrics can be optimized (metric_names)
    - how to run the model (run_simulation)
    - how to compute a scalar metric from a model result (compute_metric)
    - how to detect a failed/collapsed run (is_failed)
    """

    key: str
    label: str
    default_params: Dict[str, Any]
    metric_names: List[str]
    run_simulation: Callable[[Dict[str, Any]], Dict[str, Any]]
    compute_metric: Callable[[Dict[str, Any], str], float]
    is_failed: Callable[[Optional[Dict[str, Any]]], bool]
    homogeneous_param_names: List[str] = None


def get_model_registry() -> List[OptimizationModelSpec]:
    """Return the list of available model specs for optimization."""
    from simulation.core import run_simulation_wrapper

    simulation_defaults = {
        # IMPORTANT: Cost of Life must be > 0 for this model, otherwise energy only increases
        # and the population can explode via duplication.
        "Independent Traits": False,
        "Cost of Life": 0.1,
        "Mutation Rate": 0.01,
        "Mutation Scale": 0.1,
        "Number of Generations": 1000,
        "Initial Organism Count": 100,
        "Chemostat Volume": 10000.0,
        "Flow Percentage": 0.0,
        "Average In_Flow": 100.0,
        "Average In_Flow (Acetate)": 0.0,
        "Investment Modifier": 1.0,
        "Duplication Sigmoid Midpoint": 2.5,
        "Duplication Sigmoid Intensity": 5.0,
        "Constant Probability": 0.5,
        "Diffusion Constant": 0.01,
        "Acetate Ratio": 1.0,
        "Death Decay Rate": 10.0,
        "Cost of Transport": 0.0,
        "Initial Energy": 0.0,
        "Intermediate Costs": 0.0,
        "Initial A": 0.5,
        "Initial B": 0.5,
        "Initial Facilitation": 0.5,
    }

    def simulation_is_failed(result: Optional[Dict[str, Any]]) -> bool:
        return (result is None) or bool(result.get("collapsed", False))

    return [
        OptimizationModelSpec(
            key="simulation",
            label="Simulation",
            default_params=simulation_defaults,
            metric_names=list(SIMULATION_METRIC_NAMES),
            run_simulation=run_simulation_wrapper,
            compute_metric=compute_metric_from_simulation_result,
            is_failed=simulation_is_failed,
            homogeneous_param_names=["Initial A", "Initial B", "Initial Facilitation"],
        ),
    ]


def get_model_by_key(key: str) -> OptimizationModelSpec:
    """Return model spec by key, falling back to the first registered model."""
    for spec in get_model_registry():
        if spec.key == key:
            return spec
    return get_model_registry()[0]
