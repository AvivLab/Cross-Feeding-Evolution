"""
Monte Carlo helpers: count random draws meeting one or more metric thresholds (AND),
for a parameter box vs "Neutral Sets" (other boxes).

Used by gui.apps.neutral_comparison.gui (standalone) and kept here as shared logic.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gui.metrics import metric_compute_cost_rank, metric_supports_seed_sweep_light_mode, normalize_metric_name
from gui.apps.neutral_comparison.offload import NeutralComparisonOffloadWriter
from gui.common.simulation_settings import RANDOM_SEED_OPTIONAL, normalize_simulation_params


def _parse_bounds_map(raw: Any, required_keys: List[str]) -> Dict[str, Tuple[float, float]]:
    if not isinstance(raw, dict):
        raise ValueError("Bounds must be a JSON object mapping parameter names to [min, max].")
    out: Dict[str, Tuple[float, float]] = {}
    for k in required_keys:
        if k not in raw:
            raise ValueError(f"Missing bound for parameter {k!r}.")
        pair = raw[k]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Bounds for {k!r} must be a length-2 [min, max].")
        lo, hi = float(pair[0]), float(pair[1])
        if not (np.isfinite(lo) or lo == float("-inf")):
            raise ValueError(f"Invalid min for {k!r}: {pair[0]!r}")
        if not np.isfinite(hi):
            raise ValueError(f"Invalid max for {k!r}: {pair[1]!r}")
        if lo != float("-inf") and lo >= hi:
            raise ValueError(f"min must be < max for {k!r} (got {lo}, {hi}).")
        out[k] = (lo, hi)
    return out


def _draw_within_bounds(
    bounds: Dict[str, Tuple[float, float]],
    rng: np.random.Generator,
) -> Dict[str, float]:
    draw: Dict[str, float] = {}
    for name, (lo, hi) in bounds.items():
        if lo == float("-inf"):
            draw[name] = float(rng.uniform(0.0, hi))
        else:
            draw[name] = float(rng.uniform(lo, hi))
    return draw


def _merge_params(
    numeric_base: Dict[str, Any],
    toggles: Dict[str, Any],
    draw: Dict[str, float],
    *,
    simulation_light_batch: bool = False,
    tracking_metric_names_union: Optional[Tuple[str, ...]] = None,
    keep_optional_final_arrays: Optional[bool] = None,
) -> Dict[str, Any]:
    out = dict(numeric_base)
    out.update(toggles)
    out.update(draw)
    if simulation_light_batch and tracking_metric_names_union:
        # Same contract as seed-sweep reruns: final-generation payloads plus targeted histories
        # (see simulation.core minimal_payload_mode + tracking_metric_names OR-merge).
        out["store_history"] = False
        out["minimal_tracking"] = True
        out["tracking_metric_names"] = list(tracking_metric_names_union)
        out.pop("tracking_metric_name", None)
    else:
        out["store_history"] = True
        out["minimal_tracking"] = False
        out.pop("tracking_metric_names", None)
        out.pop("tracking_metric_name", None)
    if keep_optional_final_arrays is not None:
        out["keep_optional_final_arrays"] = bool(keep_optional_final_arrays)
    out["silent"] = True
    return normalize_simulation_params(out)


def _metric_passes(value: float, op: str, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    op = str(op or ">").strip()
    if op == ">":
        return float(value) > float(threshold)
    if op == ">=":
        return float(value) >= float(threshold)
    if op == "<":
        return float(value) < float(threshold)
    if op == "<=":
        return float(value) <= float(threshold)
    raise ValueError(f"Unsupported operator {op!r} (use >, >=, <, <=).")


def simulation_light_tracking_plan(
    model_spec,
    metric_checks: List[Tuple[str, str, float]],
) -> Tuple[bool, Tuple[str, ...]]:
    """
    Whether neutral / Monte Carlo hit-count batches can use simulation light tracking, and the
    canonical metric-name tuple passed through to ``simulation.core`` (OR-merge of retention).
    """
    canon = tuple(dict.fromkeys(normalize_metric_name(str(m[0])) for m in metric_checks))
    use = getattr(model_spec, "key", None) == "simulation" and all(
        metric_supports_seed_sweep_light_mode(m) for m in canon
    )
    return bool(use), canon


def _metric_checks_ordered_cheapest_first_many(
    raw: List[Tuple[str, str, float]],
) -> List[Tuple[str, str, float]]:
    """(canonical name, op, threshold) pairs sorted by estimated compute cost, then name (stable)."""
    checks = [
        (normalize_metric_name(str(m)), str(op or ">").strip(), float(thr))
        for m, op, thr in raw
    ]
    checks.sort(key=lambda t: (metric_compute_cost_rank(t[0]), t[0]))
    return checks


def simulation_seed_for_run(base_seed: Optional[int], run_index: int) -> Optional[int]:
    """Deterministic per-run simulation seed derived from the batch ``base_seed``."""
    if base_seed is None:
        return None
    return int(base_seed) + 19 + int(run_index) * 7919


_simulation_seed_for_run = simulation_seed_for_run


def run_hit_count_batch(
    *,
    model_spec,
    n_runs: int,
    bounds: Dict[str, Tuple[float, float]],
    numeric_base: Dict[str, Any],
    toggles: Dict[str, Any],
    metric_checks: List[Tuple[str, str, float]],
    base_seed: Optional[int],
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    resume_event: Optional[threading.Event] = None,
    offload_writer: Optional[NeutralComparisonOffloadWriter] = None,
    offload_stage: str = "primary",
) -> int:
    """
    Optional progress_callback(done_sims, n_runs, hits_so_far) throttled to avoid excessive calls
    (roughly up to ~200 updates per batch plus a final call).

    Optional resume_event: while cleared, the batch blocks between simulations (pause); when set,
    work proceeds (resume). If omitted, runs without pause checks.

    Optional offload_writer: when set, each simulation is appended to Full Save-style offload batches
    under the writer's folder (see ``NeutralComparisonOffloadWriter``). offload_stage labels records
    (e.g. ``primary``, ``neutral_0``).

    For the Simulation model, when every selected metric supports seed-sweep light tracking, runs use
    ``store_history=False`` and ``minimal_tracking=True`` with a merged ``tracking_metric_names`` list
    so ``simulation.core`` retains only what the metric AND-chain needs.

    When ``base_seed`` is set, each run also receives a deterministic ``Random Seed (optional)``
    derived from ``base_seed + 19 + run_index * 7919``.
    """
    if not metric_checks:
        raise ValueError("metric_checks must be a non-empty list of (metric_name, op, threshold).")
    rng = np.random.default_rng(base_seed)
    hits = 0
    ordered = _metric_checks_ordered_cheapest_first_many(metric_checks)
    primary_metric = normalize_metric_name(str(metric_checks[0][0]))
    use_sim_light, canon_for_light = simulation_light_tracking_plan(model_spec, metric_checks)
    tracking_union: Optional[Tuple[str, ...]] = canon_for_light if use_sim_light else None
    n = int(n_runs)
    # Throttle progress callbacks to ~200 updates per batch (plus a final call).
    stride = max(1, min(500, n // 200)) if n > 0 else 1

    def _maybe_report(i1: int) -> None:
        if progress_callback is None:
            return
        if i1 >= n or i1 % stride == 0:
            progress_callback(i1, n, hits)

    for _i in range(n):
        if resume_event is not None:
            while not resume_event.is_set():
                time.sleep(0.05)
        draw = _draw_within_bounds(bounds, rng)
        params = _merge_params(
            numeric_base,
            toggles,
            draw,
            simulation_light_batch=bool(use_sim_light),
            tracking_metric_names_union=tracking_union,
            keep_optional_final_arrays=True if offload_writer is not None else None,
        )
        sim_seed = simulation_seed_for_run(base_seed, _i)
        if sim_seed is not None:
            params[RANDOM_SEED_OPTIONAL] = int(sim_seed)
        try:
            res = model_spec.run_simulation(params)
        except Exception:
            res = None
        seed_used = params.get(RANDOM_SEED_OPTIONAL) if isinstance(params, dict) else None
        evaluated_metrics: Dict[str, float] = {}
        hit = False
        if res is None or model_spec.is_failed(res):
            _maybe_report(_i + 1)
            if offload_writer is not None:
                offload_writer.add_simulation_record(
                    merged_params=params,
                    result_obj=res if isinstance(res, dict) else None,
                    hit=False,
                    evaluated_metrics={},
                    primary_metric_value=float("nan"),
                    offload_stage=offload_stage,
                    seed_used=seed_used,
                )
            continue
        # Metric filters are AND-ed; cheapest metrics are evaluated first to fail fast.
        ok = True
        for mname, op, thr in ordered:
            try:
                v = float(model_spec.compute_metric(res, mname))
            except Exception:
                v = float("nan")
            evaluated_metrics[str(mname)] = v
            if not _metric_passes(v, op, thr):
                ok = False
                break
        if primary_metric not in evaluated_metrics:
            try:
                evaluated_metrics[str(primary_metric)] = float(model_spec.compute_metric(res, primary_metric))
            except Exception:
                evaluated_metrics[str(primary_metric)] = float("nan")
        if ok:
            hits += 1
            hit = True
        primary_mv = float(evaluated_metrics.get(str(primary_metric), float("nan")))
        if offload_writer is not None:
            offload_writer.add_simulation_record(
                merged_params=params,
                result_obj=res,
                hit=hit,
                evaluated_metrics=dict(evaluated_metrics),
                primary_metric_value=primary_mv,
                offload_stage=offload_stage,
                seed_used=seed_used,
            )
        _maybe_report(_i + 1)
    if progress_callback is not None and n > 0:
        progress_callback(n, n, hits)
    return int(hits)
