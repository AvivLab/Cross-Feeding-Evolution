"""Offload filename helpers and reconstruction of aligned datasets from batches."""

from __future__ import annotations

import math
import os
import re
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np


OFFLOAD_FILENAME_RE = re.compile(r"^offload_(?P<session>[^_]+_\d{6}|\d{8}-\d{6})_(?P<idx>\d+)\.json(?:\.gz)?$")


def parse_offload_filename(path_or_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse (session_id, batch_index) from an offload path or basename."""
    m = OFFLOAD_FILENAME_RE.match(os.path.basename(path_or_name))
    if not m:
        return None, None
    session = m.group("session")
    try:
        idx = int(m.group("idx"))
    except Exception:
        idx = None
    return session, idx


def list_offload_batch_files(
    folder: str,
    session_id: Optional[str] = None,
    *,
    sort_by_index: bool = False,
) -> List[str]:
    """List offload batch file paths, optionally filtered by session id."""
    if not folder or not os.path.isdir(folder):
        return []
    rows: List[Tuple[int, str]] = []
    for fn in os.listdir(folder):
        sess, idx = parse_offload_filename(fn)
        if not sess:
            continue
        if session_id and sess != session_id:
            continue
        order_idx = int(idx) if isinstance(idx, int) else 10**12
        rows.append((order_idx, os.path.join(folder, fn)))
    if sort_by_index:
        rows.sort(key=lambda x: (x[0], x[1]))
        return [p for _, p in rows]
    return sorted([p for _, p in rows])



def _safe_float_or_nan(value) -> float:
    try:
        fv = float(value)
        return np.nan if math.isnan(fv) else fv
    except Exception:
        return np.nan


def _coerce_vector(value) -> List[float]:
    vec = value
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    if not isinstance(vec, list):
        try:
            vec = list(vec)
        except Exception:
            vec = []
    try:
        return [float(v) for v in vec]
    except Exception:
        return []


def _coerce_optional_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def infer_primary_metric_name(batch_payload: Mapping[str, object], fallback: str) -> str:
    """Resolve primary metric name consistently across all offload consumers."""
    candidate = str(batch_payload.get("metric_name_current", "") or "").strip()
    if not candidate:
        candidate = str(batch_payload.get("metric_name_at_offload", "") or "").strip()
    return candidate or fallback


def reconstruct_from_offload_batches(
    *,
    batch_paths: List[str],
    read_payload: Callable[[str], Mapping[str, object]],
    optimization_goal: str = "Maximize",
    metric_name_fallback: str = "Unknown",
    include_replicate_info: bool = True,
    include_metric_inputs: bool = True,
    normalize_metric_input: Optional[Callable[[object], object]] = None,
    collect_metric_cache_by_name: bool = False,
    on_batch: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, object]:
    """Reconstruct aligned point/run arrays from offload batch payloads."""
    goal_maximize = str(optimization_goal).strip().lower() != "minimize"
    total_batches = len(batch_paths)

    all_param_vectors: List[List[float]] = []
    all_metric_values: List[float] = []
    all_replicate_info: List[object] = []
    all_metric_inputs: List[object] = []
    all_run_ids: List[int] = []
    all_run_indices: List[int] = []
    offload_point_sources: List[tuple[str, int]] = []
    metric_cache_by_name: Dict[str, List[float]] = {}
    run_states: Dict[int, Dict[str, object]] = {}

    param_names_list: List[str] = []
    model_key = "simulation"
    model_label = "Simulation"
    metric_name = metric_name_fallback
    next_run_id = 0
    next_run_index = 0

    for batch_i, batch_path in enumerate(batch_paths):
        if should_cancel and should_cancel():
            raise InterruptedError("cancelled")
        if on_batch:
            on_batch(batch_i, total_batches, batch_path)
        payload = read_payload(batch_path) or {}
        if batch_i == 0:
            metric_name = infer_primary_metric_name(payload, metric_name_fallback)
            pnames = payload.get("param_names_list", [])
            if isinstance(pnames, list) and pnames:
                param_names_list = [str(x) for x in pnames]
            mkey = payload.get("model_key")
            if isinstance(mkey, str) and mkey.strip():
                model_key = mkey
                model_label = mkey.capitalize()

        records = payload.get("records", [])
        if not isinstance(records, list):
            continue

        for rec_i, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            vec = _coerce_vector(record.get("param_vector", []))
            metric_value = _safe_float_or_nan(record.get("metric_value", np.nan))

            rid = _coerce_optional_int(record.get("run_id"))
            if rid is None:
                rid = next_run_id
                next_run_id += 1
            else:
                next_run_id = max(next_run_id, rid + 1)

            rix = _coerce_optional_int(record.get("run_index"))
            if rix is None:
                rix = next_run_index
                next_run_index += 1
            else:
                next_run_index = max(next_run_index, rix + 1)

            replicate_info = record.get("replicate_info", {}) if include_replicate_info else {}
            metric_input = record.get("metric_input") if include_metric_inputs else None
            if include_metric_inputs and metric_input is not None and normalize_metric_input is not None:
                normalized = normalize_metric_input(metric_input)
                if normalized is not None:
                    metric_input = normalized

            all_param_vectors.append(vec)
            all_metric_values.append(metric_value)
            all_replicate_info.append(replicate_info)
            all_metric_inputs.append(metric_input)
            all_run_ids.append(rid)
            all_run_indices.append(rix)
            offload_point_sources.append((batch_path, rec_i))

            if collect_metric_cache_by_name:
                point_idx = len(all_param_vectors) - 1
                for vals in metric_cache_by_name.values():
                    vals.append(np.nan)
                metric_map = None
                if isinstance(metric_input, dict):
                    metric_map = metric_input.get("metric_values_by_name")
                if isinstance(metric_map, dict):
                    for metric_key, raw_metric in metric_map.items():
                        key = str(metric_key)
                        arr = metric_cache_by_name.get(key)
                        if arr is None:
                            arr = [np.nan] * (point_idx + 1)
                            metric_cache_by_name[key] = arr
                        arr[point_idx] = _safe_float_or_nan(raw_metric)

            state = run_states.get(rix)
            if state is None:
                run_states[rix] = {
                    "best_metric": metric_value,
                    "best_vec": vec,
                    "final_metric": metric_value,
                    "final_vec": vec,
                }
                continue

            state["final_metric"] = metric_value
            state["final_vec"] = vec
            best_metric = _safe_float_or_nan(state.get("best_metric", np.nan))
            if np.isnan(best_metric):
                if not np.isnan(metric_value):
                    state["best_metric"] = metric_value
                    state["best_vec"] = vec
            elif not np.isnan(metric_value):
                better = metric_value > best_metric if goal_maximize else metric_value < best_metric
                if better:
                    state["best_metric"] = metric_value
                    state["best_vec"] = vec

    if not all_param_vectors:
        raise ValueError("No records found in offload batches.")

    if not param_names_list:
        dim = len(all_param_vectors[0]) if all_param_vectors else 0
        param_names_list = [f"param_{i}" for i in range(dim)]

    all_results: List[Dict[str, object]] = []
    for run_index in sorted(run_states.keys()):
        state = run_states[run_index]
        best_vec = _coerce_vector(state.get("best_vec", []))
        final_vec = _coerce_vector(state.get("final_vec", []))
        best_params = {p: float(v) for p, v in zip(param_names_list, best_vec)}
        final_params = {p: float(v) for p, v in zip(param_names_list, final_vec)}
        best_metric = _safe_float_or_nan(state.get("best_metric", np.nan))
        final_metric = _safe_float_or_nan(state.get("final_metric", np.nan))
        all_results.append(
            {
                "start_index": int(run_index),
                "descent_index": 0,
                "run_index": int(run_index),
                "final_params": final_params,
                "final_metric": float(final_metric) if not np.isnan(final_metric) else np.nan,
                "best_params": best_params,
                "best_metric": float(best_metric) if not np.isnan(best_metric) else np.nan,
                "history": [],
                "fixed_params": {},
            }
        )

    return {
        "metric_name": metric_name,
        "param_names_list": param_names_list,
        "model_key": model_key,
        "model_label": model_label,
        "all_param_vectors": all_param_vectors,
        "all_metric_values": all_metric_values,
        "all_replicate_info": all_replicate_info,
        "all_metric_inputs": all_metric_inputs,
        "all_run_ids": all_run_ids,
        "all_run_indices": all_run_indices,
        "offload_point_sources": offload_point_sources,
        "lightweight_metric_cache": metric_cache_by_name,
        "all_results": all_results,
        "all_runs_history": [[] for _ in all_results],
    }
