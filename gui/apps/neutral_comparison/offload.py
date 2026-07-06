"""
Full Save-style offload batches for Batch Runner and headless neutral comparison Monte Carlo runs.

Matches Gradient Descent layout: ``full_save_manifest_<session>.json`` plus
``offload_<session>_<batchindex>.json.gz`` batches so manifests and offload files
discover the same session ids (see ``gui.persistence.full_save``).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from gui.common.data_utils import to_json_serializable
from gui.persistence.full_save import (
    full_save_manifest_path_json,
    load_full_save_manifest,
)
from gui.persistence.json_io import make_write_json_maybe_gz_atomic_fn

_write_json = make_write_json_maybe_gz_atomic_fn(indent=2)

# Match gradient_descent_gui buffering for batched offload writes.
OFFLOAD_FLUSH_RECORDS = 250

# Parallel Slurm ranks stream offload into <session>/_parallel_shards/<shard>/ before the parent merges.
NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR = "_parallel_shards"

# Keys match gradient_descent metric input cache layout for reanalysis.
_METRIC_INPUT_RESULT_KEYS = [
    "A_history",
    "B_history",
    "energy_history",
    "M2_export_history",
    "M2_import_history",
    "task1_performance",
    "task2_performance",
    "metabs_history",
    "storage_history",
    "change_history",
    "mutation_rate",
    "mutation_scale",
    "initial_A",
    "independent_traits",
    "investment_modifier",
    "final_metabolite_environment",
    "metabolite_environment_after_inflow_final_generation",
    "enable_m1_diffusion",
    "enable_m2_diffusion",
    "enable_intermediate_costs",
]


def param_vector_for_offload(merged_params: Dict[str, Any], param_names_list: List[str]) -> List[float]:
    """Parameter vector order used in Full Save offload batches."""
    vec: List[float] = []
    for name in param_names_list:
        raw = merged_params.get(name, None)
        try:
            vec.append(float(raw))
        except Exception:
            vec.append(float("nan"))
    return vec


def param_dict_from_offload_vector(
    param_vector: Sequence[Any],
    param_names_list: Sequence[str],
    *,
    strict: bool = False,
) -> Dict[str, float]:
    """Decode stored ``param_vector`` back to a name→value map (inverse of ``param_vector_for_offload``)."""
    out: Dict[str, float] = {}
    vec = list(param_vector or [])
    names = [str(x) for x in (param_names_list or [])]
    if strict and len(vec) != len(names):
        raise ValueError(
            f"param_vector length {len(vec)} != param_names_list length {len(names)}"
        )
    for i, name in enumerate(names):
        if i >= len(vec):
            if strict:
                raise ValueError(f"missing param_vector value for {name!r}")
            break
        try:
            v = float(vec[i])
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"param {name!r} is not numeric: {vec[i]!r}") from exc
            continue
        if not np.isfinite(v):
            if strict:
                raise ValueError(f"param {name!r} is not finite: {v!r}")
            continue
        out[name] = v
    if strict and len(out) != len(names):
        missing = [n for n in names if n not in out]
        raise ValueError(f"param_vector decode missing keys: {missing!r}")
    return out


def build_neutral_comparison_offload_record(
    param_names_list: List[str],
    metric_name_at_offload: str,
    *,
    merged_params: Dict[str, Any],
    result_obj: Optional[dict],
    hit: bool,
    evaluated_metrics: Dict[str, float],
    primary_metric_value: float,
    offload_stage: str,
    seed_used: Any = None,
    run_index: int,
) -> dict:
    """
    One offload row (same schema as ``NeutralComparisonOffloadWriter.add_simulation_record``).
    ``run_index`` is stored as run_id/run_index; callers may rewrite when merging batches.
    """
    cache = extract_simulation_metric_input_cache(result_obj if isinstance(result_obj, dict) else None, seed_used)
    if isinstance(cache, dict):
        mm = cache.get("metric_values_by_name")
        if not isinstance(mm, dict):
            mm = {}
            cache["metric_values_by_name"] = mm
        for mk, mv in evaluated_metrics.items():
            try:
                fv = float(mv)
                mm[str(mk)] = None if not np.isfinite(fv) else fv
            except Exception:
                mm[str(mk)] = None
        pm = float(primary_metric_value)
        mm[str(metric_name_at_offload)] = None if not np.isfinite(pm) else pm

    succeeded = 1 if (isinstance(result_obj, dict) and not result_obj.get("collapsed")) else 0
    failed = 1 - succeeded
    return {
        "param_vector": param_vector_for_offload(merged_params, param_names_list),
        "metric_value": None if not np.isfinite(float(primary_metric_value)) else float(primary_metric_value),
        "replicate_info": [1, succeeded, failed],
        "metric_input": cache,
        "run_id": int(run_index),
        "run_index": int(run_index),
        "neutral_comparison": {"stage": str(offload_stage or ""), "hit": bool(hit)},
    }


def extract_simulation_metric_input_cache(result_obj: Any, seed_used: Any = None) -> dict:
    """Compact per-simulation cache for offload (same role as GD evaluate_metric)."""
    if not isinstance(result_obj, dict) or result_obj.get("collapsed"):
        return {"collapsed": True}
    cache: Dict[str, Any] = {"collapsed": bool(result_obj.get("collapsed", False))}
    for k in _METRIC_INPUT_RESULT_KEYS:
        if k in result_obj:
            cache[k] = to_json_serializable(result_obj.get(k))
    if seed_used is not None:
        try:
            cache["random_seed_used"] = int(seed_used)
        except Exception:
            cache["random_seed_used"] = seed_used
    cache.setdefault("metric_values_by_name", {})
    return cache


class NeutralComparisonOffloadWriter:
    """Append offload batches and update the session manifest (GD-compatible filenames)."""

    def __init__(
        self,
        folder: str,
        session_id: str,
        *,
        model_key: str,
        param_names_list: List[str],
        metric_name_at_offload: str,
        job_settings_json_path: Optional[str] = None,
    ) -> None:
        self.folder = os.path.abspath(os.path.expanduser(str(folder or "").strip()))
        self.session_id = str(session_id or "").strip()
        self.model_key = str(model_key or "simulation")
        self.param_names_list = [str(x) for x in (param_names_list or [])]
        self.metric_name_at_offload = str(metric_name_at_offload or "").strip() or "Unknown"
        self._batch_counter = 0
        self._run_index = 0
        self._pending: List[dict] = []
        self.manifest: Dict[str, Any] = {
            "session_id": self.session_id,
            "batches": [],
            "kind": "neutral_set_comparison",
            "model_key": self.model_key,
            "metric_name_at_offload": self.metric_name_at_offload,
        }
        _jp = (str(job_settings_json_path).strip() if job_settings_json_path is not None else "")
        if _jp:
            _abs = os.path.abspath(os.path.expanduser(_jp))
            self.manifest["job_settings_json_path"] = _abs
            self.manifest["job_settings_json_basename"] = os.path.basename(_abs)
        os.makedirs(self.folder, exist_ok=True)
        mp = full_save_manifest_path_json(self.folder, self.session_id)
        _write_json(mp, self.manifest)

    def _param_vector(self, merged_params: Dict[str, Any]) -> List[float]:
        return param_vector_for_offload(merged_params, self.param_names_list)

    def _flush_batch(self) -> None:
        if not self._pending:
            return
        filename = f"offload_{self.session_id}_{self._batch_counter:05d}.json.gz"
        batch_path = os.path.join(self.folder, filename)
        records = list(self._pending)
        payload: Dict[str, Any] = {
            "model_key": self.model_key,
            "session_id": self.session_id,
            "batch_index": int(self._batch_counter),
            "metric_name_at_offload": self.metric_name_at_offload,
            "param_names_list": list(self.param_names_list),
            "records": records,
        }
        _write_json(batch_path, payload)
        metric_name = self.metric_name_at_offload
        cached_points = 0
        for rec in records:
            mi = rec.get("metric_input")
            if isinstance(mi, dict):
                metric_map = mi.get("metric_values_by_name")
                if isinstance(metric_map, dict) and metric_name in metric_map:
                    cached_points += 1
        self.manifest.setdefault("batches", [])
        self.manifest["batches"].append(
            {
                "batch_index": int(self._batch_counter),
                "path": os.path.basename(batch_path),
                "num_records": len(records),
                "cached_metric_counts": {metric_name: int(cached_points)},
                "updated_at_epoch": int(time.time()),
            }
        )
        mp = full_save_manifest_path_json(self.folder, self.session_id)
        _write_json(mp, self.manifest)
        self._batch_counter += 1
        self._pending.clear()

    def add_simulation_record(
        self,
        *,
        merged_params: Dict[str, Any],
        result_obj: Optional[dict],
        hit: bool,
        evaluated_metrics: Dict[str, float],
        primary_metric_value: float,
        offload_stage: str,
        seed_used: Any = None,
    ) -> None:
        """Queue one Monte Carlo draw; flush when buffer reaches OFFLOAD_FLUSH_RECORDS."""
        rec = build_neutral_comparison_offload_record(
            self.param_names_list,
            self.metric_name_at_offload,
            merged_params=merged_params,
            result_obj=result_obj if isinstance(result_obj, dict) else None,
            hit=hit,
            evaluated_metrics=evaluated_metrics,
            primary_metric_value=primary_metric_value,
            offload_stage=offload_stage,
            seed_used=seed_used,
            run_index=int(self._run_index),
        )
        self._run_index += 1
        self._pending.append(rec)
        if len(self._pending) >= OFFLOAD_FLUSH_RECORDS:
            self._flush_batch()

    def ingest_offload_records(self, records: List[dict]) -> None:
        """Append pre-built offload rows (e.g. from worker processes), assigning run_id/run_index in order."""
        for rec in records:
            rec["run_id"] = int(self._run_index)
            rec["run_index"] = int(self._run_index)
            self._run_index += 1
            self._pending.append(rec)
            if len(self._pending) >= OFFLOAD_FLUSH_RECORDS:
                self._flush_batch()

    def finalize(self) -> None:
        """Flush any remaining buffered records."""
        self._flush_batch()


def merge_parallel_neutral_comparison_shards_into_writer(
    main_writer: NeutralComparisonOffloadWriter,
    session_folder: str,
    session_id: str,
    n_neutral_batches: int,
    n_runs: int,
    read_json: Callable[[str], Any],
) -> None:
    """
    After parallel headless workers finish, merge shard folders into ``main_writer`` in order:
    ``primary``, then ``neutral_0`` … ``neutral_{n-1}``.

    Each shard folder contains its own manifest + offload batches (same filenames as the session
    root layout, but isolated per process). Records are re-numbered via ``ingest_offload_records``.
    """
    root = os.path.abspath(os.path.expanduser(str(session_folder or "").strip()))
    sid = str(session_id or "").strip()
    n_neutral_batches = int(max(0, n_neutral_batches))
    n_runs = int(n_runs)
    shard_root = os.path.join(root, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR)
    shard_names = ["primary"] + [f"neutral_{r}" for r in range(n_neutral_batches)]
    for name in shard_names:
        shard_dir = os.path.join(shard_root, name)
        if not os.path.isdir(shard_dir):
            raise RuntimeError(
                f"Parallel neutral offload merge: missing shard directory {shard_dir!r} "
                f"(expected after worker batch {name!r})."
            )
        man = load_full_save_manifest(shard_dir, sid, read_json)
        batches = man.get("batches") if isinstance(man, dict) else None
        if not isinstance(batches, list) or not batches:
            raise RuntimeError(
                f"Parallel neutral offload merge: shard {name!r} has no batches in manifest under {shard_dir!r}."
            )
        batches_sorted = sorted(batches, key=lambda b: int(b.get("batch_index", 0)) if isinstance(b, dict) else 0)
        total_recs = 0
        for b in batches_sorted:
            if not isinstance(b, dict):
                continue
            rel = b.get("path")
            if not rel or not isinstance(rel, str):
                raise RuntimeError(f"Parallel neutral offload merge: shard {name!r} batch missing path: {b!r}")
            batch_path = os.path.join(shard_dir, rel)
            if not os.path.isfile(batch_path):
                raise RuntimeError(
                    f"Parallel neutral offload merge: missing offload file {batch_path!r} (shard {name!r})."
                )
            payload = read_json(batch_path)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Parallel neutral offload merge: invalid JSON in {batch_path!r}")
            records = payload.get("records")
            if not isinstance(records, list):
                raise RuntimeError(f"Parallel neutral offload merge: no records list in {batch_path!r}")
            total_recs += len(records)
            main_writer.ingest_offload_records(list(records))
        if total_recs != n_runs:
            raise RuntimeError(
                f"Parallel neutral offload merge: shard {name!r} expected {n_runs} records, got {total_recs} "
                f"(under {shard_dir!r})."
            )


def count_shard_offload_batches(
    session_folder: str,
    session_id: str,
    n_neutral_batches: int,
    read_json: Callable[[str], Any],
) -> int:
    """
    Total number of offload batch files listed across shard manifests under
    ``_parallel_shards/{primary,neutral_*,...}/`` (used when session-root merge is skipped).
    """
    root = os.path.abspath(os.path.expanduser(str(session_folder or "").strip()))
    sid = str(session_id or "").strip()
    n_neutral_batches = int(max(0, n_neutral_batches))
    shard_root = os.path.join(root, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR)
    shard_names = ["primary"] + [f"neutral_{r}" for r in range(n_neutral_batches)]
    total = 0
    for name in shard_names:
        shard_dir = os.path.join(shard_root, name)
        if not os.path.isdir(shard_dir):
            continue
        man = load_full_save_manifest(shard_dir, sid, read_json)
        batches = man.get("batches") if isinstance(man, dict) else None
        if isinstance(batches, list):
            total += len(batches)
    return int(total)
