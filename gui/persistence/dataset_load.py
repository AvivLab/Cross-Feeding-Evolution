"""Dataset file validation, load alignment, and UI restore helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

from gui.persistence.full_save import is_full_save_dataset_snapshot_basename


DATASET_FILETYPES = [
    ("Optimization Datasets", "optimization_dataset_*.json"),
    ("Full Save Dataset Snapshots", "full_save_dataset_*.json.gz"),
]


def is_supported_dataset_file(path: str) -> bool:
    """Return True when filename matches supported dataset naming patterns."""
    lower_name = str(path or "").lower()
    base_name = os.path.basename(lower_name)
    is_opt_dataset = base_name.startswith("optimization_dataset_") and base_name.endswith(".json")
    is_full_save_snapshot = is_full_save_dataset_snapshot_basename(base_name)
    return bool(is_opt_dataset or is_full_save_snapshot)


def invalid_dataset_file_message() -> str:
    """User-facing invalid selection message."""
    return (
        "Please select one of these dataset files:\n"
        "- optimization_dataset_*.json\n"
        "- full_save_dataset_*.json.gz"
    )



def _safe_set_entry(entry: Any, value: Any) -> None:
    """Best-effort Tk entry update; swallow UI state errors during restore."""
    try:
        entry.delete(0, "end")
        entry.insert(0, str(value))
    except Exception:
        pass


def _safe_set_bool_var(var_obj: Any, value: Any) -> None:
    """Best-effort boolean variable update for Tk variables."""
    try:
        var_obj.set(bool(value))
    except Exception:
        pass


def _safe_set_var(var_obj: Any, value: Any) -> None:
    """Best-effort generic variable update for Tk variables."""
    try:
        var_obj.set(value)
    except Exception:
        pass


def load_dataset_payload(path: str, read_json_maybe_gz_fn: Any, gzip_module: Any) -> Dict[str, Any]:
    """Load dataset payload and provide clearer message for truncated/corrupted gz files."""
    try:
        payload = read_json_maybe_gz_fn(path)
    except (EOFError, gzip_module.BadGzipFile, OSError) as gz_err:
        if str(path).lower().endswith(".gz"):
            raise ValueError(
                "Selected .json.gz file appears incomplete or corrupted.\n"
                "If this is a live/full-save output, wait for saving to finish and try again.\n"
                "Otherwise load an earlier dataset snapshot."
            ) from gz_err
        raise
    return payload if isinstance(payload, dict) else {}


def warn_if_model_mismatch(
    dataset: Dict[str, Any],
    current_model_key: str,
    current_model_label: str,
    messagebox_module: Any,
) -> None:
    """Warn user if loaded dataset was saved from a different model."""
    saved_model = dataset.get("model", {}) if isinstance(dataset, dict) else {}
    saved_key = saved_model.get("key") if isinstance(saved_model, dict) else None
    saved_label = saved_model.get("label") if isinstance(saved_model, dict) else None
    if saved_key and saved_key != current_model_key:
        messagebox_module.showwarning(
            "Model Mismatch",
            f"This dataset was saved for model:\n"
            f"- {saved_label or saved_key}\n\n"
            f"You are currently in model:\n"
            f"- {current_model_label}\n\n"
            "Some fields/plots may be inconsistent.",
        )


def _is_nan_like(value: Any) -> bool:
    """Return True for NaN-like numeric values."""
    try:
        return bool(value != value)
    except Exception:
        return False


def select_best_run_index_by_metric(all_results: Sequence[Any], maximize: bool) -> int:
    """Pick the run index with the best finite scalar using ``best_metric`` else ``final_metric``."""
    best_run_index = 0
    best_metric = None
    for i, result in enumerate(all_results):
        if not isinstance(result, dict):
            continue
        metric = result.get("best_metric", result.get("final_metric", None))
        if metric is None or _is_nan_like(metric):
            continue
        if best_metric is None:
            best_metric = metric
            best_run_index = i
            continue
        try:
            is_better = (metric > best_metric) if maximize else (metric < best_metric)
        except Exception:
            continue
        if is_better:
            best_metric = metric
            best_run_index = i
    return int(best_run_index)


def resolve_loaded_best_run_index(
    all_results: Sequence[Any],
    loaded_best_run_index: int,
    maximize: bool,
) -> int:
    """Resolve best run index from loaded data, tolerating stale/out-of-range index."""
    if 0 <= int(loaded_best_run_index) < len(all_results):
        return int(loaded_best_run_index)
    return select_best_run_index_by_metric(all_results, maximize)


def decode_loaded_point_arrays(dataset: Dict[str, Any], np_module: Any):
    """Decode point-level arrays from dataset payload."""
    param_vectors_data = dataset.get("all_param_vectors", [])
    all_param_vectors = []
    for pv in param_vectors_data:
        if isinstance(pv, list):
            all_param_vectors.append(np_module.array(pv))
        elif isinstance(pv, np_module.ndarray):
            all_param_vectors.append(pv)
        else:
            all_param_vectors.append(np_module.array(list(pv)))

    all_metric_values = [float(m) if m is not None else np_module.nan for m in dataset.get("all_metric_values", [])]
    all_replicate_info = dataset.get("all_replicate_info", [])
    all_metric_inputs = dataset.get("all_metric_inputs", [])
    all_run_ids = dataset.get("all_run_ids", [])
    all_run_indices = dataset.get("all_run_indices", [])
    return (
        all_param_vectors,
        all_metric_values,
        all_replicate_info,
        all_metric_inputs,
        all_run_ids,
        all_run_indices,
    )


def initialize_loaded_dataset_state(dataset: Dict[str, Any], np_module: Any) -> Dict[str, Any]:
    """Build the base in-memory state used by load_dataset before validation/alignment."""
    all_results = dataset.get("all_results", [])
    all_runs_history = dataset.get("all_runs_history", [])
    (
        all_param_vectors,
        all_metric_values,
        all_replicate_info,
        all_metric_inputs,
        all_run_ids,
        all_run_indices,
    ) = decode_loaded_point_arrays(dataset=dataset, np_module=np_module)

    transient_state = cleared_load_transient_state()
    loaded_best_run_index = dataset.get("best_run_index", 0)
    param_names_list = dataset.get("param_names_list", [])

    return {
        "all_results": all_results,
        "all_runs_history": all_runs_history,
        "all_param_vectors": all_param_vectors,
        "all_metric_values": all_metric_values,
        "all_replicate_info": all_replicate_info,
        "all_metric_inputs": all_metric_inputs,
        "all_run_ids": all_run_ids,
        "all_run_indices": all_run_indices,
        "loaded_best_run_index": loaded_best_run_index,
        "param_names_list": param_names_list,
        "transient_state": transient_state,
    }


def cleared_load_transient_state() -> Dict[str, Any]:
    """Return canonical transient state values when loading a regular dataset."""
    return {
        "offload_point_sources": [],
        "offload_pending_records": [],
        "offload_last_batch_path": None,
        "offload_last_batch_payload": None,
        "full_save_lightweight_mode": False,
        "full_save_lightweight_metric_name": None,
        "lightweight_metric_cache": {},
        "loaded_full_save_session_id": None,
        "loaded_full_save_settings_snapshot": None,
    }


def collect_length_inconsistencies(
    all_results: Sequence[Any],
    all_runs_history: Sequence[Any],
    all_param_vectors: Sequence[Any],
    all_metric_values: Sequence[Any],
    all_replicate_info: Sequence[Any],
    all_metric_inputs: Sequence[Any],
    all_run_ids: Sequence[Any],
    all_run_indices: Sequence[Any],
) -> List[str]:
    """Build user-facing messages for length mismatches in loaded datasets."""
    num_results = len(all_results)
    num_histories = len(all_runs_history)
    num_vectors = len(all_param_vectors)
    num_metrics = len(all_metric_values)
    num_replicate_info = len(all_replicate_info)
    num_metric_inputs = len(all_metric_inputs)
    num_run_ids = len(all_run_ids)
    num_run_indices = len(all_run_indices)

    inconsistencies: List[str] = []
    if num_results != num_histories:
        inconsistencies.append(f"Results ({num_results}) vs Histories ({num_histories})")
    if num_vectors != num_metrics:
        inconsistencies.append(f"Param vectors ({num_vectors}) vs Metrics ({num_metrics})")
    if num_vectors != num_replicate_info:
        inconsistencies.append(f"Param vectors ({num_vectors}) vs Replicate info ({num_replicate_info})")
    if num_vectors != num_metric_inputs:
        inconsistencies.append(f"Param vectors ({num_vectors}) vs Metric inputs ({num_metric_inputs})")
    if num_vectors != num_run_ids:
        inconsistencies.append(f"Param vectors ({num_vectors}) vs Run IDs ({num_run_ids})")
    if num_vectors != num_run_indices:
        inconsistencies.append(f"Param vectors ({num_vectors}) vs Run indices ({num_run_indices})")
    return inconsistencies


def format_inconsistency_warning(inconsistencies: Sequence[str]) -> str:
    """Format inconsistency lines for the load warning dialog."""
    warning_msg = "Warning: Loaded dataset has inconsistent data lengths:\n" + "\n".join(inconsistencies)
    warning_msg += "\n\nSome features may not work correctly. The dataset may be corrupted or from an older version."
    return warning_msg


def is_empty_loaded_dataset(num_results: int, num_vectors: int) -> bool:
    """Return True if loaded dataset has no run-level or point-level content."""
    return bool(num_results == 0 or num_vectors == 0)


def restore_ui_state(
    dataset: Dict[str, Any],
    entry_by_key: Dict[str, Any],
    raw_var_by_key: Dict[str, Any],
    bool_var_by_key: Dict[str, Any],
    str_var_by_key: Dict[str, Any],
) -> bool:
    """Restore ui_state fields into mapped widgets/vars; returns True if ui_state existed."""
    ui_state = dataset.get("ui_state", {}) if isinstance(dataset, dict) else {}
    if not (isinstance(ui_state, dict) and ui_state):
        return False

    for key, entry in entry_by_key.items():
        if key in ui_state:
            _safe_set_entry(entry, ui_state[key])

    for key, var_obj in raw_var_by_key.items():
        if key in ui_state:
            _safe_set_var(var_obj, ui_state[key])

    for key, var_obj in bool_var_by_key.items():
        if key in ui_state:
            _safe_set_bool_var(var_obj, ui_state[key])

    for key, var_obj in str_var_by_key.items():
        if key in ui_state:
            _safe_set_var(var_obj, str(ui_state[key]))

    return True


def restore_param_ui_state_and_fixed_values(
    dataset: Dict[str, Any],
    param_fix_checkboxes: Dict[str, Any],
    param_initial_entries: Dict[str, Any],
    param_min_entries: Dict[str, Any],
    param_max_entries: Dict[str, Any],
    fixed_entries: Dict[str, Any],
    refresh_optimizable_params: Any,
    refresh_fixed_params: Any,
) -> None:
    """Restore param checkbox/min/max state and fixed-parameter values from dataset payload."""
    param_ui_state = dataset.get("param_ui_state", {}) if isinstance(dataset, dict) else {}
    if isinstance(param_ui_state, dict) and param_ui_state:
        for pname, st in param_ui_state.items():
            if pname not in param_fix_checkboxes or not isinstance(st, dict):
                continue
            if "fixed" in st:
                _safe_set_bool_var(param_fix_checkboxes[pname], st["fixed"])
            if "initial" in st and pname in param_initial_entries:
                _safe_set_entry(param_initial_entries[pname], st["initial"])
            if "min" in st and pname in param_min_entries:
                _safe_set_entry(param_min_entries[pname], st["min"])
            if "max" in st and pname in param_max_entries:
                _safe_set_entry(param_max_entries[pname], st["max"])

        refresh_optimizable_params()
        refresh_fixed_params()

        fixed_param_values = dataset.get("fixed_param_values", {}) if isinstance(dataset, dict) else {}
        if isinstance(fixed_param_values, dict) and fixed_param_values:
            for pname, val in fixed_param_values.items():
                if pname in fixed_entries:
                    _safe_set_entry(fixed_entries[pname], val)


def extract_loaded_fixed_params(all_results: Sequence[Any], best_run_index: int) -> Dict[str, Any]:
    """Extract fixed params from the best loaded run, with safe fallback."""
    loaded_best_result = (
        all_results[best_run_index]
        if 0 <= int(best_run_index) < len(all_results)
        else (all_results[0] if all_results else {})
    )
    loaded_fixed = loaded_best_result.get("fixed_params", {}) if isinstance(loaded_best_result, dict) else {}
    return loaded_fixed if isinstance(loaded_fixed, dict) else {}


def apply_loaded_fixed_params_to_widgets(
    loaded_fixed: Dict[str, Any],
    fixed_entries: Dict[str, Any],
    param_initial_entries: Dict[str, Any],
    seed_entry: Any,
    simulation_setting_vars: Dict[str, Any],
) -> bool:
    """Apply loaded fixed params to fixed/initial entries and simulation setting widgets."""
    if not (isinstance(loaded_fixed, dict) and loaded_fixed):
        return False

    for pname, entry in fixed_entries.items():
        if pname in loaded_fixed:
            _safe_set_entry(entry, loaded_fixed[pname])

    for pname, entry in param_initial_entries.items():
        if pname in loaded_fixed:
            _safe_set_entry(entry, loaded_fixed[pname])

    if "Random Seed (optional)" in loaded_fixed:
        _safe_set_entry(seed_entry, loaded_fixed.get("Random Seed (optional)", ""))

    for setting_name, var_obj in simulation_setting_vars.items():
        if setting_name in loaded_fixed:
            _safe_set_bool_var(var_obj, loaded_fixed[setting_name])
    return True


def restore_results_summary_text(
    results_text_widget: Any,
    dataset: Dict[str, Any],
    all_results: Sequence[Any],
    best_run_index: int,
    build_summary_text_fn: Any,
    np_module: Any,
) -> None:
    """Restore saved summary text, or rebuild fallback summary from loaded results."""
    results_text_widget.delete(1.0, "end")
    saved_summary = dataset.get("summary_text", "")
    if isinstance(saved_summary, str) and saved_summary.strip():
        results_text_widget.insert("end", saved_summary)
    else:
        if all_results:
            best_result_for_load = (
                all_results[best_run_index]
                if 0 <= int(best_run_index) < len(all_results)
                else all_results[0]
            )
            best_params_for_load = best_result_for_load.get("best_params", best_result_for_load.get("final_params", {}))
            best_metric_for_load = best_result_for_load.get(
                "best_metric", best_result_for_load.get("final_metric", np_module.nan)
            )
            results_text_widget.insert(
                "end",
                build_summary_text_fn(all_results, best_run_index, best_params_for_load, best_metric_for_load),
            )
        else:
            results_text_widget.insert("end", "Loaded dataset contains no run results.\n")
    results_text_widget.see("end")


def show_loaded_dataset_info_dialog(
    dataset: Dict[str, Any],
    all_results: Sequence[Any],
    all_param_vectors: Sequence[Any],
    filename: str,
    messagebox_module: Any,
) -> None:
    """Show success/info dialog after loading optimization dataset."""
    metadata = dataset.get("metadata", {}) if isinstance(dataset, dict) else {}
    if metadata:
        info_msg = "Dataset loaded successfully!\n\n"
        info_msg += f"Runs: {metadata.get('num_runs', len(all_results))}\n"
        info_msg += f"Data points: {metadata.get('num_points', len(all_param_vectors))}\n"
        info_msg += f"Metric: {metadata.get('metric_name', 'Unknown')}\n"
        messagebox_module.showinfo("Dataset Loaded", info_msg)
    else:
        messagebox_module.showinfo("Success", f"Dataset loaded successfully from:\n{filename}")


def set_loaded_dataset_button_states(
    all_results: Sequence[Any],
    param_heatmaps_button: Any,
    metric_hist_button: Any,
) -> None:
    """Enable/disable post-load action buttons based on available results."""
    state = "normal" if all_results else "disabled"
    param_heatmaps_button.config(state=state)
    metric_hist_button.config(state=state)


def run_post_loaded_fixed_refreshes(
    refresh_diffusion_mutation_state_fn: Any,
    refresh_optimizable_params_fn: Any,
    refresh_fixed_params_fn: Any,
    refresh_metric_options_fn: Any,
    refresh_chemostat_flow_state_fn: Any,
    refresh_initial_energy_state_fn: Any,
    refresh_intermediate_costs_state_fn: Any,
    refresh_acetate_addition_state_fn: Any,
    refresh_death_dup_state_fn: Any = None,
) -> None:
    """
    Run UI refresh sequence after applying loaded fixed params.

    Callers should pass feature refresh lambdas that update ``hidden_params`` only
    (``rebuild_param_tables=False`` / equivalent). This function rebuilds the
    optimizable/fixed tables once at the end to avoid multi-destroy flashes.
    """
    refresh_diffusion_mutation_state_fn()
    for fn in (
        refresh_death_dup_state_fn,
        refresh_chemostat_flow_state_fn,
        refresh_initial_energy_state_fn,
        refresh_intermediate_costs_state_fn,
        refresh_acetate_addition_state_fn,
    ):
        if fn is None:
            continue
        try:
            fn()
        except Exception:
            pass
    refresh_optimizable_params_fn()
    refresh_fixed_params_fn()
    refresh_metric_options_fn()


def optimization_goal_from_dataset(dataset: Dict[str, Any], goal_fallback: str = "Maximize") -> str:
    """Resolve optimization goal string: metadata, then top-level ui_state, then fallback."""
    md = dataset.get("metadata", {})
    if isinstance(md, dict):
        g = md.get("optimization_goal")
        if isinstance(g, str) and g.strip():
            return str(g).strip()
    ui = dataset.get("ui_state", {})
    if isinstance(ui, dict):
        g = ui.get("optimization_goal")
        if isinstance(g, str) and g.strip():
            return str(g).strip()
    if isinstance(goal_fallback, str) and goal_fallback.strip():
        return str(goal_fallback).strip()
    return "Maximize"


def prepare_decoded_aligned_dataset(
    dataset: Dict[str, Any],
    np_module: Any,
    *,
    goal_fallback: str = "Maximize",
) -> tuple[Dict[str, Any] | None, List[str]]:
    """Decode point arrays, optionally warn on length mismatch, align, resolve best run.

    Returns ``(state, warning_messages)``. ``state`` is None only when the dataset has
    no runs or no points. ``warning_messages`` are full user-facing strings (caller may
    show on the UI thread); empty when lengths are consistent.
    """
    if not isinstance(dataset, dict):
        return None, []

    base = initialize_loaded_dataset_state(dataset=dataset, np_module=np_module)
    all_results = base["all_results"]
    all_runs_history = base["all_runs_history"]
    all_param_vectors = base["all_param_vectors"]
    all_metric_values = base["all_metric_values"]
    all_replicate_info = base["all_replicate_info"]
    all_metric_inputs = base["all_metric_inputs"]
    all_run_ids = base["all_run_ids"]
    all_run_indices = base["all_run_indices"]
    loaded_best_run_index = int(base["loaded_best_run_index"])
    param_names_list = base["param_names_list"]
    transient_state = base["transient_state"]

    inconsistencies = collect_length_inconsistencies(
        all_results,
        all_runs_history,
        all_param_vectors,
        all_metric_values,
        all_replicate_info,
        all_metric_inputs,
        all_run_ids,
        all_run_indices,
    )
    warnings: List[str] = []
    if inconsistencies:
        warnings.append(format_inconsistency_warning(inconsistencies))

    num_results = len(all_results)
    num_vectors = len(all_param_vectors)
    if is_empty_loaded_dataset(num_results, num_vectors):
        return None, warnings

    (
        all_results,
        all_runs_history,
        all_param_vectors,
        all_metric_values,
        all_replicate_info,
        all_metric_inputs,
        all_run_ids,
        all_run_indices,
    ) = align_loaded_dataset_arrays(
        all_results=all_results,
        all_runs_history=all_runs_history,
        all_param_vectors=all_param_vectors,
        all_metric_values=all_metric_values,
        all_replicate_info=all_replicate_info,
        all_metric_inputs=all_metric_inputs,
        all_run_ids=all_run_ids,
        all_run_indices=all_run_indices,
        nan_value=np_module.nan,
    )

    resolved_goal = optimization_goal_from_dataset(dataset, goal_fallback=goal_fallback)
    maximize = resolved_goal == "Maximize"
    best_run_index = resolve_loaded_best_run_index(
        all_results=all_results,
        loaded_best_run_index=loaded_best_run_index,
        maximize=maximize,
    )

    state: Dict[str, Any] = {
        "all_results": all_results,
        "all_runs_history": all_runs_history,
        "all_param_vectors": all_param_vectors,
        "all_metric_values": all_metric_values,
        "all_replicate_info": all_replicate_info,
        "all_metric_inputs": all_metric_inputs,
        "all_run_ids": all_run_ids,
        "all_run_indices": all_run_indices,
        "best_run_index": int(best_run_index),
        "param_names_list": param_names_list,
        "transient_state": transient_state,
        "resolved_goal_name": resolved_goal,
    }
    return state, warnings


def align_loaded_dataset_arrays(
    all_results: List[Any],
    all_runs_history: List[Any],
    all_param_vectors: List[Any],
    all_metric_values: List[Any],
    all_replicate_info: List[Any],
    all_metric_inputs: List[Any],
    all_run_ids: List[Any],
    all_run_indices: List[Any],
    nan_value: Any,
) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any], List[Any], List[Any], List[Any]]:
    """Pad/trim loaded arrays so run-level and point-level structures stay aligned."""
    num_results = len(all_results)
    num_histories = len(all_runs_history)
    if num_histories < num_results:
        all_runs_history = all_runs_history + ([[]] * (num_results - num_histories))
    elif num_histories > num_results:
        all_runs_history = all_runs_history[:num_results]

    num_vectors = len(all_param_vectors)
    num_metrics = len(all_metric_values)
    if num_metrics < num_vectors:
        all_metric_values = all_metric_values + [nan_value] * (num_vectors - num_metrics)
    elif num_metrics > num_vectors:
        all_metric_values = all_metric_values[:num_vectors]

    if len(all_replicate_info) < num_vectors:
        all_replicate_info = all_replicate_info + ([{}] * (num_vectors - len(all_replicate_info)))
    elif len(all_replicate_info) > num_vectors:
        all_replicate_info = all_replicate_info[:num_vectors]

    if len(all_metric_inputs) < num_vectors:
        all_metric_inputs = all_metric_inputs + ([None] * (num_vectors - len(all_metric_inputs)))
    elif len(all_metric_inputs) > num_vectors:
        all_metric_inputs = all_metric_inputs[:num_vectors]

    if len(all_run_ids) < num_vectors:
        start_id = (max(all_run_ids) + 1) if all_run_ids else 0
        all_run_ids = all_run_ids + list(range(start_id, start_id + (num_vectors - len(all_run_ids))))
    elif len(all_run_ids) > num_vectors:
        all_run_ids = all_run_ids[:num_vectors]

    if len(all_run_indices) < num_vectors:
        all_run_indices = all_run_indices + list(range(len(all_run_indices), num_vectors))
    elif len(all_run_indices) > num_vectors:
        all_run_indices = all_run_indices[:num_vectors]

    return (
        all_results,
        all_runs_history,
        all_param_vectors,
        all_metric_values,
        all_replicate_info,
        all_metric_inputs,
        all_run_ids,
        all_run_indices,
    )

