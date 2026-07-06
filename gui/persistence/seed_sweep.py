"""Shared schema + helpers for filtered seed-sweep result persistence."""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np

from gui.persistence.json_io import make_read_json_maybe_gz_fn, make_write_json_maybe_gz_atomic_fn

_read_seed_sweep_json = make_read_json_maybe_gz_fn(plain_twin_fallback=False)
_write_seed_sweep_json = make_write_json_maybe_gz_atomic_fn(indent=2)


SEED_SWEEP_RESULTS_VERSION = 1

SCRATCH_MANIFEST_KIND = "gradient_gui_seed_sweep_scratch"


def load_seed_sweep_from_scratch_dir(dir_path: str) -> dict:
    """
    Rebuild the decoded payload shape used by the seed-sweep UI from a Run Sweep scratch folder
    (manifest.json + per_point_seeds.jsonl + all_resim_values.f64).

    Unlike exported ``.json.gz`` bundles, scratch folders do not store full point metadata
    (run_index, params, …); those entries are omitted so the histograms still work.
    """
    root = os.path.abspath(os.path.expanduser(str(dir_path or "").strip()))
    if not root or not os.path.isdir(root):
        raise ValueError(f"Not a directory: {dir_path!r}")

    man_path = os.path.join(root, "manifest.json")
    pp_path = os.path.join(root, "per_point_seeds.jsonl")
    vals_path = os.path.join(root, "all_resim_values.f64")
    if not os.path.isfile(man_path):
        raise ValueError("Scratch folder must contain manifest.json.")
    if not os.path.isfile(pp_path):
        raise ValueError("Scratch folder must contain per_point_seeds.jsonl.")
    if not os.path.isfile(vals_path):
        raise ValueError("Scratch folder must contain all_resim_values.f64.")

    with open(man_path, "r", encoding="utf-8") as mf:
        manifest = json.load(mf)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json is not a JSON object.")
    if str(manifest.get("kind", "") or "") != SCRATCH_MANIFEST_KIND:
        raise ValueError("manifest.json is not a gradient GUI seed-sweep scratch manifest.")

    metric_name = str(manifest.get("metric_name", "") or "").strip()
    seeds_per_point = int(manifest.get("seeds_per_point", 0) or 0)
    n_rows = int(manifest.get("n_rows", 0) or 0)
    light_mode = manifest.get("light_mode", None)

    by_row: dict[int, list[float]] = {}
    with open(pp_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            try:
                ri = int(rec.get("row_index", -1))
            except Exception:
                continue
            vals = rec.get("values", [])
            if not isinstance(vals, list):
                vals = []
            row: list[float] = []
            for x in vals:
                try:
                    row.append(float(x))
                except Exception:
                    continue
            by_row[ri] = row

    if n_rows <= 0:
        n_rows = (max(by_row.keys()) + 1) if by_row else 0
    per_point_values: list[list[float]] = []
    for i in range(n_rows):
        per_point_values.append(list(by_row.get(i, [])))

    flat: list[float] = []
    per_point_means: list[float] = []
    for row in per_point_values:
        if row:
            flat.extend(row)
            per_point_means.append(float(sum(row) / len(row)))
        else:
            per_point_means.append(float("nan"))

    try:
        all_vals_np = np.fromfile(vals_path, dtype=np.float64)
        all_values = [float(x) for x in all_vals_np.tolist() if np.isfinite(x)]
    except Exception:
        all_values = list(flat)

    settings = {
        "metric_name": metric_name,
        "seeds_per_point": int(seeds_per_point),
        "point_count": int(n_rows),
        "model_key": str(manifest.get("model_key", "") or ""),
        "source_full_save_dir": "",
        "source_session_id": "",
        "light_mode": light_mode,
        "sweep_scratch_dir": root,
        "scratch_reload": True,
    }

    points = [{"scratch_row_index": i} for i in range(n_rows)]

    return {
        "version": SEED_SWEEP_RESULTS_VERSION,
        "saved_at_epoch": int(manifest.get("created_epoch", 0) or 0),
        "settings": settings,
        "points": points,
        "per_point_values": per_point_values,
        "per_point_means": per_point_means,
        "all_values": all_values,
    }


def suggested_seed_sweep_filename(metric_name: str) -> str:
    """Build a stable default filename for seed-sweep result exports."""
    metric_part = re.sub(r"[^A-Za-z0-9._-]+", "_", str(metric_name or "").strip()).strip("_")
    if not metric_part:
        metric_part = "metric"
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"seed_sweep_results_{metric_part}_{ts}.json.gz"


def _encode_numeric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return float(numeric)


def encode_per_point_values(per_point_values: Iterable[Iterable[Any]]) -> list[list[float | None]]:
    """Encode nested per-seed values for JSON storage."""
    encoded: list[list[float | None]] = []
    for values in per_point_values:
        row: list[float | None] = []
        for value in values:
            row.append(_encode_numeric_or_none(value))
        encoded.append(row)
    return encoded


def build_seed_sweep_results_payload(
    *,
    settings: Mapping[str, Any],
    points: Iterable[Mapping[str, Any]],
    per_point_values: Iterable[Iterable[Any]],
    saved_at_epoch: int | None = None,
) -> dict:
    """Build canonical persisted payload for seed-sweep results."""
    encoded_values = encode_per_point_values(per_point_values)
    points_out = [dict(p) for p in points]
    per_point_means: list[float | None] = []
    flat_values: list[float] = []
    for values in encoded_values:
        valid = [float(v) for v in values if v is not None]
        if valid:
            per_point_means.append(float(sum(valid) / len(valid)))
            flat_values.extend(valid)
        else:
            per_point_means.append(None)
    global_mean = float(sum(flat_values) / len(flat_values)) if flat_values else None
    global_std = None
    if len(flat_values) > 1:
        mean = float(global_mean)
        var = sum((x - mean) ** 2 for x in flat_values) / (len(flat_values) - 1)
        global_std = float(math.sqrt(var))

    return {
        "version": SEED_SWEEP_RESULTS_VERSION,
        "saved_at_epoch": int(saved_at_epoch if saved_at_epoch is not None else int(time.time())),
        "settings": dict(settings),
        "points": points_out,
        "per_point_values": encoded_values,
        "summary": {
            "point_count": int(len(points_out)),
            "valid_re_sim_value_count": int(len(flat_values)),
            "per_point_means": per_point_means,
            "global_mean": global_mean,
            "global_std": global_std,
        },
    }


def decode_seed_sweep_results_payload(payload: Any) -> dict:
    """Validate + decode a saved seed-sweep payload."""
    if not isinstance(payload, dict):
        raise ValueError("Seed sweep file is not a JSON object.")
    version = int(payload.get("version", 0) or 0)
    if version <= 0:
        raise ValueError("Seed sweep file missing version.")
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    points = payload.get("points", [])
    if not isinstance(points, list):
        points = []
    raw_values = payload.get("per_point_values", [])
    if not isinstance(raw_values, list):
        raise ValueError("Seed sweep file has invalid per_point_values.")

    decoded_values: list[list[float]] = []
    per_point_means: list[float] = []
    all_values: list[float] = []
    for item in raw_values:
        if not isinstance(item, list):
            item = []
        row_valid: list[float] = []
        for value in item:
            encoded = _encode_numeric_or_none(value)
            if encoded is not None:
                row_valid.append(float(encoded))
                all_values.append(float(encoded))
        decoded_values.append(row_valid)
        if row_valid:
            per_point_means.append(float(sum(row_valid) / len(row_valid)))
        else:
            per_point_means.append(float("nan"))

    return {
        "version": version,
        "saved_at_epoch": int(payload.get("saved_at_epoch", 0) or 0),
        "settings": settings,
        "points": points,
        "per_point_values": decoded_values,
        "per_point_means": per_point_means,
        "all_values": all_values,
    }


def save_seed_sweep_results(path: str, payload: Mapping[str, Any]) -> None:
    """Persist seed-sweep payload to json/json.gz path."""
    _write_seed_sweep_json(path, payload)


def load_seed_sweep_results(path: str) -> dict:
    """Load and decode a seed-sweep payload from disk."""
    payload = _read_seed_sweep_json(path)
    return decode_seed_sweep_results_payload(payload)
