"""Metric sidecar files on disk and cached metric-input payload helpers."""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Iterable, Optional

import numpy as np

from gui.metrics import compute_metric_from_simulation_result


def metric_name_digest(metric_name: str) -> str:
    """Stable short digest used in sidecar filenames."""
    return hashlib.sha1(str(metric_name).encode("utf-8")).hexdigest()[:12]


def metric_column_sidecar_path(folder: str, session_id: str, metric_name: str) -> str:
    """Path to full metric-column sidecar file."""
    return os.path.join(
        folder,
        f"full_save_metric_column_{session_id}_{metric_name_digest(metric_name)}.json.gz",
    )


def metric_column_shard_path(folder: str, session_id: str, metric_name: str, batch_base: str) -> str:
    """Path to per-batch metric-column shard sidecar file."""
    batch_digest = hashlib.sha1(str(batch_base).encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        folder,
        (
            "full_save_metric_column_shard_"
            f"{session_id}_{metric_name_digest(metric_name)}_{batch_digest}.json.gz"
        ),
    )


def extract_metric_digest_from_sidecar_filename(filename: str) -> Optional[str]:
    """Extract metric digest from a full (non-shard) metric sidecar filename."""
    base = os.path.basename(str(filename))
    if not base.startswith("full_save_metric_column_"):
        return None
    if base.startswith("full_save_metric_column_shard_"):
        return None
    if not base.endswith(".json.gz"):
        return None
    stem = base[:-8]
    digest = stem.rsplit("_", 1)[-1]
    if len(digest) != 12:
        return None
    return digest


def encode_metric_sidecar_values(values: Iterable[Any]) -> list[Optional[float]]:
    """Normalize metric values for JSON sidecar payload (NaN/invalid -> None)."""
    encoded: list[Optional[float]] = []
    for value in values:
        if value is None:
            encoded.append(None)
            continue
        try:
            numeric = float(value)
            if math.isnan(numeric):
                encoded.append(None)
            else:
                encoded.append(numeric)
        except Exception:
            encoded.append(None)
    return encoded


def decode_metric_sidecar_values(
    payload: Any,
    expected_len: int,
    *,
    missing_value: Any,
) -> Optional[list[Any]]:
    """Decode sidecar payload into list of numbers with caller-selected missing marker."""
    if not isinstance(payload, dict):
        return None
    values = payload.get("values", [])
    expected = int(expected_len)
    if not isinstance(values, list) or len(values) != expected:
        return None
    out = [missing_value] * expected
    for i, value in enumerate(values):
        if value is None:
            out[i] = missing_value
            continue
        try:
            out[i] = float(value)
        except Exception:
            out[i] = missing_value
    return out


def build_metric_sidecar_payload(
    *,
    session_id: str,
    metric_name: str,
    values: Iterable[Any],
    saved_at_epoch: int,
    batch_path: str | None = None,
) -> dict:
    """Construct canonical metric sidecar payload schema."""
    encoded = encode_metric_sidecar_values(values)
    payload = {
        "session_id": str(session_id),
        "metric_name": str(metric_name),
        "num_points": int(len(encoded)),
        "saved_at_epoch": int(saved_at_epoch),
        "values": encoded,
    }
    if batch_path is not None:
        payload["batch_path"] = str(batch_path)
    return payload


def normalize_metric_input(metric_input: Any) -> dict | None:
    """Normalize metric-input payload shapes (optional single-element list wrapper) to a dict."""
    while isinstance(metric_input, list) and len(metric_input) == 1:
        metric_input = metric_input[0]
    if isinstance(metric_input, dict):
        return metric_input
    if isinstance(metric_input, list):
        for item in metric_input:
            if isinstance(item, dict):
                return item
    return None


def compute_metric_from_cached_input(metric_input: Any, metric_name: str) -> float:
    """Compute a metric from cached simulation output without rerunning simulation."""
    rec = normalize_metric_input(metric_input)
    if not rec or not isinstance(rec, dict):
        return np.nan
    if bool(rec.get("collapsed", False)):
        return np.nan
    try:
        return float(compute_metric_from_simulation_result(rec, metric_name))
    except Exception:
        return np.nan
