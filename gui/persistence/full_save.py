"""Canonical full-save directory layout: paths, discovery, manifest batch ordering."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Set, Tuple

from gui.persistence.offload_reconstruct import parse_offload_filename

# Manifest filenames produced with timestamp session ids (folder discovery).
FULL_SAVE_MANIFEST_TIMESTAMP_RE = re.compile(
    r"^full_save_manifest_(\d{8}-\d{6})\.json(?:\.gz)?$"
)


def full_save_manifest_path_json(folder: str, session_id: str) -> str:
    return os.path.join(folder, f"full_save_manifest_{session_id}.json")


def full_save_manifest_path_gz(folder: str, session_id: str) -> str:
    return full_save_manifest_path_json(folder, session_id) + ".gz"


def load_full_save_manifest(
    folder: str,
    session_id: str,
    read_json: Callable[[str], Any],
) -> Dict[str, Any]:
    """
    Load ``full_save_manifest_<session>.json`` or ``...json.gz`` using ``read_json``.

    Prefers the plain ``.json`` path when both exist. On missing paths or read errors,
    returns ``{"session_id": session_id, "batches": []}`` with a normalized ``batches``
    list when a dict payload is loaded.
    """
    path_json = full_save_manifest_path_json(folder, session_id)
    path_gz = full_save_manifest_path_gz(folder, session_id)
    data = None
    try:
        if os.path.exists(path_json):
            data = read_json(path_json)
        elif os.path.exists(path_gz):
            data = read_json(path_gz)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {"session_id": session_id, "batches": []}
    if "batches" not in data or not isinstance(data["batches"], list):
        data["batches"] = []
    data.setdefault("session_id", session_id)
    return data


def full_save_settings_path_json(folder: str, session_id: str) -> str:
    return os.path.join(folder, f"full_save_settings_{session_id}.json")


def full_save_settings_path_gz(folder: str, session_id: str) -> str:
    return full_save_settings_path_json(folder, session_id) + ".gz"


def full_save_dataset_snapshot_path(folder: str, session_id: str) -> str:
    return os.path.join(folder, f"full_save_dataset_{session_id}.json.gz")


def is_full_save_manifest_filename(filename: str) -> bool:
    base = os.path.basename(str(filename or ""))
    return bool(
        base.startswith("full_save_manifest_")
        and (base.endswith(".json") or base.endswith(".json.gz"))
    )


def is_full_save_dataset_snapshot_basename(base_name: str) -> bool:
    b = str(base_name or "")
    return b.startswith("full_save_dataset_") and b.endswith(".json.gz")


def discover_full_save_session_ids(folder: str) -> Set[str]:
    """Session ids from manifest timestamps and offload batch filenames under ``folder``."""
    session_ids: Set[str] = set()
    for fn in os.listdir(folder):
        mm = FULL_SAVE_MANIFEST_TIMESTAMP_RE.match(fn)
        if mm:
            session_ids.add(mm.group(1))
            continue
        offload_session_id, _ = parse_offload_filename(fn)
        if offload_session_id:
            session_ids.add(offload_session_id)
    return session_ids


def manifest_batch_entries_ordered(manifest: Any) -> List[Tuple[int, str, int]]:
    """
    Manifest ``batches`` entries as ``(batch_index, rel_path, num_records)``, sorted by index.

    Invalid/missing indices sort last (``10**12``). Skips entries without a non-empty string path.
    """
    batches = manifest.get("batches", []) if isinstance(manifest, dict) else []
    ordered: List[Tuple[int, str, int]] = []
    for b in batches:
        if not isinstance(b, dict):
            continue
        rel = b.get("path")
        if not isinstance(rel, str) or not rel:
            continue
        try:
            bidx = int(b.get("batch_index", 10**12))
        except Exception:
            bidx = 10**12
        try:
            nrec = int(b.get("num_records", 0) or 0)
        except Exception:
            nrec = 0
        ordered.append((bidx, rel, nrec))
    ordered.sort(key=lambda x: x[0])
    return ordered


def ordered_existing_batch_paths_from_manifest(folder: str, manifest: Any) -> List[str]:
    """Absolute paths for manifest batch files that exist on disk, in batch_index order."""
    out: List[str] = []
    for _, rel, _ in manifest_batch_entries_ordered(manifest):
        p = os.path.join(folder, rel)
        if os.path.exists(p):
            out.append(p)
    return out


def manifest_declared_batch_path_count(manifest: Any) -> int:
    """How many batch entries have a usable relative path (existence not checked)."""
    return len(manifest_batch_entries_ordered(manifest))
