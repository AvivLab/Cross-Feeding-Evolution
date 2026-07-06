"""Session metadata helpers for primary batch campaigns and re-screening."""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, Optional, Tuple

from gui.persistence.json_io import make_read_json_maybe_gz_fn

_COMPARE_CONFIG_ORDER: Tuple[str, ...] = (
    "trueNeutral",
    "trueNeutral2",
    "justDeath",
    "justDup",
    "Death+Dup",
)
_FIXED_LABEL_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_Fixed_\d+(?:\.\d+)?$")
_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)


def _plot_label_config_key(label: str) -> str:
    base = str(label or "").split(" [", 1)[0].strip()
    base = base.replace("Death-plus-Dup", "Death+Dup")
    m = _FIXED_LABEL_SUFFIX_RE.match(base)
    if m:
        base = m.group("base")
    for cfg in _COMPARE_CONFIG_ORDER:
        if base == cfg or base.startswith(f"{cfg}_"):
            return cfg
    if "Death" in base and "Dup" in base:
        return "Death+Dup"
    return base


def _read_summary(path: str) -> Optional[Dict[str, Any]]:
    try:
        data = _read_json_maybe_gz(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _session_id_from_summary(data: Dict[str, Any], session_dir: str) -> str:
    sid = str(data.get("full_save_session_id") or "").strip()
    if sid:
        return sid
    summaries = glob.glob(os.path.join(session_dir, "primary_batch_campaign_*.json*"))
    if summaries:
        base = os.path.basename(summaries[0])
        if base.startswith("primary_batch_campaign_"):
            stem = base[len("primary_batch_campaign_") :]
            if stem.endswith(".json.gz"):
                return stem[: -8]
            if stem.endswith(".json"):
                return stem[: -5]
            return stem
    return os.path.basename(session_dir.rstrip("/"))


def _label_from_summary(
    data: Dict[str, Any],
    session_dir: str,
    *,
    suite_tag: str = "",
) -> str:
    stem = data.get("headless_primary_job_file_stem")
    if isinstance(stem, str) and stem.strip():
        label = stem.strip()
    else:
        sid = str(data.get("full_save_session_id") or "").strip()
        m = re.match(r"^primary_\d{8}_(.+)$", sid, re.IGNORECASE)
        if m:
            label = m.group(1).strip()
        else:
            m = re.match(r"^[a-e]_(.+)$", sid, re.IGNORECASE)
            if m:
                label = m.group(1).strip()
            else:
                m = re.match(r"^aa_(.+)$", sid, re.IGNORECASE)
                if m:
                    label = m.group(1).strip()
                else:
                    label = os.path.basename(session_dir.rstrip("/"))
    tag = str(suite_tag or "").strip()
    if tag:
        return f"{label} [{tag}]"
    return label


def _sid_from_sidecar_glob(session_dir: str, prefix: str) -> str:
    hits = sorted(glob.glob(os.path.join(session_dir, f"{prefix}*.npz")))
    if not hits:
        return ""
    base = os.path.basename(hits[0])
    stem = base[: -4] if base.endswith(".npz") else base
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return ""


def _infer_label_from_session_dir(session_dir: str, *, suite_tag: str = "") -> str:
    base = os.path.basename(session_dir.rstrip("/"))
    m = re.match(r"^job\d+-\d{8}-\d{6}_(.+)$", base)
    if not m:
        m = re.match(r"^\d{8}-\d{6}_(.+)$", base)
    label = m.group(1).strip() if m else base
    tag = str(suite_tag or "").strip()
    if tag:
        return f"{label} [{tag}]"
    return label


def _resolve_session_meta(
    session_dir: str,
    suite_tag: str,
) -> Optional[Tuple[str, str, Optional[Dict[str, Any]]]]:
    """Return ``(session_id, plot_label, summary_dict_or_none)`` for a session folder."""
    summaries = sorted(glob.glob(os.path.join(session_dir, "primary_batch_campaign_*.json*")))
    data: Optional[Dict[str, Any]] = None
    if summaries:
        data = _read_summary(summaries[0])
    if data:
        sid = _session_id_from_summary(data, session_dir)
        label = _label_from_summary(data, session_dir, suite_tag=suite_tag)
        return sid, label, data
    sid = _sid_from_sidecar_glob(session_dir, "primary_final_energies_")
    if not sid:
        sid = _sid_from_sidecar_glob(session_dir, "primary_population_stats_")
    if not sid:
        return None
    label = _infer_label_from_session_dir(session_dir, suite_tag=suite_tag)
    return sid, label, None


def load_campaign_summary(session_dir: str) -> Optional[Dict[str, Any]]:
    """Load the first ``primary_batch_campaign_*.json*`` in a session folder."""
    summaries = sorted(glob.glob(os.path.join(session_dir, "primary_batch_campaign_*.json*")))
    if not summaries:
        return None
    return _read_summary(summaries[0])
