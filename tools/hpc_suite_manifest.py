"""Minimal HPC suite manifest helpers used by re-screening."""

from __future__ import annotations

import re
from typing import Optional, Tuple

COMPARE_CONFIG_ORDER: Tuple[str, ...] = (
    "trueNeutral",
    "trueNeutral2",
    "justDeath",
    "justDup",
    "Death+Dup",
)

_LEGACY_PRIMARY_STEM_RE = re.compile(r"^primary_\d{8}_(.+)$", re.IGNORECASE)
_LETTER_PRIMARY_STEM_RE = re.compile(r"^[a-e]_(.+)$", re.IGNORECASE)
_OPTIONAL_PRIMARY_STEM_RE = re.compile(r"^aa_(.+)$", re.IGNORECASE)
_FIXED_STEM_RE = re.compile(r"^(?P<base>.+)_Fixed_(?P<ratio>\d+(?:\.\d+)?)$")


def strip_primary_json_prefix(stem: str) -> str:
    s = str(stem or "").strip()
    for rx in (_LEGACY_PRIMARY_STEM_RE, _LETTER_PRIMARY_STEM_RE, _OPTIONAL_PRIMARY_STEM_RE):
        m = rx.match(s)
        if m:
            return m.group(1).strip()
    return s


def primary_job_key(stem: str) -> str:
    """Map job stems (incl. ``a_*_Fixed_N`` / legacy ``primary_*``) to profile keys."""
    m = _FIXED_STEM_RE.match(stem)
    base = m.group("base") if m else stem
    return strip_primary_json_prefix(base)
