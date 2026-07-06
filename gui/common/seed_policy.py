"""Shared seed policy helpers for optimization and GUI handoff."""

from __future__ import annotations

from typing import Any, Optional


def parse_optional_seed(value: Any) -> Optional[int]:
    """Parse an optional seed value to int, or None for blank/None."""
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    return int(value)


def descent_seed_for_run(base_seed: Optional[int], start_idx: int, descent_idx: int) -> Optional[int]:
    """
    Seed policy for each descent run.

    Current policy keeps the same seed across starts/descents.
    """
    return base_seed


def replicate_seed_for_run(run_seed: Optional[int], rep_idx: int, num_replicates: int) -> Optional[int]:
    """
    Seed policy for each replicate within one run.

    - Single replicate: keep seed unchanged.
    - Multiple replicates: offset by replicate index.
    """
    if run_seed is None:
        return None
    return int(run_seed + rep_idx) if int(num_replicates) > 1 else int(run_seed)


def choose_launch_seed(
    descent_seed: Any,
    metric_input_seed: Any,
    derived_seed: Any,
    fallback_seed_text: str,
) -> str:
    """
    Choose seed text for launching an individual run from stored data.

    Priority:
    1) per-point metric_input random_seed_used
    2) per-run descent seed
    3) derived seed from run index policy
    4) fallback seed text (snapshot/UI)
    """
    for candidate in (metric_input_seed, descent_seed, derived_seed):
        if candidate in (None, ""):
            continue
        try:
            return str(int(candidate))
        except Exception:
            return str(candidate)
    txt = (fallback_seed_text or "").strip()
    return txt

