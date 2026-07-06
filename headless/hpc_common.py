"""Shared headless helpers for primary batch campaigns and re-screening."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from gui.common.simulation_settings import (
    BINARY_DEATH_AT_ZERO_ENERGY,
    CONSTANT_DEATH_PROBABILITY,
    CONSTANT_DUPLICATION_PROBABILITY,
    CONSTANT_PROBABILITY,
    NO_DEATH,
    any_constant_probability_mode,
    diffusion_transport_active_from_toggles,
)

_DIFFUSION_CONSTANT = "Diffusion Constant"
_CHEMOSTAT_VOLUME = "Chemostat Volume"
PARALLEL_SHARD_SUBDIR = "_parallel_shards"


def is_primary_batch_session_dir(session_dir: str) -> bool:
    """True when ``session_dir`` looks like a primary batch campaign output folder."""
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    if not os.path.isdir(session_dir):
        return False
    if glob.glob(os.path.join(session_dir, "primary_batch_campaign_*.json*")):
        return True
    if glob.glob(os.path.join(session_dir, "full_save_manifest_*.json*")):
        return True
    if glob.glob(os.path.join(session_dir, "full_save_settings_*.json*")):
        return True
    if glob.glob(os.path.join(session_dir, "primary_final_energies_*.npz")):
        return True
    if glob.glob(os.path.join(session_dir, "primary_population_stats_*.npz")):
        return True
    return os.path.isdir(os.path.join(session_dir, PARALLEL_SHARD_SUBDIR, "primary"))


is_primary_batch_session_folder = is_primary_batch_session_dir


def discover_primary_session_dirs(
    output_root: str,
    *,
    scan_suite_subdirs: bool = False,
) -> List[Tuple[str, str]]:
    """Return sorted ``(session_dir, suite_tag)`` pairs under an output root."""
    root = os.path.abspath(os.path.expanduser(output_root))
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return out

    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out

    for entry in entries:
        if entry.startswith("."):
            continue
        session_dir = os.path.join(root, entry)
        if not os.path.isdir(session_dir):
            continue
        if is_primary_batch_session_dir(session_dir):
            out.append((session_dir, ""))
            continue
        # Layout: Output/<suite>/<session>/ when scan_suite_subdirs is enabled.
        if scan_suite_subdirs:
            try:
                sub_entries = sorted(os.listdir(session_dir))
            except OSError:
                continue
            for sub in sub_entries:
                if sub.startswith("."):
                    continue
                sub_dir = os.path.join(session_dir, sub)
                if os.path.isdir(sub_dir) and is_primary_batch_session_dir(sub_dir):
                    out.append((sub_dir, entry))
    return out


def prune_irrelevant_bounds_for_toggles(
    bounds: Dict[str, Tuple[float, float]],
    toggles: Dict[str, Any],
) -> Dict[str, Tuple[float, float]]:
    """Drop Monte Carlo bound keys unused under the given toggle regime."""
    out = dict(bounds)
    if not diffusion_transport_active_from_toggles(toggles):
        out.pop(_DIFFUSION_CONSTANT, None)
        out.pop(_CHEMOSTAT_VOLUME, None)
    if bool(toggles.get(NO_DEATH)) or bool(toggles.get(BINARY_DEATH_AT_ZERO_ENERGY)) or bool(
        toggles.get(CONSTANT_DEATH_PROBABILITY)
    ):
        out.pop("Death Decay Rate", None)
    if any_constant_probability_mode(toggles):
        out.pop("Death Probability", None)
        out.pop("Duplication Probability", None)
    else:
        out.pop(CONSTANT_PROBABILITY, None)
    if bool(toggles.get(CONSTANT_DUPLICATION_PROBABILITY)):
        out.pop("Duplication Sigmoid Midpoint", None)
        out.pop("Duplication Sigmoid Intensity", None)
    if not bool(toggles.get("Enable Chemostat Flow")):
        out.pop("Flow Percentage", None)
    return out


def effective_process_pool_workers(parallel_workers: int, n_jobs: int) -> int:
    """Process pool size: ``min(parallel_workers, n_jobs, SLURM_CPUS_PER_TASK)``."""
    pw = max(1, int(parallel_workers))
    jobs = max(0, int(n_jobs))
    if jobs <= 0:
        return 1
    workers = min(pw, jobs)
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if cpus > 0:
            workers = min(workers, cpus)
    except Exception:
        pass
    return max(1, workers)
