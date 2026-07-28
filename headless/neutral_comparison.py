# MCCM_HEADLESS_NEUTRAL_FILE_VERSION=1.5.10
"""
Headless Neutral Set Comparison (Monte Carlo primary vs neutral batches).

Load a job description from the same JSON shape written by the standalone GUI
(``kind: neutral_set_comparison``), then write a new session under an output
folder: ``full_save_manifest_*``, ``full_save_settings_*``, gzipped offload
batches, and ``neutral_set_comparison_<session>.json``.

Typical workflow: configure and save once from the GUI locally, copy the JSON
to the cluster, point ``--output-dir`` at scratch storage, and run::

    cd /path/to/MCCP_Enzymes
    python3 -m headless.neutral_comparison /path/job.json --output-dir /scratch/out

The first stdout line reports ``version=`` (``NEUTRAL_HEADLESS_VERSION``) and the runner path.
The same version is stored in session settings and the summary JSON.

Progress lines (timestamped) go to stdout (Slurm ``.out``) and, by default, to
``<output_dir>/neutral_progress.log`` unless ``--no-progress-log`` is set. Override the path with
``--progress-log`` or env ``NEUTRAL_PROGRESS_LOG``. If ``--progress-every`` is 0 but a log file is
used, an interval is chosen automatically from ``n_runs``.

HPC-style layout: ``--output-root /path/to/Output`` creates a subfolder. If the logical
``session_id`` matches ``neutral_YYYYMMDD_<name>`` (from ``--session-id`` or a
``neutral_YYYYMMDD_<name>.json`` stem), the directory is
``/path/to/Output/<time>_<name>/`` (only the final ``<name>`` segment, e.g. ``justDeath``), not the
full stem. For other session ids, the directory is ``/path/to/Output/<time>_<session_id>/``. The
``<time>`` segment is `YYYYMMDD-HHMMSS` at process start, or ``NEUTRAL_SESSION_FOLDER_STAMP`` (set the
same in every Slurm task so all ranks use one folder). If there is no session id, the folder is
``/path/to/Output/<YYYYMMDD-HHMMSS>/`` or ``/path/to/Output/<NEUTRAL_SESSION_FOLDER_STAMP>/``. On-disk
file names (manifest, summary, histogram) still use the full logical session id
(e.g. ``neutral_20260423_justDeath``), not the short folder label. The job
JSON's ``save_folder`` is ignored when ``--output-root`` or ``--output-dir`` is passed.
Unless ``--no-progress-log`` is set, progress defaults to ``<output_dir>/neutral_progress.log``
(``--progress-log`` / ``NEUTRAL_PROGRESS_LOG`` still override). Use ``PYTHONUNBUFFERED=1`` in Slurm
if your site buffers batch stdout despite line flushing.

For hangs with empty Slurm ``.out``, set ``NEUTRAL_HEADLESS_TRACE=1`` or pass ``--debug-trace``: lines go to
stderr (often ``.err`` or merged into ``.out``) and to ``<output_dir>/headless_debug_trace.log`` with
monotonic timestamps between major steps.

Parallel HPC: use ``--parallel-neutral-workers N`` (or env ``NEUTRAL_PARALLEL_NEUTRAL_WORKERS``). When
``N > 1``, the **primary** batch and **all neutral** batches run **at the same time** (up to
``min(N, n_neutral_batches + 1)`` worker processes). Each worker streams offload to its own folder
under ``_parallel_shards/<primary|neutral_r>/`` (periodic gzip batches, bounded RAM), then the parent
merges those shards into the usual session-root manifest and ``offload_<session>_*.json.gz`` files so
reanalysis matches the serial layout. When ``N == 1``, the job stays fully serial (primary, then each
neutral batch). Match Slurm ``--cpus-per-task`` to ``N`` (or higher) so processes are not oversubscribed on one CPU.

**Multi-node Slurm:** use ``--multi-node-index I`` and ``--multi-node-count T`` together (same job JSON
and **same** ``--session-id`` / output folder on **shared** storage). Neutral batch indices are split across
ranks; rank ``0`` runs the primary batch plus its neutral slice, other ranks run their neutral slices only.
Each rank may use ``--parallel-neutral-workers`` / ``NEUTRAL_PARALLEL_NEUTRAL_WORKERS`` up to that node's
``SLURM_CPUS_PER_TASK``. Rank ``0`` waits for all ranks' ``.done`` markers, merges shards, and writes the
summary JSON and histogram. Requires a fixed session id: ``--session-id``, or a job file whose basename
matches ``neutral_YYYYMMDD_<name>.json`` (same id inferred for every rank).

After a successful run, writes ``neutral_hit_histogram_<session_id>.png``,
``neutral_primary_events_<session_id>.png`` (mean deaths / duplications / mutations per primary
simulation for metric hits vs misses), and ``neutral_param_heatmap_<session_id>.png`` (1D bin
counts of threshold-crossing primary runs in ``primary_bounds`` parameter space) when matplotlib is
installed; use ``--no-histogram-png`` to skip the histogram PNG only.

**Python version:** use **3.9+** (same as ``requirements.txt`` / numba). Do not use the
cluster default ``python3`` if it is 3.6; load a newer module and call that interpreter
(see your site's ``module avail python``). On some HPC installs a given ``python/3.11.x`` module may
lack ``_ssl``; if ``import ssl`` fails, use another module (e.g. 3.12) and recreate your venv.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gui.metrics import filter_metric_options_for_simulation_settings
from gui.persistence.full_save import full_save_manifest_path_json, full_save_settings_path_json
from gui.persistence.json_io import make_read_json_maybe_gz_fn, make_write_json_maybe_gz_atomic_fn
from gui.models.registry import get_model_by_key
from gui.apps.neutral_comparison.offload import (
    NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR,
    NeutralComparisonOffloadWriter,
    count_shard_offload_batches,
    merge_parallel_neutral_comparison_shards_into_writer,
)
from gui.apps.neutral_comparison.batch import run_hit_count_batch, simulation_light_tracking_plan
from gui.apps.neutral_comparison.primary_event_chart import (
    load_primary_offload_records,
    summarize_primary_events_by_hit,
    write_primary_event_rates_png,
)
from gui.apps.neutral_comparison.parameter_heatmap import (
    load_param_names_from_offload,
    write_parameter_heatmap_png,
)
from gui.common.simulation_settings import normalize_simulation_params, prune_irrelevant_numeric_parameters_for_export

_write_results_json = make_write_json_maybe_gz_atomic_fn(indent=2)
_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)

METRIC_NONE_LABEL = "(No metric)"

# Headless runner only (not the GUI job JSON schema ``version`` field). Bump when behavior or
# summary/session JSON fields change so HPC logs and artifacts identify the code revision.
NEUTRAL_HEADLESS_VERSION = "1.5.11"


def _normalized_job_settings_json_path(raw: Optional[Any]) -> Optional[str]:
    """Absolute path for manifest / worker payloads, or None if unset."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return os.path.abspath(os.path.expanduser(s))


def _job_json_stem_for_filename_hint(config_path: str) -> str:
    """File stem: strip ``.json`` or ``.json.gz`` from basename (for neutral_* session hints)."""
    base = os.path.basename((config_path or "").replace("\r", "").replace("\n", "").strip())
    if not base:
        return ""
    lo = base.lower()
    if lo.endswith(".json.gz"):
        return base[: -8]
    if lo.endswith(".json"):
        return base[: -5]
    return base


def neutral_job_filename_session_id(stem: str) -> Optional[str]:
    """
    If the job file stem matches ``neutral_YYYYMMDD_<name>``, return that stem as the session id.

    ``<name>`` is any non-empty suffix (often camelCase or snake_case) without path separators.
    The ``neutral_`` prefix is matched case-insensitively; the returned id preserves the file stem.
    """
    s = (stem or "").strip()
    if not s or "/" in s or "\\" in s:
        return None
    m = re.match(r"^neutral_(\d{8})_(.+)$", s, re.IGNORECASE)
    if not m:
        return None
    tail = m.group(2).strip()
    if not tail:
        return None
    for ch in tail:
        if ch in "/\\\0:":
            return None
    return s


def neutral_session_id_output_root_folder_suffix(session_id: str) -> Optional[str]:
    """
    For ``session_id`` matching ``neutral_YYYYMMDD_<name>``, return ``<name>`` for a shorter
    ``--output-root`` subfolder. Otherwise return ``None`` (caller should use the full session id).
    """
    s = (session_id or "").strip()
    if not s:
        return None
    m = re.match(r"^neutral_(\d{8})_(.+)$", s, re.IGNORECASE)
    if not m:
        return None
    name = m.group(2).strip()
    if not name:
        return None
    for ch in name:
        if ch in "/\\\0:":
            return None
    return name


def _output_root_time_prefix() -> str:
    """Folder name time segment under ``--output-root``; override for multi-rank same folder."""
    v = (os.environ.get("NEUTRAL_SESSION_FOLDER_STAMP") or os.environ.get("NEUTRAL_OUTPUT_STAMP") or "").strip()
    if v:
        return v
    return time.strftime("%Y%m%d-%H%M%S")


# Serialize prints + optional progress file writes (Slurm .out + heartbeat thread in parallel mode).
_EMIT_LOCK = threading.Lock()
_TRACE_LOCK = threading.Lock()
# When enabled (``--debug-trace`` or env ``NEUTRAL_HEADLESS_TRACE=1``): monotonic timestamps to stderr
# and to ``<output_dir>/headless_debug_trace.log`` once the output dir is known.
_trace_cfg: Dict[str, Any] = {"on": False, "t0": None, "fp": None, "path": "", "main_pid": None}


def _trace(msg: str) -> None:
    if not _trace_cfg.get("on"):
        return
    if _trace_cfg.get("t0") is None:
        _trace_cfg["t0"] = time.monotonic()
    t0 = float(_trace_cfg["t0"])
    dt = time.monotonic() - t0
    wall = datetime.now().isoformat(timespec="seconds")
    line = f"[headless trace +{dt:08.3f}s wall={wall}] {msg}\n"
    with _TRACE_LOCK:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass
        fp = _trace_cfg.get("fp")
        mpid = _trace_cfg.get("main_pid")
        if fp is not None and (mpid is None or os.getpid() == int(mpid)):
            try:
                fp.write(line)
                fp.flush()
            except Exception:
                pass


def _trace_open_out_dir(out_dir: str) -> None:
    if not _trace_cfg.get("on"):
        return
    with _TRACE_LOCK:
        old = _trace_cfg.get("fp")
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
            _trace_cfg["fp"] = None
        try:
            folder = os.path.abspath(os.path.expanduser(str(out_dir).strip()))
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "headless_debug_trace.log")
            _trace_cfg["fp"] = open(path, "w", encoding="utf-8", buffering=1)
            _trace_cfg["path"] = path
            _trace_cfg["main_pid"] = int(os.getpid())
        except Exception as exc:
            _trace_cfg["fp"] = None
            _trace_cfg["path"] = ""
            try:
                sys.stderr.write(f"[headless trace] could not open trace log in out_dir: {exc!r}\n")
                sys.stderr.flush()
            except Exception:
                pass
    _trace(f"trace log file={_trace_cfg.get('path')!r} (also mirrored on stderr)")


def _trace_close() -> None:
    if not _trace_cfg.get("on"):
        return
    with _TRACE_LOCK:
        fp = _trace_cfg.get("fp")
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass
            _trace_cfg["fp"] = None


def _configure_stdio_for_job() -> None:
    """Slurm/cron often attach non-TTY stdout; prefer line buffering so .out updates promptly."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "isatty") and not stream.isatty():
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def _metric_slot_active(name: str) -> bool:
    s = (name or "").strip()
    return bool(s) and s != METRIC_NONE_LABEL


def _bounds_from_save_json(bounds_json: Any, *, label: str) -> Dict[str, Tuple[float, float]]:
    if not isinstance(bounds_json, dict):
        raise ValueError(f"{label} must be a JSON object mapping parameter names to [min, max].")
    out: Dict[str, Tuple[float, float]] = {}
    for pname, raw_pair in bounds_json.items():
        key = str(pname)
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) < 2:
            raise ValueError(f"Invalid bounds entry for {key!r} in {label}.")
        lo_raw, hi_raw = raw_pair[0], raw_pair[1]
        if lo_raw in ("-inf", "-Infinity", None):
            lo = float("-inf")
        else:
            try:
                lo = float(lo_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid min for {key!r} in {label}.") from exc
        try:
            hi = float(hi_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid max for {key!r} in {label}.") from exc
        if lo != float("-inf") and not np.isfinite(lo):
            raise ValueError(f"Invalid min for {key!r} in {label}.")
        if not np.isfinite(hi):
            raise ValueError(f"Invalid max for {key!r} in {label}.")
        if lo != float("-inf") and lo >= hi:
            raise ValueError(f"min must be < max for {key!r} in {label} (got {lo}, {hi}).")
        out[key] = (lo, hi)
    return out


def _jsonable_value(v: Any) -> Any:
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return {str(kk): _jsonable_value(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable_value(x) for x in v]
    if isinstance(v, int):
        return int(v)
    try:
        if hasattr(v, "item"):
            return _jsonable_value(v.item())
    except Exception:
        pass
    try:
        x = float(v)
        if x != x:
            return None
        return x
    except Exception:
        return str(v)


def _jsonable_params(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _jsonable_value(v) for k, v in d.items()}


def _bounds_jsonable(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for k, (lo, hi) in bounds.items():
        lo_j: Any = "-inf" if lo == float("-inf") else float(lo)
        out[str(k)] = [lo_j, float(hi)]
    return out


def _metric_filter_row(label: str, mname: str, op: str, thr: float) -> Dict[str, Any]:
    active = label == "A" or _metric_slot_active(mname)
    return {
        "label": label,
        "active": active,
        "metric": mname if active else None,
        "operator": str(op) if active else None,
        "threshold": float(thr) if active else None,
    }


def _metric_checks_from_filters(metric_filters: Any) -> Tuple[List[Tuple[str, str, float]], List[Dict[str, Any]]]:
    if not isinstance(metric_filters, list) or not metric_filters:
        raise ValueError("metric_filters must be a non-empty list (same format as GUI saves).")
    by_label = {str(r.get("label", "")): r for r in metric_filters if isinstance(r, dict) and r.get("label")}
    checks: List[Tuple[str, str, float]] = []
    snapshot: List[Dict[str, Any]] = []
    slots = [
        ("A", by_label.get("A")),
        ("B", by_label.get("B")),
        ("C", by_label.get("C")),
        ("D", by_label.get("D")),
    ]
    for lab, row in slots:
        if row is None:
            if lab == "A":
                raise ValueError("metric_filters must include label 'A'.")
            mname = METRIC_NONE_LABEL
            op = ">"
            thr = 0.0
            snapshot.append(_metric_filter_row(lab, mname, op, thr))
            continue
        active = bool(row.get("active", row.get("metric") is not None))
        mval = row.get("metric")
        if lab != "A" and (not active or not mval):
            mname = METRIC_NONE_LABEL
            op = str(row.get("operator") or ">")
            thr = 0.0
            snapshot.append(_metric_filter_row(lab, mname, op, thr))
            continue
        if lab == "A" and not mval:
            raise ValueError("Metric A must be a non-empty name in metric_filters.")
        mname = str(mval)
        op = str(row.get("operator") or ">")
        th = row.get("threshold")
        if th is None:
            raise ValueError(f"metric_filters row {lab!r} missing threshold.")
        thr = float(th)
        snapshot.append(_metric_filter_row(lab, mname, op, thr))
        if lab == "A" or _metric_slot_active(mname):
            checks.append((mname, op, thr))
    if not checks:
        raise ValueError("No active metric filters (metric A is required).")
    return checks, snapshot


def _allowed_metrics_for_side(
    model_spec,
    numeric: Dict[str, Any],
    toggles: Dict[str, Any],
) -> set:
    merged = {**numeric, **toggles}
    m2 = bool(merged.get("Enable M2 Diffusion", True))
    m1 = bool(merged.get("Enable M1 Diffusion", False))
    ic = bool(merged.get("Enable Intermediate Costs", False))
    names = list(getattr(model_spec, "metric_names", []) or [])
    filtered = filter_metric_options_for_simulation_settings(
        names,
        enable_m2_diffusion=m2,
        enable_m1_diffusion=m1,
        enable_intermediate_costs=ic,
    )
    return set(filtered)


def _comparison_batch_worker(payload: Dict[str, Any]) -> Tuple[str, Optional[int], int]:
    """
    Child-process entry: one primary or one neutral Monte Carlo batch.

    Returns ``(role, batch_index, hits)``. Offload rows are written under ``shard_offload_folder``
    via ``NeutralComparisonOffloadWriter`` (streaming flushes); the parent merges shards in order.
    """
    from gui.models.registry import get_model_by_key
    from gui.apps.neutral_comparison.batch import run_hit_count_batch

    role = str(payload.get("role", "neutral"))
    model_spec = get_model_by_key(str(payload["model_key"]))
    metric_checks = [(str(t[0]), str(t[1]), float(t[2])) for t in payload["metric_checks"]]
    shard_folder = str(payload.get("shard_offload_folder") or "").strip()
    if not shard_folder:
        raise ValueError("internal: parallel batch worker requires shard_offload_folder in payload")
    sid = str(payload.get("session_id") or "").strip()
    if not sid:
        raise ValueError("internal: parallel batch worker requires session_id in payload")
    if role == "primary":
        bounds = payload["primary_bounds"]
        numeric_base = payload["numeric_base"]
        toggles = payload["toggles"]
        stage = "primary"
        batch_index: Optional[int] = None
    else:
        bounds = payload["neutral_bounds"]
        numeric_base = payload["numeric_base"]
        toggles = payload["toggles"]
        stage = str(payload["offload_stage"])
        batch_index = int(payload["r"])
    if not isinstance(bounds, dict):
        raise TypeError("bounds must be a dict")
    pe = max(0, int(payload.get("progress_every") or 0))
    plab = str(payload.get("progress_label") or stage)
    hv = str(payload.get("headless_runner_version") or "").strip()
    if hv:
        print(
            f"[{plab}] headless_version={hv} n_runs={int(payload['n_runs'])} worker_pid={os.getpid()}",
            flush=True,
        )
    prog_cb = None
    if pe > 0:
        last_t = [0.0]

        def _worker_progress(done: int, total: int, hits: int) -> None:
            now = time.time()
            if done < total and done % int(pe) != 0 and (now - last_t[0]) < 30.0:
                return
            last_t[0] = now
            # Child stdout → same Slurm .out as parent (not the parent's progress_log file).
            print(f"[{plab}] {done}/{total} simulations, hits so far: {hits}", flush=True)

        prog_cb = _worker_progress

    jp = _normalized_job_settings_json_path(payload.get("job_settings_json_path"))
    shard_writer = NeutralComparisonOffloadWriter(
        shard_folder,
        sid,
        model_key=str(payload.get("model_key") or "simulation"),
        param_names_list=list(payload["param_names_list"]),
        metric_name_at_offload=str(payload["offload_metric_name"]),
        job_settings_json_path=jp,
    )
    hits = run_hit_count_batch(
        model_spec=model_spec,
        n_runs=int(payload["n_runs"]),
        bounds=bounds,
        numeric_base=numeric_base,
        toggles=toggles,
        metric_checks=metric_checks,
        base_seed=payload.get("base_seed"),
        progress_callback=prog_cb,
        offload_writer=shard_writer,
        offload_stage=stage,
    )
    shard_writer.finalize()
    return (role, batch_index, int(hits))


def _multi_node_neutral_batch_range(n_neutral: int, rank: int, n_tasks: int) -> Tuple[int, int]:
    """Inclusive-exclusive neutral batch indices [lo, hi) owned by this Slurm task rank."""
    nn = int(n_neutral)
    r = int(rank)
    t = int(n_tasks)
    lo = (nn * r) // t
    hi = (nn * (r + 1)) // t
    return lo, hi


def _multi_node_done_path(folder: str, rank: int) -> str:
    return os.path.join(folder, f"multi_node_rank_{int(rank)}.done")


def _multi_node_hits_path(folder: str, rank: int) -> str:
    return os.path.join(folder, f"multi_node_rank_{int(rank)}_hits.json")


def _multi_node_wait_for_settings_json(folder: str, sid: str, *, emit, timeout_s: float = 86400.0) -> None:
    """Non-zero ranks wait until rank 0 has written full_save_settings (shared filesystem)."""
    path = full_save_settings_path_json(folder, sid)
    t0 = time.monotonic()
    last_emit = 0.0
    while time.monotonic() - t0 < timeout_s:
        if os.path.isfile(path):
            return
        now = time.monotonic()
        if now - last_emit > 120.0:
            emit(f"[multi-node] waiting for {os.path.basename(path)} … ({int(now - t0)}s)")
            last_emit = now
        time.sleep(2.0)
    raise TimeoutError(f"Multi-node: settings JSON never appeared: {path!r}")


def _multi_node_wait_all_dones(folder: str, n_tasks: int, *, emit, timeout_s: float = 86400.0 * 7) -> None:
    """Rank 0 waits until every rank has touched its .done marker."""
    emit(f"[multi-node] waiting for all {int(n_tasks)} rank(s) to finish and write .done markers…")
    t0 = time.monotonic()
    last_emit = 0.0
    while time.monotonic() - t0 < timeout_s:
        ok = all(os.path.isfile(_multi_node_done_path(folder, i)) for i in range(int(n_tasks)))
        if ok:
            return
        now = time.monotonic()
        if now - last_emit > 300.0:
            missing = [i for i in range(int(n_tasks)) if not os.path.isfile(_multi_node_done_path(folder, i))]
            emit(f"[multi-node] waiting for ranks {missing} to finish … ({int(now - t0)}s)")
            last_emit = now
        time.sleep(5.0)
    raise TimeoutError("Multi-node: timeout waiting for all rank .done markers.")


def _multi_node_write_hits_json(
    folder: str,
    rank: int,
    hits_primary: Optional[int],
    neutral_hits_by_r: Dict[int, int],
) -> None:
    payload: Dict[str, Any] = {
        "rank": int(rank),
        "hits_primary": int(hits_primary) if hits_primary is not None else None,
        "neutral_hits_by_r": {str(k): int(v) for k, v in sorted(neutral_hits_by_r.items())},
    }
    _write_results_json(_multi_node_hits_path(folder, rank), payload)


def _multi_node_load_merged_hits(folder: str, n_tasks: int, n_neutral: int) -> Tuple[int, List[int]]:
    """Combine per-rank hit JSON into primary hit count and full neutral_counts list."""
    primary: Optional[int] = None
    neutral_counts = [0] * int(n_neutral)
    for i in range(int(n_tasks)):
        hp = _multi_node_hits_path(folder, i)
        data = _read_json_maybe_gz(hp)
        if not isinstance(data, dict):
            raise RuntimeError(f"Multi-node: missing or invalid hits file {hp!r}")
        hpv = data.get("hits_primary")
        if hpv is not None:
            if primary is not None and int(primary) != int(hpv):
                raise RuntimeError("Multi-node: conflicting hits_primary across ranks.")
            primary = int(hpv)
        nmap = data.get("neutral_hits_by_r")
        if isinstance(nmap, dict):
            for ks, vs in nmap.items():
                r = int(ks)
                neutral_counts[r] = int(vs)
    if primary is None:
        raise RuntimeError("Multi-node: no rank reported hits_primary (rank 0 must run primary).")
    return int(primary), neutral_counts


def _run_multi_node_parallel_shards_on_rank(
    *,
    folder: str,
    sid: str,
    mn_idx: int,
    include_primary: bool,
    neutral_r_indices: List[int],
    n_runs: int,
    model_spec,
    primary_bounds,
    neutral_bounds,
    num_p,
    tog_p,
    num_n,
    tog_n,
    metric_checks: List[Tuple[str, str, float]],
    base_seed: Optional[int],
    param_names_list: List[str],
    neutral_metric_name: str,
    progress_every: int,
    emit,
    parallel_neutral_workers: int,
    job_settings_json_path: Optional[str] = None,
) -> Tuple[Optional[int], Dict[int, int]]:
    """
    Run ProcessPoolExecutor shard workers on this node for primary (optional) + listed neutral indices.

    Returns (hits_primary or None if no primary, neutral_hits_by_r).
    """
    from concurrent.futures import ProcessPoolExecutor

    neutral_hits_by_r: Dict[int, int] = {}
    hits_primary: Optional[int] = None
    payloads: List[Dict[str, Any]] = []
    if include_primary:
        _p0: Dict[str, Any] = {
            "role": "primary",
            "model_key": str(getattr(model_spec, "key", "simulation")),
            "session_id": sid,
            "shard_offload_folder": os.path.join(
                folder, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, "primary"
            ),
            "n_runs": int(n_runs),
            "primary_bounds": primary_bounds,
            "numeric_base": num_p,
            "toggles": tog_p,
            "metric_checks": list(metric_checks),
            "base_seed": base_seed,
            "param_names_list": list(param_names_list),
            "offload_metric_name": neutral_metric_name,
            "progress_every": int(progress_every),
            "progress_label": "primary",
            "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
        }
        if job_settings_json_path:
            _p0["job_settings_json_path"] = job_settings_json_path
        payloads.append(_p0)
    for r in neutral_r_indices:
        if base_seed is None:
            local_seed = None
        else:
            local_seed = int(base_seed) + 19 + int(r) * 10_007
        _pn: Dict[str, Any] = {
            "role": "neutral",
            "r": int(r),
            "model_key": str(getattr(model_spec, "key", "simulation")),
            "session_id": sid,
            "shard_offload_folder": os.path.join(
                folder, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, f"neutral_{r}"
            ),
            "n_runs": int(n_runs),
            "neutral_bounds": neutral_bounds,
            "numeric_base": num_n,
            "toggles": tog_n,
            "metric_checks": list(metric_checks),
            "base_seed": local_seed,
            "param_names_list": list(param_names_list),
            "offload_metric_name": neutral_metric_name,
            "offload_stage": f"neutral_{r}",
            "progress_every": int(progress_every),
            "progress_label": f"neutral_{r}",
            "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
        }
        if job_settings_json_path:
            _pn["job_settings_json_path"] = job_settings_json_path
        payloads.append(_pn)
    if not payloads:
        return None, {}

    max_jobs = len(payloads)
    pnw = max(1, int(parallel_neutral_workers))
    try:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if cpus > 0:
            pnw = min(pnw, cpus)
    except Exception:
        pass
    workers = min(pnw, max_jobs)
    emit(
        f"[multi-node rank {mn_idx}] process pool max_workers={workers} for {max_jobs} shard batch(es) "
        f"(primary={'yes' if include_primary else 'no'})."
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_comparison_batch_worker, p) for p in payloads]
        for fut in futs:
            role, batch_index, hits_n = fut.result()
            if role == "primary":
                hits_primary = int(hits_n)
            else:
                ridx = int(batch_index) if batch_index is not None else -1
                neutral_hits_by_r[ridx] = int(hits_n)
    if include_primary and hits_primary is None:
        raise RuntimeError("internal: primary batch missing from multi-node pool results.")
    return hits_primary, neutral_hits_by_r


def _write_neutral_comparison_histogram_png(
    *,
    folder: str,
    session_id: str,
    primary_hits: int,
    neutral_hit_counts: List[int],
    percentile_lt_percent: float,
    run_headline: Optional[str] = None,
) -> Optional[str]:
    """
    Match ``neutral_set_comparison_gui._draw_histogram``: neutral batch hit counts + primary vline.

    Returns the absolute path written, or None if matplotlib is unavailable or save fails.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=100)
        nc = [int(x) for x in neutral_hit_counts]
        if nc:
            bins = max(8, min(40, len(nc)))
            ax.hist(nc, bins=bins, color="#0072B2", alpha=0.85, edgecolor="white")
        ax.axvline(
            int(primary_hits),
            color="#D55E00",
            linewidth=2.0,
            label=f"Primary count ({int(primary_hits)})",
        )
        ax.set_xlabel("Hit count per batch (N simulations)")
        ax.set_ylabel("# of neutral batches")
        ax.legend(loc="upper right")
        p = float(percentile_lt_percent)
        if np.isfinite(p):
            ax.set_title(f"Neutral batch hit counts — percentile rank of primary: {p:.2f}%")
        else:
            ax.set_title("Neutral batch hit counts")
        h = (run_headline or "").strip()
        if h:
            fig.suptitle(h, fontsize=12, fontweight="semibold", y=1.03)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
        else:
            fig.tight_layout()
        fname = f"neutral_hit_histogram_{session_id}.png"
        out_abs = os.path.abspath(os.path.join(folder, fname))
        fig.savefig(out_abs, dpi=100, bbox_inches="tight")
        return out_abs
    except Exception:
        return None
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass


def _validate_metrics_for_job(model_spec, metric_checks: List[Tuple[str, str, float]], num_p, tog_p, num_n, tog_n) -> None:
    base_names = list(getattr(model_spec, "metric_names", []) or [])
    pa = _allowed_metrics_for_side(model_spec, num_p, tog_p)
    pb = _allowed_metrics_for_side(model_spec, num_n, tog_n)
    common = {m for m in base_names if m in pa and m in pb}
    for mname, _op, _thr in metric_checks:
        if mname not in common:
            raise ValueError(
                f"Metric {mname!r} is not valid for the current primary and neutral simulation "
                "settings (transport / pool regime), or is unknown."
            )


def run_from_neutral_comparison_json(
    data: Dict[str, Any],
    *,
    output_dir: str,
    session_id: Optional[str] = None,
    progress_every: int = 0,
    progress_log_path: Optional[str] = None,
    parallel_neutral_workers: int = 1,
    write_histogram_png: bool = True,
    multi_node_index: Optional[int] = None,
    multi_node_count: Optional[int] = None,
    job_settings_json_path: Optional[str] = None,
    skip_offload_merge: bool = False,
) -> Optional[Dict[str, Any]]:
    kind = str(data.get("kind", "") or "")
    if kind != "neutral_set_comparison":
        raise ValueError(
            f"Expected kind 'neutral_set_comparison' (primary vs neutral job JSON), got {kind!r}. "
            "Build a neutral_set_comparison job (headless/HPC export), not a Batch Runner primary_batch_campaign JSON."
        )
    _trace("run_from: validated job kind neutral_set_comparison")

    model_key = str((data.get("model") or {}).get("key", "simulation") if isinstance(data.get("model"), dict) else "simulation")
    if model_key != "simulation":
        raise ValueError("Headless runner only supports the Simulation model (model.key must be 'simulation').")

    _trace("run_from: calling get_model_by_key('simulation')")
    model_spec = get_model_by_key("simulation")
    _trace("run_from: get_model_by_key returned")
    n_runs = int(data["n_runs"])
    n_neutral = int(data["n_neutral_batches"])
    if n_runs < 1 or n_runs > 500_000:
        raise ValueError("n_runs must be in [1, 500000].")
    if n_neutral < 1 or n_neutral > 10_000:
        raise ValueError("n_neutral_batches must be in [1, 10000].")

    mn_c0 = multi_node_count
    mn_i0 = multi_node_index
    if (mn_c0 is not None) ^ (mn_i0 is not None):
        raise ValueError("Provide both multi_node_index and multi_node_count, or neither.")
    if mn_c0 is not None:
        if int(mn_c0) < 2:
            raise ValueError("multi_node_count must be at least 2.")
        if mn_i0 is None or not (0 <= int(mn_i0) < int(mn_c0)):
            raise ValueError("multi_node_index must satisfy 0 <= index < multi_node_count.")
        if not (session_id or "").strip():
            raise ValueError(
                "Multi-node mode requires a session_id on every rank: pass --session-id, or use a job file "
                "named neutral_YYYYMMDD_name.json whose stem is used as the session id."
            )

    base_seed = data.get("base_seed")
    if base_seed is not None:
        base_seed = int(base_seed)

    num_p = data.get("primary_numeric_parameters")
    tog_p = data.get("primary_toggles")
    num_n = data.get("neutral_numeric_parameters")
    tog_n = data.get("neutral_toggles")
    if not isinstance(num_p, dict) or not isinstance(tog_p, dict):
        raise ValueError("primary_numeric_parameters and primary_toggles must be objects.")
    if not isinstance(num_n, dict) or not isinstance(tog_n, dict):
        raise ValueError("neutral_numeric_parameters and neutral_toggles must be objects.")

    pb_raw = data.get("primary_bounds")
    nb_raw = data.get("neutral_bounds")
    primary_bounds = _bounds_from_save_json(pb_raw, label="primary_bounds")
    neutral_bounds = _bounds_from_save_json(nb_raw, label="neutral_bounds")

    metric_checks, metric_filters_snapshot = _metric_checks_from_filters(data.get("metric_filters"))
    _validate_metrics_for_job(model_spec, metric_checks, num_p, tog_p, num_n, tog_n)
    _trace("run_from: bounds + metric_filters validated")

    folder = os.path.abspath(os.path.expanduser(str(output_dir).strip()))
    os.makedirs(folder, exist_ok=True)
    if _trace_cfg.get("on") and not str(_trace_cfg.get("path") or "").strip():
        _trace_open_out_dir(folder)
    _trace(f"run_from: output folder ready folder={folder!r}")

    sid = (session_id or "").strip() or time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(folder, f"neutral_set_comparison_{sid}.json")
    job_json_path = _normalized_job_settings_json_path(job_settings_json_path)
    _stem = _job_json_stem_for_filename_hint(str(job_settings_json_path or ""))
    _sid_from_stem = neutral_job_filename_session_id(_stem)
    if _sid_from_stem:
        # Short label (e.g. justDup, Death+Dup) for PNG suptitles — matches output folder suffix.
        neutral_run_label = neutral_session_id_output_root_folder_suffix(_sid_from_stem) or _stem
    else:
        neutral_run_label = None

    log_path = (progress_log_path or "").strip()
    log_fp = None
    log_abs: Optional[str] = None
    if log_path:
        log_abs = os.path.abspath(os.path.expanduser(log_path))
        os.makedirs(os.path.dirname(log_abs) or ".", exist_ok=True)
        log_fp = open(log_abs, "w", encoding="utf-8", newline="\n")
        if neutral_run_label:
            log_fp.write(f"# {neutral_run_label}\n")
        if int(progress_every) <= 0:
            progress_every = max(1, min(200, int(n_runs) // 50 or 1))
    _trace(f"run_from: progress log opened={bool(log_fp)} progress_every={progress_every}")

    def _emit(msg: str) -> None:
        with _EMIT_LOCK:
            print(msg, flush=True)
            if log_fp is not None:
                log_fp.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
                log_fp.flush()

    try:
        _trace("run_from: try-block entered (session setup + batches)")
        param_names_list = list(model_spec.default_params.keys())

        sim_light_used, sim_light_canon = simulation_light_tracking_plan(model_spec, metric_checks)
        _trace(f"run_from: simulation_light_tracking_plan done use_light={bool(sim_light_used)}")

        pnw = max(1, int(parallel_neutral_workers))
        _trace(f"run_from: parallel_neutral_workers effective pnw={pnw}")
        if skip_offload_merge:
            if multi_node_count is None and pnw <= 1:
                raise ValueError(
                    "skip_offload_merge is only valid with multi-node mode or parallel_neutral_workers > 1 "
                    "(there is no shard merge to skip otherwise)."
                )

        mn_c = multi_node_count
        mn_i = multi_node_index
        _skip_standard_batches = False
        neutral_metric_name = str(metric_checks[0][0])
        neutral_counts: List[int]
        hits_primary: int
        offload_writer: Optional[NeutralComparisonOffloadWriter] = None

        if mn_c is not None:
            n_lo, n_hi = _multi_node_neutral_batch_range(n_neutral, int(mn_i), int(mn_c))
            neutral_r_indices = list(range(n_lo, n_hi))
            include_primary = int(mn_i) == 0
            _emit(
                f"[multi-node] rank {int(mn_i)}/{int(mn_c)} neutral_batches=[{n_lo},{n_hi}) "
                f"primary={'yes' if include_primary else 'no'} session={sid}"
            )
            if int(mn_i) == 0:
                offload_writer = NeutralComparisonOffloadWriter(
                    folder,
                    sid,
                    model_key=str(getattr(model_spec, "key", "simulation")),
                    param_names_list=param_names_list,
                    metric_name_at_offload=str(metric_checks[0][0]),
                    job_settings_json_path=job_json_path,
                )
                settings_snapshot_mn: Dict[str, Any] = {
                    "kind": "neutral_set_comparison_session",
                    "version": 1,
                    "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
                    "session_id": sid,
                }
                if neutral_run_label:
                    settings_snapshot_mn["headless_neutral_job_file_stem"] = neutral_run_label
                settings_snapshot_mn.update(
                    {
                    "model": {"key": model_spec.key, "label": getattr(model_spec, "label", "")},
                    "n_runs": int(n_runs),
                    "n_neutral_batches": int(n_neutral),
                    "base_seed": base_seed,
                    "match_neutral_to_primary": bool(data.get("match_neutral_to_primary", False)),
                    "metric_filters": list(metric_filters_snapshot),
                    "simulation_light_tracking": bool(sim_light_used),
                    "simulation_light_tracking_metrics": list(sim_light_canon),
                    "parallel_neutral_workers": int(pnw),
                    "parallel_primary_with_neutral_batches": bool(pnw > 1),
                    "parallel_offload_shard_subdir": NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR,
                    "parallel_offload_merged_from_shards": not skip_offload_merge,
                    "multi_node_slurm": {
                        "task_count": int(mn_c),
                        "this_rank": int(mn_i),
                        "driver_rank": 0,
                        "note": (
                            "Rank 0 merges offload after all ranks write _parallel_shards/ and .done markers."
                            if not skip_offload_merge
                            else "Rank 0 skipped session-root offload merge (--skip-offload-merge); "
                            "per-simulation rows remain under _parallel_shards/ only."
                        ),
                    },
                    "full_save_folder": folder,
                    "saved_at_epoch": int(time.time()),
                    }
                )
                _write_results_json(full_save_settings_path_json(folder, sid), settings_snapshot_mn)
                _trace("run_from: multi-node rank 0 settings written")
            else:
                _multi_node_wait_for_settings_json(folder, sid, emit=_emit)

            hp_part, nh_map = _run_multi_node_parallel_shards_on_rank(
                folder=folder,
                sid=sid,
                mn_idx=int(mn_i),
                include_primary=include_primary,
                neutral_r_indices=neutral_r_indices,
                n_runs=int(n_runs),
                model_spec=model_spec,
                primary_bounds=primary_bounds,
                neutral_bounds=neutral_bounds,
                num_p=num_p,
                tog_p=tog_p,
                num_n=num_n,
                tog_n=tog_n,
                metric_checks=metric_checks,
                base_seed=base_seed,
                param_names_list=param_names_list,
                neutral_metric_name=neutral_metric_name,
                progress_every=int(progress_every),
                emit=_emit,
                parallel_neutral_workers=int(pnw),
                job_settings_json_path=job_json_path,
            )
            _multi_node_write_hits_json(folder, int(mn_i), hp_part if include_primary else None, nh_map)
            with open(_multi_node_done_path(folder, int(mn_i)), "w", encoding="utf-8") as _df:
                _df.write(f"ok t={int(time.time())}\n")
            if int(mn_i) != 0:
                _emit(f"[multi-node] rank {int(mn_i)} finished (shards + hits + done). Exiting without merge.")
                return None

            _multi_node_wait_all_dones(folder, int(mn_c), emit=_emit)
            hits_primary, neutral_counts = _multi_node_load_merged_hits(folder, int(mn_c), int(n_neutral))
            if offload_writer is None:
                raise RuntimeError("internal: multi-node rank 0 missing offload_writer for merge.")
            if skip_offload_merge:
                _emit(
                    f"Session {sid}: skipping multi-node offload merge "
                    f"(per-simulation rows remain under {NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/ only)."
                )
            else:
                _emit(
                    f"Session {sid}: merging multi-node offload shards from "
                    f"{NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/ into session-root batches…"
                )
                merge_parallel_neutral_comparison_shards_into_writer(
                    offload_writer,
                    folder,
                    sid,
                    n_neutral,
                    n_runs,
                    _read_json_maybe_gz,
                )
                _emit(
                    f"Session {sid}: merged multi-node offload shards from "
                    f"{NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/ into session-root batches."
                )
            _skip_standard_batches = True

        if not _skip_standard_batches:
            offload_writer = NeutralComparisonOffloadWriter(
                folder,
                sid,
                model_key=str(getattr(model_spec, "key", "simulation")),
                param_names_list=param_names_list,
                metric_name_at_offload=str(metric_checks[0][0]),
                job_settings_json_path=job_json_path,
            )
            _trace("run_from: NeutralComparisonOffloadWriter constructed")

            settings_snapshot: Dict[str, Any] = {
                "kind": "neutral_set_comparison_session",
                "version": 1,
                "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
                "session_id": sid,
                "model": {"key": model_spec.key, "label": getattr(model_spec, "label", "")},
                "n_runs": int(n_runs),
                "n_neutral_batches": int(n_neutral),
                "base_seed": base_seed,
                "match_neutral_to_primary": bool(data.get("match_neutral_to_primary", False)),
                "metric_filters": list(metric_filters_snapshot),
                "simulation_light_tracking": bool(sim_light_used),
                "simulation_light_tracking_metrics": list(sim_light_canon),
                "parallel_neutral_workers": int(pnw),
                "parallel_primary_with_neutral_batches": bool(pnw > 1),
                "parallel_offload_shard_subdir": NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR,
                "parallel_offload_merged_from_shards": bool(pnw > 1) and not skip_offload_merge,
                "full_save_folder": folder,
                "saved_at_epoch": int(time.time()),
            }
            if neutral_run_label:
                settings_snapshot["headless_neutral_job_file_stem"] = neutral_run_label
            _write_results_json(full_save_settings_path_json(folder, sid), settings_snapshot)
            _trace("run_from: full_save_settings JSON written")
    
            last_log_t = [0.0]
    
            def _progress_cb(done: int, total: int, hits: int, phase: str) -> None:
                if progress_every <= 0:
                    return
                now = time.time()
                if done < total and done % int(progress_every) != 0 and (now - last_log_t[0]) < 30.0:
                    return
                last_log_t[0] = now
                _emit(f"[{phase}] {done}/{total} simulations, hits so far: {hits}")
    
            intro_bits: List[str] = []
            if neutral_run_label:
                intro_bits.append(f"job_file_stem={neutral_run_label}")
            intro_bits += [
                f"headless_version={NEUTRAL_HEADLESS_VERSION}",
                f"session={sid}",
                f"n_runs={n_runs}",
                f"n_neutral_batches={n_neutral}",
                f"parallel_neutral_workers={pnw}",
                f"output_dir={folder}",
            ]
            if log_fp is not None and log_abs is not None:
                intro_bits.append(f"progress_every={progress_every}")
                intro_bits.append(f"progress_log={log_abs}")
            _trace("run_from: about to emit Ready line and start Monte Carlo batches")
            _emit("Ready: " + ", ".join(intro_bits))

            if pnw <= 1:
                _trace("run_from: serial mode — starting primary run_hit_count_batch")
                _emit(f"Session {sid}: primary batch ({n_runs} runs)…")
                hits_primary = run_hit_count_batch(
                    model_spec=model_spec,
                    n_runs=n_runs,
                    bounds=primary_bounds,
                    numeric_base=num_p,
                    toggles=tog_p,
                    metric_checks=metric_checks,
                    base_seed=base_seed,
                    progress_callback=(lambda d, t, h: _progress_cb(d, t, h, "primary")) if progress_every > 0 else None,
                    offload_writer=offload_writer,
                    offload_stage="primary",
                )
                _trace(f"run_from: serial primary batch finished hits_primary={hits_primary}")
                neutral_counts = []
                for r in range(n_neutral):
                    if base_seed is None:
                        local_seed = None
                    else:
                        local_seed = int(base_seed) + 19 + r * 10_007
                    phase = f"neutral {r + 1}/{n_neutral}"
                    _trace(f"run_from: serial neutral batch r={r} starting")
                    _emit(f"Session {sid}: {phase} ({n_runs} runs)…")
                    hc = run_hit_count_batch(
                        model_spec=model_spec,
                        n_runs=n_runs,
                        bounds=neutral_bounds,
                        numeric_base=num_n,
                        toggles=tog_n,
                        metric_checks=metric_checks,
                        base_seed=local_seed,
                        progress_callback=(lambda d, t, h, _p=phase: _progress_cb(d, t, h, _p)) if progress_every > 0 else None,
                        offload_writer=offload_writer,
                        offload_stage=f"neutral_{r}",
                    )
                    neutral_counts.append(hc)
                    _trace(f"run_from: serial neutral batch r={r} done hits={hc}")
            else:
                # Separate processes (not threads): no shared writer/RNG. Parent merges offload in one thread.
                from concurrent.futures import ProcessPoolExecutor
    
                max_jobs = int(n_neutral) + 1
                workers = min(pnw, max_jobs)
                _trace(
                    f"run_from: parallel mode max_jobs={max_jobs} workers={workers} "
                    f"(next: ProcessPoolExecutor startup can be slow)"
                )
                _emit(
                    f"Session {sid}: primary + {n_neutral} neutral batch(es) at once "
                    f"(up to {workers} parallel workers, {n_runs} sims each; neutrals do not wait on primary)…"
                )
                _emit(
                    f"Starting process pool (max_workers={workers}); first worker activity may be slow "
                    f"(imports / Numba JIT). Progress lines with batch tags go to this .out file."
                )
                primary_payload: Dict[str, Any] = {
                    "role": "primary",
                    "model_key": str(getattr(model_spec, "key", "simulation")),
                    "session_id": sid,
                    "shard_offload_folder": os.path.join(
                        folder, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, "primary"
                    ),
                    "n_runs": int(n_runs),
                    "primary_bounds": primary_bounds,
                    "numeric_base": num_p,
                    "toggles": tog_p,
                    "metric_checks": list(metric_checks),
                    "base_seed": base_seed,
                    "param_names_list": list(param_names_list),
                    "offload_metric_name": neutral_metric_name,
                    "progress_every": int(progress_every),
                    "progress_label": "primary",
                    "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
                }
                if job_json_path:
                    primary_payload["job_settings_json_path"] = job_json_path
                neutral_payloads: List[Dict[str, Any]] = []
                for r in range(n_neutral):
                    if base_seed is None:
                        local_seed = None
                    else:
                        local_seed = int(base_seed) + 19 + r * 10_007
                    _np: Dict[str, Any] = {
                        "role": "neutral",
                        "r": int(r),
                        "model_key": str(getattr(model_spec, "key", "simulation")),
                        "session_id": sid,
                        "shard_offload_folder": os.path.join(
                            folder, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, f"neutral_{r}"
                        ),
                        "n_runs": int(n_runs),
                        "neutral_bounds": neutral_bounds,
                        "numeric_base": num_n,
                        "toggles": tog_n,
                        "metric_checks": list(metric_checks),
                        "base_seed": local_seed,
                        "param_names_list": list(param_names_list),
                        "offload_metric_name": neutral_metric_name,
                        "offload_stage": f"neutral_{r}",
                        "progress_every": int(progress_every),
                        "progress_label": f"neutral_{r}",
                        "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
                    }
                    if job_json_path:
                        _np["job_settings_json_path"] = job_json_path
                    neutral_payloads.append(_np)
                _trace("run_from: entering ProcessPoolExecutor context manager")
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    _trace("run_from: executor ready, submitting primary + neutral futures")
                    fut_primary = pool.submit(_comparison_batch_worker, primary_payload)
                    fut_neutral = [pool.submit(_comparison_batch_worker, p) for p in neutral_payloads]
                    _trace("run_from: all futures submitted; starting heartbeat + blocking on results")
                    hb_stop = threading.Event()
                    t_hb0 = time.monotonic()
    
                    def _parallel_heartbeat() -> None:
                        first_wait = 10.0
                        later = 30.0
                        pending = first_wait
                        while True:
                            if hb_stop.wait(pending):
                                break
                            prim_done = fut_primary.done()
                            n_done = sum(1 for f in fut_neutral if f.done())
                            _emit(
                                f"[heartbeat] elapsed {time.monotonic() - t_hb0:.0f}s: "
                                f"primary_future_done={prim_done}, neutral_futures_done={n_done}/{len(fut_neutral)} "
                                f"(per-sim lines also come from worker stdout when progress_every>0)"
                            )
                            pending = later
    
                    hb_thread = threading.Thread(
                        target=_parallel_heartbeat,
                        name="neutral_headless_hb",
                        daemon=True,
                    )
                    _emit(
                        "[heartbeat] parallel worker futures submitted; "
                        "status lines every 10s then 30s until all batches finish."
                    )
                    hb_thread.start()
                    try:
                        _trace("run_from: blocking on fut_primary.result()")
                        _role_p, _idx_p, hits_primary = fut_primary.result()
                        _trace(f"run_from: primary future completed hits={hits_primary}")
                        if _role_p != "primary":
                            raise RuntimeError(f"internal: expected primary batch worker, got role={_role_p!r}")
                        neutral_counts = [0] * n_neutral
                        for r in range(n_neutral):
                            _trace(f"run_from: blocking on fut_neutral[{r}].result()")
                            _role_n, ridx, hits_n = fut_neutral[r].result()
                            _trace(f"run_from: neutral future r={r} done hits={hits_n}")
                            if _role_n != "neutral" or int(ridx) != int(r):
                                raise RuntimeError(
                                    f"internal: neutral batch mismatch at index {r} (got role={_role_n!r}, ridx={ridx!r})"
                                )
                            neutral_counts[r] = int(hits_n)
                        _trace("run_from: merging parallel shard offload folders into session-root writer")
                        if skip_offload_merge:
                            _emit(
                                f"Session {sid}: skipping worker offload merge "
                                f"(per-simulation rows remain under {NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/ only)."
                            )
                        else:
                            _emit(
                                f"Session {sid}: merging worker offload shards from "
                                f"{NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/ into session-root batches…"
                            )
                            merge_parallel_neutral_comparison_shards_into_writer(
                                offload_writer,
                                folder,
                                sid,
                                n_neutral,
                                n_runs,
                                _read_json_maybe_gz,
                            )
                            _trace("run_from: shard merge into offload_writer complete")
                            _emit(
                                f"Session {sid}: merged worker offload shards from "
                                f"{NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/ into session-root batches."
                            )
                    finally:
                        hb_stop.set()
                        hb_thread.join(timeout=5.0)
                _trace("run_from: parallel ProcessPoolExecutor block exited")
                _emit(
                    f"Session {sid}: parallel phase done — primary_hits={hits_primary}, "
                    f"neutral hits per batch: {neutral_counts}"
                )

        _emit(
            f"Session {sid}: finishing session files "
            f"(percentile, offload finalize, histogram, summary JSON)…"
        )
        arr = np.asarray(neutral_counts, dtype=float)
        pct = float("nan") if arr.size == 0 else 100.0 * float(np.mean(arr < float(hits_primary)))
        _trace(f"run_from: percentile_lt_percent computed pct={pct!r}")

        _trace("run_from: calling offload_writer.finalize()")
        if offload_writer is None:
            raise RuntimeError("internal: offload_writer missing before finalize().")
        _emit(f"Session {sid}: finalizing offload (flushing any buffered rows to gzip batches)…")
        offload_writer.finalize()
        _trace("run_from: offload_writer.finalize() returned")
        _nb = len(offload_writer.manifest.get("batches", []))
        _shard_batch_files_for_summary = 0
        if skip_offload_merge:
            _shard_batch_files_for_summary = int(
                count_shard_offload_batches(folder, sid, n_neutral, _read_json_maybe_gz)
            )
        if skip_offload_merge:
            _emit(
                f"Session {sid}: offload finalize complete (session-root batches: {_nb}; "
                f"shard offload batch files: {_shard_batch_files_for_summary}; session-root merge was skipped — "
                f"per-simulation rows remain under {NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/)."
            )
        else:
            _emit(f"Session {sid}: offload finalize complete ({_nb} offload batch file(s) in manifest).")

        hist_abs: Optional[str] = None
        primary_events_abs: Optional[str] = None
        primary_event_summary: Optional[Dict[str, Any]] = None
        if write_histogram_png:
            _emit(f"Session {sid}: writing neutral hit histogram PNG…")
            _trace("run_from: writing histogram PNG (if matplotlib available)")
            hist_abs = _write_neutral_comparison_histogram_png(
                folder=folder,
                session_id=sid,
                primary_hits=int(hits_primary),
                neutral_hit_counts=[int(x) for x in neutral_counts],
                percentile_lt_percent=float(pct),
                run_headline=neutral_run_label,
            )
            if hist_abs is None:
                _emit(
                    "Note: histogram PNG not written (install matplotlib in the venv, or use --no-histogram-png)."
                )
            else:
                _emit(f"Session {sid}: histogram PNG complete: {os.path.basename(hist_abs)}")

        _emit(
            f"Session {sid}: summarizing primary per-generation deaths / duplications / "
            "mutations / flow removal by filter outcome…"
        )
        _primary_recs = load_primary_offload_records(folder, sid, _read_json_maybe_gz)
        primary_event_summary = summarize_primary_events_by_hit(_primary_recs)
        _pe_meets = int((primary_event_summary.get("meets_threshold") or {}).get("n") or 0)
        _pe_below = int((primary_event_summary.get("below_threshold") or {}).get("n") or 0)
        primary_events_abs = write_primary_event_rates_png(
            folder=folder,
            session_id=sid,
            summary=primary_event_summary,
            run_headline=neutral_run_label,
        )
        if primary_events_abs is None:
            if not _primary_recs:
                _emit(
                    "Note: primary events PNG not written — no primary offload records found "
                    f"(session root or {NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR}/primary/). "
                    "If multi-node used --skip-offload-merge, sync an updated mccm_light bundle."
                )
            elif _pe_meets + _pe_below == 0:
                _emit(
                    "Note: primary events PNG not written — "
                    f"{len(_primary_recs)} primary offload row(s) but none had change_history "
                    f"({primary_event_summary.get('skipped_no_change_history', 0)} skipped)."
                )
            else:
                _emit("Note: primary events PNG not written (matplotlib unavailable or save failed).")
        else:
            _emit(f"Session {sid}: primary events PNG complete: {os.path.basename(primary_events_abs)}")

        _emit(f"Session {sid}: writing parameter heatmap PNG (threshold-crossing primary runs)…")
        param_heatmap_abs: Optional[str] = None
        _param_names_offload = load_param_names_from_offload(folder, sid, _read_json_maybe_gz) or list(
            param_names_list
        )
        param_heatmap_abs = write_parameter_heatmap_png(
            folder=folder,
            session_id=sid,
            records=_primary_recs,
            param_names_list=_param_names_offload,
            primary_bounds=_bounds_jsonable(primary_bounds),
            run_headline=neutral_run_label,
        )
        if param_heatmap_abs is None:
            if not _primary_recs:
                _emit("Note: parameter heatmap PNG not written — no primary offload records.")
            else:
                _emit("Note: parameter heatmap PNG not written (matplotlib unavailable or save failed).")
        else:
            _emit(f"Session {sid}: parameter heatmap PNG complete: {os.path.basename(param_heatmap_abs)}")

        save_payload: Dict[str, Any] = {
            "kind": "neutral_set_comparison",
            "version": 1,
            "headless_runner_version": NEUTRAL_HEADLESS_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "full_save_session_id": sid,
            "full_save_manifest": os.path.basename(full_save_manifest_path_json(folder, sid)),
            "full_save_settings": os.path.basename(full_save_settings_path_json(folder, sid)),
            "offload_batches_written": (
                int(_shard_batch_files_for_summary)
                if skip_offload_merge
                else len(offload_writer.manifest.get("batches", []))
            ),
            "save_folder": folder,
            "output_path": out_path,
            "n_runs": int(n_runs),
            "n_neutral_batches": int(n_neutral),
            "base_seed": base_seed,
            "match_neutral_to_primary": bool(data.get("match_neutral_to_primary", False)),
            "metric_filters": list(metric_filters_snapshot),
            "metric_filters_logical": "AND",
            "evaluation_note": (
                "Within each simulation, active metric conditions are evaluated in an order sorted by "
                "estimated metric compute cost (cheaper checks first), same spirit as other Monte Carlo filters. "
                "When simulation_light_tracking is true in the session settings, each run used "
                "store_history=False and minimal_tracking=True with merged tracking_metric_names in simulation.core."
            ),
            "simulation_light_tracking": bool(sim_light_used),
            "simulation_light_tracking_metrics": list(sim_light_canon),
            "primary_hit_count": int(hits_primary),
            "neutral_hit_counts": [int(x) for x in neutral_counts],
            "percentile_lt_percent": float(pct),
            "primary_bounds": _bounds_jsonable(primary_bounds),
            "neutral_bounds": _bounds_jsonable(neutral_bounds),
            "primary_numeric_parameters": _jsonable_params(
                prune_irrelevant_numeric_parameters_for_export(num_p, tog_p)
            ),
            "primary_toggles": _jsonable_params(tog_p),
            "neutral_numeric_parameters": _jsonable_params(
                prune_irrelevant_numeric_parameters_for_export(num_n, tog_n)
            ),
            "neutral_toggles": _jsonable_params(tog_n),
            "model": {"key": model_spec.key, "label": getattr(model_spec, "label", "")},
        }
        if neutral_run_label:
            save_payload["headless_neutral_job_file_stem"] = neutral_run_label
        if skip_offload_merge:
            save_payload["offload_merge_skipped"] = True
            save_payload["offload_shard_batch_files"] = int(_shard_batch_files_for_summary)
            save_payload["offload_shard_subdir"] = NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR
        if hist_abs:
            save_payload["histogram_png"] = os.path.basename(hist_abs)
        if primary_event_summary is not None:
            save_payload["primary_event_summary"] = primary_event_summary
        if primary_events_abs:
            save_payload["primary_events_png"] = os.path.basename(primary_events_abs)
        if param_heatmap_abs:
            save_payload["parameter_heatmap_png"] = os.path.basename(param_heatmap_abs)
        _emit(f"Session {sid}: writing summary JSON {os.path.basename(out_path)!r}…")
        _trace(f"run_from: writing summary JSON -> {out_path!r}")
        _write_results_json(out_path, save_payload)
        _trace("run_from: summary JSON write complete")
        _emit(f"Session {sid}: summary JSON write complete.")

        done_extra_parts = []
        if hist_abs:
            done_extra_parts.append(f"histogram: {hist_abs}")
        if primary_events_abs:
            done_extra_parts.append(f"primary events: {primary_events_abs}")
        if param_heatmap_abs:
            done_extra_parts.append(f"parameter heatmap: {param_heatmap_abs}")
        done_extra = ("\nWrote " + "; ".join(done_extra_parts)) if done_extra_parts else ""
        _emit(
            f"Done (headless_version={NEUTRAL_HEADLESS_VERSION}). primary_hits={hits_primary} "
            f"neutral_counts={neutral_counts} percentile_lt_percent={pct:.4g}\nWrote: {out_path}{done_extra}"
        )
        _trace("run_from: returning successfully to main()")
        return save_payload
    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio_for_job()
    if os.environ.get("NEUTRAL_HEADLESS_TRACE", "").strip().lower() in ("1", "true", "yes"):
        _trace_cfg["on"] = True
        try:
            sys.stderr.write(
                f"[headless trace] NEUTRAL_HEADLESS_TRACE enabled pid={os.getpid()} "
                f"(timestamps start at first trace line)\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
    print(
        f"[headless neutral] version={NEUTRAL_HEADLESS_VERSION} runner={os.path.abspath(__file__)}",
        flush=True,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--debug-trace",
        action="store_true",
        help=(
            "Timestamped trace to stderr and <output_dir>/headless_debug_trace.log (same as env "
            "NEUTRAL_HEADLESS_TRACE=1). Use to locate hangs when batch logs look empty."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {NEUTRAL_HEADLESS_VERSION}",
        help="Print headless runner version and exit.",
    )
    p.add_argument(
        "config",
        help="Path to neutral_set_comparison JSON (same kind as the GUI summary save).",
    )
    p.add_argument(
        "--output-dir",
        default="",
        help="Exact directory for offload batches and summary JSON (default: save_folder from the config, or cwd).",
    )
    p.add_argument(
        "--output-root",
        default="",
        metavar="DIR",
        help=(
            "Parent directory: for session ids of the form neutral_YYYYMMDD_<name>, create "
            "DIR/<time>_<name>/; otherwise DIR/<time>_<session_id>/ (or DIR/<time>/ with no id). "
            "Here <time> is YYYYMMDD-HHMMSS or env NEUTRAL_SESSION_FOLDER_STAMP. Manifest/summary filenames still "
            "use the full session id. Ignores job save_folder. Takes precedence over --output-dir."
        ),
    )
    p.add_argument(
        "--session-id",
        default="",
        help=(
            "Session / folder name (default: new timestamp YYYYMMDD-HHMMSS when using --output-root; if omitted "
            "and the config basename matches neutral_YYYYMMDD_name.json, that stem is used, e.g. "
            "neutral_20260423_justDeath)."
        ),
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        metavar="N",
        help="Print progress every N finished simulations per batch (0 disables unless --progress-log is set).",
    )
    p.add_argument(
        "--progress-log",
        default="",
        metavar="PATH",
        help=(
            "Append timestamped progress lines to this file (and still print to stdout). "
            "If set and --progress-every is 0, a default interval is chosen from n_runs. "
            "Environment variable NEUTRAL_PROGRESS_LOG overrides the default file path when set; "
            "use --no-progress-log to skip the default file under the output directory."
        ),
    )
    p.add_argument(
        "--no-progress-log",
        action="store_true",
        help="Do not default to <output_dir>/neutral_progress.log (stdout + Slurm .out only unless --progress-log is set).",
    )
    p.add_argument(
        "--parallel-neutral-workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "When N>1: run primary and all neutral batches concurrently (up to min(N, n_neutral_batches+1) "
            "processes). When N==1: serial primary then serial neutrals. Default: env "
            "NEUTRAL_PARALLEL_NEUTRAL_WORKERS, else 1."
        ),
    )
    p.add_argument(
        "--no-histogram-png",
        action="store_true",
        help=(
            "Do not write neutral_hit_histogram_<session>.png (neutral batch hit histogram only). "
            "neutral_primary_events_<session>.png is still written when offload data allow."
        ),
    )
    p.add_argument(
        "--skip-offload-merge",
        action="store_true",
        help=(
            "After parallel or multi-node Monte Carlo batches, skip copying shard offload gzip batches "
            "into session-root offload_<session>_*.json.gz (saves time and duplicate I/O). Summary JSON "
            "and histogram still run. Requires --parallel-neutral-workers > 1 or multi-node mode. "
            "Per-simulation rows remain only under _parallel_shards/."
        ),
    )
    p.add_argument(
        "--multi-node-index",
        type=int,
        default=None,
        metavar="I",
        help=(
            "Slurm multi-node: this task's rank in [0, T). Requires --multi-node-count T and the same "
            "--session-id on every task. Rank 0 merges offload and writes the summary."
        ),
    )
    p.add_argument(
        "--multi-node-count",
        type=int,
        default=None,
        metavar="T",
        help="Slurm multi-node: total number of concurrent tasks (e.g. 2 for two nodes).",
    )
    args = p.parse_args(argv)
    if bool(getattr(args, "debug_trace", False)):
        _trace_cfg["on"] = True
    if _trace_cfg.get("on"):
        _trace("main: argparse done")
    # Remove embedded CR/LF from config paths (e.g. when a path is split across lines).
    args.config = (args.config or "").replace("\r", "").replace("\n", "").strip()
    if not args.config:
        print("Error: config path is empty after stripping newlines.", file=sys.stderr)
        return 1

    if _trace_cfg.get("on"):
        _trace(f"main: reading config path={args.config!r}")
    try:
        raw = _read_json_maybe_gz(args.config)
    except Exception as exc:
        print(f"Error reading config: {exc}", file=sys.stderr)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 2:
            cfg = args.config
            if "path/to" in cfg.replace("\\", "/").lower():
                print(
                    "Hint: the config path still contains a documentation placeholder (e.g. /path/to/...). "
                    "Use the real JSON path on disk, or set CONFIG_JSON to that path before sbatch.",
                    file=sys.stderr,
                )
            print(
                "Hint: confirm the file exists and CONFIG_JSON is a single line (no line break in the path). "
                f"Verify: ls -l {cfg!r}",
                file=sys.stderr,
            )
        return 1
    if not isinstance(raw, dict):
        print("Config must be a JSON object.", file=sys.stderr)
        return 1
    if _trace_cfg.get("on"):
        _trace("main: config JSON parsed successfully")

    root = (args.output_root or "").strip()
    explicit_dir = (args.output_dir or "").strip()
    sid_arg = (args.session_id or "").strip() or None
    cfg_stem = _job_json_stem_for_filename_hint(args.config)
    if sid_arg is None:
        _inferred = neutral_job_filename_session_id(cfg_stem)
        if _inferred is not None:
            sid_arg = _inferred
            print(
                f"[headless neutral] session_id from job filename stem: {_inferred!r} "
                f"(pass --session-id to use a different id).",
                file=sys.stderr,
                flush=True,
            )

    if root:
        if explicit_dir:
            print(
                "Note: --output-root is set; ignoring --output-dir (outputs go under <root>/<time>_<label>/: "
                "for neutral_YYYYMMDD_*, <label> is the final name segment; otherwise full session_id; or "
                "<root>/<time>/ with no id).",
                file=sys.stderr,
            )
        root_abs = os.path.abspath(os.path.expanduser(root))
        tpre = _output_root_time_prefix()
        if sid_arg is not None:
            _folder_suffix = neutral_session_id_output_root_folder_suffix(sid_arg)
            if _folder_suffix is not None:
                out_dir = os.path.join(root_abs, f"{tpre}_{_folder_suffix}")
                print(
                    f"[headless neutral] --output-root folder: {tpre!r} + {_folder_suffix!r} "
                    f"(full session id for file names: {sid_arg!r})",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                out_dir = os.path.join(root_abs, f"{tpre}_{sid_arg}")
        else:
            out_dir = os.path.join(root_abs, tpre)
        session_for_run: Optional[str] = sid_arg or tpre
    elif explicit_dir:
        out_dir = os.path.abspath(os.path.expanduser(explicit_dir))
        session_for_run = sid_arg
    else:
        out_dir = str(raw.get("save_folder") or "").strip() or os.getcwd()
        out_dir = os.path.abspath(os.path.expanduser(out_dir))
        session_for_run = sid_arg

    log_arg = (args.progress_log or "").strip()
    if not log_arg:
        log_arg = (os.environ.get("NEUTRAL_PROGRESS_LOG") or "").strip()
    if not log_arg and not bool(getattr(args, "no_progress_log", False)):
        log_arg = os.path.join(out_dir, "neutral_progress.log")

    if args.parallel_neutral_workers is not None:
        parallel_neutral_workers = max(1, int(args.parallel_neutral_workers))
    else:
        ev = (os.environ.get("NEUTRAL_PARALLEL_NEUTRAL_WORKERS") or "").strip()
        parallel_neutral_workers = max(1, int(ev)) if ev else 1

    mn_idx = getattr(args, "multi_node_index", None)
    mn_cnt = getattr(args, "multi_node_count", None)
    if (mn_idx is not None) ^ (mn_cnt is not None):
        print("Error: set both --multi-node-index and --multi-node-count, or neither.", file=sys.stderr)
        return 1
    if mn_cnt is not None and not sid_arg:
        print(
            "Error: multi-node mode requires a fixed session id: pass --session-id, or use a job file named "
            "like neutral_YYYYMMDD_name.json (stem becomes the session id).",
            file=sys.stderr,
        )
        return 1

    if bool(getattr(args, "skip_offload_merge", False)):
        if mn_cnt is None and parallel_neutral_workers <= 1:
            print(
                "Error: --skip-offload-merge requires multi-node mode (--multi-node-count) or "
                "--parallel-neutral-workers > 1.",
                file=sys.stderr,
            )
            return 1

    _jstem = neutral_job_filename_session_id(cfg_stem)
    _start_extra = f" job_file_stem={_jstem!r}" if _jstem else ""
    print(
        f"[headless neutral] starting: version={NEUTRAL_HEADLESS_VERSION} config={args.config!r} "
        f"output_dir={out_dir!r} parallel_workers={parallel_neutral_workers} "
        f"skip_offload_merge={bool(getattr(args, 'skip_offload_merge', False))} "
        f"progress_log={log_arg or '(none)'}{_start_extra}",
        flush=True,
    )

    if _trace_cfg.get("on"):
        _trace_open_out_dir(out_dir)
        _trace(
            f"main: invoking run_from_neutral_comparison_json "
            f"(parallel_neutral_workers={parallel_neutral_workers} progress_every_arg={args.progress_every})"
        )

    rc = 0
    try:
        result = run_from_neutral_comparison_json(
            raw,
            output_dir=out_dir,
            session_id=session_for_run,
            progress_every=max(0, int(args.progress_every)),
            progress_log_path=log_arg or None,
            parallel_neutral_workers=parallel_neutral_workers,
            write_histogram_png=not bool(getattr(args, "no_histogram_png", False)),
            multi_node_index=mn_idx,
            multi_node_count=mn_cnt,
            job_settings_json_path=_normalized_job_settings_json_path(args.config),
            skip_offload_merge=bool(getattr(args, "skip_offload_merge", False)),
        )
        if result is None:
            print("[headless neutral] multi-node non-zero rank finished.", flush=True)
        if _trace_cfg.get("on"):
            _trace("main: run_from_neutral_comparison_json returned without exception")
    except Exception as exc:
        if _trace_cfg.get("on"):
            _trace(f"main: run_from_neutral_comparison_json raised {type(exc).__name__}: {exc!r}")
        print(f"Run failed: {exc}", file=sys.stderr)
        rc = 1
    finally:
        _trace_close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
