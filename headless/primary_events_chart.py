#!/usr/bin/env python3
"""
Regenerate the primary events bar chart (PNG) from existing Neutral Set Comparison session folder(s).

Does not re-run simulations. Reads primary offload rows from session-root batches and/or
``_parallel_shards/primary/`` (multi-node jobs with ``--skip-offload-merge``).

Usage (one session folder)::

    python -m headless.primary_events_chart --session-dir /path/to/Output/<session_folder>

Usage (every session under an Output root)::

    python -m headless.primary_events_chart --output-root /path/to/Output

Or pass a path and let the tool auto-detect (session folder vs parent Output directory)::

    ./run_primary_events_chart.sh /gs/gsfs0/users/srosean/MCCP_Enzymes/Output

Optional::

    --session-id neutral_20260423_justDup   # single-folder mode only; if not inferred
    --update-summary                        # patch neutral_set_comparison_*.json in each folder
    --headline "..."                        # override suptitle (default: infer justDup / Death+Dup per folder)
    --skip-existing                         # skip folders that already have the PNG
    --no-png                                # stats only
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gui.apps.neutral_comparison.offload import NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR
from gui.apps.neutral_comparison.primary_event_chart import (
    load_primary_offload_records,
    pop_last_primary_event_png_error,
    summarize_primary_events_by_hit,
    write_primary_event_rates_png,
)
from gui.persistence.full_save import full_save_manifest_path_json
from gui.persistence.json_io import make_read_json_maybe_gz_fn, make_write_json_maybe_gz_atomic_fn
from headless.neutral_comparison import neutral_session_id_output_root_folder_suffix

PRIMARY_EVENTS_CHART_VERSION = "1.2.0"

_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)
_write_json = make_write_json_maybe_gz_atomic_fn(indent=2)

_MANIFEST_RE = re.compile(r"^full_save_manifest_(.+)\.json(?:\.gz)?$")
_SUMMARY_RE = re.compile(r"^neutral_set_comparison_(.+)\.json(?:\.gz)?$")
_SETTINGS_RE = re.compile(r"^full_save_settings_(.+)\.json(?:\.gz)?$")


def infer_session_id(session_dir: str) -> Optional[str]:
    """Infer logical session id from manifest, summary, or settings filenames in ``session_dir``."""
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    if not os.path.isdir(session_dir):
        return None
    candidates: List[str] = []
    for name in os.listdir(session_dir):
        base = os.path.basename(name)
        for rx in (_MANIFEST_RE, _SUMMARY_RE, _SETTINGS_RE):
            m = rx.match(base)
            if m:
                candidates.append(m.group(1))
                break
    if not candidates:
        return None
    uniq = list(dict.fromkeys(candidates))
    if len(uniq) == 1:
        return uniq[0]
    for c in uniq:
        if os.path.isfile(full_save_manifest_path_json(session_dir, c)):
            return c
    return uniq[0]


def is_neutral_session_folder(session_dir: str) -> bool:
    """True if ``session_dir`` looks like a neutral comparison session output folder."""
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    if not os.path.isdir(session_dir):
        return False
    if infer_session_id(session_dir):
        return True
    return os.path.isdir(os.path.join(session_dir, NEUTRAL_COMPARISON_PARALLEL_SHARD_SUBDIR, "primary"))


def discover_session_dirs_under_output_root(output_root: str) -> List[str]:
    """Return sorted session folder paths directly under ``output_root``."""
    output_root = os.path.abspath(os.path.expanduser(output_root))
    if not os.path.isdir(output_root):
        return []
    found: List[str] = []
    try:
        names = sorted(os.listdir(output_root))
    except OSError:
        return []
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(output_root, name)
        if os.path.isdir(path) and is_neutral_session_folder(path):
            found.append(path)
    return found


def find_summary_json_path(session_dir: str, session_id: str) -> Optional[str]:
    session_dir = os.path.abspath(session_dir)
    exact = os.path.join(session_dir, f"neutral_set_comparison_{session_id}.json")
    if os.path.isfile(exact):
        return exact
    exact_gz = exact + ".gz"
    if os.path.isfile(exact_gz):
        return exact_gz
    hits = sorted(glob.glob(os.path.join(session_dir, "neutral_set_comparison_*.json*")))
    return hits[0] if hits else None


def expected_png_path(session_dir: str, session_id: str) -> str:
    return os.path.join(session_dir, f"neutral_primary_events_{session_id}.png")


def infer_run_headline(session_dir: str, session_id: str) -> Optional[str]:
    """
    Short job label for PNG suptitle (e.g. ``justDup``, ``Death+Dup``), matching histogram PNGs.

    Prefer ``headless_neutral_job_file_stem`` from the session summary, then the logical session id,
    then the output folder basename suffix (``job<id>-<time>_<label>``).
    """
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    sid = (session_id or "").strip()

    sp = find_summary_json_path(session_dir, sid) if sid else None
    if sp:
        try:
            payload = _read_json_maybe_gz(sp)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            stem = payload.get("headless_neutral_job_file_stem")
            if isinstance(stem, str) and stem.strip():
                short = neutral_session_id_output_root_folder_suffix(stem.strip())
                if short:
                    return short
                short = neutral_session_id_output_root_folder_suffix(sid)
                if short:
                    return short
                return stem.strip()

    if sid:
        short = neutral_session_id_output_root_folder_suffix(sid)
        if short:
            return short

    base = os.path.basename(session_dir)
    if "_" in base:
        tail = base.rsplit("_", 1)[-1].strip()
        if tail:
            return tail
    return None


def build_primary_events_chart_for_session(
    session_dir: str,
    session_id: str,
    *,
    write_png: bool = True,
    run_headline: Optional[str] = None,
    log_png_errors: bool = False,
) -> Tuple[Dict[str, Any], Optional[str], int]:
    """
    Load offload, summarize, optionally write PNG.

    Returns ``(summary_dict, png_abs_path_or_none, n_primary_records)``.
    """
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is empty")
    records = load_primary_offload_records(session_dir, sid, _read_json_maybe_gz)
    summary = summarize_primary_events_by_hit(records)
    summary["chart_tool_version"] = PRIMARY_EVENTS_CHART_VERSION
    summary["session_dir"] = session_dir
    summary["session_id"] = sid
    summary["n_primary_offload_records"] = len(records)
    png_path: Optional[str] = None
    if write_png:
        png_path = write_primary_event_rates_png(
            folder=session_dir,
            session_id=sid,
            summary=summary,
            run_headline=run_headline,
            log_errors=log_png_errors,
        )
    return summary, png_path, len(records)


def _print_summary_human(summary: Dict[str, Any], *, prefix: str = "") -> None:
    head = f"{prefix}\n" if prefix else ""
    if head:
        print(head.rstrip())
    for label, key in (("Meets threshold", "meets_threshold"), ("Below threshold", "below_threshold")):
        side = summary.get(key) if isinstance(summary.get(key), dict) else {}
        n = int(side.get("n") or 0)
        print(f"\n{label} (n={n}):")
        if n == 0:
            print("  (no runs)")
            continue
        for mk, name in (
            ("deaths_mean", "deaths"),
            ("duplications_mean", "duplications"),
            ("mutations_mean", "mutations"),
            ("flow_removed_mean", "flow_removed"),
        ):
            mu = side.get(mk)
            ci_key = mk.replace("_mean", "_ci95")
            err = side.get(ci_key)
            if mu is None:
                continue
            if err is not None:
                print(f"  {name}: {float(mu):.4f} ± {float(err):.4f} per generation (95% CI)")
            else:
                print(f"  {name}: {float(mu):.4f} per generation")
    skipped = int(summary.get("skipped_no_change_history") or 0)
    if skipped:
        print(f"\nSkipped {skipped} primary row(s) with no change_history.")


def _update_summary_json(session_dir: str, sid: str, summary: Dict[str, Any], png_path: Optional[str]) -> None:
    sp = find_summary_json_path(session_dir, sid)
    if not sp:
        print("  Warning: --update-summary: no neutral_set_comparison_*.json in session dir.", file=sys.stderr)
        return
    payload = _read_json_maybe_gz(sp)
    if not isinstance(payload, dict):
        raise ValueError("summary file is not a JSON object")
    payload["primary_event_summary"] = summary
    if png_path:
        payload["primary_events_png"] = os.path.basename(png_path)
    payload["primary_events_chart_regenerated"] = PRIMARY_EVENTS_CHART_VERSION
    _write_json(sp, payload)
    print(f"  Updated summary: {sp}")


def process_one_session(
    session_dir: str,
    *,
    session_id_override: Optional[str] = None,
    write_png: bool = True,
    run_headline: Optional[str] = None,
    update_summary: bool = False,
    summary_json: Optional[str] = None,
    skip_existing: bool = False,
    verbose: bool = True,
) -> Tuple[int, str]:
    """
    Process a single session folder. Returns ``(exit_code, status_message)``.
    exit_code 0 = success, 1 = failure, 2 = skipped.
    """
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    sid = (session_id_override or "").strip() or infer_session_id(session_dir)
    if not sid:
        return 1, "no session id (missing manifest/summary)"

    if skip_existing and write_png and os.path.isfile(expected_png_path(session_dir, sid)):
        return 2, "skipped (PNG already exists)"

    headline = (run_headline or "").strip() or infer_run_headline(session_dir, sid)

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"[primary_events_chart] {session_dir}")
        print(f"[primary_events_chart] session_id={sid!r}")
        if headline:
            print(f"[primary_events_chart] headline={headline!r}")

    try:
        summary, png_path, n_recs = build_primary_events_chart_for_session(
            session_dir,
            sid,
            write_png=write_png,
            run_headline=headline,
            log_png_errors=verbose,
        )
    except Exception as exc:
        return 1, f"error: {exc}"

    if verbose:
        print(f"[primary_events_chart] primary offload records loaded: {n_recs}")
        _print_summary_human(summary)

    if write_png and not png_path:
        meets = summary.get("meets_threshold") or {}
        below = summary.get("below_threshold") or {}
        n_ok = int(meets.get("n") or 0) + int(below.get("n") or 0)
        if n_recs == 0:
            return 1, "no primary offload records"
        if n_ok == 0:
            return 1, f"{n_recs} records but no change_history"
        png_err = pop_last_primary_event_png_error()
        return 1, png_err or "matplotlib missing or PNG save failed"

    if verbose and png_path:
        print(f"  Wrote PNG: {png_path}")

    if summary_json:
        out = os.path.abspath(os.path.expanduser(summary_json))
        _write_json(out, summary)
        if verbose:
            print(f"  Wrote summary JSON: {out}")

    if update_summary:
        try:
            _update_summary_json(session_dir, sid, summary, png_path)
        except Exception as exc:
            return 1, f"summary update failed: {exc}"

    if write_png and png_path:
        return 0, os.path.basename(png_path)
    if not write_png:
        return 0, "ok (no png)"
    return 1, "unknown failure"


def process_session_dirs_batch(
    session_dirs: Sequence[str],
    *,
    output_root_label: Optional[str] = None,
    session_id_override: Optional[str] = None,
    write_png: bool = True,
    run_headline: Optional[str] = None,
    update_summary: bool = False,
    skip_existing: bool = False,
) -> int:
    """Process a list of session folder paths (batch mode)."""
    session_dirs = [os.path.abspath(os.path.expanduser(d)) for d in session_dirs]
    if not session_dirs:
        print("Error: no session folders to process.", file=sys.stderr)
        return 1

    if output_root_label:
        print(f"[primary_events_chart] output-root={output_root_label}")
    print(f"[primary_events_chart] processing {len(session_dirs)} session folder(s)")

    ok_n = fail_n = skip_n = 0
    for session_dir in session_dirs:
        code, msg = process_one_session(
            session_dir,
            session_id_override=session_id_override,
            write_png=write_png,
            run_headline=run_headline,
            update_summary=update_summary,
            skip_existing=skip_existing,
            verbose=True,
        )
        if code == 0:
            ok_n += 1
            print(f"  => OK: {msg}")
        elif code == 2:
            skip_n += 1
            print(f"  => SKIP: {msg}")
        else:
            fail_n += 1
            print(f"  => FAIL: {msg}", file=sys.stderr)

    print(f"\n{'=' * 72}")
    print(f"[primary_events_chart] done: {ok_n} ok, {skip_n} skipped, {fail_n} failed (of {len(session_dirs)} folders)")
    return 0 if fail_n == 0 else 1


def _resolve_paths(
    path: Optional[str],
    session_dir: Optional[str],
    output_root: Optional[str],
) -> Tuple[str, List[str], Optional[str]]:
    """
    Return ``(mode, session_dirs, output_root_label)``.

    * batch: one or more session folders (siblings under an Output directory)
    * single: exactly one session folder
    """
    if session_dir and output_root:
        raise ValueError("Use only one of --session-dir or --output-root, not both.")
    if session_dir:
        d = os.path.abspath(os.path.expanduser(session_dir.strip()))
        return "single", [d], None
    if output_root:
        root = os.path.abspath(os.path.expanduser(output_root.strip()))
        subs = discover_session_dirs_under_output_root(root)
        if not subs:
            raise ValueError(
                f"No neutral session folders found under {root!r} "
                "(expected subdirs with full_save_manifest_* or _parallel_shards/primary/)."
            )
        return "batch", subs, root

    if not (path or "").strip():
        raise ValueError("Provide PATH, --session-dir, or --output-root.")

    p = os.path.abspath(os.path.expanduser(path.strip()))
    if not os.path.isdir(p):
        raise ValueError(f"not a directory: {p!r}")

    subs = discover_session_dirs_under_output_root(p)
    if len(subs) > 1:
        return "batch", subs, p
    if len(subs) == 1:
        return "single", subs, p
    if is_neutral_session_folder(p):
        return "single", [p], None
    raise ValueError(
        f"No neutral session folders found at {p!r} or in its immediate subdirectories."
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build neutral_primary_events_<session>.png from existing neutral comparison "
            "session folder(s) (no re-simulation)."
        ),
    )
    ap.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Session folder or Output parent directory (auto-detect if neither --session-dir nor --output-root).",
    )
    ap.add_argument(
        "--session-dir",
        default=None,
        metavar="DIR",
        help="Single session output folder.",
    )
    ap.add_argument(
        "--output-root",
        default=None,
        metavar="DIR",
        help="Process every neutral session folder directly under this directory.",
    )
    ap.add_argument(
        "--session-id",
        default=None,
        help="Logical session id (single-folder mode only; default: infer per folder).",
    )
    ap.add_argument(
        "--headline",
        default=None,
        help="Optional suptitle on each PNG.",
    )
    ap.add_argument(
        "--update-summary",
        action="store_true",
        help="Merge primary_event_summary and primary_events_png into neutral_set_comparison_*.json.",
    )
    ap.add_argument(
        "--summary-json",
        default=None,
        metavar="PATH",
        help="Also write numeric summary to this JSON path (single-folder mode only).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip folders that already contain neutral_primary_events_<session>.png.",
    )
    ap.add_argument(
        "--no-png",
        action="store_true",
        help="Do not write PNG; print summary statistics only.",
    )
    ap.add_argument(
        "--version",
        action="store_true",
        help="Print tool version and exit.",
    )
    args = ap.parse_args(argv)

    if args.version:
        print(f"primary_events_chart {PRIMARY_EVENTS_CHART_VERSION}")
        return 0

    try:
        mode, session_dirs, root_label = _resolve_paths(args.path, args.session_dir, args.output_root)
    except ValueError as exc:
        ap.error(str(exc))
        return 1

    headline = (args.headline or "").strip() or None
    write_png = not bool(args.no_png)

    if mode == "batch":
        if args.session_id:
            print(
                "Warning: --session-id is ignored in batch mode (each folder uses its own inferred id).",
                file=sys.stderr,
            )
        if args.summary_json:
            ap.error("--summary-json is only valid with a single --session-dir.")
        return process_session_dirs_batch(
            session_dirs,
            output_root_label=root_label,
            write_png=write_png,
            run_headline=headline,
            update_summary=bool(args.update_summary),
            skip_existing=bool(args.skip_existing),
        )

    # single
    session_dir = session_dirs[0]
    if not os.path.isdir(session_dir):
        print(f"Error: not a directory: {session_dir!r}", file=sys.stderr)
        return 1

    code, msg = process_one_session(
        session_dir,
        session_id_override=(args.session_id or "").strip() or None,
        write_png=write_png,
        run_headline=headline,
        update_summary=bool(args.update_summary),
        summary_json=(args.summary_json or "").strip() or None,
        skip_existing=bool(args.skip_existing),
        verbose=True,
    )
    if code == 2:
        print(f"Skipped: {msg}")
        return 0
    if code != 0:
        print(f"Failed: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
