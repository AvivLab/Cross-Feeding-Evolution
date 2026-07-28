#!/usr/bin/env python3
"""Build Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv from campaign sessions.

Scans an output root for Batch Runner / headless campaign folders, aggregates
per-batch hit counts, and writes the compare CSV that ``assemble_figure_data.py``
expects. Optionally stages in-session ``Re-Runs/`` (and ``Re-Runs-NonHits/``)
into the ``Re-Runs/sessions/`` layout used by figure assembly.

Typical layout after running paper settings into one root::

    OUTPUT_ROOT/
      Fixed_1_ratio_Neutral/          # or any folder names
        primary_batch_campaign_*.json
        batch_hit_counts_*.csv
        Re-Runs/primary_hit_rescreen_*.csv
      ...

Usage (from ``tools/Figure Reconstruction``)::

    export PYTHONPATH="scripts${PYTHONPATH:+:$PYTHONPATH}"
    python3 scripts/build_summary_hit_counts.py --output-root /path/to/OUTPUT_ROOT
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from plot_primary_batch_violins import MAIN_CONFIG_ORDER, workspaces_output
from plot_ratio_supplementary import SUITE_ORDER

PAPER_SUITE_TAGS = frozenset(suite for suite, _, _ in SUITE_ORDER)
PAPER_CONFIG_KEYS = frozenset(key for key, _, _ in MAIN_CONFIG_ORDER)

COMPARE_FIELDS: Sequence[str] = (
    "plot_label",
    "configuration",
    "suite",
    "session_dir",
    "batch_index",
    "hit_count",
    "n_runs",
)

CONFIG_ALIASES = {
    "neutral": "Neutral",
    "trueneutral": "Neutral",
    "death+dup": "Death+Dup",
    "death+duplication": "Death+Dup",
    "selection": "Death+Dup",
    "selection regime": "Death+Dup",
    "justdeath": "justDeath",
    "justdup": "justDup",
}

CONFIG_TO_SESSION_STEM = {
    "Neutral": ("a", "trueNeutral"),
    "Death+Dup": ("e", "Death+Dup"),
    "justDeath": ("c", "justDeath"),
    "justDup": ("d", "justDup"),
    "trueNeutral2": ("aa", "trueNeutral2"),
    "binary_death": ("b", "binary_death"),
}

_SUITE_FROM_PATH_RE = re.compile(r"(Fixed_[0-9.]+_ratio)")
_CAMPAIGN_JSON_RE = re.compile(r"^primary_batch_campaign_.+\.json$")
_HIT_CSV_GLOBS = (
    "batch_hit_counts_*.csv",
    "primary_batch_hit_counts_*.csv",
)


def _normalize_config(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    key = re.sub(r"\s+", " ", text).lower()
    key = key.replace(" ", "")
    if key in CONFIG_ALIASES:
        return CONFIG_ALIASES[key]
    spaced = re.sub(r"\s+", " ", text).strip().lower()
    if spaced in CONFIG_ALIASES:
        return CONFIG_ALIASES[spaced]
    # Preserve known paper keys as-is.
    for known in PAPER_CONFIG_KEYS | set(CONFIG_TO_SESSION_STEM):
        if text == known or text.lower() == known.lower():
            return known
    return text


def _format_y(acetate_ratio: float) -> str:
    if acetate_ratio == int(acetate_ratio):
        return str(int(acetate_ratio))
    return f"{acetate_ratio:g}"


def _suite_from_acetate(numeric: Mapping[str, Any] | None) -> Tuple[str, str]:
    if not numeric or "Acetate Ratio" not in numeric:
        return "", ""
    try:
        val = float(numeric["Acetate Ratio"])
    except (TypeError, ValueError):
        return "", ""
    y = _format_y(val)
    return f"Fixed_{y}_ratio", y


def _suite_from_path(path: Path) -> str:
    for part in path.parts:
        m = _SUITE_FROM_PATH_RE.fullmatch(part)
        if m:
            return m.group(1)
    return ""


def _infer_config_from_toggles(toggles: Mapping[str, Any] | None, bounds: Mapping[str, Any] | None) -> str:
    """Match gui.apps.batch_runner.csv_output.infer_configuration for paper keys."""
    tog = toggles or {}
    bound_keys = set((bounds or {}).keys())

    def _flag(name: str) -> bool:
        raw = tog.get(name, False)
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    no_death = _flag("No Death")
    const_death = _flag("Constant Death Probability")
    const_dup = _flag("Constant Duplication Probability")
    has_death_decay = "Death Decay Rate" in bound_keys
    has_dup_sigmoid = (
        "Duplication Sigmoid Intensity" in bound_keys
        or "Duplication Sigmoid Midpoint" in bound_keys
    )
    if no_death and const_dup and not has_death_decay:
        return "Neutral"
    if has_death_decay and has_dup_sigmoid:
        return "Death+Dup"
    if has_death_decay:
        return "justDeath"
    if no_death and has_dup_sigmoid:
        return "justDup"
    if const_death and const_dup:
        return "Death+Dup"
    if const_death:
        return "justDeath"
    if const_dup and not no_death:
        return "justDup"
    return ""


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_campaign_jsons(output_root: Path) -> Iterator[Path]:
    skip = {"Summary", "Re-Runs", "Re-Runs-NonHits", "data", "output", ".git"}
    for path in sorted(output_root.rglob("primary_batch_campaign_*.json")):
        if any(part in skip for part in path.parts):
            continue
        if path.name.endswith(".gz"):
            continue
        if not _CAMPAIGN_JSON_RE.match(path.name):
            continue
        yield path


def _hit_csv_candidates(session_dir: Path) -> List[Path]:
    found: List[Path] = []
    for pattern in _HIT_CSV_GLOBS:
        found.extend(session_dir.glob(pattern))
    return sorted({p.resolve() for p in found if p.is_file()})


def _tally_hit_counts_from_csv(csv_path: Path) -> List[int]:
    """Return per-batch hit counts from a session hit CSV (legacy or hit-only)."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []

    # Legacy / compare schema: one row per batch with hit_count.
    if "hit_count" in rows[0] and all(
        (row.get("hit_index") in (None, "", "nan") for row in rows[:5])
    ):
        by_batch: Dict[int, int] = {}
        for row in rows:
            try:
                batch_idx = int(row.get("batch_index") or 0)
                by_batch[batch_idx] = int(float(row["hit_count"]))
            except (TypeError, ValueError):
                continue
        if by_batch:
            max_b = max(by_batch)
            return [int(by_batch.get(i, 0)) for i in range(1, max_b + 1)]

    # Hit-only schema: one row per primary hit.
    tallies: Dict[int, int] = defaultdict(int)
    for row in rows:
        # Skip explicit non-hit rows if present.
        row_type = str(row.get("row_type") or "").strip().lower()
        if row_type in {"batch", "primary_batch"}:
            try:
                batch_idx = int(row.get("batch_index") or 0)
                tallies[batch_idx] = int(float(row.get("hit_count") or 0))
            except (TypeError, ValueError):
                continue
            continue
        if row_type and row_type not in {"hit", "rescreen_hit", ""}:
            continue
        try:
            batch_idx = int(row.get("batch_index") or 0)
        except (TypeError, ValueError):
            continue
        if batch_idx < 1:
            continue
        tallies[batch_idx] += 1
    if not tallies:
        return []
    max_b = max(tallies)
    return [int(tallies.get(i, 0)) for i in range(1, max_b + 1)]


def _counts_from_campaign_json(data: Mapping[str, Any]) -> List[int]:
    raw = data.get("primary_hit_counts")
    if not isinstance(raw, list) or not raw:
        return []
    out: List[int] = []
    for item in raw:
        out.append(int(item))
    return out


def _metadata_from_session(
    session_dir: Path,
    campaign_path: Optional[Path],
) -> Tuple[str, str, str, List[int], int]:
    """Return configuration, suite, plot_label, hit_counts, n_runs."""
    configuration = ""
    suite = ""
    plot_label = ""
    counts: List[int] = []
    n_runs = 0
    data: Dict[str, Any] = {}

    if campaign_path and campaign_path.is_file():
        data = _load_json(campaign_path)
        counts = _counts_from_campaign_json(data)
        try:
            n_runs = int(data.get("n_runs") or 0)
        except (TypeError, ValueError):
            n_runs = 0
        configuration = _normalize_config(data.get("configuration_name"))
        if not configuration:
            configuration = _normalize_config(
                _infer_config_from_toggles(
                    data.get("primary_toggles") if isinstance(data.get("primary_toggles"), dict) else {},
                    data.get("primary_bounds") if isinstance(data.get("primary_bounds"), dict) else {},
                )
            )
        suite, _y = _suite_from_acetate(
            data.get("primary_numeric_parameters")
            if isinstance(data.get("primary_numeric_parameters"), dict)
            else {}
        )

    hit_csvs = _hit_csv_candidates(session_dir)
    hit_csv = hit_csvs[0] if hit_csvs else None
    if hit_csv is not None:
        with hit_csv.open(newline="", encoding="utf-8") as fh:
            sample = next(csv.DictReader(fh), None)
        if sample:
            configuration = configuration or _normalize_config(sample.get("configuration"))
            suite = suite or str(sample.get("suite") or "").strip()
            plot_label = str(sample.get("plot_label") or "").strip()
        if not counts:
            counts = _tally_hit_counts_from_csv(hit_csv)

    suite = suite or _suite_from_path(session_dir)
    if not suite and campaign_path:
        suite = _suite_from_path(campaign_path)

    if not plot_label:
        sid = ""
        if campaign_path:
            sid = campaign_path.stem.replace("primary_batch_campaign_", "", 1)
        plot_label = f"{configuration or 'campaign'}_{sid or session_dir.name}"
        if suite:
            plot_label = f"{plot_label} [{suite}]"

    if n_runs < 1 and counts:
        # Fallback: at least as large as the max observed hit count.
        n_runs = max(int(max(counts)), 1)

    return configuration, suite, plot_label, counts, n_runs


def collect_campaigns(output_root: Path) -> List[Tuple[str, str, str, str, List[int], int]]:
    """Return (plot_label, configuration, suite, session_dir, hit_counts, n_runs)."""
    best: Dict[Tuple[str, str], Tuple[float, Path, str, List[int], int]] = {}

    campaign_jsons = list(_iter_campaign_jsons(output_root))
    sessions: Dict[Path, Optional[Path]] = {}
    for path in campaign_jsons:
        sessions[path.parent.resolve()] = path

    # Also pick up sessions that only have hit CSVs (no campaign JSON).
    for pattern in _HIT_CSV_GLOBS:
        for csv_path in output_root.rglob(pattern):
            if any(part in {"Summary", "Re-Runs", "Re-Runs-NonHits"} for part in csv_path.parts):
                continue
            sessions.setdefault(csv_path.parent.resolve(), None)

    for session_dir, campaign_path in sorted(sessions.items(), key=lambda item: str(item[0])):
        configuration, suite, plot_label, counts, n_runs = _metadata_from_session(
            session_dir, campaign_path
        )
        if not counts:
            continue
        if suite not in PAPER_SUITE_TAGS:
            continue
        if configuration not in PAPER_CONFIG_KEYS:
            continue
        key = (suite, configuration)
        mtime = 0.0
        if campaign_path and campaign_path.is_file():
            mtime = campaign_path.stat().st_mtime
        else:
            hit_csvs = _hit_csv_candidates(session_dir)
            if hit_csvs:
                mtime = hit_csvs[0].stat().st_mtime
        prev = best.get(key)
        if prev is None or mtime >= prev[0]:
            best[key] = (mtime, session_dir, plot_label, counts, n_runs)

    rows: List[Tuple[str, str, str, str, List[int], int]] = []
    for suite, _, _ in SUITE_ORDER:
        for configuration, _, _ in MAIN_CONFIG_ORDER:
            entry = best.get((suite, configuration))
            if entry is None:
                continue
            _mtime, session_dir, plot_label, counts, n_runs = entry
            rows.append(
                (plot_label, configuration, suite, str(session_dir), list(counts), int(n_runs))
            )
    return rows


def write_compare_csv(
    path: Path,
    campaigns: Sequence[Tuple[str, str, str, str, List[int], int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COMPARE_FIELDS))
        writer.writeheader()
        for plot_label, configuration, suite, session_dir, counts, n_runs in campaigns:
            for batch_idx, hit_count in enumerate(counts, start=1):
                writer.writerow(
                    {
                        "plot_label": plot_label,
                        "configuration": configuration,
                        "suite": suite,
                        "session_dir": session_dir,
                        "batch_index": batch_idx,
                        "hit_count": int(hit_count),
                        "n_runs": int(n_runs) if n_runs > 0 else "",
                    }
                )


def _session_stem(configuration: str, suite: str) -> Optional[str]:
    mapping = CONFIG_TO_SESSION_STEM.get(configuration)
    if mapping is None:
        return None
    prefix, stem = mapping
    m = re.match(r"^Fixed_(?P<y>.+)_ratio$", suite)
    if not m:
        return None
    return f"{prefix}_{stem}_Fixed_{m.group('y')}"


def stage_rescreen_sessions(
    campaigns: Sequence[Tuple[str, str, str, str, List[int], int]],
    output_root: Path,
) -> Tuple[int, int]:
    """Copy/symlink in-session Re-Runs into output_root/Re-Runs/sessions/<stem>/."""
    staged_hits = 0
    staged_non = 0
    for _plot, configuration, suite, session_dir_s, _counts, _n_runs in campaigns:
        stem = _session_stem(configuration, suite)
        if not stem:
            continue
        session_dir = Path(session_dir_s)
        for mode, counter_name in (("Re-Runs", "hits"), ("Re-Runs-NonHits", "non")):
            src = session_dir / mode
            if not src.is_dir():
                continue
            patterns = (
                ("primary_hit_rescreen_*.csv", "primary_hit_rescreen_*.json")
                if mode == "Re-Runs"
                else ("primary_non_hit_rescreen_*.csv", "primary_non_hit_rescreen_*.json")
            )
            files = []
            for pattern in patterns:
                files.extend(src.glob(pattern))
            # Also accept files written with session id in the name.
            if not files:
                files = list(src.glob("*.csv")) + list(src.glob("*.json"))
            if not files:
                continue
            dest = output_root / mode / "sessions" / stem
            dest.mkdir(parents=True, exist_ok=True)
            for path in files:
                target = dest / path.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                try:
                    target.symlink_to(path.resolve())
                except OSError:
                    shutil.copy2(path, target)
            if counter_name == "hits":
                staged_hits += 1
            else:
                staged_non += 1
    return staged_hits, staged_non


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Folder containing campaign session directories (default: Workspaces/Output if present)",
    )
    parser.add_argument(
        "--write-ratio",
        type=Path,
        default=None,
        help="Output CSV path (default: <output-root>/Summary/Summary_Ratio/primary_batch_compare_hit_counts.csv)",
    )
    parser.add_argument(
        "--stage-rescreens",
        action="store_true",
        default=True,
        help="Stage Re-Runs/ and Re-Runs-NonHits/ into sessions/ layout (default: on)",
    )
    parser.add_argument(
        "--no-stage-rescreens",
        action="store_false",
        dest="stage_rescreens",
        help="Do not copy/symlink re-screen outputs into Re-Runs/sessions/",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_root = (args.output_root or workspaces_output()).resolve()
    if not output_root.is_dir():
        raise SystemExit(f"Output root not found: {output_root}")

    campaigns = collect_campaigns(output_root)
    if not campaigns:
        raise SystemExit(
            f"No paper campaign sessions found under {output_root}. "
            "Run Batch Runner / headless campaigns for Neutral and Death+Dup "
            f"across suites {', '.join(sorted(PAPER_SUITE_TAGS))}."
        )

    ratio_path = args.write_ratio or (
        output_root / "Summary" / "Summary_Ratio" / "primary_batch_compare_hit_counts.csv"
    )
    write_compare_csv(ratio_path, campaigns)
    print(f"Output root: {output_root}")
    print(f"Campaigns: {len(campaigns)}")
    print(f"Wrote: {ratio_path}")

    if args.stage_rescreens:
        n_hits, n_non = stage_rescreen_sessions(campaigns, output_root)
        print(f"Staged re-screen sessions: hits={n_hits} non-hits={n_non}")
        print(f"  {output_root / 'Re-Runs' / 'sessions'}")
        if n_non:
            print(f"  {output_root / 'Re-Runs-NonHits' / 'sessions'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
