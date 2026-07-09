#!/usr/bin/env python3
"""Embed neutral hit parameters (PCA / UMAP / t-SNE) colored by re-screen outcome.

Caches expensive embeddings and bare panel PNGs under
``figures/Extra/html/cache/true_neutral_param_embedding/`` so layout tweaks
(colorbar, labels, suptitle) can re-assemble quickly without re-running UMAP/t-SNE.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.image import imread
from matplotlib.lines import Line2D

from plot_hit_rescreen_panel import N_RESCREEN, workspaces_re_runs
from plot_ratio_supplementary import SUITE_ORDER, panel_label

# Varying sampled parameters (fixed campaign settings excluded).
PARAM_FEATURES: Tuple[str, ...] = (
    "Cost of Life",
    "Flow Percentage",
    "Initial A",
    "Initial Energy",
    "Mutation Rate",
    "Mutation Scale",
)
LOG10_FEATURES = frozenset({"Flow Percentage", "Mutation Rate"})
RF_NO_FLOW_EXCLUDE = "Flow Percentage"
EXCLUDED_FLOW_AND_MUTATION = frozenset({"Flow Percentage", "Mutation Rate"})
EMBEDDING_PANEL_LETTERS = ("a", "b", "c", "d", "e", "f")
RF_PANEL_LETTERS = ("a", "b", "c", "d", "e", "f")
LR_PANEL_LETTERS = ("a", "b", "c", "d", "e", "f")
DT_PANEL_LETTERS = ("a", "b", "c")
PANEL_FIGSIZE = (5.6, 4.4)
DT_PANEL_FIGSIZE = (11.0, 4.2)
COMPOSITE_FIGSIZE = (12.0, 14.0)
RF_FIGSIZE = (12.0, 15.5)
LR_FIGSIZE = (12.0, 15.5)
DT_FIGSIZE = (12.0, 15.0)
DT_ROW_HEIGHT_IN = 4.8
DT_ROW_GAP_IN = 2.6
DT_FIG_OVERHEAD_IN = 3.4
DT_PANEL_DPI = 260
DT_MAX_DEPTH = 4
DT_MIN_SAMPLES_LEAF = 50
DT_EXTREME_LOW_CEILING = 10
DT_EXTREME_HIGH_FLOOR = 18
PARAM_3D_FIGSIZE = (9.5, 7.5)
PARAM_3D_SUCCESS_THRESHOLD = 10
PARAM_3D_AXES = ("Cost of Life", "Initial A", "Initial Energy")
PARAM_3D_AXIS_LO_FLOOR: Dict[str, float] = {
    "Initial A": 0.0,
    "Initial Energy": 0.0,
}
PARAM_FILTER_COLORS: Tuple[str, ...] = (
    "#c0392b",
    "#1e8449",
    "#2471a3",
    "#8e44ad",
    "#d68910",
    "#117a65",
)
# Leave room for axis labels when caching bare panels (re-stitched with imshow).
PANEL_MARGINS_SCATTER = dict(left=0.18, right=0.96, top=0.88, bottom=0.16)
PANEL_MARGINS_HIST = dict(left=0.16, right=0.96, top=0.88, bottom=0.22)
PANEL_MARGINS_BARH = dict(left=0.34, right=0.96, top=0.88, bottom=0.16)
PANEL_MARGINS_TREE = dict(left=0.01, right=0.99, top=0.93, bottom=0.03)
CACHE_VERSION = 11

HitRecord = Dict[str, object]
CacheData = Dict[str, object]


def _positive_class_proba(clf, X: np.ndarray) -> np.ndarray:
    """Return P(positive class); degenerate when fit saw only one class."""
    proba = clf.predict_proba(X)
    if proba.shape[1] == 1:
        only = int(clf.classes_[0])
        fill = 1.0 if only == 1 else 0.0
        return np.full(len(X), fill, dtype=np.float32)
    pos = 1 if 1 in clf.classes_ else int(clf.classes_[-1])
    col = list(clf.classes_).index(pos)
    return proba[:, col].astype(np.float32)


def _degenerate_binary_cv(y: np.ndarray) -> Tuple[float, float]:
    """CV accuracy when ``y`` has a single class (StratifiedKFold is undefined)."""
    return 1.0, 0.0


def default_html_figures_dir(root: Path) -> Path:
    return root / "figures/Extra/html/figures"


def default_cache_dir(root: Path) -> Path:
    return root / "figures/Extra/html/cache/true_neutral_param_embedding"


def load_true_neutral_hits(sessions_dir: Path) -> List[HitRecord]:
    """Load all primary-hit re-screen rows for ``a_trueNeutral*`` sessions."""
    rows: List[HitRecord] = []
    if not sessions_dir.is_dir():
        return rows
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        name = session_dir.name
        if not name.startswith("a_trueNeutral") or name.startswith("aa_"):
            continue
        csvs = sorted(session_dir.glob("primary_hit_rescreen_*.csv"))
        if not csvs:
            continue
        with csvs[0].open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                params = json.loads(row["params_json"])
                rows.append(
                    {
                        "session_id": row["session_id"],
                        "suite_tag": true_neutral_suite_tag(row["session_id"]),
                        "hit_index": int(row["hit_index"]),
                        "n_hit_again": int(row["n_hit_again"]),
                        "hit_rate": float(row["hit_rate"]),
                        "params": params,
                    }
                )
    return rows


def split_extreme_gap_rows(
    rows: Sequence[HitRecord],
    *,
    low_ceiling: int = DT_EXTREME_LOW_CEILING,
    high_floor: int = DT_EXTREME_HIGH_FLOOR,
) -> Tuple[List[HitRecord], np.ndarray, int]:
    """Keep hits with re-hit count $<$ low_ceiling or $>$ high_floor."""
    kept: List[HitRecord] = []
    high_success: List[bool] = []
    n_excluded = 0
    for row in rows:
        n_hit = int(row["n_hit_again"])
        if n_hit < low_ceiling:
            kept.append(row)
            high_success.append(False)
        elif n_hit > high_floor:
            kept.append(row)
            high_success.append(True)
        else:
            n_excluded += 1
    return kept, np.array(high_success, dtype=bool), n_excluded


def compute_dt_models_three_variants(
    rows: Sequence[HitRecord],
    high_success: np.ndarray,
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    X, _ = build_feature_matrix(rows)
    no_flow_features = param_features_without_flow()
    X_no_flow, _ = build_feature_matrix(rows, features=no_flow_features)
    no_flow_mut_features = param_features_without_flow_and_mutation()
    X_no_flow_mut, _ = build_feature_matrix(rows, features=no_flow_mut_features)
    return (
        compute_decision_tree(X, high_success),
        compute_decision_tree(X_no_flow, high_success),
        compute_decision_tree(X_no_flow_mut, high_success),
    )


def build_feature_matrix(
    rows: Sequence[HitRecord],
    *,
    features: Sequence[str] = PARAM_FEATURES,
) -> Tuple[np.ndarray, List[str]]:
    """Return scaled feature matrix and feature labels."""
    from sklearn.preprocessing import StandardScaler

    raw = np.array(
        [
            [
                float(r["params"][name])
                if name not in LOG10_FEATURES
                else np.log10(max(float(r["params"][name]), 1.0e-12))
                for name in features
            ]
            for r in rows
        ],
        dtype=float,
    )
    labels = [
        f"{name} (log10)" if name in LOG10_FEATURES else name for name in features
    ]
    scaled = StandardScaler().fit_transform(raw)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, labels


def build_raw_feature_matrix(
    rows: Sequence[HitRecord],
    *,
    features: Sequence[str] = PARAM_FEATURES,
) -> Tuple[np.ndarray, List[str]]:
    """Physical parameter values from ``params_json`` (for scatter-plot axes)."""
    raw = np.array(
        [[float(r["params"][name]) for name in features] for r in rows],
        dtype=float,
    )
    return raw, list(features)


def feature_display_names(
    *,
    features: Sequence[str] = PARAM_FEATURES,
    raw: bool = False,
) -> List[str]:
    names: List[str] = []
    for name in features:
        if name == "Initial A":
            names.append("Initial Task A")
        elif name in LOG10_FEATURES and not raw:
            names.append(f"{name} (log10)")
        else:
            names.append(name)
    return names


def tree_feature_display_names(
    *,
    features: Sequence[str] = PARAM_FEATURES,
) -> List[str]:
    """Short labels for decision-tree nodes to reduce box overlap."""
    short = {
        "Cost of Life": "Cost life",
        "Flow Percentage": "Flow log10",
        "Initial A": "Init. A",
        "Initial Energy": "Init. E",
        "Mutation Rate": "Mut. log10",
        "Mutation Scale": "Mut. scale",
    }
    return [short.get(name, name) for name in features]


def tree_panel_layout(clf: object) -> Tuple[Tuple[float, float], int]:
    """Return (figsize, fontsize) so node boxes stay large and legible."""
    tree = clf.tree_
    n_leaves = max(2, int(np.sum(tree.children_left == -1)))
    depth = int(tree.max_depth) + 1
    fontsize = 14
    if n_leaves > 18:
        fontsize = 12
    if n_leaves > 26:
        fontsize = 11
    # Keep canvas modest; scale with fontsize so boxes do not shrink on wide trees.
    leaf_pitch_in = 0.9 * (fontsize / 12.0)
    level_pitch_in = 1.0 * (fontsize / 12.0)
    width = max(14.0, min(32.0, 2.0 + n_leaves * leaf_pitch_in))
    height = max(8.0, min(13.0, 1.5 + depth * level_pitch_in))
    return (width, height), fontsize


def param_features_without_flow() -> Tuple[str, ...]:
    return tuple(name for name in PARAM_FEATURES if name != RF_NO_FLOW_EXCLUDE)


def param_features_without_flow_and_mutation() -> Tuple[str, ...]:
    return tuple(name for name in PARAM_FEATURES if name not in EXCLUDED_FLOW_AND_MUTATION)


def param_3d_axis_label(name: str) -> str:
    if name == "Initial A":
        return "Initial Task A"
    if name == "Initial Energy":
        return "Initial energy"
    if name == "Cost of Life":
        return "Cost of life"
    return name


def param_feature_display_label(name: str) -> str:
    return param_3d_axis_label(name)


def true_neutral_suite_tag(session_id: str) -> str:
    if "_Fixed_" in session_id:
        return f"Fixed_{session_id.split('_Fixed_', 1)[1]}_ratio"
    return "Variable_Ratio"


def suite_tag_display_label(tag: str) -> str:
    for suite, _y, latex in SUITE_ORDER:
        if suite == tag:
            return latex.replace("$", "")
    if tag == "Variable_Ratio":
        return "Variable Y"
    if tag.startswith("Fixed_") and tag.endswith("_ratio"):
        y_text = tag[len("Fixed_") : -len("_ratio")]
        return f"Y={y_text}"
    return tag


def _build_ratio_options(rows: Sequence[HitRecord]) -> List[Dict[str, object]]:
    counts: Counter[str] = Counter(str(r["suite_tag"]) for r in rows)
    ordered_tags: List[str] = []
    seen: set[str] = set()
    for suite, _y, _latex in SUITE_ORDER:
        if suite in counts:
            ordered_tags.append(suite)
            seen.add(suite)
    for tag in sorted(counts.keys()):
        if tag not in seen:
            ordered_tags.append(tag)
    return [
        {
            "tag": tag,
            "label": suite_tag_display_label(tag),
            "count": int(counts[tag]),
        }
        for tag in ordered_tags
    ]


def _ratio_filter_checkboxes_html(ratio_options: Sequence[Dict[str, object]]) -> str:
    blocks: List[str] = []
    for opt in ratio_options:
        tag = str(opt["tag"])
        label = str(opt["label"])
        count = int(opt["count"])
        blocks.append(
            f"""          <label class="ratio-filter-row">
            <input
              type="checkbox"
              class="ratio-filter-cb"
              data-tag="{tag}"
              checked
            />
            <span class="ratio-filter-label" data-tag="{tag}" data-base-label="{label}">{label} ({count:,})</span>
          </label>"""
        )
    return (
        "      <div class=\"ratio-filter-grid\">\n"
        + "\n".join(blocks)
        + "\n      </div>"
    )


def _param_interactive_filter_payload(
    rows: Sequence[HitRecord],
) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    labels: List[str] = []
    ranges: List[List[float]] = []
    values: List[List[float]] = []
    for name in PARAM_FEATURES:
        arr = np.array([float(r["params"][name]) for r in rows], dtype=float)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        floor = PARAM_3D_AXIS_LO_FLOOR.get(name)
        if floor is not None:
            lo = floor
        labels.append(param_feature_display_label(name))
        ranges.append([lo, hi])
        values.append(arr.astype(np.float64).tolist())
    return labels, ranges, values


def _hist_filter_panel_html(labels: Sequence[str], colors: Sequence[str]) -> str:
    blocks: List[str] = []
    for idx, (label, color) in enumerate(zip(labels, colors)):
        blocks.append(
            f"""        <div class="param-filter" data-hist-axis="{idx}">
          <div class="param-filter-head">
            <span class="param-swatch" style="background:{color}"></span>
            <span class="param-filter-name">{label}</span>
          </div>
          <label class="param-range-row">
            Range
            <div class="dual-range">
              <div class="dual-range-track">
                <div
                  class="dual-range-fill"
                  id="hist-fill-{idx}"
                  style="background:{color}"
                ></div>
              </div>
              <input
                type="range"
                class="dual-range-lo"
                id="hist-lo-{idx}"
                min="0"
                max="1000"
                step="1"
                value="0"
              />
              <input
                type="range"
                class="dual-range-hi"
                id="hist-hi-{idx}"
                min="0"
                max="1000"
                step="1"
                value="1000"
              />
            </div>
          </label>
          <div class="param-range-values" id="hist-range-{idx}"></div>
        </div>"""
        )
    return "\n".join(blocks)


def extract_param_3d_arrays(
    rows: Sequence[HitRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cost = np.array([float(r["params"]["Cost of Life"]) for r in rows], dtype=float)
    initial_a = np.array([float(r["params"]["Initial A"]) for r in rows], dtype=float)
    energy = np.array([float(r["params"]["Initial Energy"]) for r in rows], dtype=float)
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    return cost, initial_a, energy, n_hit


def plot_param_3d_scatter_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
) -> int:
    """3D scatter of cost of life, initial Task~A, and initial energy vs re-hit count."""
    rows = load_true_neutral_hits(sessions_dir)
    if not rows:
        raise ValueError(f"No neutral hits found under {sessions_dir}")
    cost, initial_a, energy, n_hit = extract_param_3d_arrays(rows)

    fig = plt.figure(figsize=PARAM_3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    ax.scatter(
        cost,
        initial_a,
        energy,
        c=n_hit,
        cmap="viridis",
        norm=norm,
        s=7.0,
        alpha=0.48,
        linewidths=0.0,
        depthshade=False,
        rasterized=True,
    )
    ax.set_xlabel(param_3d_axis_label(PARAM_3D_AXES[0]), labelpad=8)
    ax.set_ylabel(param_3d_axis_label(PARAM_3D_AXES[1]), labelpad=8)
    ax.set_zlabel(param_3d_axis_label(PARAM_3D_AXES[2]), labelpad=8)
    ax.tick_params(labelsize=8.0)
    ax.view_init(elev=22, azim=-58)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.68, pad=0.10)
    cbar.set_label(f"Re-run successes (0–{N_RESCREEN})", fontsize=9.0)
    cbar.ax.tick_params(labelsize=8.0)

    fig.suptitle(
        f"Neutral parameter space ({len(rows):,} hits, all fixed-$Y$ suites)",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.92, top=0.90, bottom=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return len(rows)


def _param_3d_viridis_rgb(n_hit: np.ndarray) -> np.ndarray:
    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    cmap = plt.get_cmap("viridis")
    return cmap(norm(n_hit))[:, :3].astype(np.float32)


def _param_3d_axis_ranges(
    cost: np.ndarray,
    initial_a: np.ndarray,
    energy: np.ndarray,
) -> List[List[float]]:
    ranges: List[List[float]] = []
    for name, arr in zip(PARAM_3D_AXES, (cost, initial_a, energy)):
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        floor = PARAM_3D_AXIS_LO_FLOOR.get(name)
        if floor is not None:
            lo = floor
        ranges.append([lo, hi])
    return ranges


def _param_3d_scene_coords(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mid = 0.5 * (lo + hi)
    span = max(hi - lo, 1e-12)
    return ((arr - mid) / span * 2.0).astype(np.float32)


def _load_param_3d_vendor_scripts() -> Tuple[str, str]:
    vendor_dir = Path(__file__).resolve().parent / "vendor"
    three_js = (vendor_dir / "three.min.js").read_text(encoding="utf-8")
    orbit_js = (vendor_dir / "OrbitControls.js").read_text(encoding="utf-8")
    return three_js, orbit_js


def compute_rf_only_data(
    rows: Sequence[HitRecord],
    *,
    success_threshold: int,
) -> Dict[str, object]:
    """Compute random-forest models only (for HTML RF tab / panel cache)."""
    X, _ = build_feature_matrix(rows)
    X_raw, _ = build_raw_feature_matrix(rows)
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    high_success = n_hit >= success_threshold
    rf = compute_random_forest(X, high_success, X_plot=X_raw)
    rf["feature_names"] = feature_display_names(raw=True)
    no_flow_features = param_features_without_flow()
    X_no_flow, _ = build_feature_matrix(rows, features=no_flow_features)
    X_raw_no_flow, _ = build_raw_feature_matrix(rows, features=no_flow_features)
    rf_no_flow = compute_random_forest(X_no_flow, high_success, X_plot=X_raw_no_flow)
    rf_no_flow["feature_names"] = feature_display_names(features=no_flow_features, raw=True)
    no_flow_mut_features = param_features_without_flow_and_mutation()
    X_no_flow_mut, _ = build_feature_matrix(rows, features=no_flow_mut_features)
    X_raw_no_flow_mut, _ = build_raw_feature_matrix(rows, features=no_flow_mut_features)
    rf_no_flow_mut = compute_random_forest(
        X_no_flow_mut, high_success, X_plot=X_raw_no_flow_mut
    )
    rf_no_flow_mut["feature_names"] = feature_display_names(
        features=no_flow_mut_features, raw=True
    )
    return {
        "n_hit": n_hit,
        "n_hits": len(rows),
        "rf": rf,
        "rf_no_flow": rf_no_flow,
        "rf_no_flow_mut": rf_no_flow_mut,
    }


def compute_lr_only_data(
    rows: Sequence[HitRecord],
    *,
    success_threshold: int,
) -> Dict[str, object]:
    """Compute logistic-regression models only (for HTML LR tab / panel cache)."""
    X, _ = build_feature_matrix(rows)
    X_raw, _ = build_raw_feature_matrix(rows)
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    high_success = n_hit >= success_threshold
    lr = compute_logistic_regression(X, high_success, X_plot=X_raw)
    lr["feature_names"] = feature_display_names(raw=True)
    no_flow_features = param_features_without_flow()
    X_no_flow, _ = build_feature_matrix(rows, features=no_flow_features)
    X_raw_no_flow, _ = build_raw_feature_matrix(rows, features=no_flow_features)
    lr_no_flow = compute_logistic_regression(X_no_flow, high_success, X_plot=X_raw_no_flow)
    lr_no_flow["feature_names"] = feature_display_names(features=no_flow_features, raw=True)
    no_flow_mut_features = param_features_without_flow_and_mutation()
    X_no_flow_mut, _ = build_feature_matrix(rows, features=no_flow_mut_features)
    X_raw_no_flow_mut, _ = build_raw_feature_matrix(rows, features=no_flow_mut_features)
    lr_no_flow_mut = compute_logistic_regression(
        X_no_flow_mut, high_success, X_plot=X_raw_no_flow_mut
    )
    lr_no_flow_mut["feature_names"] = feature_display_names(
        features=no_flow_mut_features, raw=True
    )
    return {
        "n_hit": n_hit,
        "n_hits": len(rows),
        "lr": lr,
        "lr_no_flow": lr_no_flow,
        "lr_no_flow_mut": lr_no_flow_mut,
    }


def _png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rf_oob_scores_from_meta(meta: Dict[str, object]) -> Dict[str, float]:
    return {
        "rf_oob_score": float(meta["rf_oob_score"]),
        "rf_no_flow_oob_score": float(meta["rf_no_flow_oob_score"]),
        "rf_no_flow_mut_oob_score": float(meta["rf_no_flow_mut_oob_score"]),
    }


def _html_classifier_meta_fresh(
    meta: Dict[str, object],
    *,
    success_threshold: int,
    n_hits: int,
) -> bool:
    return (
        int(meta.get("success_threshold", -1)) == success_threshold
        and int(meta.get("n_hits", -1)) == n_hits
    )


def _patch_html_classifier_meta(cache_dir: Path, updates: Dict[str, object]) -> None:
    meta_path = cache_meta_path(cache_dir)
    meta: Dict[str, object] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(updates)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _ensure_rf_panels_for_html(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    success_threshold: int,
) -> Tuple[Path, Dict[str, float]]:
    rf_panels_dir = cache_rf_panels_dir(cache_dir)
    meta_path = cache_meta_path(cache_dir)
    if rf_panels_complete(rf_panels_dir) and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _html_classifier_meta_fresh(
            meta, success_threshold=success_threshold, n_hits=len(rows)
        ):
            return rf_panels_dir, _rf_oob_scores_from_meta(meta)
    data = compute_rf_only_data(rows, success_threshold=success_threshold)
    render_rf_panel_pngs(cache_dir, data, success_threshold=success_threshold)
    scores = {
        "rf_oob_score": float(data["rf"]["oob_score"]),
        "rf_no_flow_oob_score": float(data["rf_no_flow"]["oob_score"]),
        "rf_no_flow_mut_oob_score": float(data["rf_no_flow_mut"]["oob_score"]),
    }
    _patch_html_classifier_meta(
        cache_dir,
        {"success_threshold": success_threshold, "n_hits": len(rows), **scores},
    )
    return rf_panels_dir, scores


def _rf_interactive_payload(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    success_threshold: int,
) -> Dict[str, object]:
    rf_panels_dir, scores = _ensure_rf_panels_for_html(
        cache_dir=cache_dir,
        rows=rows,
        success_threshold=success_threshold,
    )
    panels = [
        {
            "letter": letter,
            "src": _png_data_uri(rf_panels_dir / f"{letter}.png"),
        }
        for letter in RF_PANEL_LETTERS
    ]
    target_txt = (
        f"binary target ≥{success_threshold}/{N_RESCREEN} re-run successes"
    )
    row_notes = [
        (
            f"OOB accuracy = {scores['rf_oob_score']:.3f} "
            f"(all six features; {target_txt})"
        ),
        (
            f"OOB accuracy = {scores['rf_no_flow_oob_score']:.3f} "
            f"(flow percentage excluded; {target_txt})"
        ),
        (
            f"OOB accuracy = {scores['rf_no_flow_mut_oob_score']:.3f} "
            f"(flow percentage and mutation rate excluded; {target_txt})"
        ),
    ]
    return {
        "title": f"Random forest re-hit classifier ({len(rows):,} neutral hits)",
        "panels": panels,
        "row_notes": row_notes,
        "footer": "200 trees, balanced class weights",
    }


def _lr_cv_scores_from_meta(meta: Dict[str, object]) -> Dict[str, Tuple[float, float]]:
    return {
        "lr": (
            float(meta["lr_cv_accuracy"]),
            float(meta["lr_cv_accuracy_std"]),
        ),
        "lr_no_flow": (
            float(meta["lr_no_flow_cv_accuracy"]),
            float(meta["lr_no_flow_cv_accuracy_std"]),
        ),
        "lr_no_flow_mut": (
            float(meta["lr_no_flow_mut_cv_accuracy"]),
            float(meta["lr_no_flow_mut_cv_accuracy_std"]),
        ),
    }


def _ensure_lr_panels_for_html(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    success_threshold: int,
) -> Tuple[Path, Dict[str, Tuple[float, float]]]:
    lr_panels_dir = cache_lr_panels_dir(cache_dir)
    meta_path = cache_meta_path(cache_dir)
    if lr_panels_complete(lr_panels_dir) and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _html_classifier_meta_fresh(
            meta, success_threshold=success_threshold, n_hits=len(rows)
        ):
            return lr_panels_dir, _lr_cv_scores_from_meta(meta)
    data = compute_lr_only_data(rows, success_threshold=success_threshold)
    render_lr_panel_pngs(cache_dir, data, success_threshold=success_threshold)
    scores = {
        "lr": (
            float(data["lr"]["cv_accuracy"]),
            float(data["lr"]["cv_accuracy_std"]),
        ),
        "lr_no_flow": (
            float(data["lr_no_flow"]["cv_accuracy"]),
            float(data["lr_no_flow"]["cv_accuracy_std"]),
        ),
        "lr_no_flow_mut": (
            float(data["lr_no_flow_mut"]["cv_accuracy"]),
            float(data["lr_no_flow_mut"]["cv_accuracy_std"]),
        ),
    }
    _patch_html_classifier_meta(
        cache_dir,
        {
            "success_threshold": success_threshold,
            "n_hits": len(rows),
            "lr_cv_accuracy": scores["lr"][0],
            "lr_cv_accuracy_std": scores["lr"][1],
            "lr_no_flow_cv_accuracy": scores["lr_no_flow"][0],
            "lr_no_flow_cv_accuracy_std": scores["lr_no_flow"][1],
            "lr_no_flow_mut_cv_accuracy": scores["lr_no_flow_mut"][0],
            "lr_no_flow_mut_cv_accuracy_std": scores["lr_no_flow_mut"][1],
        },
    )
    return lr_panels_dir, scores


def _lr_interactive_payload(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    success_threshold: int,
) -> Dict[str, object]:
    lr_panels_dir, scores = _ensure_lr_panels_for_html(
        cache_dir=cache_dir,
        rows=rows,
        success_threshold=success_threshold,
    )
    panels = [
        {
            "letter": letter,
            "src": _png_data_uri(lr_panels_dir / f"{letter}.png"),
        }
        for letter in LR_PANEL_LETTERS
    ]
    target_txt = (
        f"binary target ≥{success_threshold}/{N_RESCREEN} re-run successes"
    )

    def _cv_note(key: str, feature_note: str) -> str:
        acc, std = scores[key]
        return (
            f"5-fold CV accuracy = {acc:.3f} ± {std:.3f} "
            f"({feature_note}; {target_txt})"
        )

    row_notes = [
        _cv_note("lr", "all six features"),
        _cv_note("lr_no_flow", "flow percentage excluded"),
        _cv_note(
            "lr_no_flow_mut",
            "flow percentage and mutation rate excluded",
        ),
    ]
    return {
        "title": (
            f"Logistic regression re-hit classifier ({len(rows):,} neutral hits)"
        ),
        "panels": panels,
        "row_notes": row_notes,
        "footer": "Balanced class weights; features standardized before fitting",
    }


def compute_dt_only_data(
    rows: Sequence[HitRecord],
    *,
    success_threshold: int,
) -> Dict[str, object]:
    """Compute decision-tree models only (for HTML DT tab / panel cache)."""
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    high_success = n_hit >= success_threshold
    dt, dt_no_flow, dt_no_flow_mut = compute_dt_models_three_variants(rows, high_success)
    return {
        "n_hit": n_hit,
        "n_hits": len(rows),
        "dt": dt,
        "dt_no_flow": dt_no_flow,
        "dt_no_flow_mut": dt_no_flow_mut,
    }


def _dt_cv_scores_from_meta(meta: Dict[str, object]) -> Dict[str, Tuple[float, float]]:
    return {
        "dt": (
            float(meta["dt_cv_accuracy"]),
            float(meta["dt_cv_accuracy_std"]),
        ),
        "dt_no_flow": (
            float(meta["dt_no_flow_cv_accuracy"]),
            float(meta["dt_no_flow_cv_accuracy_std"]),
        ),
        "dt_no_flow_mut": (
            float(meta["dt_no_flow_mut_cv_accuracy"]),
            float(meta["dt_no_flow_mut_cv_accuracy_std"]),
        ),
    }


def _dt_cv_scores_from_models(
    dt: Dict[str, object],
    dt_no_flow: Dict[str, object],
    dt_no_flow_mut: Dict[str, object],
) -> Dict[str, Tuple[float, float]]:
    return {
        "dt": (float(dt["cv_accuracy"]), float(dt["cv_accuracy_std"])),
        "dt_no_flow": (
            float(dt_no_flow["cv_accuracy"]),
            float(dt_no_flow["cv_accuracy_std"]),
        ),
        "dt_no_flow_mut": (
            float(dt_no_flow_mut["cv_accuracy"]),
            float(dt_no_flow_mut["cv_accuracy_std"]),
        ),
    }


def _dt_panel_notes(
    scores: Dict[str, Tuple[float, float]],
    target_txt: str,
) -> List[str]:
    keys = ("dt", "dt_no_flow", "dt_no_flow_mut")
    feature_notes = (
        "all six features",
        "flow percentage excluded",
        "flow percentage and mutation rate excluded",
    )
    return [
        (
            f"5-fold CV accuracy = {scores[key][0]:.3f} ± {scores[key][1]:.3f} "
            f"({note}; {target_txt})"
        )
        for key, note in zip(keys, feature_notes)
    ]


def _dt_section_payload(
    panels_dir: Path,
    scores: Dict[str, Tuple[float, float]],
    *,
    title: str,
    subtitle: str,
    target_txt: str,
) -> Dict[str, object]:
    return {
        "title": title,
        "subtitle": subtitle,
        "panels": [
            {
                "letter": letter,
                "src": _png_data_uri(panels_dir / f"{letter}.png"),
            }
            for letter in DT_PANEL_LETTERS
        ],
        "panel_notes": _dt_panel_notes(scores, target_txt),
    }


def _ensure_dt_panels_for_html(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    success_threshold: int,
) -> Tuple[Path, Dict[str, Tuple[float, float]]]:
    dt_panels_dir = cache_dt_panels_dir(cache_dir)
    meta_path = cache_meta_path(cache_dir)
    if dt_panels_complete(dt_panels_dir) and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _html_classifier_meta_fresh(
            meta, success_threshold=success_threshold, n_hits=len(rows)
        ):
            return dt_panels_dir, _dt_cv_scores_from_meta(meta)
    data = compute_dt_only_data(rows, success_threshold=success_threshold)
    render_dt_panel_pngs(cache_dir, rows, success_threshold=success_threshold)
    scores = _dt_cv_scores_from_models(
        data["dt"],
        data["dt_no_flow"],
        data["dt_no_flow_mut"],
    )
    _patch_html_classifier_meta(
        cache_dir,
        {
            "success_threshold": success_threshold,
            "n_hits": len(rows),
            "dt_cv_accuracy": scores["dt"][0],
            "dt_cv_accuracy_std": scores["dt"][1],
            "dt_no_flow_cv_accuracy": scores["dt_no_flow"][0],
            "dt_no_flow_cv_accuracy_std": scores["dt_no_flow"][1],
            "dt_no_flow_mut_cv_accuracy": scores["dt_no_flow_mut"][0],
            "dt_no_flow_mut_cv_accuracy_std": scores["dt_no_flow_mut"][1],
        },
    )
    return dt_panels_dir, scores


def _ensure_dt_extreme_panels_for_html(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    low_ceiling: int = DT_EXTREME_LOW_CEILING,
    high_floor: int = DT_EXTREME_HIGH_FLOOR,
) -> Tuple[Path, Dict[str, Tuple[float, float]], int, int, int]:
    rows_all = list(rows)
    kept, high_success, n_excluded = split_extreme_gap_rows(
        rows_all,
        low_ceiling=low_ceiling,
        high_floor=high_floor,
    )
    if not kept:
        raise ValueError("No hits remain after extreme-gap filtering")
    dt_panels_dir = cache_dt_extreme_panels_dir(cache_dir)
    extreme_meta_path = cache_dt_extreme_meta_path(cache_dir)
    if dt_panels_complete(dt_panels_dir) and extreme_meta_path.is_file():
        meta = json.loads(extreme_meta_path.read_text(encoding="utf-8"))
        if (
            int(meta.get("low_ceiling", -1)) == low_ceiling
            and int(meta.get("high_floor", -1)) == high_floor
            and int(meta.get("n_all", -1)) == len(rows_all)
        ):
            return (
                dt_panels_dir,
                _dt_cv_scores_from_meta(meta),
                int(meta["n_retained"]),
                int(meta["n_excluded"]),
                int(meta["n_all"]),
            )
    dt, dt_no_flow, dt_no_flow_mut = compute_dt_models_three_variants(kept, high_success)
    scores = _dt_cv_scores_from_models(dt, dt_no_flow, dt_no_flow_mut)
    if not dt_panels_complete(dt_panels_dir):
        titles = (
            f"Decision tree ($>{high_floor}$ vs.\\ $<{low_ceiling}$)",
            f"Decision tree, no flow ($>{high_floor}$ vs.\\ $<{low_ceiling}$)",
            (
                "Decision tree, no flow/mutation "
                f"($>{high_floor}$ vs.\\ $<{low_ceiling}$)"
            ),
        )
        render_dt_panel_pngs_to_dir(
            dt_panels_dir,
            kept,
            high_success=high_success,
            panel_titles=titles,
            class_names=(f"<{low_ceiling}", f">{high_floor}"),
        )
    extreme_meta_path.write_text(
        json.dumps(
            {
                "low_ceiling": int(low_ceiling),
                "high_floor": int(high_floor),
                "n_retained": len(kept),
                "n_excluded": int(n_excluded),
                "n_all": len(rows_all),
                "dt_cv_accuracy": scores["dt"][0],
                "dt_cv_accuracy_std": scores["dt"][1],
                "dt_no_flow_cv_accuracy": scores["dt_no_flow"][0],
                "dt_no_flow_cv_accuracy_std": scores["dt_no_flow"][1],
                "dt_no_flow_mut_cv_accuracy": scores["dt_no_flow_mut"][0],
                "dt_no_flow_mut_cv_accuracy_std": scores["dt_no_flow_mut"][1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return dt_panels_dir, scores, len(kept), n_excluded, len(rows_all)


def _dt_interactive_payload(
    *,
    cache_dir: Path,
    rows: Sequence[HitRecord],
    success_threshold: int,
) -> Dict[str, object]:
    dt_panels_dir, std_scores = _ensure_dt_panels_for_html(
        cache_dir=cache_dir,
        rows=rows,
        success_threshold=success_threshold,
    )
    std_target_txt = (
        f"binary target ≥{success_threshold}/{N_RESCREEN} re-run successes"
    )
    extreme_panels_dir, extreme_scores, n_retained, n_excluded, n_all = (
        _ensure_dt_extreme_panels_for_html(cache_dir=cache_dir, rows=rows)
    )
    extreme_target_txt = (
        f"binary target >{DT_EXTREME_HIGH_FLOOR} vs "
        f"<{DT_EXTREME_LOW_CEILING} re-run successes"
    )
    return {
        "title": "Decision tree re-hit classifiers",
        "sections": [
            _dt_section_payload(
                dt_panels_dir,
                std_scores,
                title=(
                    f"Standard target (≥{success_threshold} vs "
                    f"<{success_threshold})"
                ),
                subtitle=f"{len(rows):,} neutral hits",
                target_txt=std_target_txt,
            ),
            _dt_section_payload(
                extreme_panels_dir,
                extreme_scores,
                title=(
                    f"Extreme classes only "
                    f"(>{DT_EXTREME_HIGH_FLOOR} vs <{DT_EXTREME_LOW_CEILING})"
                ),
                subtitle=(
                    f"{n_retained:,} hits retained "
                    f"({n_excluded:,} excluded from {n_all:,}; "
                    f"gap {DT_EXTREME_LOW_CEILING}–{DT_EXTREME_HIGH_FLOOR} "
                    f"re-run successes removed)"
                ),
                target_txt=extreme_target_txt,
            ),
        ],
        "footer": (
            "Max depth 4, min leaf 50, balanced class weights; "
            "features standardized before fitting"
        ),
    }


def plot_param_3d_interactive_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
    cache_dir: Path,
    open_browser: bool = False,
    success_threshold: int = PARAM_3D_SUCCESS_THRESHOLD,
) -> int:
    """Write a rotatable Three.js HTML viewer for the 3D parameter scatter."""
    rows = load_true_neutral_hits(sessions_dir)
    if not rows:
        raise ValueError(f"No neutral hits found under {sessions_dir}")
    cost, initial_a, energy, n_hit = extract_param_3d_arrays(rows)
    axis_labels = [param_3d_axis_label(name) for name in PARAM_3D_AXES]
    axis_ranges = _param_3d_axis_ranges(cost, initial_a, energy)
    filter_labels, filter_ranges, filter_values = _param_interactive_filter_payload(rows)
    ratio_options = _build_ratio_options(rows)
    ratio_filter_html = _ratio_filter_checkboxes_html(ratio_options)
    colors = _param_3d_viridis_rgb(n_hit)
    hist_filter_html = _hist_filter_panel_html(filter_labels, PARAM_FILTER_COLORS)
    rf_payload = _rf_interactive_payload(
        cache_dir=cache_dir,
        rows=rows,
        success_threshold=success_threshold,
    )
    lr_payload = _lr_interactive_payload(
        cache_dir=cache_dir,
        rows=rows,
        success_threshold=success_threshold,
    )
    dt_payload = _dt_interactive_payload(
        cache_dir=cache_dir,
        rows=rows,
        success_threshold=success_threshold,
    )
    payload = {
        "title_base": "Neutral parameter space",
        "title": (
            f"Neutral parameter space ({len(rows):,} hits)"
        ),
        "hist_title": (
            f"Re-run success histogram ({len(rows):,} neutral hits)"
        ),
        "n_rescreen": int(N_RESCREEN),
        "success_threshold": int(success_threshold),
        "axes": axis_labels,
        "ranges": axis_ranges,
        "filter_labels": filter_labels,
        "filter_ranges": filter_ranges,
        "filter_values": filter_values,
        "suite_tags": [str(r["suite_tag"]) for r in rows],
        "ratio_options": ratio_options,
        "cost": cost.astype(np.float64).tolist(),
        "initial_a": initial_a.astype(np.float64).tolist(),
        "energy": energy.astype(np.float64).tolist(),
        "n_hit": n_hit.astype(int).tolist(),
        "x": _param_3d_scene_coords(cost, *axis_ranges[0]).tolist(),
        "y": _param_3d_scene_coords(initial_a, *axis_ranges[1]).tolist(),
        "z": _param_3d_scene_coords(energy, *axis_ranges[2]).tolist(),
        "r": colors[:, 0].tolist(),
        "g": colors[:, 1].tolist(),
        "b": colors[:, 2].tolist(),
        "rf": rf_payload,
        "lr": lr_payload,
        "dt": dt_payload,
    }
    data_json = json.dumps(payload)
    three_js, orbit_js = _load_param_3d_vendor_scripts()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Neutral parameter explorer</title>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f8;
      color: #1a1a1a;
    }}
    #app {{
      position: relative;
      width: 100%;
      height: 100%;
    }}
    #view-nav {{
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 40px;
      display: flex;
      align-items: stretch;
      gap: 0;
      padding: 0 12px;
      background: rgba(255, 255, 255, 0.97);
      border-bottom: 1px solid #ccc;
      z-index: 10;
      box-sizing: border-box;
    }}
    .view-tab {{
      border: none;
      border-bottom: 3px solid transparent;
      background: transparent;
      padding: 0 14px;
      font-size: 12px;
      font-weight: 600;
      color: #555;
      cursor: pointer;
    }}
    .view-tab:hover {{
      color: #1a1a1a;
      background: rgba(0, 0, 0, 0.03);
    }}
    .view-tab.active {{
      color: #1a1a1a;
      border-bottom-color: #2471a3;
    }}
    .view {{
      display: none;
      position: absolute;
      inset: 40px 0 0 0;
      overflow: hidden;
      visibility: hidden;
      pointer-events: none;
    }}
    .view.active {{
      display: block;
      visibility: visible;
      pointer-events: auto;
    }}
    #view-3d canvas {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }}
    #title {{
      position: absolute;
      top: 12px;
      left: 252px;
      right: 240px;
      font-size: 15px;
      font-weight: 600;
      z-index: 2;
      pointer-events: none;
    }}
    #hint {{
      position: absolute;
      bottom: 12px;
      left: 16px;
      font-size: 12px;
      color: #444;
      z-index: 2;
      pointer-events: none;
    }}
    #error {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      max-width: 520px;
      padding: 14px 16px;
      border-radius: 8px;
      background: #fff4f4;
      border: 1px solid #e0a0a0;
      color: #7a1010;
      font-size: 13px;
      line-height: 1.45;
      z-index: 4;
      display: none;
      white-space: pre-wrap;
    }}
    #tooltip {{
      position: absolute;
      display: none;
      max-width: 260px;
      padding: 8px 10px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid #ccc;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
      font-size: 12px;
      line-height: 1.45;
      z-index: 3;
      pointer-events: none;
      white-space: nowrap;
    }}
    #colorbar-wrap {{
      position: absolute;
      top: 56px;
      right: 16px;
      width: 18px;
      height: 220px;
      border: 1px solid #bbb;
      border-radius: 3px;
      overflow: hidden;
      z-index: 2;
      pointer-events: none;
    }}
    #colorbar {{
      width: 100%;
      height: 100%;
      background: linear-gradient(
        to top,
        #440154 0%,
        #3b528b 25%,
        #21918c 50%,
        #5ec962 75%,
        #fde725 100%
      );
    }}
    #colorbar-label {{
      position: absolute;
      top: 56px;
      right: 42px;
      width: 150px;
      height: 220px;
      font-size: 11px;
      z-index: 2;
      pointer-events: none;
    }}
    #colorbar-label .top, #colorbar-label .bottom {{
      position: absolute;
      right: 0;
    }}
    #colorbar-label .top {{ top: 0; }}
    #colorbar-label .bottom {{ bottom: 0; }}
    #colorbar-title {{
      position: absolute;
      top: 36px;
      right: 16px;
      width: 170px;
      font-size: 11px;
      text-align: right;
      z-index: 2;
      pointer-events: none;
    }}
    #axis-labels {{
      position: absolute;
      left: 252px;
      bottom: 36px;
      font-size: 11px;
      color: #333;
      z-index: 2;
      pointer-events: none;
      line-height: 1.5;
    }}
    #exclude-panel {{
      position: relative;
      top: auto;
      right: auto;
      width: auto;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid #ccc;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
      font-size: 12px;
      line-height: 1.5;
      flex-shrink: 0;
    }}
    #right-panel-stack {{
      position: absolute;
      top: 292px;
      right: 16px;
      width: 228px;
      max-height: calc(100% - 308px);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 4;
    }}
    #ratio-filter-panel {{
      position: relative;
      width: auto;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid #ccc;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
      font-size: 12px;
      line-height: 1.45;
      flex-shrink: 0;
    }}
    #ratio-filter-panel h3 {{
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    #exclude-panel h3 {{
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    #exclude-panel label {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      cursor: pointer;
    }}
    #exclude-panel .gap-row {{
      display: flex;
      gap: 8px;
      margin-bottom: 8px;
    }}
    #exclude-panel .gap-row label {{
      flex: 1;
      flex-direction: column;
      align-items: stretch;
      gap: 4px;
      margin-bottom: 0;
      cursor: default;
    }}
    #exclude-panel input[type="number"] {{
      width: 100%;
      box-sizing: border-box;
      padding: 4px 6px;
      border: 1px solid #bbb;
      border-radius: 4px;
      font-size: 12px;
    }}
    #exclude-panel button {{
      width: 100%;
      padding: 6px 8px;
      border: 1px solid #888;
      border-radius: 4px;
      background: #f3f3f3;
      font-size: 12px;
      cursor: pointer;
    }}
    #exclude-panel button:hover {{
      background: #e8e8e8;
    }}
    #exclude-stats {{
      margin-top: 8px;
      font-size: 11px;
      color: #444;
      line-height: 1.45;
    }}
    #exclude-legend {{
      margin-top: 8px;
      font-size: 11px;
      line-height: 1.6;
    }}
    #exclude-legend .swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 6px;
      vertical-align: -1px;
    }}
    #viridis-legend {{
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 2;
    }}
    #exclude-legend.hidden {{
      display: none;
    }}
    #param-filter-panel {{
      position: absolute;
      top: 48px;
      left: 16px;
      width: 220px;
      max-height: calc(100% - 96px);
      overflow-y: auto;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid #ccc;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
      font-size: 12px;
      line-height: 1.45;
      z-index: 4;
    }}
    #param-filter-panel h3 {{
      margin: 0 0 10px;
      font-size: 12px;
      font-weight: 600;
    }}
    #param-filter-panel .threshold-row {{
      display: flex;
      flex-direction: row;
      align-items: center;
      flex-wrap: wrap;
      gap: 4px;
      margin-bottom: 12px;
      font-size: 11px;
    }}
    #param-filter-panel .threshold-row input {{
      width: 3.25rem;
      flex: 0 0 auto;
      box-sizing: border-box;
      padding: 4px 6px;
      border: 1px solid #bbb;
      border-radius: 4px;
      font-size: 12px;
    }}
    .param-filter {{
      margin-bottom: 14px;
      padding-top: 2px;
      border-top: 1px solid #e4e4e4;
    }}
    .param-filter:first-of-type {{
      border-top: none;
      padding-top: 0;
    }}
    .param-filter-head {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      font-weight: 600;
      font-size: 11px;
    }}
    .param-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    .param-range-row {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 8px;
      font-size: 10px;
      color: #555;
    }}
    .dual-range {{
      position: relative;
      height: 34px;
    }}
    .dual-range-track {{
      position: absolute;
      left: 0;
      right: 0;
      top: 50%;
      transform: translateY(-50%);
      height: 4px;
      background: #ddd;
      border-radius: 2px;
    }}
    .dual-range-fill {{
      position: absolute;
      top: 0;
      height: 100%;
      border-radius: 2px;
      opacity: 0.85;
    }}
    .dual-range input[type="range"] {{
      position: absolute;
      width: 100%;
      top: 0;
      left: 0;
      margin: 0;
      padding: 0;
      pointer-events: none;
      -webkit-appearance: none;
      appearance: none;
      background: transparent;
      height: 34px;
      z-index: 2;
    }}
    .dual-range input[type="range"]::-webkit-slider-runnable-track {{
      -webkit-appearance: none;
      height: 4px;
      background: transparent;
    }}
    .dual-range input[type="range"]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      pointer-events: auto;
      width: 14px;
      height: 14px;
      margin-top: -5px;
      border-radius: 50%;
      background: #fff;
      border: 2px solid #666;
      cursor: pointer;
    }}
    .dual-range input[type="range"]::-moz-range-track {{
      height: 4px;
      background: transparent;
      border: none;
    }}
    .dual-range input[type="range"]::-moz-range-thumb {{
      pointer-events: auto;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #fff;
      border: 2px solid #666;
      cursor: pointer;
    }}
    .dual-range-lo {{
      z-index: 3;
    }}
    .dual-range-hi {{
      z-index: 4;
    }}
    .param-range-values {{
      font-size: 11px;
      color: #333;
      margin-bottom: 4px;
    }}
    .param-success-count {{
      font-size: 11px;
      color: #1a1a1a;
    }}
    .param-success-count strong {{
      font-size: 13px;
    }}
    .param-in-range {{
      font-size: 10px;
      color: #666;
      margin-top: 2px;
    }}
    #param-combined-stats {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid #e4e4e4;
    }}
    .ratio-filter-section {{
      margin-bottom: 12px;
      padding-bottom: 10px;
      border-bottom: 1px solid #e4e4e4;
    }}
    .ratio-filter-section h4 {{
      margin: 0 0 8px;
      font-size: 11px;
      font-weight: 600;
    }}
    .ratio-filter-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 10px;
      align-items: start;
    }}
    .ratio-filter-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 0;
      font-size: 10px;
      cursor: pointer;
      min-width: 0;
    }}
    .ratio-filter-row span {{
      line-height: 1.3;
    }}
    .ratio-filter-row input {{
      margin: 0;
      flex-shrink: 0;
    }}
    #view-hist.active {{
      display: flex;
      flex-direction: row;
      align-items: stretch;
      gap: 16px;
      padding: 8px 16px 16px;
      box-sizing: border-box;
    }}
    #hist-filter-panel {{
      position: relative;
      top: auto;
      left: auto;
      flex: 0 0 240px;
      width: 240px;
      max-height: none;
      align-self: stretch;
      overflow-y: auto;
      padding: 10px 12px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid #ccc;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.10);
      font-size: 12px;
      line-height: 1.45;
      z-index: 1;
    }}
    #hist-filter-panel h3 {{
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    #hist-filter-panel .panel-note {{
      margin: 0 0 10px;
      font-size: 10px;
      color: #666;
      line-height: 1.4;
    }}
    #hist-reset-filters {{
      width: 100%;
      margin-bottom: 10px;
      padding: 6px 8px;
      border: 1px solid #888;
      border-radius: 4px;
      background: #f3f3f3;
      font-size: 12px;
      cursor: pointer;
    }}
    #hist-reset-filters:hover {{
      background: #e8e8e8;
    }}
    #hist-main {{
      position: relative;
      top: auto;
      left: auto;
      right: auto;
      bottom: auto;
      flex: 1 1 auto;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    #hist-title-bar {{
      font-size: 15px;
      font-weight: 600;
      flex: 0 0 auto;
    }}
    #hist-stats {{
      font-size: 12px;
      color: #333;
      line-height: 1.5;
      flex: 0 0 auto;
    }}
    #hist-chart-wrap {{
      flex: 1 1 auto;
      min-height: 220px;
      border: 1px solid #ccc;
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      box-sizing: border-box;
    }}
    #hist-canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    #view-rf.active {{
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 8px 16px 16px;
      box-sizing: border-box;
    }}
    #rf-main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 28px;
      gap: 16px;
      align-items: stretch;
      flex: 1 1 auto;
      min-height: 0;
    }}
    #rf-content {{
      min-width: 0;
    }}
    #rf-title-bar {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    #rf-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .rf-panel {{
      position: relative;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }}
    .rf-panel img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .rf-panel-label {{
      position: absolute;
      top: 8px;
      left: 8px;
      font-weight: 700;
      font-size: 12px;
      background: rgba(255, 255, 255, 0.92);
      padding: 2px 6px;
      border-radius: 3px;
      border: 1px solid #ddd;
    }}
    #rf-notes {{
      margin-top: 10px;
      font-size: 11px;
      color: #333;
      line-height: 1.6;
    }}
    #rf-footer {{
      margin-top: 6px;
      font-size: 11px;
      color: #555;
    }}
    #rf-colorbar-wrap {{
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 6px;
      min-width: 28px;
    }}
    #rf-colorbar-title {{
      font-size: 10px;
      text-align: center;
      line-height: 1.3;
      color: #333;
    }}
    #rf-colorbar {{
      flex: 1 1 auto;
      min-height: 180px;
      border: 1px solid #bbb;
      border-radius: 3px;
      background: linear-gradient(
        to top,
        #440154 0%,
        #3b528b 25%,
        #21918c 50%,
        #5ec962 75%,
        #fde725 100%
      );
    }}
    #rf-colorbar-labels {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      font-size: 10px;
      text-align: center;
      color: #333;
      min-height: 180px;
    }}
    #view-lr.active {{
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 8px 16px 16px;
      box-sizing: border-box;
    }}
    #lr-main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 28px;
      gap: 16px;
      align-items: stretch;
      flex: 1 1 auto;
      min-height: 0;
    }}
    #lr-content {{
      min-width: 0;
    }}
    #lr-title-bar {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    #lr-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .lr-panel {{
      position: relative;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }}
    .lr-panel img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .lr-panel-label {{
      position: absolute;
      top: 8px;
      left: 8px;
      font-weight: 700;
      font-size: 12px;
      background: rgba(255, 255, 255, 0.92);
      padding: 2px 6px;
      border-radius: 3px;
      border: 1px solid #ddd;
    }}
    #lr-notes {{
      margin-top: 10px;
      font-size: 11px;
      color: #333;
      line-height: 1.6;
    }}
    #lr-footer {{
      margin-top: 6px;
      font-size: 11px;
      color: #555;
    }}
    #lr-colorbar-wrap {{
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 6px;
      min-width: 28px;
    }}
    #lr-colorbar-title {{
      font-size: 10px;
      text-align: center;
      line-height: 1.3;
      color: #333;
    }}
    #lr-colorbar {{
      flex: 1 1 auto;
      min-height: 180px;
      border: 1px solid #bbb;
      border-radius: 3px;
      background: linear-gradient(
        to top,
        #440154 0%,
        #3b528b 25%,
        #21918c 50%,
        #5ec962 75%,
        #fde725 100%
      );
    }}
    #lr-colorbar-labels {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      font-size: 10px;
      text-align: center;
      color: #333;
      min-height: 180px;
    }}
    #view-dt.active {{
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 8px 16px 16px;
      box-sizing: border-box;
    }}
    #dt-content {{
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
    }}
    #dt-title-bar {{
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 12px;
    }}
    .dt-section {{
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid #ddd;
    }}
    .dt-section:last-of-type {{
      border-bottom: none;
      padding-bottom: 0;
    }}
    .dt-section-title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    .dt-section-subtitle {{
      font-size: 11px;
      color: #555;
      margin-bottom: 10px;
      line-height: 1.5;
    }}
    .dt-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .dt-panel-item {{
      min-width: 0;
    }}
    .dt-panel {{
      position: relative;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: #fff;
      overflow: hidden;
    }}
    .dt-panel img {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .dt-panel-label {{
      position: absolute;
      top: 8px;
      left: 8px;
      font-weight: 700;
      font-size: 12px;
      background: rgba(255, 255, 255, 0.92);
      padding: 2px 6px;
      border-radius: 3px;
      border: 1px solid #ddd;
    }}
    .dt-panel-note {{
      margin-top: 6px;
      font-size: 11px;
      color: #333;
      line-height: 1.5;
    }}
    #dt-footer {{
      margin-top: 4px;
      font-size: 11px;
      color: #555;
    }}
  </style>
</head>
<body>
  <div id="app">
    <nav id="view-nav">
      <button type="button" class="view-tab active" data-view="3d">
        3D parameter space
      </button>
      <button type="button" class="view-tab" data-view="hist">
        Re-run histogram
      </button>
      <button type="button" class="view-tab" data-view="rf">
        Random forest
      </button>
      <button type="button" class="view-tab" data-view="lr">
        Logistic regression
      </button>
      <button type="button" class="view-tab" data-view="dt">
        Decision tree
      </button>
    </nav>
    <div id="view-3d" class="view active">
    <div id="title"></div>
    <div id="viridis-legend">
      <div id="colorbar-title"></div>
      <div id="colorbar-wrap"><div id="colorbar"></div></div>
      <div id="colorbar-label">
        <div class="top"></div>
        <div class="bottom">0</div>
      </div>
    </div>
    <div id="right-panel-stack">
    <div id="exclude-panel">
      <h3>Exclude mode</h3>
      <label>
        <input type="checkbox" id="exclude-toggle" />
        Hide a re-hit gap
      </label>
      <div class="gap-row">
        <label>
          Gap low
          <input type="number" id="gap-lo" min="0" max="20" step="1" value="14" />
        </label>
        <label>
          Gap high
          <input type="number" id="gap-hi" min="0" max="20" step="1" value="17" />
        </label>
      </div>
      <button type="button" id="apply-exclude">Apply</button>
      <div id="exclude-stats"></div>
      <div id="exclude-legend" class="hidden">
        <div><span class="swatch" style="background:#4a86e8"></span>Below gap (blue)</div>
        <div><span class="swatch" style="background:#fde725"></span>Above gap (yellow)</div>
        <div><span class="swatch" style="background:#ddd"></span>Gap hidden</div>
      </div>
    </div>
    <div id="ratio-filter-panel">
      <h3>Task energy yield ratio (Y)</h3>
{ratio_filter_html}
    </div>
    </div>
    <div id="param-filter-panel">
      <h3>Successes by parameter</h3>
      <label class="threshold-row">
        <span>Threshold (re-runs ≥</span>
        <input
          type="number"
          id="success-threshold"
          min="0"
          max="20"
          step="1"
          value="10"
        />
        <span>)</span>
      </label>
      <div id="param-filter-list">
        <div class="param-filter" data-axis="0">
          <div class="param-filter-head">
            <span class="param-swatch" style="background:#c0392b"></span>
            <span class="param-filter-name" id="param-name-0"></span>
          </div>
          <label class="param-range-row">
            Range
            <div class="dual-range">
              <div class="dual-range-track">
                <div
                  class="dual-range-fill"
                  id="param-fill-0"
                  style="background:#c0392b"
                ></div>
              </div>
              <input
                type="range"
                class="dual-range-lo"
                id="param-lo-0"
                min="0"
                max="1000"
                step="1"
                value="0"
              />
              <input
                type="range"
                class="dual-range-hi"
                id="param-hi-0"
                min="0"
                max="1000"
                step="1"
                value="1000"
              />
            </div>
          </label>
          <div class="param-range-values" id="param-range-0"></div>
        </div>
        <div class="param-filter" data-axis="1">
          <div class="param-filter-head">
            <span class="param-swatch" style="background:#1e8449"></span>
            <span class="param-filter-name" id="param-name-1"></span>
          </div>
          <label class="param-range-row">
            Range
            <div class="dual-range">
              <div class="dual-range-track">
                <div
                  class="dual-range-fill"
                  id="param-fill-1"
                  style="background:#1e8449"
                ></div>
              </div>
              <input
                type="range"
                class="dual-range-lo"
                id="param-lo-1"
                min="0"
                max="1000"
                step="1"
                value="0"
              />
              <input
                type="range"
                class="dual-range-hi"
                id="param-hi-1"
                min="0"
                max="1000"
                step="1"
                value="1000"
              />
            </div>
          </label>
          <div class="param-range-values" id="param-range-1"></div>
        </div>
        <div class="param-filter" data-axis="2">
          <div class="param-filter-head">
            <span class="param-swatch" style="background:#2471a3"></span>
            <span class="param-filter-name" id="param-name-2"></span>
          </div>
          <label class="param-range-row">
            Range
            <div class="dual-range">
              <div class="dual-range-track">
                <div
                  class="dual-range-fill"
                  id="param-fill-2"
                  style="background:#2471a3"
                ></div>
              </div>
              <input
                type="range"
                class="dual-range-lo"
                id="param-lo-2"
                min="0"
                max="1000"
                step="1"
                value="0"
              />
              <input
                type="range"
                class="dual-range-hi"
                id="param-hi-2"
                min="0"
                max="1000"
                step="1"
                value="1000"
              />
            </div>
          </label>
          <div class="param-range-values" id="param-range-2"></div>
        </div>
      </div>
      <div id="param-combined-stats">
        <div class="param-success-count" id="param-combined-success"></div>
        <div class="param-in-range" id="param-combined-total"></div>
      </div>
    </div>
    <div id="axis-labels"></div>
    <div id="hint">Drag to rotate · scroll to zoom · right-drag to pan · hover for values</div>
    <div id="tooltip"></div>
    </div>
    <div id="view-hist" class="view">
      <div id="hist-filter-panel">
        <h3>Filter parameters</h3>
        <p class="panel-note">
          Narrow which hits contribute to the histogram. All filters apply together.
        </p>
        <div class="ratio-filter-section">
          <h4>Task energy yield ratio (Y)</h4>
{ratio_filter_html}
        </div>
        <button type="button" id="hist-reset-filters">Reset all filters</button>
        <div id="hist-filter-list">
{hist_filter_html}
        </div>
      </div>
      <div id="hist-main">
        <div id="hist-title-bar"></div>
        <div id="hist-stats"></div>
        <div id="hist-chart-wrap">
          <canvas id="hist-canvas"></canvas>
        </div>
      </div>
    </div>
    <div id="view-rf" class="view">
      <div id="rf-main">
        <div id="rf-content">
          <div id="rf-title-bar"></div>
          <div id="rf-grid"></div>
          <div id="rf-notes"></div>
          <div id="rf-footer"></div>
        </div>
        <div id="rf-colorbar-wrap">
          <div id="rf-colorbar-title"></div>
          <div id="rf-colorbar"></div>
          <div id="rf-colorbar-labels">
            <div id="rf-colorbar-top"></div>
            <div id="rf-colorbar-bottom">0</div>
          </div>
        </div>
      </div>
    </div>
    <div id="view-lr" class="view">
      <div id="lr-main">
        <div id="lr-content">
          <div id="lr-title-bar"></div>
          <div id="lr-grid"></div>
          <div id="lr-notes"></div>
          <div id="lr-footer"></div>
        </div>
        <div id="lr-colorbar-wrap">
          <div id="lr-colorbar-title"></div>
          <div id="lr-colorbar"></div>
          <div id="lr-colorbar-labels">
            <div id="lr-colorbar-top"></div>
            <div id="lr-colorbar-bottom">0</div>
          </div>
        </div>
      </div>
    </div>
    <div id="view-dt" class="view">
      <div id="dt-content">
        <div id="dt-title-bar"></div>
        <div id="dt-sections"></div>
        <div id="dt-footer"></div>
      </div>
    </div>
    <div id="error"></div>
  </div>
  <script id="param-3d-data" type="application/json">{data_json}</script>
  <script>{three_js}</script>
  <script>{orbit_js}</script>
  <script>
    (function () {{
      function showError(message) {{
        const box = document.getElementById("error");
        box.textContent = message;
        box.style.display = "block";
      }}

      try {{
        if (typeof THREE === "undefined") {{
          throw new Error("Three.js failed to load.");
        }}

        const DATA = JSON.parse(
          document.getElementById("param-3d-data").textContent,
        );

        document.getElementById("title").textContent = DATA.title_base;
        document.getElementById("colorbar-title").textContent =
          "Re-run successes (0–" + DATA.n_rescreen + ")";
        document.querySelector("#colorbar-label .top").textContent =
          String(DATA.n_rescreen);
        document.getElementById("axis-labels").innerHTML =
          "<div><strong>Axes</strong> (red: " +
          DATA.axes[0] +
          "; green: " +
          DATA.axes[1] +
          "; blue: " +
          DATA.axes[2] +
          ")</div>" +
          DATA.axes
          .map(function (label, i) {{
            return (
              "<div>" +
              label +
              ": " +
              DATA.ranges[i][0].toPrecision(4) +
              " – " +
              DATA.ranges[i][1].toPrecision(4) +
              "</div>"
            );
          }})
          .join("");

        const app = document.getElementById("app");
        const view3d = document.getElementById("view-3d");
        const tooltip = document.getElementById("tooltip");
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(view3d.clientWidth, Math.max(view3d.clientHeight, 1));
        renderer.setClearColor(0xf7f7f8, 1);
        view3d.insertBefore(renderer.domElement, view3d.firstChild);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(
          48,
          view3d.clientWidth / Math.max(view3d.clientHeight, 1),
          0.01,
          100,
        );
        camera.position.set(2.4, 1.6, 2.2);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.target.set(0, 0, 0);

        const ambient = new THREE.AmbientLight(0xffffff, 0.85);
        scene.add(ambient);
        const key = new THREE.DirectionalLight(0xffffff, 0.35);
        key.position.set(2, 3, 4);
        scene.add(key);

        const n = DATA.x.length;
        const PARAM_VALUE_ARRAYS = [DATA.cost, DATA.initial_a, DATA.energy];
        const PARAM_SLIDER_STEPS = 1000;
        const COLOR_BELOW = [0.29, 0.53, 0.91];
        const COLOR_ABOVE = [0.99, 0.84, 0.15];
        let excludeMode = false;
        let indexMap = new Array(n);
        for (let i = 0; i < n; i += 1) {{
          indexMap[i] = i;
        }}

        function sliderToPhysical(axisIdx, sliderVal) {{
          const lo = DATA.ranges[axisIdx][0];
          const hi = DATA.ranges[axisIdx][1];
          return lo + (sliderVal / PARAM_SLIDER_STEPS) * (hi - lo);
        }}

        function clampParamSliders(axisIdx) {{
          const loEl = document.getElementById("param-lo-" + axisIdx);
          const hiEl = document.getElementById("param-hi-" + axisIdx);
          let lo = parseInt(loEl.value, 10);
          let hi = parseInt(hiEl.value, 10);
          if (Number.isNaN(lo) || Number.isNaN(hi)) {{
            throw new Error("Parameter sliders must be numeric.");
          }}
          if (lo > hi) {{
            if (document.activeElement === loEl) {{
              hiEl.value = String(lo);
            }} else {{
              loEl.value = String(hi);
            }}
          }}
        }}

        function getParamPhysicalBounds(axisIdx) {{
          clampParamSliders(axisIdx);
          const loEl = document.getElementById("param-lo-" + axisIdx);
          const hiEl = document.getElementById("param-hi-" + axisIdx);
          const loSlider = Math.min(
            parseInt(loEl.value, 10),
            parseInt(hiEl.value, 10),
          );
          const hiSlider = Math.max(
            parseInt(loEl.value, 10),
            parseInt(hiEl.value, 10),
          );
          return {{
            lo: sliderToPhysical(axisIdx, loSlider),
            hi: sliderToPhysical(axisIdx, hiSlider),
          }};
        }}

        function getAllParamPhysicalBounds() {{
          return [
            getParamPhysicalBounds(0),
            getParamPhysicalBounds(1),
            getParamPhysicalBounds(2),
          ];
        }}

        let selectedRatioTags = new Set(
          DATA.ratio_options.map(function (opt) {{
            return opt.tag;
          }}),
        );

        function pointPassesRatioFilter(i) {{
          return selectedRatioTags.has(DATA.suite_tags[i]);
        }}

        function syncRatioCheckboxes(tag, checked) {{
          document
            .querySelectorAll('.ratio-filter-cb[data-tag="' + tag + '"]')
            .forEach(function (cb) {{
              cb.checked = checked;
            }});
        }}

        function onRatioFilterChange(changedCb) {{
          const tag = changedCb.dataset.tag;
          const checked = changedCb.checked;
          syncRatioCheckboxes(tag, checked);
          if (checked) {{
            selectedRatioTags.add(tag);
          }} else {{
            selectedRatioTags.delete(tag);
          }}
          updateParamSuccessCounts();
          updateHistView();
        }}

        function resetRatioFilters() {{
          selectedRatioTags = new Set(
            DATA.ratio_options.map(function (opt) {{
              return opt.tag;
            }}),
          );
          document.querySelectorAll(".ratio-filter-cb").forEach(function (cb) {{
            cb.checked = true;
          }});
        }}

        function initRatioFilters() {{
          resetRatioFilters();
          document.querySelectorAll(".ratio-filter-cb").forEach(function (cb) {{
            cb.addEventListener("change", function () {{
              onRatioFilterChange(cb);
            }});
          }});
        }}

        function pointPassesParamRangesOnly(i, valueArrays, paramBounds, axisCount) {{
          for (let axisIdx = 0; axisIdx < axisCount; axisIdx += 1) {{
            const value = valueArrays[axisIdx][i];
            const bounds = paramBounds[axisIdx];
            if (value < bounds.lo || value > bounds.hi) {{
              return false;
            }}
          }}
          return true;
        }}

        function countHitsForTagInParamRanges(
          tag,
          valueArrays,
          paramBounds,
          axisCount,
        ) {{
          let count = 0;
          for (let i = 0; i < n; i += 1) {{
            if (DATA.suite_tags[i] !== tag) {{
              continue;
            }}
            if (
              pointPassesParamRangesOnly(i, valueArrays, paramBounds, axisCount)
            ) {{
              count += 1;
            }}
          }}
          return count;
        }}

        function updateRatioFilterLabels(
          containerSelector,
          valueArrays,
          paramBounds,
          axisCount,
        ) {{
          const container = document.querySelector(containerSelector);
          if (!container) {{
            return;
          }}
          container.querySelectorAll(".ratio-filter-label").forEach(function (span) {{
            const tag = span.dataset.tag;
            const label = span.dataset.baseLabel;
            const count = countHitsForTagInParamRanges(
              tag,
              valueArrays,
              paramBounds,
              axisCount,
            );
            span.textContent = label + " (" + count.toLocaleString() + ")";
          }});
        }}

        function pointPassesParamFilters(i, paramBounds) {{
          if (!pointPassesRatioFilter(i)) {{
            return false;
          }}
          for (let axisIdx = 0; axisIdx < 3; axisIdx += 1) {{
            const value = PARAM_VALUE_ARRAYS[axisIdx][i];
            const bounds = paramBounds[axisIdx];
            if (value < bounds.lo || value > bounds.hi) {{
              return false;
            }}
          }}
          return true;
        }}

        const geometry = new THREE.BufferGeometry();
        const material = new THREE.PointsMaterial({{
          size: 0.035,
          vertexColors: THREE.VertexColors,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.78,
          depthWrite: false,
        }});
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        function gapBounds() {{
          const lo = parseInt(document.getElementById("gap-lo").value, 10);
          const hi = parseInt(document.getElementById("gap-hi").value, 10);
          if (Number.isNaN(lo) || Number.isNaN(hi)) {{
            throw new Error("Gap low and high must be integers.");
          }}
          return {{
            lo: Math.min(lo, hi),
            hi: Math.max(lo, hi),
          }};
        }}

        function countParamFilteredHits(paramBounds) {{
          let count = 0;
          for (let i = 0; i < n; i += 1) {{
            if (pointPassesParamFilters(i, paramBounds)) {{
              count += 1;
            }}
          }}
          return count;
        }}

        function updateFilteredTitle(nFiltered) {{
          document.getElementById("title").textContent =
            DATA.title_base + " (" + nFiltered.toLocaleString() + " hits)";
        }}

        function rebuildPointCloud() {{
          const gap = gapBounds();
          const paramBounds = getAllParamPhysicalBounds();
          const nFiltered = countParamFilteredHits(paramBounds);
          updateFilteredTitle(nFiltered);
          const newPositions = [];
          const newColors = [];
          const newIndexMap = [];
          let nBelow = 0;
          let nAbove = 0;
          let nHidden = 0;

          for (let i = 0; i < n; i += 1) {{
            if (!pointPassesParamFilters(i, paramBounds)) {{
              continue;
            }}
            const hit = DATA.n_hit[i];
            if (excludeMode && hit >= gap.lo && hit <= gap.hi) {{
              nHidden += 1;
              continue;
            }}
            newPositions.push(DATA.x[i], DATA.y[i], DATA.z[i]);
            if (excludeMode) {{
              if (hit > gap.hi) {{
                newColors.push(
                  COLOR_ABOVE[0],
                  COLOR_ABOVE[1],
                  COLOR_ABOVE[2],
                );
                nAbove += 1;
              }} else {{
                newColors.push(
                  COLOR_BELOW[0],
                  COLOR_BELOW[1],
                  COLOR_BELOW[2],
                );
                nBelow += 1;
              }}
            }} else {{
              newColors.push(DATA.r[i], DATA.g[i], DATA.b[i]);
            }}
            newIndexMap.push(i);
          }}

          geometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(newPositions, 3),
          );
          geometry.setAttribute(
            "color",
            new THREE.Float32BufferAttribute(newColors, 3),
          );
          indexMap = newIndexMap;

          const stats = document.getElementById("exclude-stats");
          if (excludeMode) {{
            stats.textContent =
              "Showing " +
              indexMap.length.toLocaleString() +
              " / " +
              nFiltered.toLocaleString() +
              " hits\\n" +
              "Blue: " +
              nBelow.toLocaleString() +
              " (< " +
              gap.lo +
              ")\\n" +
              "Yellow: " +
              nAbove.toLocaleString() +
              " (> " +
              gap.hi +
              ")\\n" +
              "Hidden gap: " +
              gap.lo +
              "–" +
              gap.hi +
              " (" +
              nHidden.toLocaleString() +
              ")";
          }} else {{
            stats.textContent =
              "Showing " + nFiltered.toLocaleString() + " hits";
          }}
        }}

        function updateExcludeUi() {{
          const viridisLegend = document.getElementById("viridis-legend");
          const excludeLegend = document.getElementById("exclude-legend");
          viridisLegend.classList.toggle("hidden", excludeMode);
          excludeLegend.classList.toggle("hidden", !excludeMode);
        }}

        function applyExcludeMode() {{
          excludeMode = document.getElementById("exclude-toggle").checked;
          rebuildPointCloud();
          updateExcludeUi();
          hovered = -1;
          tooltip.style.display = "none";
        }}

        rebuildPointCloud();

        const box = new THREE.Box3().setFromObject(points);
        const center = box.getCenter(new THREE.Vector3());
        points.position.sub(center);

        const AXIS_LEN = 1.28;
        const AXIS_BAR_THICKNESS = 0.045;
        const TICK_LABEL_OFFSET = 0.42;
        const AXIS_NAME_LABEL_OFFSET = 0.72;
        const TICK_SCENE = [-1, 0, 1];

        function axisOutwardAnchor(axisMid, outward, extra) {{
          return axisMid.clone().add(
            outward.clone().multiplyScalar(AXIS_BAR_THICKNESS * 0.55 + extra),
          );
        }}

        function tickLabelOutward(axis, t) {{
          if (t === -1) {{
            if (axis.idx === 0) return new THREE.Vector3(0, -1, -1).normalize();
            if (axis.idx === 1) return new THREE.Vector3(-1, 0, -1).normalize();
            return new THREE.Vector3(-1, -1, 0).normalize();
          }}
          return axis.tickOutward.clone();
        }}

        function tickLabelOffset(t) {{
          return t === -1 ? TICK_LABEL_OFFSET + 0.12 : TICK_LABEL_OFFSET;
        }}

        function addLeaderLine(group, from, to, color, opacity) {{
          const leaderMat = new THREE.LineBasicMaterial({{
            color: color,
            transparent: true,
            opacity: opacity || 0.9,
          }});
          const leaderGeom = new THREE.BufferGeometry().setFromPoints([from, to]);
          const leader = new THREE.Line(leaderGeom, leaderMat);
          leader.renderOrder = 9;
          group.add(leader);
        }}

        function sceneToPhysical(axisIndex, sceneCoord) {{
          const lo = DATA.ranges[axisIndex][0];
          const hi = DATA.ranges[axisIndex][1];
          const mid = 0.5 * (lo + hi);
          const span = Math.max(hi - lo, 1e-12);
          return mid + sceneCoord * span * 0.5;
        }}

        function formatPhysical(value) {{
          if (value === 0 || Math.abs(value) < 1e-12) return "0";
          if (Math.abs(value) >= 100) return value.toFixed(0);
          if (Math.abs(value) >= 10) return value.toFixed(1);
          if (Math.abs(value) >= 1) return value.toFixed(2);
          return value.toPrecision(3);
        }}

        function buildPointTooltip(origIdx) {{
          const lines = [
            "<strong>Re-run successes:</strong> " + DATA.n_hit[origIdx],
          ];
          for (let p = 0; p < DATA.filter_labels.length; p += 1) {{
            lines.push(
              "<strong>" +
                DATA.filter_labels[p] +
                ":</strong> " +
                formatPhysical(DATA.filter_values[p][origIdx]),
            );
          }}
          return lines.join("<br>");
        }}

        function readSuccessThreshold() {{
          const threshold = parseInt(
            document.getElementById("success-threshold").value,
            10,
          );
          if (Number.isNaN(threshold)) {{
            throw new Error("Success threshold must be an integer.");
          }}
          return Math.max(0, Math.min(DATA.n_rescreen, threshold));
        }}

        function updateDualRangeFill(axisIdx) {{
          clampParamSliders(axisIdx);
          const loEl = document.getElementById("param-lo-" + axisIdx);
          const hiEl = document.getElementById("param-hi-" + axisIdx);
          const lo = parseInt(loEl.value, 10);
          const hi = parseInt(hiEl.value, 10);
          const loSlider = Math.min(lo, hi);
          const hiSlider = Math.max(lo, hi);
          const fill = document.getElementById("param-fill-" + axisIdx);
          const loPct = (loSlider / PARAM_SLIDER_STEPS) * 100;
          const hiPct = (hiSlider / PARAM_SLIDER_STEPS) * 100;
          fill.style.left = loPct + "%";
          fill.style.width = Math.max(0, hiPct - loPct) + "%";
        }}

        function formatPercent(count, total) {{
          if (total <= 0) return "0%";
          const pct = (100 * count) / total;
          if (pct >= 10) return pct.toFixed(1) + "%";
          if (pct >= 1) return pct.toFixed(1) + "%";
          return pct.toFixed(2) + "%";
        }}

        function updateParamSuccessCounts() {{
          const threshold = readSuccessThreshold();
          const totalHits = DATA.x.length;
          let ratioFilteredTotal = 0;
          for (let i = 0; i < totalHits; i += 1) {{
            if (pointPassesRatioFilter(i)) {{
              ratioFilteredTotal += 1;
            }}
          }}
          const statsTotal = ratioFilteredTotal || totalHits;
          const paramBounds = getAllParamPhysicalBounds();
          updateRatioFilterLabels(
            "#ratio-filter-panel",
            PARAM_VALUE_ARRAYS,
            paramBounds,
            3,
          );
          for (let axisIdx = 0; axisIdx < 3; axisIdx += 1) {{
            updateDualRangeFill(axisIdx);
            const bounds = getParamPhysicalBounds(axisIdx);
            document.getElementById("param-range-" + axisIdx).textContent =
              formatPhysical(bounds.lo) + " – " + formatPhysical(bounds.hi);
          }}
          let inRange = 0;
          let successes = 0;
          for (let i = 0; i < totalHits; i += 1) {{
            if (!pointPassesParamFilters(i, paramBounds)) {{
              continue;
            }}
            inRange += 1;
            if (DATA.n_hit[i] >= threshold) {{
              successes += 1;
            }}
          }}
          document.getElementById("param-combined-success").innerHTML =
            "<strong>" +
            successes.toLocaleString() +
            "</strong> successes (≥ " +
            threshold +
            " re-runs) · " +
            formatPercent(successes, statsTotal) +
            " of selected ratios";
          document.getElementById("param-combined-total").textContent =
            inRange.toLocaleString() +
            " hits in range (" +
            formatPercent(inRange, statsTotal) +
            " of selected ratios) · " +
            statsTotal.toLocaleString() +
            " in selected ratios";
          rebuildPointCloud();
          hovered = -1;
          tooltip.style.display = "none";
        }}

        function initParamFilterPanel() {{
          document.getElementById("success-threshold").value = String(
            DATA.success_threshold,
          );
          for (let axisIdx = 0; axisIdx < 3; axisIdx += 1) {{
            document.getElementById("param-name-" + axisIdx).textContent =
              DATA.axes[axisIdx];
            const loEl = document.getElementById("param-lo-" + axisIdx);
            const hiEl = document.getElementById("param-hi-" + axisIdx);
            loEl.addEventListener("input", updateParamSuccessCounts);
            hiEl.addEventListener("input", updateParamSuccessCounts);
          }}
          document
            .getElementById("success-threshold")
            .addEventListener("input", updateParamSuccessCounts);
          document
            .getElementById("success-threshold")
            .addEventListener("change", updateParamSuccessCounts);
          updateParamSuccessCounts();
        }}

        function makeAxisNameSprite(text, hexColor, scaleX, scaleY) {{
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          const fontSize = 28;
          const barHeight = 8;
          const gap = 10;
          const font = "700 " + fontSize + "px -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.font = font;
          const metrics = ctx.measureText(text);
          const contentWidth = Math.max(metrics.width, 56);
          canvas.width = Math.ceil(contentWidth + 24);
          canvas.height = barHeight + gap + fontSize + 16;
          const barY = 10;
          ctx.fillStyle = hexColor;
          ctx.fillRect(12, barY, contentWidth, barHeight);
          ctx.font = font;
          ctx.fillStyle = hexColor;
          ctx.textBaseline = "top";
          ctx.fillText(text, 12, barY + barHeight + gap);
          const texture = new THREE.CanvasTexture(canvas);
          texture.minFilter = THREE.LinearFilter;
          const material = new THREE.SpriteMaterial({{
            map: texture,
            transparent: true,
            depthTest: false,
          }});
          const sprite = new THREE.Sprite(material);
          sprite.scale.set(scaleX || 0.72, scaleY || 0.16, 1);
          sprite.renderOrder = 10;
          return sprite;
        }}

        function makeTextSprite(text, hexColor, scaleX, scaleY) {{
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          const fontSize = 28;
          ctx.font = "600 " + fontSize + "px -apple-system, BlinkMacSystemFont, sans-serif";
          const metrics = ctx.measureText(text);
          canvas.width = Math.ceil(metrics.width + 24);
          canvas.height = 48;
          ctx.font = "600 " + fontSize + "px -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.fillStyle = hexColor;
          ctx.textBaseline = "middle";
          ctx.fillText(text, 12, 24);
          const texture = new THREE.CanvasTexture(canvas);
          texture.minFilter = THREE.LinearFilter;
          const material = new THREE.SpriteMaterial({{
            map: texture,
            transparent: true,
            depthTest: false,
          }});
          const sprite = new THREE.Sprite(material);
          sprite.scale.set(scaleX || 0.55, scaleY || 0.11, 1);
          sprite.renderOrder = 10;
          return sprite;
        }}

        function boxPoint(x, y, z) {{
          return new THREE.Vector3(x, y, z);
        }}

        function addLabeledAxes() {{
          const axisGroup = new THREE.Group();
          const BOX_LO = -AXIS_LEN;
          const BOX_HI = AXIS_LEN;

          const grayMat = new THREE.LineBasicMaterial({{
            color: 0xbbbbbb,
            linewidth: 1,
          }});
          const boxEdgePairs = [
            [boxPoint(BOX_HI, BOX_LO, BOX_LO), boxPoint(BOX_HI, BOX_HI, BOX_LO)],
            [boxPoint(BOX_HI, BOX_HI, BOX_LO), boxPoint(BOX_LO, BOX_HI, BOX_LO)],
            [boxPoint(BOX_LO, BOX_LO, BOX_HI), boxPoint(BOX_HI, BOX_LO, BOX_HI)],
            [boxPoint(BOX_HI, BOX_LO, BOX_HI), boxPoint(BOX_HI, BOX_HI, BOX_HI)],
            [boxPoint(BOX_HI, BOX_HI, BOX_HI), boxPoint(BOX_LO, BOX_HI, BOX_HI)],
            [boxPoint(BOX_LO, BOX_HI, BOX_HI), boxPoint(BOX_LO, BOX_LO, BOX_HI)],
            [boxPoint(BOX_HI, BOX_LO, BOX_LO), boxPoint(BOX_HI, BOX_LO, BOX_HI)],
            [boxPoint(BOX_HI, BOX_HI, BOX_LO), boxPoint(BOX_HI, BOX_HI, BOX_HI)],
            [boxPoint(BOX_LO, BOX_HI, BOX_LO), boxPoint(BOX_LO, BOX_HI, BOX_HI)],
          ];
          boxEdgePairs.forEach(function (pair) {{
            const geom = new THREE.BufferGeometry().setFromPoints(pair);
            axisGroup.add(new THREE.Line(geom, grayMat));
          }});

          const axisDefs = [
            {{
              idx: 0,
              color: 0xc0392b,
              hex: "#c0392b",
              label: DATA.axes[0],
              start: boxPoint(BOX_LO, BOX_LO, BOX_LO),
              end: boxPoint(BOX_HI, BOX_LO, BOX_LO),
              tickOutward: new THREE.Vector3(0, -1, 0),
            }},
            {{
              idx: 1,
              color: 0x1e8449,
              hex: "#1e8449",
              label: DATA.axes[1],
              start: boxPoint(BOX_LO, BOX_LO, BOX_LO),
              end: boxPoint(BOX_LO, BOX_HI, BOX_LO),
              tickOutward: new THREE.Vector3(-1, 0, 0),
            }},
            {{
              idx: 2,
              color: 0x2471a3,
              hex: "#2471a3",
              label: DATA.axes[2],
              start: boxPoint(BOX_LO, BOX_LO, BOX_LO),
              end: boxPoint(BOX_LO, BOX_LO, BOX_HI),
              tickOutward: new THREE.Vector3(0, -1, 0),
              nameLabelOffset: 0.88,
            }},
          ];

          axisDefs.forEach(function (axis) {{
            const lineMat = new THREE.LineBasicMaterial({{
              color: axis.color,
              linewidth: 1,
            }});
            const axisSpan = axis.start.distanceTo(axis.end);
            const axisMid = axis.start.clone().lerp(axis.end, 0.5);
            const barThickness = AXIS_BAR_THICKNESS;
            let barSize;
            if (axis.idx === 0) {{
              barSize = [axisSpan, barThickness, barThickness];
            }} else if (axis.idx === 1) {{
              barSize = [barThickness, axisSpan, barThickness];
            }} else {{
              barSize = [barThickness, barThickness, axisSpan];
            }}
            const barGeom = new THREE.BoxGeometry(
              barSize[0],
              barSize[1],
              barSize[2],
            );
            const barMat = new THREE.MeshBasicMaterial({{ color: axis.color }});
            const barMesh = new THREE.Mesh(barGeom, barMat);
            barMesh.position.copy(axisMid);
            barMesh.renderOrder = 0;
            axisGroup.add(barMesh);

            const outward = axis.tickOutward.clone();
            const nameOffset = axis.nameLabelOffset || AXIS_NAME_LABEL_OFFSET;
            const anchor = axisOutwardAnchor(axisMid, outward, 0.03);
            const labelPos = axisMid.clone().add(
              outward.multiplyScalar(nameOffset),
            );
            addLeaderLine(axisGroup, anchor, labelPos, axis.color, 0.95);
            const axisLabel = makeAxisNameSprite(axis.label, axis.hex, 0.72, 0.16);
            axisLabel.position.copy(labelPos);
            axisLabel.renderOrder = 20;
            axisGroup.add(axisLabel);

            TICK_SCENE.forEach(function (t) {{
              const frac = 0.5 * (t + 1);
              const pos = axis.start.clone().lerp(axis.end, frac);
              const tickLen = 0.05;
              const tickOutward = axis.tickOutward.clone();
              const tickOut = tickOutward.clone().multiplyScalar(tickLen);
              const tickGeom = new THREE.BufferGeometry().setFromPoints([
                pos.clone().sub(tickOut),
                pos.clone().add(tickOut),
              ]);
              axisGroup.add(new THREE.Line(tickGeom, lineMat));

              if (t !== 0) {{
                const labelOutward = tickLabelOutward(axis, t);
                const labelOffset = tickLabelOffset(t);
                const tickLabel = makeTextSprite(
                  formatPhysical(sceneToPhysical(axis.idx, t)),
                  axis.hex,
                  0.42,
                  0.085,
                );
                const tickAnchor = axisOutwardAnchor(pos, labelOutward, 0.02);
                const tickLabelPos = pos.clone().add(
                  labelOutward.multiplyScalar(labelOffset),
                );
                addLeaderLine(axisGroup, tickAnchor, tickLabelPos, axis.color, 0.85);
                tickLabel.position.copy(tickLabelPos);
                tickLabel.renderOrder = 18;
                axisGroup.add(tickLabel);
              }}
            }});
          }});

          scene.add(axisGroup);
        }}

        addLabeledAxes();

        const raycaster = new THREE.Raycaster();
        raycaster.params.Points = raycaster.params.Points || {{}};
        raycaster.params.Points.threshold = 0.05;
        const mouse = new THREE.Vector2();
        let hovered = -1;

        function onResize() {{
          const w = view3d.clientWidth;
          const h = Math.max(view3d.clientHeight, 1);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }}
        window.addEventListener("resize", function () {{
          if (activeView === "3d") {{
            onResize();
          }} else if (activeView === "hist") {{
            updateHistView();
          }}
        }});

        function onPointerMove(event) {{
          const rect = renderer.domElement.getBoundingClientRect();
          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
          raycaster.setFromCamera(mouse, camera);
          const hits = raycaster.intersectObject(points);
          if (!hits.length) {{
            hovered = -1;
            tooltip.style.display = "none";
            return;
          }}
          const idx = hits[0].index;
          const origIdx = indexMap[idx];
          if (origIdx === undefined) {{
            hovered = -1;
            tooltip.style.display = "none";
            return;
          }}
          if (origIdx === hovered) {{
            tooltip.style.left = event.clientX + 14 + "px";
            tooltip.style.top = event.clientY + 14 + "px";
            return;
          }}
          hovered = origIdx;
          tooltip.innerHTML = buildPointTooltip(origIdx);
          tooltip.style.display = "block";
          tooltip.style.left = event.clientX + 14 + "px";
          tooltip.style.top = event.clientY + 14 + "px";
        }}
        renderer.domElement.addEventListener("pointermove", onPointerMove);
        renderer.domElement.addEventListener("pointerleave", function () {{
          hovered = -1;
          tooltip.style.display = "none";
        }});

        document.getElementById("exclude-toggle").addEventListener("change", applyExcludeMode);
        document.getElementById("apply-exclude").addEventListener("click", applyExcludeMode);
        document.getElementById("gap-lo").addEventListener("change", function () {{
          if (excludeMode) applyExcludeMode();
        }});
        document.getElementById("gap-hi").addEventListener("change", function () {{
          if (excludeMode) applyExcludeMode();
        }});
        updateExcludeUi();
        initRatioFilters();
        initParamFilterPanel();

        let activeView = "3d";
        const HIST_FILTER_COUNT = DATA.filter_values.length;
        const HIST_SLIDER_STEPS = 1000;

        function histSliderToPhysical(axisIdx, sliderVal) {{
          const lo = DATA.filter_ranges[axisIdx][0];
          const hi = DATA.filter_ranges[axisIdx][1];
          return lo + (sliderVal / HIST_SLIDER_STEPS) * (hi - lo);
        }}

        function clampHistSliders(axisIdx) {{
          const loEl = document.getElementById("hist-lo-" + axisIdx);
          const hiEl = document.getElementById("hist-hi-" + axisIdx);
          let lo = parseInt(loEl.value, 10);
          let hi = parseInt(hiEl.value, 10);
          if (Number.isNaN(lo) || Number.isNaN(hi)) {{
            throw new Error("Histogram filters must be numeric.");
          }}
          if (lo > hi) {{
            if (document.activeElement === loEl) {{
              hiEl.value = String(lo);
            }} else {{
              loEl.value = String(hi);
            }}
          }}
        }}

        function getHistPhysicalBounds(axisIdx) {{
          clampHistSliders(axisIdx);
          const loEl = document.getElementById("hist-lo-" + axisIdx);
          const hiEl = document.getElementById("hist-hi-" + axisIdx);
          const loSlider = Math.min(
            parseInt(loEl.value, 10),
            parseInt(hiEl.value, 10),
          );
          const hiSlider = Math.max(
            parseInt(loEl.value, 10),
            parseInt(hiEl.value, 10),
          );
          return {{
            lo: histSliderToPhysical(axisIdx, loSlider),
            hi: histSliderToPhysical(axisIdx, hiSlider),
          }};
        }}

        function getAllHistPhysicalBounds() {{
          const bounds = [];
          for (let axisIdx = 0; axisIdx < HIST_FILTER_COUNT; axisIdx += 1) {{
            bounds.push(getHistPhysicalBounds(axisIdx));
          }}
          return bounds;
        }}

        function updateHistDualRangeFill(axisIdx) {{
          clampHistSliders(axisIdx);
          const loEl = document.getElementById("hist-lo-" + axisIdx);
          const hiEl = document.getElementById("hist-hi-" + axisIdx);
          const lo = parseInt(loEl.value, 10);
          const hi = parseInt(hiEl.value, 10);
          const loSlider = Math.min(lo, hi);
          const hiSlider = Math.max(lo, hi);
          const fill = document.getElementById("hist-fill-" + axisIdx);
          const loPct = (loSlider / HIST_SLIDER_STEPS) * 100;
          const hiPct = (hiSlider / HIST_SLIDER_STEPS) * 100;
          fill.style.left = loPct + "%";
          fill.style.width = Math.max(0, hiPct - loPct) + "%";
        }}

        function pointPassesHistFilters(i, paramBounds) {{
          if (!pointPassesRatioFilter(i)) {{
            return false;
          }}
          for (let axisIdx = 0; axisIdx < HIST_FILTER_COUNT; axisIdx += 1) {{
            const value = DATA.filter_values[axisIdx][i];
            const bounds = paramBounds[axisIdx];
            if (value < bounds.lo || value > bounds.hi) {{
              return false;
            }}
          }}
          return true;
        }}

        function viridisHex(t) {{
          const stops = [
            [0.0, 68, 1, 84],
            [0.25, 59, 82, 139],
            [0.5, 33, 145, 140],
            [0.75, 94, 201, 98],
            [1.0, 253, 231, 37],
          ];
          const x = Math.max(0, Math.min(1, t));
          let i = 0;
          while (i < stops.length - 2 && x > stops[i + 1][0]) {{
            i += 1;
          }}
          const a = stops[i];
          const b = stops[i + 1];
          const u = (x - a[0]) / Math.max(b[0] - a[0], 1e-12);
          const r = Math.round(a[1] + u * (b[1] - a[1]));
          const g = Math.round(a[2] + u * (b[2] - a[2]));
          const bl = Math.round(a[3] + u * (b[3] - a[3]));
          return "rgb(" + r + "," + g + "," + bl + ")";
        }}

        function drawHistogram(counts, filteredCount, totalHits) {{
          const canvas = document.getElementById("hist-canvas");
          const wrap = document.getElementById("hist-chart-wrap");
          const dpr = window.devicePixelRatio || 1;
          const w = Math.max(wrap.clientWidth - 24, 320);
          const h = Math.max(wrap.clientHeight - 24, 220);
          canvas.width = Math.round(w * dpr);
          canvas.height = Math.round(h * dpr);
          canvas.style.width = w + "px";
          canvas.style.height = h + "px";
          const ctx = canvas.getContext("2d");
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, w, h);

          const margin = {{
            top: 18,
            right: 16,
            bottom: 42,
            left: 52,
          }};
          const plotW = w - margin.left - margin.right;
          const plotH = h - margin.top - margin.bottom;
          const nBins = DATA.n_rescreen + 1;
          const maxCount = Math.max.apply(null, counts.concat([1]));

          ctx.strokeStyle = "#bbb";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(margin.left, margin.top);
          ctx.lineTo(margin.left, margin.top + plotH);
          ctx.lineTo(margin.left + plotW, margin.top + plotH);
          ctx.stroke();

          const barGap = 2;
          const barW = plotW / nBins - barGap;
          for (let b = 0; b < nBins; b += 1) {{
            const barH = (counts[b] / maxCount) * plotH;
            const x = margin.left + b * (plotW / nBins) + barGap * 0.5;
            const y = margin.top + plotH - barH;
            ctx.fillStyle = viridisHex(b / DATA.n_rescreen);
            ctx.fillRect(x, y, Math.max(barW, 1), barH);
          }}

          ctx.fillStyle = "#333";
          ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.textAlign = "center";
          for (let b = 0; b < nBins; b += 1) {{
            if (b % 2 === 0 || nBins <= 11) {{
              const x = margin.left + (b + 0.5) * (plotW / nBins);
              ctx.fillText(String(b), x, margin.top + plotH + 16);
            }}
          }}
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          const yTicks = 4;
          for (let t = 0; t <= yTicks; t += 1) {{
            const val = Math.round((maxCount * t) / yTicks);
            const y = margin.top + plotH - (plotH * t) / yTicks;
            ctx.fillText(val.toLocaleString(), margin.left - 8, y);
            ctx.strokeStyle = "#eee";
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(margin.left + plotW, y);
            ctx.stroke();
          }}
          ctx.textAlign = "center";
          ctx.textBaseline = "alphabetic";
          ctx.fillText(
            "Re-run successes (out of " + DATA.n_rescreen + ")",
            margin.left + plotW * 0.5,
            h - 8,
          );
          ctx.textAlign = "left";
          ctx.fillText("Hits per bin", 0, margin.top - 4);
        }}

        function updateHistView() {{
          const paramBounds = getAllHistPhysicalBounds();
          updateRatioFilterLabels(
            "#hist-filter-panel .ratio-filter-section",
            DATA.filter_values,
            paramBounds,
            HIST_FILTER_COUNT,
          );
          const totalHits = DATA.n_hit.length;
          const counts = new Array(DATA.n_rescreen + 1).fill(0);
          let filteredCount = 0;
          let sumHits = 0;

          for (let axisIdx = 0; axisIdx < HIST_FILTER_COUNT; axisIdx += 1) {{
            updateHistDualRangeFill(axisIdx);
            const bounds = paramBounds[axisIdx];
            document.getElementById("hist-range-" + axisIdx).textContent =
              formatPhysical(bounds.lo) + " – " + formatPhysical(bounds.hi);
          }}

          for (let i = 0; i < totalHits; i += 1) {{
            if (!pointPassesHistFilters(i, paramBounds)) {{
              continue;
            }}
            filteredCount += 1;
            const hit = DATA.n_hit[i];
            counts[hit] += 1;
            sumHits += hit;
          }}

          const meanHits = filteredCount ? sumHits / filteredCount : 0;
          document.getElementById("hist-stats").innerHTML =
            "<strong>" +
            filteredCount.toLocaleString() +
            "</strong> hits match filters (" +
            formatPercent(filteredCount, totalHits) +
            " of " +
            totalHits.toLocaleString() +
            ")<br>Mean re-runs: <strong>" +
            meanHits.toFixed(2) +
            "</strong> · Total re-runs counted: <strong>" +
            sumHits.toLocaleString() +
            "</strong>";
          drawHistogram(counts, filteredCount, totalHits);
        }}

        function resetHistFilters() {{
          resetRatioFilters();
          for (let axisIdx = 0; axisIdx < HIST_FILTER_COUNT; axisIdx += 1) {{
            document.getElementById("hist-lo-" + axisIdx).value = "0";
            document.getElementById("hist-hi-" + axisIdx).value = String(
              HIST_SLIDER_STEPS,
            );
          }}
          updateParamSuccessCounts();
          updateHistView();
        }}

        function initHistView() {{
          document.getElementById("hist-title-bar").textContent = DATA.hist_title;
          for (let axisIdx = 0; axisIdx < HIST_FILTER_COUNT; axisIdx += 1) {{
            const loEl = document.getElementById("hist-lo-" + axisIdx);
            const hiEl = document.getElementById("hist-hi-" + axisIdx);
            loEl.addEventListener("input", updateHistView);
            hiEl.addEventListener("input", updateHistView);
          }}
          document
            .getElementById("hist-reset-filters")
            .addEventListener("click", resetHistFilters);
          updateHistView();
        }}

        function initRfView() {{
          document.getElementById("rf-title-bar").textContent = DATA.rf.title;
          document.getElementById("rf-colorbar-title").textContent =
            "Re-run successes (0–" + DATA.n_rescreen + ")";
          document.getElementById("rf-colorbar-top").textContent =
            String(DATA.n_rescreen);
          document.getElementById("rf-grid").innerHTML = DATA.rf.panels
            .map(function (panel) {{
              return (
                '<div class="rf-panel">' +
                '<span class="rf-panel-label">' +
                panel.letter +
                "</span>" +
                '<img src="' +
                panel.src +
                '" alt="Panel ' +
                panel.letter +
                '" />' +
                "</div>"
              );
            }})
            .join("");
          document.getElementById("rf-notes").innerHTML = DATA.rf.row_notes
            .map(function (note) {{
              return "<div>" + note + "</div>";
            }})
            .join("");
          document.getElementById("rf-footer").textContent = DATA.rf.footer;
        }}

        function initLrView() {{
          document.getElementById("lr-title-bar").textContent = DATA.lr.title;
          document.getElementById("lr-colorbar-title").textContent =
            "Re-run successes (0–" + DATA.n_rescreen + ")";
          document.getElementById("lr-colorbar-top").textContent =
            String(DATA.n_rescreen);
          document.getElementById("lr-grid").innerHTML = DATA.lr.panels
            .map(function (panel) {{
              return (
                '<div class="lr-panel">' +
                '<span class="lr-panel-label">' +
                panel.letter +
                "</span>" +
                '<img src="' +
                panel.src +
                '" alt="Panel ' +
                panel.letter +
                '" />' +
                "</div>"
              );
            }})
            .join("");
          document.getElementById("lr-notes").innerHTML = DATA.lr.row_notes
            .map(function (note) {{
              return "<div>" + note + "</div>";
            }})
            .join("");
          document.getElementById("lr-footer").textContent = DATA.lr.footer;
        }}

        function initDtView() {{
          document.getElementById("dt-title-bar").textContent = DATA.dt.title;
          document.getElementById("dt-sections").innerHTML = DATA.dt.sections
            .map(function (section) {{
              const panelsHtml = section.panels
                .map(function (panel, panelIdx) {{
                  return (
                    '<div class="dt-panel-item">' +
                    '<div class="dt-panel">' +
                    '<span class="dt-panel-label">' +
                    panel.letter +
                    "</span>" +
                    '<img src="' +
                    panel.src +
                    '" alt="Panel ' +
                    panel.letter +
                    '" />' +
                    "</div>" +
                    '<div class="dt-panel-note">' +
                    section.panel_notes[panelIdx] +
                    "</div>" +
                    "</div>"
                  );
                }})
                .join("");
              return (
                '<div class="dt-section">' +
                '<div class="dt-section-title">' +
                section.title +
                "</div>" +
                '<div class="dt-section-subtitle">' +
                section.subtitle +
                "</div>" +
                '<div class="dt-grid">' +
                panelsHtml +
                "</div>" +
                "</div>"
              );
            }})
            .join("");
          document.getElementById("dt-footer").textContent = DATA.dt.footer;
        }}

        function switchView(viewName) {{
          activeView = viewName;
          document.querySelectorAll(".view-tab").forEach(function (btn) {{
            btn.classList.toggle("active", btn.dataset.view === viewName);
          }});
          document.getElementById("view-3d").classList.toggle("active", viewName === "3d");
          document.getElementById("view-hist").classList.toggle(
            "active",
            viewName === "hist",
          );
          document.getElementById("view-rf").classList.toggle(
            "active",
            viewName === "rf",
          );
          document.getElementById("view-lr").classList.toggle(
            "active",
            viewName === "lr",
          );
          document.getElementById("view-dt").classList.toggle(
            "active",
            viewName === "dt",
          );
          hovered = -1;
          tooltip.style.display = "none";
          if (viewName === "hist") {{
            updateHistView();
          }} else if (viewName === "3d") {{
            onResize();
          }}
        }}

        document.querySelectorAll(".view-tab").forEach(function (btn) {{
          btn.addEventListener("click", function () {{
            switchView(btn.dataset.view);
          }});
        }});
        initHistView();
        initRfView();
        initLrView();
        initDtView();

        function animate() {{
          requestAnimationFrame(animate);
          if (activeView !== "3d") {{
            return;
          }}
          controls.update();
          renderer.render(scene, camera);
        }}
        animate();
      }} catch (err) {{
        showError(String(err && err.message ? err.message : err));
        if (typeof console !== "undefined" && console.error) {{
          console.error(err);
        }}
      }}
    }})();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return len(rows)


def compute_random_forest(
    X: np.ndarray,
    high_success: np.ndarray,
    *,
    X_plot: np.ndarray | None = None,
) -> Dict[str, object]:
    """Random forest classifier: high vs low re-hit from scaled parameters."""
    from sklearn.ensemble import RandomForestClassifier

    y = high_success.astype(int)
    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        class_weight="balanced",
        oob_score=len(np.unique(y)) >= 2,
        random_state=0,
        n_jobs=-1,
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    top2_idx = np.argsort(importances)[-2:][::-1]
    plot_x = X if X_plot is None else X_plot
    oob = clf.oob_score_ if len(np.unique(y)) >= 2 else 1.0
    return {
        "importances": importances,
        "top2_idx": top2_idx,
        "top2_coords": plot_x[:, top2_idx],
        "oob_score": float(oob),
        "predict_proba": _positive_class_proba(clf, X),
    }


def compute_logistic_regression(
    X: np.ndarray,
    high_success: np.ndarray,
    *,
    X_plot: np.ndarray | None = None,
) -> Dict[str, object]:
    """Logistic regression: high vs low re-hit from scaled parameters."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = high_success.astype(int)
    plot_x = X if X_plot is None else X_plot
    if len(np.unique(y)) < 2:
        cv_accuracy, cv_accuracy_std = _degenerate_binary_cv(y)
        zeros = np.zeros(X.shape[1], dtype=float)
        top2_idx = np.array([0, min(1, X.shape[1] - 1)])
        return {
            "coefficients": zeros,
            "importances": zeros,
            "top2_idx": top2_idx,
            "top2_coords": plot_x[:, top2_idx],
            "cv_accuracy": cv_accuracy,
            "cv_accuracy_std": cv_accuracy_std,
            "predict_proba": np.zeros(len(X), dtype=np.float32),
        }
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_accuracy = float(np.mean(cv_scores))
    cv_accuracy_std = float(np.std(cv_scores))
    clf.fit(X, y)
    coefs = clf.coef_[0] if clf.coef_.size else np.zeros(X.shape[1])
    abs_coefs = np.abs(coefs)
    top2_idx = np.argsort(abs_coefs)[-2:][::-1]
    plot_x = X if X_plot is None else X_plot
    return {
        "coefficients": coefs,
        "importances": abs_coefs,
        "top2_idx": top2_idx,
        "top2_coords": plot_x[:, top2_idx],
        "cv_accuracy": cv_accuracy,
        "cv_accuracy_std": cv_accuracy_std,
        "predict_proba": _positive_class_proba(clf, X),
    }


def compute_decision_tree(
    X: np.ndarray,
    high_success: np.ndarray,
    *,
    X_plot: np.ndarray | None = None,
) -> Dict[str, object]:
    """Decision tree classifier: high vs low re-hit from scaled parameters."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    y = high_success.astype(int)
    clf = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=0,
    )
    if len(np.unique(y)) < 2:
        cv_accuracy, cv_accuracy_std = _degenerate_binary_cv(y)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_accuracy = float(np.mean(cv_scores))
        cv_accuracy_std = float(np.std(cv_scores))
    clf.fit(X, y)
    importances = clf.feature_importances_
    top2_idx = np.argsort(importances)[-2:][::-1]
    plot_x = X if X_plot is None else X_plot
    return {
        "importances": importances,
        "top2_idx": top2_idx,
        "top2_coords": plot_x[:, top2_idx],
        "cv_accuracy": cv_accuracy,
        "cv_accuracy_std": cv_accuracy_std,
        "predict_proba": _positive_class_proba(clf, X),
    }


def fit_decision_tree_model(
    X: np.ndarray,
    high_success: np.ndarray,
):
    """Fit the same decision tree used for panel rendering."""
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(X, high_success.astype(int))
    return clf


def stratified_subsample(
    rows: Sequence[HitRecord],
    *,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[List[HitRecord], np.ndarray]:
    """Subsample up to ``max_points`` hits, stratified by ``n_hit_again``."""
    if len(rows) <= max_points:
        idx = np.arange(len(rows))
        return list(rows), idx
    by_count: Dict[int, List[int]] = {}
    for i, row in enumerate(rows):
        by_count.setdefault(int(row["n_hit_again"]), []).append(i)
    chosen: List[int] = []
    counts = sorted(by_count)
    remaining = max_points
    while remaining > 0 and counts:
        progressed = False
        for c in counts:
            if remaining <= 0:
                break
            bucket = by_count[c]
            if not bucket:
                continue
            j = int(rng.integers(0, len(bucket)))
            chosen.append(bucket.pop(j))
            remaining -= 1
            progressed = True
        counts = [c for c in counts if by_count[c]]
        if not progressed:
            break
    chosen = np.array(sorted(chosen), dtype=int)
    return [rows[i] for i in chosen], chosen


def rehit_lda_classes(n_hit: np.ndarray) -> np.ndarray:
    """Three re-screen tiers for 2D LDA (low / mid / high success count)."""
    classes = np.zeros(len(n_hit), dtype=int)
    classes[n_hit >= 7] = 1
    classes[n_hit >= 14] = 2
    return classes


def compute_lda_embedding(X: np.ndarray, n_hit: np.ndarray) -> np.ndarray:
    """LDA on parameters supervised by 3-class re-hit bins (up to 2 axes)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    y = rehit_lda_classes(n_hit)
    n_comp = min(2, len(np.unique(y)) - 1)
    if n_comp <= 0:
        return np.zeros((len(X), 2), dtype=float)
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    coords = lda.fit_transform(X, y)
    if n_comp == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords


def compute_supervised_umap_embedding(
    X: np.ndarray,
    high_success: np.ndarray,
    *,
    umap_neighbors: int,
) -> np.ndarray:
    """UMAP with categorical supervision on high vs low re-hit (``target_weight=0.5``)."""
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=0.25,
        target_metric="categorical",
        target_weight=0.5,
        random_state=0,
    )
    return reducer.fit_transform(X, y=high_success.astype(int))


def compute_embeddings_for_rows(
    rows: Sequence[HitRecord],
    X: np.ndarray,
    *,
    success_threshold: int,
    tsne_max_points: int,
    umap_neighbors: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    out: Dict[str, np.ndarray] = {"pca": PCA(n_components=2, random_state=0).fit_transform(X)}

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=0.25,
            metric="euclidean",
            random_state=0,
        )
        out["umap"] = reducer.fit_transform(X)
    except Exception as exc:
        print(f"UMAP skipped: {exc}", file=sys.stderr)

    _, tsne_idx = stratified_subsample(rows, max_points=tsne_max_points, rng=rng)
    tsne = TSNE(
        n_components=2,
        perplexity=min(40, max(5, len(tsne_idx) // 50)),
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    out["tsne_idx"] = tsne_idx
    out["tsne"] = tsne.fit_transform(X[tsne_idx])
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    out["lda"] = compute_lda_embedding(X, n_hit)
    out["sup_umap"] = compute_supervised_umap_embedding(
        X,
        n_hit >= success_threshold,
        umap_neighbors=umap_neighbors,
    )
    return out


def kmeans_labels(X: np.ndarray, *, k: int, rng: np.random.Generator) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(
        n_clusters=k, n_init=10, random_state=int(rng.integers(0, 2**31 - 1))
    ).fit_predict(X)


def summarize_separation(
    rows: Sequence[HitRecord],
    cluster_ids: np.ndarray,
    *,
    high_success: np.ndarray,
) -> str:
    """Return a one-line summary of how well clusters align with re-hit groups."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    y = high_success.astype(int)
    ari = adjusted_rand_score(y, cluster_ids)
    try:
        sil = silhouette_score(
            np.array([[float(r["hit_rate"])] for r in rows]),
            cluster_ids,
        )
        sil_txt = f" silhouette(hit rate)={sil:.3f}"
    except Exception:
        sil_txt = ""
    means = []
    for cid in sorted(set(cluster_ids)):
        mask = cluster_ids == cid
        rates = [float(rows[i]["hit_rate"]) for i in np.where(mask)[0]]
        means.append((cid, float(np.mean(rates)), int(mask.sum())))
    mean_txt = "; ".join(f"C{int(c)}: rate={m:.2f} (n={n})" for c, m, n in means)
    return f"ARI(high/low vs k=2)={ari:.3f}{sil_txt}. {mean_txt}"


def summarize_random_forest(rf: Dict[str, object], *, label: str = "RF") -> str:
    return f"{label} OOB accuracy={float(rf['oob_score']):.3f}"


def summarize_logistic_regression(lr: Dict[str, object], *, label: str = "LR") -> str:
    return (
        f"{label} 5-fold CV accuracy={float(lr['cv_accuracy']):.3f}"
        f" $\\pm$ {float(lr['cv_accuracy_std']):.3f}"
    )


def summarize_decision_tree(dt: Dict[str, object], *, label: str = "DT") -> str:
    return (
        f"{label} 5-fold CV accuracy={float(dt['cv_accuracy']):.3f}"
        f" $\\pm$ {float(dt['cv_accuracy_std']):.3f}"
    )


def summarize_rf_models(
    rf: Dict[str, object],
    rf_no_flow: Dict[str, object],
    rf_no_flow_mut: Dict[str, object],
) -> str:
    return "; ".join(
        [
            summarize_random_forest(rf, label="RF (6 features)"),
            summarize_random_forest(rf_no_flow, label="RF (excl. flow)"),
            summarize_random_forest(rf_no_flow_mut, label="RF (excl. flow & mut.)"),
        ]
    )


def summarize_lr_models(
    lr: Dict[str, object],
    lr_no_flow: Dict[str, object],
    lr_no_flow_mut: Dict[str, object],
) -> str:
    return "; ".join(
        [
            summarize_logistic_regression(lr, label="LR (6 features)"),
            summarize_logistic_regression(lr_no_flow, label="LR (excl. flow)"),
            summarize_logistic_regression(lr_no_flow_mut, label="LR (excl. flow & mut.)"),
        ]
    )


def summarize_dt_models(
    dt: Dict[str, object],
    dt_no_flow: Dict[str, object],
    dt_no_flow_mut: Dict[str, object],
) -> str:
    return "; ".join(
        [
            summarize_decision_tree(dt, label="DT (6 features)"),
            summarize_decision_tree(dt_no_flow, label="DT (excl. flow)"),
            summarize_decision_tree(dt_no_flow_mut, label="DT (excl. flow & mut.)"),
        ]
    )


def build_summary(
    rows: Sequence[HitRecord],
    cluster_pca: np.ndarray,
    *,
    high_success: np.ndarray,
    rf: Dict[str, object],
    rf_no_flow: Dict[str, object],
    rf_no_flow_mut: Dict[str, object],
) -> Tuple[str, str]:
    embedding_summary = summarize_separation(rows, cluster_pca, high_success=high_success)
    rf_summary = summarize_rf_models(rf, rf_no_flow, rf_no_flow_mut)
    return embedding_summary, rf_summary


def _save_panel(
    fig: plt.Figure,
    path: Path,
    *,
    margins: Dict[str, float] | None = None,
    dpi: int = 180,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if margins is not None:
        fig.subplots_adjust(**margins)
    else:
        fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def trim_image_margins(
    img: np.ndarray,
    *,
    threshold: float = 0.985,
    pad: int = 3,
) -> np.ndarray:
    """Crop near-white borders from a cached panel PNG before stitching."""
    gray = img if img.ndim == 2 else img[:, :, :3].min(axis=2)
    mask = gray < threshold
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0 = max(0, int(rows[0]) - pad)
    r1 = min(img.shape[0], int(rows[-1]) + 1 + pad)
    c0 = max(0, int(cols[0]) - pad)
    c1 = min(img.shape[1], int(cols[-1]) + 1 + pad)
    return img[r0:r1, c0:c1]


def assembly_panel_label(ax: plt.Axes, letter: str) -> None:
    """Panel letter tight to the trimmed panel image (used when stitching PNGs)."""
    ax.text(
        -0.01,
        1.01,
        f"({letter})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


def _show_panel(ax: plt.Axes, panel_path: Path) -> None:
    ax.imshow(trim_image_margins(imread(panel_path)), aspect="auto")
    ax.axis("off")


def plot_lr_coefficient_panel(
    ax: plt.Axes,
    *,
    coefficients: np.ndarray,
    feature_names: Sequence[str],
    title: str,
    panel_letter: str | None = None,
) -> None:
    order = np.argsort(coefficients)
    colors = [
        "#d62728" if float(coefficients[i]) < 0 else "#4c78a8" for i in order
    ]
    ax.barh(
        np.arange(len(coefficients)),
        coefficients[order],
        color=colors,
        alpha=0.85,
    )
    ax.set_yticks(np.arange(len(coefficients)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8.0)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.axvline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
    ax.grid(alpha=0.2, linewidth=0.7, axis="x")
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def plot_rf_importance_panel(
    ax: plt.Axes,
    *,
    importances: np.ndarray,
    feature_names: Sequence[str],
    title: str,
    panel_letter: str | None = None,
) -> None:
    order = np.argsort(importances)
    ax.barh(
        np.arange(len(importances)),
        importances[order],
        color="#4c78a8",
        alpha=0.85,
    )
    ax.set_yticks(np.arange(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8.0)
    ax.set_xlabel("Mean decrease impurity")
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.grid(alpha=0.2, linewidth=0.7, axis="x")
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def plot_rf_top_features_panel(
    ax: plt.Axes,
    *,
    coords: np.ndarray,
    n_hit_again: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    panel_letter: str | None = None,
    point_size: float = 5.0,
    alpha: float = 0.35,
) -> None:
    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=n_hit_again,
        cmap="viridis",
        norm=norm,
        s=point_size,
        alpha=alpha,
        linewidths=0.0,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.grid(alpha=0.2, linewidth=0.7)
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def plot_dt_tree_panel(
    ax: plt.Axes,
    *,
    clf: object,
    feature_names: Sequence[str],
    title: str,
    panel_letter: str | None = None,
    fontsize: int = 6,
    class_names: Sequence[str] = ("low", "high"),
) -> None:
    from sklearn.tree import plot_tree

    plot_tree(
        clf,
        feature_names=list(feature_names),
        class_names=list(class_names),
        filled=True,
        rounded=True,
        fontsize=fontsize,
        ax=ax,
        impurity=False,
        proportion=True,
        precision=2,
    )
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad_x = 0.04 * (x1 - x0)
    pad_y = 0.05 * (y1 - y0)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def _show_panel_preserve_aspect(ax: plt.Axes, panel_path: Path) -> None:
    img = trim_image_margins(imread(panel_path))
    ax.imshow(img, aspect="equal")
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.axis("off")


def _dt_panel_aspect(panel_path: Path) -> float:
    img = trim_image_margins(imread(panel_path))
    return float(img.shape[1]) / max(float(img.shape[0]), 1.0)


def _show_panel_fill_subplot(ax: plt.Axes, panel_path: Path) -> None:
    """Stretch a trimmed panel to fill the subplot (used for decision trees)."""
    img = trim_image_margins(imread(panel_path))
    ax.imshow(img, aspect="auto")
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.axis("off")


def _dt_assembly_figsize(
    dt_panels_dir: Path,
    *,
    row_height_in: float = DT_ROW_HEIGHT_IN,
    row_gap_in: float = DT_ROW_GAP_IN,
) -> Tuple[float, float]:
    """Fixed-width composite with explicit vertical gaps between tree rows."""
    del dt_panels_dir
    n_rows = len(DT_PANEL_LETTERS)
    fig_w = DT_FIGSIZE[0]
    fig_h = row_height_in * n_rows + row_gap_in * max(0, n_rows - 1) + DT_FIG_OVERHEAD_IN
    return (fig_w, fig_h)


def plot_embedding_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    n_hit_again: np.ndarray,
    title: str,
    panel_letter: str | None = None,
    cmap: str = "viridis",
    point_size: float = 5.0,
    alpha: float = 0.35,
) -> None:
    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=n_hit_again,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        linewidths=0.0,
    )
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2, linewidth=0.7)
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def plot_binary_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    high_success: np.ndarray,
    title: str,
    panel_letter: str | None = None,
    point_size: float = 5.0,
    alpha: float = 0.4,
    show_legend: bool = True,
) -> None:
    colors = np.where(high_success, "#d62728", "#4c78a8")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0.0,
    )
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2, linewidth=0.7)
    if panel_letter is not None:
        panel_label(ax, panel_letter)
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=6, color="#d62728", label="High re-hit"),
            Line2D([0], [0], marker="o", linestyle="", markersize=6, color="#4c78a8", label="Low re-hit"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.5)


def plot_cluster_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    cluster_ids: np.ndarray,
    title: str,
    panel_letter: str | None = None,
    point_size: float = 5.0,
    alpha: float = 0.4,
) -> None:
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    colors = [palette[int(c) % len(palette)] for c in cluster_ids]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0.0,
    )
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2, linewidth=0.7)
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def plot_hist_panel(
    ax: plt.Axes,
    *,
    n_hit: np.ndarray,
    cluster_pca: np.ndarray,
    title: str,
    panel_letter: str | None = None,
    show_legend: bool = True,
) -> None:
    bins = np.arange(-0.5, N_RESCREEN + 1.5, 1)
    for cid, color in enumerate(["#4c78a8", "#f58518"]):
        mask = cluster_pca == cid
        ax.hist(
            n_hit[mask],
            bins=bins,
            alpha=0.55,
            color=color,
            label=f"Cluster {cid} (n={int(mask.sum())})",
            density=True,
        )
    ax.set_xlabel(f"Re-run successes per hit (out of {N_RESCREEN})")
    ax.set_ylabel("Fraction within cluster")
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.2, linewidth=0.7)
    if panel_letter is not None:
        panel_label(ax, panel_letter)


def cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def cache_embeddings_path(cache_dir: Path) -> Path:
    return cache_dir / "embeddings.npz"


def cache_panels_dir(cache_dir: Path) -> Path:
    return cache_dir / "panels"


def cache_rf_panels_dir(cache_dir: Path) -> Path:
    return cache_dir / "rf_panels"


def cache_lr_panels_dir(cache_dir: Path) -> Path:
    return cache_dir / "lr_panels"


def cache_dt_panels_dir(cache_dir: Path) -> Path:
    return cache_dir / "dt_panels"


def cache_dt_extreme_panels_dir(cache_dir: Path) -> Path:
    return cache_dir / "dt_extreme_panels"


def cache_dt_extreme_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "dt_extreme_meta.json"


def _meta_matches(
    meta: dict,
    *,
    success_threshold: int,
    tsne_max_points: int,
    umap_neighbors: int,
) -> bool:
    return (
        int(meta.get("success_threshold", -1)) == success_threshold
        and int(meta.get("tsne_max_points", -1)) == tsne_max_points
        and int(meta.get("umap_neighbors", -1)) == umap_neighbors
    )


def save_cache(
    cache_dir: Path,
    *,
    data: CacheData,
    success_threshold: int,
    tsne_max_points: int,
    umap_neighbors: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb = data["embeddings"]
    save_kwargs = {
        "pca": emb["pca"],
        "tsne": emb["tsne"],
        "tsne_idx": emb["tsne_idx"],
        "lda": emb["lda"],
        "sup_umap": emb["sup_umap"],
        "n_hit": data["n_hit"],
        "high_success": data["high_success"],
        "cluster_pca": data["cluster_pca"],
        "rf_importances": data["rf"]["importances"],
        "rf_top2_idx": data["rf"]["top2_idx"],
        "rf_top2_coords": data["rf"]["top2_coords"],
        "rf_oob_score": np.array([float(data["rf"]["oob_score"])]),
        "rf_no_flow_importances": data["rf_no_flow"]["importances"],
        "rf_no_flow_top2_idx": data["rf_no_flow"]["top2_idx"],
        "rf_no_flow_top2_coords": data["rf_no_flow"]["top2_coords"],
        "rf_no_flow_oob_score": np.array([float(data["rf_no_flow"]["oob_score"])]),
        "rf_no_flow_mut_importances": data["rf_no_flow_mut"]["importances"],
        "rf_no_flow_mut_top2_idx": data["rf_no_flow_mut"]["top2_idx"],
        "rf_no_flow_mut_top2_coords": data["rf_no_flow_mut"]["top2_coords"],
        "rf_no_flow_mut_oob_score": np.array([float(data["rf_no_flow_mut"]["oob_score"])]),
        "lr_coefficients": data["lr"]["coefficients"],
        "lr_top2_idx": data["lr"]["top2_idx"],
        "lr_top2_coords": data["lr"]["top2_coords"],
        "lr_cv_accuracy": np.array([float(data["lr"]["cv_accuracy"])]),
        "lr_cv_accuracy_std": np.array([float(data["lr"]["cv_accuracy_std"])]),
        "lr_no_flow_coefficients": data["lr_no_flow"]["coefficients"],
        "lr_no_flow_top2_idx": data["lr_no_flow"]["top2_idx"],
        "lr_no_flow_top2_coords": data["lr_no_flow"]["top2_coords"],
        "lr_no_flow_cv_accuracy": np.array([float(data["lr_no_flow"]["cv_accuracy"])]),
        "lr_no_flow_cv_accuracy_std": np.array([float(data["lr_no_flow"]["cv_accuracy_std"])]),
        "lr_no_flow_mut_coefficients": data["lr_no_flow_mut"]["coefficients"],
        "lr_no_flow_mut_top2_idx": data["lr_no_flow_mut"]["top2_idx"],
        "lr_no_flow_mut_top2_coords": data["lr_no_flow_mut"]["top2_coords"],
        "lr_no_flow_mut_cv_accuracy": np.array([float(data["lr_no_flow_mut"]["cv_accuracy"])]),
        "lr_no_flow_mut_cv_accuracy_std": np.array(
            [float(data["lr_no_flow_mut"]["cv_accuracy_std"])]
        ),
        "dt_importances": data["dt"]["importances"],
        "dt_top2_idx": data["dt"]["top2_idx"],
        "dt_top2_coords": data["dt"]["top2_coords"],
        "dt_cv_accuracy": np.array([float(data["dt"]["cv_accuracy"])]),
        "dt_cv_accuracy_std": np.array([float(data["dt"]["cv_accuracy_std"])]),
        "dt_no_flow_importances": data["dt_no_flow"]["importances"],
        "dt_no_flow_top2_idx": data["dt_no_flow"]["top2_idx"],
        "dt_no_flow_top2_coords": data["dt_no_flow"]["top2_coords"],
        "dt_no_flow_cv_accuracy": np.array([float(data["dt_no_flow"]["cv_accuracy"])]),
        "dt_no_flow_cv_accuracy_std": np.array(
            [float(data["dt_no_flow"]["cv_accuracy_std"])]
        ),
        "dt_no_flow_mut_importances": data["dt_no_flow_mut"]["importances"],
        "dt_no_flow_mut_top2_idx": data["dt_no_flow_mut"]["top2_idx"],
        "dt_no_flow_mut_top2_coords": data["dt_no_flow_mut"]["top2_coords"],
        "dt_no_flow_mut_cv_accuracy": np.array(
            [float(data["dt_no_flow_mut"]["cv_accuracy"])]
        ),
        "dt_no_flow_mut_cv_accuracy_std": np.array(
            [float(data["dt_no_flow_mut"]["cv_accuracy_std"])]
        ),
    }
    if "umap" in emb:
        save_kwargs["umap"] = emb["umap"]
    np.savez_compressed(cache_embeddings_path(cache_dir), **save_kwargs)
    meta = {
        "cache_version": CACHE_VERSION,
        "success_threshold": success_threshold,
        "tsne_max_points": tsne_max_points,
        "umap_neighbors": umap_neighbors,
        "n_hits": int(len(data["n_hit"])),
        "has_umap": "umap" in emb,
        "embedding_summary": str(data["embedding_summary"]),
        "rf_summary": str(data["rf_summary"]),
        "lr_summary": str(data["lr_summary"]),
        "dt_summary": str(data["dt_summary"]),
        "rf_oob_score": float(data["rf"]["oob_score"]),
        "rf_no_flow_oob_score": float(data["rf_no_flow"]["oob_score"]),
        "rf_no_flow_mut_oob_score": float(data["rf_no_flow_mut"]["oob_score"]),
        "lr_cv_accuracy": float(data["lr"]["cv_accuracy"]),
        "lr_cv_accuracy_std": float(data["lr"]["cv_accuracy_std"]),
        "lr_no_flow_cv_accuracy": float(data["lr_no_flow"]["cv_accuracy"]),
        "lr_no_flow_cv_accuracy_std": float(data["lr_no_flow"]["cv_accuracy_std"]),
        "lr_no_flow_mut_cv_accuracy": float(data["lr_no_flow_mut"]["cv_accuracy"]),
        "lr_no_flow_mut_cv_accuracy_std": float(data["lr_no_flow_mut"]["cv_accuracy_std"]),
        "dt_cv_accuracy": float(data["dt"]["cv_accuracy"]),
        "dt_cv_accuracy_std": float(data["dt"]["cv_accuracy_std"]),
        "dt_no_flow_cv_accuracy": float(data["dt_no_flow"]["cv_accuracy"]),
        "dt_no_flow_cv_accuracy_std": float(data["dt_no_flow"]["cv_accuracy_std"]),
        "dt_no_flow_mut_cv_accuracy": float(data["dt_no_flow_mut"]["cv_accuracy"]),
        "dt_no_flow_mut_cv_accuracy_std": float(data["dt_no_flow_mut"]["cv_accuracy_std"]),
    }
    cache_meta_path(cache_dir).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_cache(
    cache_dir: Path,
    *,
    success_threshold: int,
    tsne_max_points: int,
    umap_neighbors: int,
) -> Optional[CacheData]:
    meta_path = cache_meta_path(cache_dir)
    emb_path = cache_embeddings_path(cache_dir)
    if not meta_path.is_file() or not emb_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not _meta_matches(
        meta,
        success_threshold=success_threshold,
        tsne_max_points=tsne_max_points,
        umap_neighbors=umap_neighbors,
    ):
        return None
    if int(meta.get("cache_version", 0)) != CACHE_VERSION:
        return None
    loaded = np.load(emb_path)
    if (
        "lda" not in loaded
        or "sup_umap" not in loaded
        or "rf_importances" not in loaded
        or "rf_no_flow_importances" not in loaded
        or "rf_no_flow_mut_importances" not in loaded
        or "lr_coefficients" not in loaded
        or "lr_no_flow_coefficients" not in loaded
        or "lr_no_flow_mut_coefficients" not in loaded
        or "dt_importances" not in loaded
        or "dt_no_flow_importances" not in loaded
        or "dt_no_flow_mut_importances" not in loaded
    ):
        return None
    embeddings: Dict[str, np.ndarray] = {
        "pca": loaded["pca"],
        "tsne": loaded["tsne"],
        "tsne_idx": loaded["tsne_idx"],
        "lda": loaded["lda"],
        "sup_umap": loaded["sup_umap"],
    }
    if "umap" in loaded:
        embeddings["umap"] = loaded["umap"]
    return {
        "embeddings": embeddings,
        "n_hit": loaded["n_hit"],
        "high_success": loaded["high_success"],
        "cluster_pca": loaded["cluster_pca"],
        "rf": {
            "importances": loaded["rf_importances"],
            "top2_idx": loaded["rf_top2_idx"],
            "top2_coords": loaded["rf_top2_coords"],
            "oob_score": float(loaded["rf_oob_score"][0]),
            "feature_names": feature_display_names(raw=True),
        },
        "rf_no_flow": {
            "importances": loaded["rf_no_flow_importances"],
            "top2_idx": loaded["rf_no_flow_top2_idx"],
            "top2_coords": loaded["rf_no_flow_top2_coords"],
            "oob_score": float(loaded["rf_no_flow_oob_score"][0]),
            "feature_names": feature_display_names(
                features=param_features_without_flow(),
                raw=True,
            ),
        },
        "rf_no_flow_mut": {
            "importances": loaded["rf_no_flow_mut_importances"],
            "top2_idx": loaded["rf_no_flow_mut_top2_idx"],
            "top2_coords": loaded["rf_no_flow_mut_top2_coords"],
            "oob_score": float(loaded["rf_no_flow_mut_oob_score"][0]),
            "feature_names": feature_display_names(
                features=param_features_without_flow_and_mutation(),
                raw=True,
            ),
        },
        "lr": {
            "coefficients": loaded["lr_coefficients"],
            "importances": np.abs(loaded["lr_coefficients"]),
            "top2_idx": loaded["lr_top2_idx"],
            "top2_coords": loaded["lr_top2_coords"],
            "cv_accuracy": float(loaded["lr_cv_accuracy"][0]),
            "cv_accuracy_std": float(loaded["lr_cv_accuracy_std"][0]),
            "feature_names": feature_display_names(raw=True),
        },
        "lr_no_flow": {
            "coefficients": loaded["lr_no_flow_coefficients"],
            "importances": np.abs(loaded["lr_no_flow_coefficients"]),
            "top2_idx": loaded["lr_no_flow_top2_idx"],
            "top2_coords": loaded["lr_no_flow_top2_coords"],
            "cv_accuracy": float(loaded["lr_no_flow_cv_accuracy"][0]),
            "cv_accuracy_std": float(loaded["lr_no_flow_cv_accuracy_std"][0]),
            "feature_names": feature_display_names(
                features=param_features_without_flow(),
                raw=True,
            ),
        },
        "lr_no_flow_mut": {
            "coefficients": loaded["lr_no_flow_mut_coefficients"],
            "importances": np.abs(loaded["lr_no_flow_mut_coefficients"]),
            "top2_idx": loaded["lr_no_flow_mut_top2_idx"],
            "top2_coords": loaded["lr_no_flow_mut_top2_coords"],
            "cv_accuracy": float(loaded["lr_no_flow_mut_cv_accuracy"][0]),
            "cv_accuracy_std": float(loaded["lr_no_flow_mut_cv_accuracy_std"][0]),
            "feature_names": feature_display_names(
                features=param_features_without_flow_and_mutation(),
                raw=True,
            ),
        },
        "dt": {
            "importances": loaded["dt_importances"],
            "top2_idx": loaded["dt_top2_idx"],
            "top2_coords": loaded["dt_top2_coords"],
            "cv_accuracy": float(loaded["dt_cv_accuracy"][0]),
            "cv_accuracy_std": float(loaded["dt_cv_accuracy_std"][0]),
            "feature_names": feature_display_names(raw=True),
        },
        "dt_no_flow": {
            "importances": loaded["dt_no_flow_importances"],
            "top2_idx": loaded["dt_no_flow_top2_idx"],
            "top2_coords": loaded["dt_no_flow_top2_coords"],
            "cv_accuracy": float(loaded["dt_no_flow_cv_accuracy"][0]),
            "cv_accuracy_std": float(loaded["dt_no_flow_cv_accuracy_std"][0]),
            "feature_names": feature_display_names(
                features=param_features_without_flow(),
                raw=True,
            ),
        },
        "dt_no_flow_mut": {
            "importances": loaded["dt_no_flow_mut_importances"],
            "top2_idx": loaded["dt_no_flow_mut_top2_idx"],
            "top2_coords": loaded["dt_no_flow_mut_top2_coords"],
            "cv_accuracy": float(loaded["dt_no_flow_mut_cv_accuracy"][0]),
            "cv_accuracy_std": float(loaded["dt_no_flow_mut_cv_accuracy_std"][0]),
            "feature_names": feature_display_names(
                features=param_features_without_flow_and_mutation(),
                raw=True,
            ),
        },
        "embedding_summary": meta.get("embedding_summary", meta.get("summary", "")),
        "rf_summary": meta.get(
            "rf_summary",
            summarize_rf_models(
                {"oob_score": float(loaded["rf_oob_score"][0])},
                {"oob_score": float(loaded["rf_no_flow_oob_score"][0])},
                {"oob_score": float(loaded["rf_no_flow_mut_oob_score"][0])},
            ),
        ),
        "lr_summary": meta.get(
            "lr_summary",
            summarize_lr_models(
                {
                    "cv_accuracy": float(loaded["lr_cv_accuracy"][0]),
                    "cv_accuracy_std": float(loaded["lr_cv_accuracy_std"][0]),
                },
                {
                    "cv_accuracy": float(loaded["lr_no_flow_cv_accuracy"][0]),
                    "cv_accuracy_std": float(loaded["lr_no_flow_cv_accuracy_std"][0]),
                },
                {
                    "cv_accuracy": float(loaded["lr_no_flow_mut_cv_accuracy"][0]),
                    "cv_accuracy_std": float(loaded["lr_no_flow_mut_cv_accuracy_std"][0]),
                },
            ),
        ),
        "dt_summary": meta.get(
            "dt_summary",
            summarize_dt_models(
                {
                    "cv_accuracy": float(loaded["dt_cv_accuracy"][0]),
                    "cv_accuracy_std": float(loaded["dt_cv_accuracy_std"][0]),
                },
                {
                    "cv_accuracy": float(loaded["dt_no_flow_cv_accuracy"][0]),
                    "cv_accuracy_std": float(loaded["dt_no_flow_cv_accuracy_std"][0]),
                },
                {
                    "cv_accuracy": float(loaded["dt_no_flow_mut_cv_accuracy"][0]),
                    "cv_accuracy_std": float(
                        loaded["dt_no_flow_mut_cv_accuracy_std"][0]
                    ),
                },
            ),
        ),
        "n_hits": int(meta["n_hits"]),
        "has_umap": bool(meta.get("has_umap", "umap" in loaded)),
    }


def panels_complete(panels_dir: Path, *, has_umap: bool) -> bool:
    needed = [
        letter for letter in EMBEDDING_PANEL_LETTERS if letter != "b" or has_umap
    ]
    return all((panels_dir / f"{letter}.png").is_file() for letter in needed)


def rf_panels_complete(rf_panels_dir: Path) -> bool:
    return all((rf_panels_dir / f"{letter}.png").is_file() for letter in RF_PANEL_LETTERS)


def lr_panels_complete(lr_panels_dir: Path) -> bool:
    return all((lr_panels_dir / f"{letter}.png").is_file() for letter in LR_PANEL_LETTERS)


def dt_panels_complete(dt_panels_dir: Path) -> bool:
    return all((dt_panels_dir / f"{letter}.png").is_file() for letter in DT_PANEL_LETTERS)


def compute_cache_data(
    rows: Sequence[HitRecord],
    *,
    success_threshold: int,
    tsne_max_points: int,
    umap_neighbors: int,
    rng: np.random.Generator,
) -> CacheData:
    X, _ = build_feature_matrix(rows)
    X_raw, _ = build_raw_feature_matrix(rows)
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    high_success = n_hit >= success_threshold
    embeddings = compute_embeddings_for_rows(
        rows,
        X,
        success_threshold=success_threshold,
        tsne_max_points=tsne_max_points,
        umap_neighbors=umap_neighbors,
        rng=rng,
    )
    cluster_pca = kmeans_labels(embeddings["pca"], k=2, rng=rng)
    rf = compute_random_forest(X, high_success, X_plot=X_raw)
    rf["feature_names"] = feature_display_names(raw=True)
    no_flow_features = param_features_without_flow()
    X_no_flow, _ = build_feature_matrix(rows, features=no_flow_features)
    X_raw_no_flow, _ = build_raw_feature_matrix(rows, features=no_flow_features)
    rf_no_flow = compute_random_forest(X_no_flow, high_success, X_plot=X_raw_no_flow)
    rf_no_flow["feature_names"] = feature_display_names(features=no_flow_features, raw=True)
    no_flow_mut_features = param_features_without_flow_and_mutation()
    X_no_flow_mut, _ = build_feature_matrix(rows, features=no_flow_mut_features)
    X_raw_no_flow_mut, _ = build_raw_feature_matrix(rows, features=no_flow_mut_features)
    rf_no_flow_mut = compute_random_forest(
        X_no_flow_mut, high_success, X_plot=X_raw_no_flow_mut
    )
    rf_no_flow_mut["feature_names"] = feature_display_names(
        features=no_flow_mut_features, raw=True
    )
    lr = compute_logistic_regression(X, high_success, X_plot=X_raw)
    lr["feature_names"] = feature_display_names(raw=True)
    lr_no_flow = compute_logistic_regression(X_no_flow, high_success, X_plot=X_raw_no_flow)
    lr_no_flow["feature_names"] = feature_display_names(features=no_flow_features, raw=True)
    lr_no_flow_mut = compute_logistic_regression(
        X_no_flow_mut, high_success, X_plot=X_raw_no_flow_mut
    )
    lr_no_flow_mut["feature_names"] = feature_display_names(
        features=no_flow_mut_features, raw=True
    )
    dt = compute_decision_tree(X, high_success, X_plot=X_raw)
    dt["feature_names"] = feature_display_names(raw=True)
    dt_no_flow = compute_decision_tree(X_no_flow, high_success, X_plot=X_raw_no_flow)
    dt_no_flow["feature_names"] = feature_display_names(features=no_flow_features, raw=True)
    dt_no_flow_mut = compute_decision_tree(
        X_no_flow_mut, high_success, X_plot=X_raw_no_flow_mut
    )
    dt_no_flow_mut["feature_names"] = feature_display_names(
        features=no_flow_mut_features, raw=True
    )
    embedding_summary, rf_summary = build_summary(
        rows,
        cluster_pca,
        high_success=high_success,
        rf=rf,
        rf_no_flow=rf_no_flow,
        rf_no_flow_mut=rf_no_flow_mut,
    )
    lr_summary = summarize_lr_models(lr, lr_no_flow, lr_no_flow_mut)
    dt_summary = summarize_dt_models(dt, dt_no_flow, dt_no_flow_mut)
    return {
        "embeddings": embeddings,
        "n_hit": n_hit,
        "high_success": high_success,
        "cluster_pca": cluster_pca,
        "rf": rf,
        "rf_no_flow": rf_no_flow,
        "rf_no_flow_mut": rf_no_flow_mut,
        "lr": lr,
        "lr_no_flow": lr_no_flow,
        "lr_no_flow_mut": lr_no_flow_mut,
        "dt": dt,
        "dt_no_flow": dt_no_flow,
        "dt_no_flow_mut": dt_no_flow_mut,
        "embedding_summary": embedding_summary,
        "rf_summary": rf_summary,
        "lr_summary": lr_summary,
        "dt_summary": dt_summary,
        "n_hits": len(rows),
        "has_umap": "umap" in embeddings,
    }


def render_panel_pngs(
    cache_dir: Path,
    data: CacheData,
    *,
    success_threshold: int,
) -> Path:
    """Render bare panels (no panel letters, legends, or colorbar) to ``panels/*.png``."""
    panels_dir = cache_panels_dir(cache_dir)
    panels_dir.mkdir(parents=True, exist_ok=True)
    emb = data["embeddings"]
    n_hit = data["n_hit"]
    cluster_pca = data["cluster_pca"]

    for stale in panels_dir.glob("*.png"):
        if stale.stem not in EMBEDDING_PANEL_LETTERS:
            stale.unlink()

    # (a) PCA
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    plot_embedding_panel(
        ax,
        emb["pca"],
        n_hit_again=n_hit,
        title="PCA (all hits)",
        panel_letter=None,
    )
    _save_panel(fig, panels_dir / "a.png", margins=PANEL_MARGINS_SCATTER)

    # (b) UMAP or placeholder
    if data.get("has_umap") and "umap" in emb:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_embedding_panel(
            ax,
            emb["umap"],
            n_hit_again=n_hit,
            title="UMAP (all hits)",
            panel_letter=None,
        )
        _save_panel(fig, panels_dir / "b.png", margins=PANEL_MARGINS_SCATTER)

    # (c) LDA (supervised: 3-class re-hit bins)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    plot_embedding_panel(
        ax,
        emb["lda"],
        n_hit_again=n_hit,
        title="LDA (3-class re-hit bins)",
        panel_letter=None,
    )
    _save_panel(fig, panels_dir / "c.png", margins=PANEL_MARGINS_SCATTER)

    # (d) supervised UMAP
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    plot_embedding_panel(
        ax,
        emb["sup_umap"],
        n_hit_again=n_hit,
        title=f"Supervised UMAP ($\\geq${success_threshold} target)",
        panel_letter=None,
    )
    _save_panel(fig, panels_dir / "d.png", margins=PANEL_MARGINS_SCATTER)

    # (e) k-means
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    plot_cluster_panel(
        ax,
        emb["pca"],
        cluster_ids=cluster_pca,
        title="PCA + $k$-means ($k{=}2$)",
        panel_letter=None,
    )
    _save_panel(fig, panels_dir / "e.png", margins=PANEL_MARGINS_SCATTER)

    # (f) histogram
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    plot_hist_panel(
        ax,
        n_hit=n_hit,
        cluster_pca=cluster_pca,
        title="$k$-means clusters vs re-hit count",
        panel_letter=None,
        show_legend=False,
    )
    _save_panel(fig, panels_dir / "f.png", margins=PANEL_MARGINS_HIST)

    return panels_dir


def render_rf_panel_pngs(
    cache_dir: Path,
    data: CacheData,
    *,
    success_threshold: int,
) -> Path:
    """Render bare RF panels (no panel letters, legends, or colorbar)."""
    rf_panels_dir = cache_rf_panels_dir(cache_dir)
    rf_panels_dir.mkdir(parents=True, exist_ok=True)
    n_hit = data["n_hit"]

    for stale in rf_panels_dir.glob("*.png"):
        if stale.stem not in RF_PANEL_LETTERS:
            stale.unlink()

    def _importance_panel(letter: str, rf: Dict[str, object], title: str) -> None:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_rf_importance_panel(
            ax,
            importances=np.asarray(rf["importances"]),
            feature_names=list(rf["feature_names"]),
            title=title,
            panel_letter=None,
        )
        _save_panel(fig, rf_panels_dir / f"{letter}.png", margins=PANEL_MARGINS_BARH)

    def _scatter_panel(letter: str, rf: Dict[str, object], title: str) -> None:
        top_idx = rf["top2_idx"]
        feat_names = list(rf["feature_names"])
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_rf_top_features_panel(
            ax,
            coords=np.asarray(rf["top2_coords"]),
            n_hit_again=n_hit,
            x_label=feat_names[int(top_idx[0])],
            y_label=feat_names[int(top_idx[1])],
            title=title,
            panel_letter=None,
        )
        _save_panel(fig, rf_panels_dir / f"{letter}.png", margins=PANEL_MARGINS_SCATTER)

    rf = data["rf"]
    rf_no_flow = data["rf_no_flow"]
    rf_no_flow_mut = data["rf_no_flow_mut"]
    _importance_panel(
        "a",
        rf,
        f"Random forest ($\\geq${success_threshold} target)",
    )
    _scatter_panel("b", rf, "Top-2 RF features")
    _importance_panel(
        "c",
        rf_no_flow,
        f"Random forest, no flow ($\\geq${success_threshold} target)",
    )
    _scatter_panel("d", rf_no_flow, "Top-2 RF features (no flow)")
    _importance_panel(
        "e",
        rf_no_flow_mut,
        f"Random forest, no flow/mutation ($\\geq${success_threshold} target)",
    )
    _scatter_panel("f", rf_no_flow_mut, "Top-2 RF features (no flow/mutation)")

    return rf_panels_dir


def render_lr_panel_pngs(
    cache_dir: Path,
    data: CacheData,
    *,
    success_threshold: int,
) -> Path:
    """Render bare logistic-regression panels (no panel letters, legends, or colorbar)."""
    lr_panels_dir = cache_lr_panels_dir(cache_dir)
    lr_panels_dir.mkdir(parents=True, exist_ok=True)
    n_hit = data["n_hit"]

    for stale in lr_panels_dir.glob("*.png"):
        if stale.stem not in LR_PANEL_LETTERS:
            stale.unlink()

    def _coefficient_panel(letter: str, lr: Dict[str, object], title: str) -> None:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_lr_coefficient_panel(
            ax,
            coefficients=np.asarray(lr["coefficients"]),
            feature_names=list(lr["feature_names"]),
            title=title,
            panel_letter=None,
        )
        _save_panel(fig, lr_panels_dir / f"{letter}.png", margins=PANEL_MARGINS_BARH)

    def _scatter_panel(letter: str, lr: Dict[str, object], title: str) -> None:
        top_idx = lr["top2_idx"]
        feat_names = list(lr["feature_names"])
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        plot_rf_top_features_panel(
            ax,
            coords=np.asarray(lr["top2_coords"]),
            n_hit_again=n_hit,
            x_label=feat_names[int(top_idx[0])],
            y_label=feat_names[int(top_idx[1])],
            title=title,
            panel_letter=None,
        )
        _save_panel(fig, lr_panels_dir / f"{letter}.png", margins=PANEL_MARGINS_SCATTER)

    lr = data["lr"]
    lr_no_flow = data["lr_no_flow"]
    lr_no_flow_mut = data["lr_no_flow_mut"]
    _coefficient_panel(
        "a",
        lr,
        f"Logistic regression ($\\geq${success_threshold} target)",
    )
    _scatter_panel("b", lr, "Top-2 LR features")
    _coefficient_panel(
        "c",
        lr_no_flow,
        f"Logistic regression, no flow ($\\geq${success_threshold} target)",
    )
    _scatter_panel("d", lr_no_flow, "Top-2 LR features (no flow)")
    _coefficient_panel(
        "e",
        lr_no_flow_mut,
        f"Logistic regression, no flow/mutation ($\\geq${success_threshold} target)",
    )
    _scatter_panel("f", lr_no_flow_mut, "Top-2 LR features (no flow/mutation)")

    return lr_panels_dir


def render_dt_panel_pngs(
    cache_dir: Path,
    rows: Sequence[HitRecord],
    *,
    success_threshold: int,
) -> Path:
    """Render standard decision-tree panels ($\\geq$ threshold vs.\\ below)."""
    n_hit = np.array([int(r["n_hit_again"]) for r in rows], dtype=int)
    titles = (
        f"Decision tree ($\\geq${success_threshold} target)",
        f"Decision tree, no flow ($\\geq${success_threshold} target)",
        f"Decision tree, no flow/mutation ($\\geq${success_threshold} target)",
    )
    return render_dt_panel_pngs_to_dir(
        cache_dt_panels_dir(cache_dir),
        rows,
        high_success=n_hit >= success_threshold,
        panel_titles=titles,
    )


def render_dt_panel_pngs_to_dir(
    dt_panels_dir: Path,
    rows: Sequence[HitRecord],
    *,
    high_success: np.ndarray,
    panel_titles: Tuple[str, str, str],
    class_names: Tuple[str, str] = ("low", "high"),
) -> Path:
    """Render bare decision-tree panels (no panel letters, legends, or colorbar)."""
    dt_panels_dir.mkdir(parents=True, exist_ok=True)

    for stale in dt_panels_dir.glob("*.png"):
        if stale.stem not in DT_PANEL_LETTERS:
            stale.unlink()

    def _tree_panel(
        letter: str,
        *,
        features: Sequence[str],
        title: str,
    ) -> None:
        X, _ = build_feature_matrix(rows, features=features)
        clf = fit_decision_tree_model(X, high_success)
        figsize, fontsize = tree_panel_layout(clf)
        tree_names = tree_feature_display_names(features=features)
        fig, ax = plt.subplots(figsize=figsize)
        plot_dt_tree_panel(
            ax,
            clf=clf,
            feature_names=tree_names,
            title=title,
            panel_letter=None,
            fontsize=fontsize,
            class_names=class_names,
        )
        _save_panel(fig, dt_panels_dir / f"{letter}.png", margins=PANEL_MARGINS_TREE, dpi=DT_PANEL_DPI)

    no_flow_features = param_features_without_flow()
    no_flow_mut_features = param_features_without_flow_and_mutation()

    _tree_panel("a", features=PARAM_FEATURES, title=panel_titles[0])
    _tree_panel("b", features=no_flow_features, title=panel_titles[1])
    _tree_panel("c", features=no_flow_mut_features, title=panel_titles[2])

    return dt_panels_dir


def plot_true_neutral_dt_extreme_gap_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
    cache_dir: Path,
    recompute_panels: bool = False,
    low_ceiling: int = DT_EXTREME_LOW_CEILING,
    high_floor: int = DT_EXTREME_HIGH_FLOOR,
) -> str:
    """Decision-tree figure: $>$ high_floor vs.\\ $<$ low_ceiling; gap excluded."""
    rows_all = load_true_neutral_hits(sessions_dir)
    if not rows_all:
        raise ValueError(f"No neutral hits found under {sessions_dir}")
    rows, high_success, n_excluded = split_extreme_gap_rows(
        rows_all,
        low_ceiling=low_ceiling,
        high_floor=high_floor,
    )
    if not rows:
        raise ValueError("No hits remain after extreme-gap filtering")

    dt_panels_dir = cache_dt_extreme_panels_dir(cache_dir)
    dt, dt_no_flow, dt_no_flow_mut = compute_dt_models_three_variants(rows, high_success)
    dt_summary = summarize_dt_models(dt, dt_no_flow, dt_no_flow_mut)

    if recompute_panels or not dt_panels_complete(dt_panels_dir):
        titles = (
            f"Decision tree ($>{high_floor}$ vs.\\ $<{low_ceiling}$)",
            f"Decision tree, no flow ($>{high_floor}$ vs.\\ $<{low_ceiling}$)",
            (
                "Decision tree, no flow/mutation "
                f"($>{high_floor}$ vs.\\ $<{low_ceiling}$)"
            ),
        )
        render_dt_panel_pngs_to_dir(
            dt_panels_dir,
            rows,
            high_success=high_success,
            panel_titles=titles,
            class_names=(f"<{low_ceiling}", f">{high_floor}"),
        )

    pool_note = (
        f"Hits with {low_ceiling}--{high_floor} re-run successes excluded "
        f"({n_excluded:,} of {len(rows_all):,}); "
        f"{len(rows):,} extreme hits retained"
    )
    assemble_dt_figure_from_panels(
        dt_panels_dir=dt_panels_dir,
        output_path=output_path,
        dt_cv_accuracy=float(dt["cv_accuracy"]),
        dt_cv_accuracy_std=float(dt["cv_accuracy_std"]),
        dt_no_flow_cv_accuracy=float(dt_no_flow["cv_accuracy"]),
        dt_no_flow_cv_accuracy_std=float(dt_no_flow["cv_accuracy_std"]),
        dt_no_flow_mut_cv_accuracy=float(dt_no_flow_mut["cv_accuracy"]),
        dt_no_flow_mut_cv_accuracy_std=float(dt_no_flow_mut["cv_accuracy_std"]),
        n_hits=len(rows),
        target_txt=(
            f"binary target $>{high_floor}$ vs.\\ $<{low_ceiling}$ "
            f"re-run successes"
        ),
        suptitle=(
            "Decision tree, extreme re-hit classes only "
            f"({len(rows):,} of {len(rows_all):,} neutral hits)"
        ),
        pool_note=pool_note,
    )
    return dt_summary


def assemble_figure_from_panels(
    *,
    panels_dir: Path,
    output_path: Path,
    summary: str,
    n_hits: int,
    has_umap: bool,
    figsize: Tuple[float, float] = COMPOSITE_FIGSIZE,
) -> None:
    """Stitch cached bare embedding panels; add panel letters, colorbar, title, footer."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for ax, letter in zip(axes.ravel(), EMBEDDING_PANEL_LETTERS):
        panel_path = panels_dir / f"{letter}.png"
        if letter == "b" and not has_umap:
            ax.axis("off")
            assembly_panel_label(ax, letter)
            continue
        if not panel_path.is_file():
            ax.axis("off")
            assembly_panel_label(ax, letter)
            continue
        _show_panel(ax, panel_path)
        assembly_panel_label(ax, letter)

    fig.subplots_adjust(left=0.07, right=0.86, top=0.96, bottom=0.08, hspace=0.12, wspace=0.10)

    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.885, 0.38, 0.018, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"Re-run successes (0–{N_RESCREEN})", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    legend_clusters = [
        Line2D([0], [0], marker="s", linestyle="", markersize=6, color="#4c78a8", label="Cluster 0"),
        Line2D([0], [0], marker="s", linestyle="", markersize=6, color="#f58518", label="Cluster 1"),
    ]
    axes[2, 1].legend(handles=legend_clusters, loc="upper right", frameon=False, fontsize=7.5)

    fig.suptitle(
        f"Neutral parameter embeddings ({n_hits:,} hits, all fixed-$Y$ suites)",
        fontsize=11.5,
        fontweight="bold",
        y=0.995,
    )
    fig.text(0.5, 0.02, summary, ha="center", va="bottom", fontsize=8.0, color="#333333")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def assemble_rf_figure_from_panels(
    *,
    rf_panels_dir: Path,
    output_path: Path,
    rf_oob_score: float,
    rf_no_flow_oob_score: float,
    rf_no_flow_mut_oob_score: float,
    n_hits: int,
    success_threshold: int,
    figsize: Tuple[float, float] = RF_FIGSIZE,
) -> None:
    """Stitch cached RF panels into a 3x2 supplementary figure."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for ax, letter in zip(axes.ravel(), RF_PANEL_LETTERS):
        panel_path = rf_panels_dir / f"{letter}.png"
        if not panel_path.is_file():
            ax.axis("off")
            assembly_panel_label(ax, letter)
            continue
        _show_panel(ax, panel_path)
        assembly_panel_label(ax, letter)

    fig.subplots_adjust(left=0.07, right=0.86, top=0.95, bottom=0.06, hspace=0.24, wspace=0.10)

    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.885, 0.24, 0.018, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"Re-run successes (0–{N_RESCREEN})", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f"Random forest re-hit classifier ({n_hits:,} neutral hits)",
        fontsize=11.0,
        fontweight="bold",
        y=0.99,
    )

    def row_note_y(row_axes: np.ndarray) -> float:
        return min(ax.get_position().y0 for ax in row_axes) - 0.008

    target_txt = (
        f"binary target $\\geq${success_threshold}/{N_RESCREEN} re-run successes"
    )
    row_notes = [
        (rf_oob_score, "all six features"),
        (rf_no_flow_oob_score, "flow percentage excluded"),
        (rf_no_flow_mut_oob_score, "flow percentage and mutation rate excluded"),
    ]
    for row_axes, (score, feature_txt) in zip(axes, row_notes):
        fig.text(
            0.46,
            row_note_y(row_axes),
            f"OOB accuracy $=$ {score:.3f} ({feature_txt}; {target_txt})",
            ha="center",
            va="top",
            fontsize=8.0,
            color="#333333",
        )
    fig.text(
        0.5,
        0.02,
        "$200$ trees, balanced class weights",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def assemble_lr_figure_from_panels(
    *,
    lr_panels_dir: Path,
    output_path: Path,
    lr_cv_accuracy: float,
    lr_cv_accuracy_std: float,
    lr_no_flow_cv_accuracy: float,
    lr_no_flow_cv_accuracy_std: float,
    lr_no_flow_mut_cv_accuracy: float,
    lr_no_flow_mut_cv_accuracy_std: float,
    n_hits: int,
    success_threshold: int,
    figsize: Tuple[float, float] = LR_FIGSIZE,
) -> None:
    """Stitch cached logistic-regression panels into a 3x2 supplementary figure."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    for ax, letter in zip(axes.ravel(), LR_PANEL_LETTERS):
        panel_path = lr_panels_dir / f"{letter}.png"
        if not panel_path.is_file():
            ax.axis("off")
            assembly_panel_label(ax, letter)
            continue
        _show_panel(ax, panel_path)
        assembly_panel_label(ax, letter)

    fig.subplots_adjust(left=0.07, right=0.86, top=0.95, bottom=0.06, hspace=0.24, wspace=0.10)

    norm = Normalize(vmin=0, vmax=N_RESCREEN)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.885, 0.24, 0.018, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"Re-run successes (0–{N_RESCREEN})", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        f"Logistic regression re-hit classifier ({n_hits:,} neutral hits)",
        fontsize=11.0,
        fontweight="bold",
        y=0.99,
    )

    def row_note_y(row_axes: np.ndarray) -> float:
        return min(ax.get_position().y0 for ax in row_axes) - 0.008

    target_txt = (
        f"binary target $\\geq${success_threshold}/{N_RESCREEN} re-run successes"
    )
    row_notes = [
        (lr_cv_accuracy, lr_cv_accuracy_std, "all six features"),
        (lr_no_flow_cv_accuracy, lr_no_flow_cv_accuracy_std, "flow percentage excluded"),
        (
            lr_no_flow_mut_cv_accuracy,
            lr_no_flow_mut_cv_accuracy_std,
            "flow percentage and mutation rate excluded",
        ),
    ]
    for row_axes, (score, score_std, feature_txt) in zip(axes, row_notes):
        fig.text(
            0.46,
            row_note_y(row_axes),
            f"5-fold CV accuracy $=$ {score:.3f} $\\pm$ {score_std:.3f} "
            f"({feature_txt}; {target_txt})",
            ha="center",
            va="top",
            fontsize=8.0,
            color="#333333",
        )
    fig.text(
        0.5,
        0.02,
        "Balanced class weights; standardized features",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def assemble_dt_figure_from_panels(
    *,
    dt_panels_dir: Path,
    output_path: Path,
    dt_cv_accuracy: float,
    dt_cv_accuracy_std: float,
    dt_no_flow_cv_accuracy: float,
    dt_no_flow_cv_accuracy_std: float,
    dt_no_flow_mut_cv_accuracy: float,
    dt_no_flow_mut_cv_accuracy_std: float,
    n_hits: int,
    success_threshold: int = 10,
    target_txt: str | None = None,
    suptitle: str | None = None,
    pool_note: str | None = None,
    figsize: Tuple[float, float] | None = None,
) -> None:
    """Stitch cached decision-tree panels into a 3x1 supplementary figure."""
    if figsize is None:
        figsize = _dt_assembly_figsize(dt_panels_dir)
    height_ratios: List[float] = []
    for i, _letter in enumerate(DT_PANEL_LETTERS):
        height_ratios.append(DT_ROW_HEIGHT_IN)
        if i < len(DT_PANEL_LETTERS) - 1:
            height_ratios.append(DT_ROW_GAP_IN)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        len(height_ratios),
        1,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.04,
        left=0.05,
        right=0.97,
        top=0.92,
        bottom=0.09,
    )
    axes: List[plt.Axes] = []
    for i, letter in enumerate(DT_PANEL_LETTERS):
        ax = fig.add_subplot(gs[i * 2, 0])
        panel_path = dt_panels_dir / f"{letter}.png"
        if not panel_path.is_file():
            ax.axis("off")
            assembly_panel_label(ax, letter)
            axes.append(ax)
            continue
        _show_panel_fill_subplot(ax, panel_path)
        assembly_panel_label(ax, letter)
        axes.append(ax)

    if suptitle is None:
        suptitle = f"Decision tree re-hit classifier ({n_hits:,} neutral hits)"
    fig.suptitle(suptitle, fontsize=11.0, fontweight="bold", y=0.985)

    def row_note_y(ax: plt.Axes) -> float:
        return ax.get_position().y0 - 0.022

    if target_txt is None:
        target_txt = (
            f"binary target $\\geq${success_threshold}/{N_RESCREEN} re-run successes"
        )
    row_notes = [
        (dt_cv_accuracy, dt_cv_accuracy_std, "all six features"),
        (dt_no_flow_cv_accuracy, dt_no_flow_cv_accuracy_std, "flow percentage excluded"),
        (
            dt_no_flow_mut_cv_accuracy,
            dt_no_flow_mut_cv_accuracy_std,
            "flow percentage and mutation rate excluded",
        ),
    ]
    for ax, (score, score_std, feature_txt) in zip(axes, row_notes):
        fig.text(
            0.5,
            row_note_y(ax),
            f"5-fold CV accuracy $=$ {score:.3f} $\\pm$ {score_std:.3f} "
            f"({feature_txt}; {target_txt})",
            ha="center",
            va="top",
            fontsize=8.0,
            color="#333333",
        )
    footer_y = 0.055 if pool_note else 0.03
    if pool_note:
        fig.text(
            0.5,
            0.025,
            pool_note,
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#333333",
        )
    fig.text(
        0.5,
        footer_y,
        f"max depth $={DT_MAX_DEPTH}$, min leaf $={DT_MIN_SAMPLES_LEAF}$; "
        "balanced class weights; standardized features",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#333333",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_true_neutral_param_embedding_figure(
    *,
    sessions_dir: Path,
    output_path: Path,
    rf_output_path: Path,
    lr_output_path: Path,
    dt_output_path: Path,
    cache_dir: Path,
    success_threshold: int = 10,
    tsne_max_points: int = 8000,
    umap_neighbors: int = 30,
    figsize: Tuple[float, float] = COMPOSITE_FIGSIZE,
    rf_figsize: Tuple[float, float] = RF_FIGSIZE,
    lr_figsize: Tuple[float, float] = LR_FIGSIZE,
    dt_figsize: Tuple[float, float] = DT_FIGSIZE,
    recompute_embeddings: bool = False,
    recompute_panels: bool = False,
    assemble_only: bool = False,
) -> Tuple[str, str, str, str]:
    panels_dir = cache_panels_dir(cache_dir)
    rf_panels_dir = cache_rf_panels_dir(cache_dir)
    lr_panels_dir = cache_lr_panels_dir(cache_dir)
    dt_panels_dir = cache_dt_panels_dir(cache_dir)

    if assemble_only:
        meta = json.loads(cache_meta_path(cache_dir).read_text(encoding="utf-8"))
        embedding_summary = str(meta.get("embedding_summary", meta.get("summary", "")))
        rf_summary = str(meta.get("rf_summary", ""))
        lr_summary = str(meta.get("lr_summary", ""))
        dt_summary = str(meta.get("dt_summary", ""))
        rf_oob_score = float(meta["rf_oob_score"]) if "rf_oob_score" in meta else None
        rf_no_flow_oob_score = (
            float(meta["rf_no_flow_oob_score"]) if "rf_no_flow_oob_score" in meta else None
        )
        rf_no_flow_mut_oob_score = (
            float(meta["rf_no_flow_mut_oob_score"])
            if "rf_no_flow_mut_oob_score" in meta
            else None
        )
        lr_cv_accuracy = float(meta["lr_cv_accuracy"]) if "lr_cv_accuracy" in meta else None
        lr_cv_accuracy_std = (
            float(meta["lr_cv_accuracy_std"]) if "lr_cv_accuracy_std" in meta else None
        )
        lr_no_flow_cv_accuracy = (
            float(meta["lr_no_flow_cv_accuracy"]) if "lr_no_flow_cv_accuracy" in meta else None
        )
        lr_no_flow_cv_accuracy_std = (
            float(meta["lr_no_flow_cv_accuracy_std"])
            if "lr_no_flow_cv_accuracy_std" in meta
            else None
        )
        lr_no_flow_mut_cv_accuracy = (
            float(meta["lr_no_flow_mut_cv_accuracy"])
            if "lr_no_flow_mut_cv_accuracy" in meta
            else None
        )
        lr_no_flow_mut_cv_accuracy_std = (
            float(meta["lr_no_flow_mut_cv_accuracy_std"])
            if "lr_no_flow_mut_cv_accuracy_std" in meta
            else None
        )
        dt_cv_accuracy = float(meta["dt_cv_accuracy"]) if "dt_cv_accuracy" in meta else None
        dt_cv_accuracy_std = (
            float(meta["dt_cv_accuracy_std"]) if "dt_cv_accuracy_std" in meta else None
        )
        dt_no_flow_cv_accuracy = (
            float(meta["dt_no_flow_cv_accuracy"])
            if "dt_no_flow_cv_accuracy" in meta
            else None
        )
        dt_no_flow_cv_accuracy_std = (
            float(meta["dt_no_flow_cv_accuracy_std"])
            if "dt_no_flow_cv_accuracy_std" in meta
            else None
        )
        dt_no_flow_mut_cv_accuracy = (
            float(meta["dt_no_flow_mut_cv_accuracy"])
            if "dt_no_flow_mut_cv_accuracy" in meta
            else None
        )
        dt_no_flow_mut_cv_accuracy_std = (
            float(meta["dt_no_flow_mut_cv_accuracy_std"])
            if "dt_no_flow_mut_cv_accuracy_std" in meta
            else None
        )
        if (
            rf_oob_score is None
            or rf_no_flow_oob_score is None
            or rf_no_flow_mut_oob_score is None
            or lr_cv_accuracy is None
            or lr_cv_accuracy_std is None
            or lr_no_flow_cv_accuracy is None
            or lr_no_flow_cv_accuracy_std is None
            or lr_no_flow_mut_cv_accuracy is None
            or lr_no_flow_mut_cv_accuracy_std is None
            or dt_cv_accuracy is None
            or dt_cv_accuracy_std is None
            or dt_no_flow_cv_accuracy is None
            or dt_no_flow_cv_accuracy_std is None
            or dt_no_flow_mut_cv_accuracy is None
            or dt_no_flow_mut_cv_accuracy_std is None
        ):
            loaded = np.load(cache_embeddings_path(cache_dir))
            if rf_oob_score is None:
                rf_oob_score = float(loaded["rf_oob_score"][0])
            if rf_no_flow_oob_score is None:
                rf_no_flow_oob_score = float(loaded["rf_no_flow_oob_score"][0])
            if rf_no_flow_mut_oob_score is None:
                rf_no_flow_mut_oob_score = float(loaded["rf_no_flow_mut_oob_score"][0])
            if lr_cv_accuracy is None:
                lr_cv_accuracy = float(loaded["lr_cv_accuracy"][0])
            if lr_cv_accuracy_std is None:
                lr_cv_accuracy_std = float(loaded["lr_cv_accuracy_std"][0])
            if lr_no_flow_cv_accuracy is None:
                lr_no_flow_cv_accuracy = float(loaded["lr_no_flow_cv_accuracy"][0])
            if lr_no_flow_cv_accuracy_std is None:
                lr_no_flow_cv_accuracy_std = float(loaded["lr_no_flow_cv_accuracy_std"][0])
            if lr_no_flow_mut_cv_accuracy is None:
                lr_no_flow_mut_cv_accuracy = float(loaded["lr_no_flow_mut_cv_accuracy"][0])
            if lr_no_flow_mut_cv_accuracy_std is None:
                lr_no_flow_mut_cv_accuracy_std = float(
                    loaded["lr_no_flow_mut_cv_accuracy_std"][0]
                )
            if dt_cv_accuracy is None:
                dt_cv_accuracy = float(loaded["dt_cv_accuracy"][0])
            if dt_cv_accuracy_std is None:
                dt_cv_accuracy_std = float(loaded["dt_cv_accuracy_std"][0])
            if dt_no_flow_cv_accuracy is None:
                dt_no_flow_cv_accuracy = float(loaded["dt_no_flow_cv_accuracy"][0])
            if dt_no_flow_cv_accuracy_std is None:
                dt_no_flow_cv_accuracy_std = float(
                    loaded["dt_no_flow_cv_accuracy_std"][0]
                )
            if dt_no_flow_mut_cv_accuracy is None:
                dt_no_flow_mut_cv_accuracy = float(
                    loaded["dt_no_flow_mut_cv_accuracy"][0]
                )
            if dt_no_flow_mut_cv_accuracy_std is None:
                dt_no_flow_mut_cv_accuracy_std = float(
                    loaded["dt_no_flow_mut_cv_accuracy_std"][0]
                )
        assemble_figure_from_panels(
            panels_dir=panels_dir,
            output_path=output_path,
            summary=embedding_summary,
            n_hits=int(meta["n_hits"]),
            has_umap=bool(meta.get("has_umap", True)),
            figsize=figsize,
        )
        assemble_rf_figure_from_panels(
            rf_panels_dir=rf_panels_dir,
            output_path=rf_output_path,
            rf_oob_score=rf_oob_score,
            rf_no_flow_oob_score=rf_no_flow_oob_score,
            rf_no_flow_mut_oob_score=rf_no_flow_mut_oob_score,
            n_hits=int(meta["n_hits"]),
            success_threshold=success_threshold,
            figsize=rf_figsize,
        )
        assemble_lr_figure_from_panels(
            lr_panels_dir=lr_panels_dir,
            output_path=lr_output_path,
            lr_cv_accuracy=lr_cv_accuracy,
            lr_cv_accuracy_std=lr_cv_accuracy_std,
            lr_no_flow_cv_accuracy=lr_no_flow_cv_accuracy,
            lr_no_flow_cv_accuracy_std=lr_no_flow_cv_accuracy_std,
            lr_no_flow_mut_cv_accuracy=lr_no_flow_mut_cv_accuracy,
            lr_no_flow_mut_cv_accuracy_std=lr_no_flow_mut_cv_accuracy_std,
            n_hits=int(meta["n_hits"]),
            success_threshold=success_threshold,
            figsize=lr_figsize,
        )
        assemble_dt_figure_from_panels(
            dt_panels_dir=dt_panels_dir,
            output_path=dt_output_path,
            dt_cv_accuracy=dt_cv_accuracy,
            dt_cv_accuracy_std=dt_cv_accuracy_std,
            dt_no_flow_cv_accuracy=dt_no_flow_cv_accuracy,
            dt_no_flow_cv_accuracy_std=dt_no_flow_cv_accuracy_std,
            dt_no_flow_mut_cv_accuracy=dt_no_flow_mut_cv_accuracy,
            dt_no_flow_mut_cv_accuracy_std=dt_no_flow_mut_cv_accuracy_std,
            n_hits=int(meta["n_hits"]),
            success_threshold=success_threshold,
            figsize=dt_figsize,
        )
        return embedding_summary, rf_summary, lr_summary, dt_summary

    rows = load_true_neutral_hits(sessions_dir)
    if not rows:
        raise ValueError(f"No neutral hits found under {sessions_dir}")

    data: Optional[CacheData] = None
    if not recompute_embeddings:
        data = load_cache(
            cache_dir,
            success_threshold=success_threshold,
            tsne_max_points=tsne_max_points,
            umap_neighbors=umap_neighbors,
        )
        if data is not None and int(data.get("n_hits", -1)) != len(rows):
            print(
                f"Embedding cache stale ({data.get('n_hits')} hits vs {len(rows):,} now); "
                "recomputing...",
                file=sys.stderr,
            )
            data = None

    if data is None:
        print(f"Computing embeddings for {len(rows):,} hits...", file=sys.stderr)
        rng = np.random.default_rng(0)
        data = compute_cache_data(
            rows,
            success_threshold=success_threshold,
            tsne_max_points=tsne_max_points,
            umap_neighbors=umap_neighbors,
            rng=rng,
        )
        save_cache(
            cache_dir,
            data=data,
            success_threshold=success_threshold,
            tsne_max_points=tsne_max_points,
            umap_neighbors=umap_neighbors,
        )
        recompute_panels = True
        print(f"Wrote embedding cache: {cache_dir}", file=sys.stderr)
    else:
        print(f"Loaded embedding cache: {cache_dir}", file=sys.stderr)

    has_umap = bool(data.get("has_umap", "umap" in data["embeddings"]))
    if recompute_panels or not panels_complete(panels_dir, has_umap=has_umap):
        print("Rendering bare embedding panel PNGs...", file=sys.stderr)
        render_panel_pngs(cache_dir, data, success_threshold=success_threshold)
        print(f"Wrote panel cache: {panels_dir}", file=sys.stderr)
    else:
        print(f"Using cached embedding panels: {panels_dir}", file=sys.stderr)

    if recompute_panels or not rf_panels_complete(rf_panels_dir):
        print("Rendering bare RF panel PNGs...", file=sys.stderr)
        render_rf_panel_pngs(cache_dir, data, success_threshold=success_threshold)
        print(f"Wrote RF panel cache: {rf_panels_dir}", file=sys.stderr)
    else:
        print(f"Using cached RF panels: {rf_panels_dir}", file=sys.stderr)

    if recompute_panels or not lr_panels_complete(lr_panels_dir):
        print("Rendering bare logistic-regression panel PNGs...", file=sys.stderr)
        render_lr_panel_pngs(cache_dir, data, success_threshold=success_threshold)
        print(f"Wrote LR panel cache: {lr_panels_dir}", file=sys.stderr)
    else:
        print(f"Using cached LR panels: {lr_panels_dir}", file=sys.stderr)

    if recompute_panels or not dt_panels_complete(dt_panels_dir):
        print("Rendering bare decision-tree panel PNGs...", file=sys.stderr)
        render_dt_panel_pngs(
            cache_dir,
            rows,
            success_threshold=success_threshold,
        )
        print(f"Wrote DT panel cache: {dt_panels_dir}", file=sys.stderr)
    else:
        print(f"Using cached DT panels: {dt_panels_dir}", file=sys.stderr)

    assemble_figure_from_panels(
        panels_dir=panels_dir,
        output_path=output_path,
        summary=str(data["embedding_summary"]),
        n_hits=int(data["n_hits"]),
        has_umap=has_umap,
        figsize=figsize,
    )
    assemble_rf_figure_from_panels(
        rf_panels_dir=rf_panels_dir,
        output_path=rf_output_path,
        rf_oob_score=float(data["rf"]["oob_score"]),
        rf_no_flow_oob_score=float(data["rf_no_flow"]["oob_score"]),
        rf_no_flow_mut_oob_score=float(data["rf_no_flow_mut"]["oob_score"]),
        n_hits=int(data["n_hits"]),
        success_threshold=success_threshold,
        figsize=rf_figsize,
    )
    assemble_lr_figure_from_panels(
        lr_panels_dir=lr_panels_dir,
        output_path=lr_output_path,
        lr_cv_accuracy=float(data["lr"]["cv_accuracy"]),
        lr_cv_accuracy_std=float(data["lr"]["cv_accuracy_std"]),
        lr_no_flow_cv_accuracy=float(data["lr_no_flow"]["cv_accuracy"]),
        lr_no_flow_cv_accuracy_std=float(data["lr_no_flow"]["cv_accuracy_std"]),
        lr_no_flow_mut_cv_accuracy=float(data["lr_no_flow_mut"]["cv_accuracy"]),
        lr_no_flow_mut_cv_accuracy_std=float(data["lr_no_flow_mut"]["cv_accuracy_std"]),
        n_hits=int(data["n_hits"]),
        success_threshold=success_threshold,
        figsize=lr_figsize,
    )
    assemble_dt_figure_from_panels(
        dt_panels_dir=dt_panels_dir,
        output_path=dt_output_path,
        dt_cv_accuracy=float(data["dt"]["cv_accuracy"]),
        dt_cv_accuracy_std=float(data["dt"]["cv_accuracy_std"]),
        dt_no_flow_cv_accuracy=float(data["dt_no_flow"]["cv_accuracy"]),
        dt_no_flow_cv_accuracy_std=float(data["dt_no_flow"]["cv_accuracy_std"]),
        dt_no_flow_mut_cv_accuracy=float(data["dt_no_flow_mut"]["cv_accuracy"]),
        dt_no_flow_mut_cv_accuracy_std=float(data["dt_no_flow_mut"]["cv_accuracy_std"]),
        n_hits=int(data["n_hits"]),
        success_threshold=success_threshold,
        figsize=dt_figsize,
    )
    return (
        str(data["embedding_summary"]),
        str(data["rf_summary"]),
        str(data["lr_summary"]),
        str(data["dt_summary"]),
    )


def main(argv: Iterable[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    html_figures = default_html_figures_dir(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=workspaces_re_runs() / "sessions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "figures/Extra/supplementary_true_neutral_param_embedding.png",
    )
    parser.add_argument(
        "--rf-output",
        type=Path,
        default=html_figures / "supplementary_true_neutral_random_forest.png",
        help="Output path for the separate random-forest figure",
    )
    parser.add_argument(
        "--lr-output",
        type=Path,
        default=html_figures / "supplementary_true_neutral_logistic_regression.png",
        help="Output path for the separate logistic-regression figure",
    )
    parser.add_argument(
        "--dt-output",
        type=Path,
        default=html_figures / "supplementary_true_neutral_decision_tree.png",
        help="Output path for the separate decision-tree figure",
    )
    parser.add_argument(
        "--dt-only",
        action="store_true",
        help="Only render the decision-tree figure (uses embedding cache)",
    )
    parser.add_argument(
        "--dt-extreme-output",
        type=Path,
        default=html_figures / "supplementary_true_neutral_decision_tree_extreme_gap.png",
        help="Output path for the extreme-gap decision-tree figure",
    )
    parser.add_argument(
        "--dt-extreme-only",
        action="store_true",
        help="Only render the extreme-gap decision-tree figure ($>18$ vs.\\ $<10$)",
    )
    parser.add_argument(
        "--param-3d-output",
        type=Path,
        default=html_figures / "supplementary_true_neutral_param_3d.png",
        help="Output path for the 3D parameter scatter figure",
    )
    parser.add_argument(
        "--param-3d-interactive-output",
        type=Path,
        default=root / "figures/Extra/html/supplementary_true_neutral_param_3d.html",
        help="Output path for the rotatable 3D HTML viewer",
    )
    parser.add_argument(
        "--param-3d-only",
        action="store_true",
        help="Only render the 3D parameter scatter (fast; reads sessions directly)",
    )
    parser.add_argument(
        "--param-3d-interactive-only",
        action="store_true",
        help="Only write the rotatable 3D HTML viewer (fast; reads sessions directly)",
    )
    parser.add_argument(
        "--open-param-3d-interactive",
        action="store_true",
        help="Open the interactive 3D HTML in the default browser after writing it",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(root),
        help="Directory for embeddings.npz and panels/*.png",
    )
    parser.add_argument(
        "--success-threshold",
        type=int,
        default=10,
        help="n_hit_again >= threshold counts as high re-hit success",
    )
    parser.add_argument(
        "--tsne-max-points",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Re-run PCA/UMAP/t-SNE and overwrite embedding cache",
    )
    parser.add_argument(
        "--recompute-panels",
        action="store_true",
        help="Re-render bare panel PNGs from cached embeddings",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Only stitch cached panels into the composite figure (fast)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.param_3d_interactive_only:
        n_hits = plot_param_3d_interactive_figure(
            sessions_dir=args.sessions_dir,
            output_path=args.param_3d_interactive_output,
            cache_dir=args.cache_dir,
            open_browser=args.open_param_3d_interactive,
            success_threshold=args.success_threshold,
        )
        print(f"Sessions: {args.sessions_dir}")
        print(f"Wrote: {args.param_3d_interactive_output} ({n_hits:,} hits)")
        if not args.open_param_3d_interactive:
            print(f"Open in browser: file://{args.param_3d_interactive_output.resolve()}")
        return 0

    if args.param_3d_only:
        n_hits = plot_param_3d_scatter_figure(
            sessions_dir=args.sessions_dir,
            output_path=args.param_3d_output,
        )
        n_hits_interactive = plot_param_3d_interactive_figure(
            sessions_dir=args.sessions_dir,
            output_path=args.param_3d_interactive_output,
            cache_dir=args.cache_dir,
            open_browser=args.open_param_3d_interactive,
            success_threshold=args.success_threshold,
        )
        print(f"Sessions: {args.sessions_dir}")
        print(f"Wrote: {args.param_3d_output} ({n_hits:,} hits)")
        print(f"Wrote: {args.param_3d_interactive_output} ({n_hits_interactive:,} hits)")
        if not args.open_param_3d_interactive:
            print(f"Open in browser: file://{args.param_3d_interactive_output.resolve()}")
        return 0

    if args.dt_only:
        rng = np.random.default_rng(0)
        data = None
        if not args.recompute_embeddings:
            data = load_cache(
                args.cache_dir,
                success_threshold=args.success_threshold,
                tsne_max_points=args.tsne_max_points,
                umap_neighbors=args.umap_neighbors,
            )
        if data is None:
            rows = load_true_neutral_hits(args.sessions_dir)
            if not rows:
                raise ValueError(f"No neutral hits found under {args.sessions_dir}")
            print(f"Computing cache for {len(rows):,} hits...", file=sys.stderr)
            data = compute_cache_data(
                rows,
                success_threshold=args.success_threshold,
                tsne_max_points=args.tsne_max_points,
                umap_neighbors=args.umap_neighbors,
                rng=rng,
            )
            save_cache(
                args.cache_dir,
                data=data,
                success_threshold=args.success_threshold,
                tsne_max_points=args.tsne_max_points,
                umap_neighbors=args.umap_neighbors,
            )
        rows = load_true_neutral_hits(args.sessions_dir)
        dt_panels_dir = cache_dt_panels_dir(args.cache_dir)
        if args.recompute_panels or not dt_panels_complete(dt_panels_dir):
            render_dt_panel_pngs(
                args.cache_dir,
                rows,
                success_threshold=args.success_threshold,
            )
        assemble_dt_figure_from_panels(
            dt_panels_dir=dt_panels_dir,
            output_path=args.dt_output,
            dt_cv_accuracy=float(data["dt"]["cv_accuracy"]),
            dt_cv_accuracy_std=float(data["dt"]["cv_accuracy_std"]),
            dt_no_flow_cv_accuracy=float(data["dt_no_flow"]["cv_accuracy"]),
            dt_no_flow_cv_accuracy_std=float(data["dt_no_flow"]["cv_accuracy_std"]),
            dt_no_flow_mut_cv_accuracy=float(data["dt_no_flow_mut"]["cv_accuracy"]),
            dt_no_flow_mut_cv_accuracy_std=float(
                data["dt_no_flow_mut"]["cv_accuracy_std"]
            ),
            n_hits=int(data["n_hits"]),
            success_threshold=args.success_threshold,
        )
        print(f"Sessions: {args.sessions_dir}")
        print(f"Wrote: {args.dt_output}")
        print(str(data["dt_summary"]))
        return 0

    if args.dt_extreme_only:
        dt_extreme_summary = plot_true_neutral_dt_extreme_gap_figure(
            sessions_dir=args.sessions_dir,
            output_path=args.dt_extreme_output,
            cache_dir=args.cache_dir,
            recompute_panels=args.recompute_panels,
        )
        print(f"Sessions: {args.sessions_dir}")
        print(f"Wrote: {args.dt_extreme_output}")
        print(dt_extreme_summary)
        return 0

    embedding_summary, rf_summary, lr_summary, dt_summary = (
        plot_true_neutral_param_embedding_figure(
        sessions_dir=args.sessions_dir,
        output_path=args.output,
        rf_output_path=args.rf_output,
        lr_output_path=args.lr_output,
        dt_output_path=args.dt_output,
        cache_dir=args.cache_dir,
        success_threshold=args.success_threshold,
        tsne_max_points=args.tsne_max_points,
        umap_neighbors=args.umap_neighbors,
        recompute_embeddings=args.recompute_embeddings,
        recompute_panels=args.recompute_panels,
        assemble_only=args.assemble_only,
        )
    )
    n_hits_3d = plot_param_3d_scatter_figure(
        sessions_dir=args.sessions_dir,
        output_path=args.param_3d_output,
    )
    n_hits_3d_interactive = plot_param_3d_interactive_figure(
        sessions_dir=args.sessions_dir,
        output_path=args.param_3d_interactive_output,
        cache_dir=args.cache_dir,
        open_browser=args.open_param_3d_interactive,
        success_threshold=args.success_threshold,
    )
    dt_extreme_summary = plot_true_neutral_dt_extreme_gap_figure(
        sessions_dir=args.sessions_dir,
        output_path=args.dt_extreme_output,
        cache_dir=args.cache_dir,
        recompute_panels=args.recompute_panels,
    )
    print(f"Sessions: {args.sessions_dir}")
    print(f"Cache: {args.cache_dir}")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.rf_output}")
    print(f"Wrote: {args.lr_output}")
    print(f"Wrote: {args.dt_output}")
    print(f"Wrote: {args.dt_extreme_output}")
    print(f"Wrote: {args.param_3d_output} ({n_hits_3d:,} hits)")
    print(f"Wrote: {args.param_3d_interactive_output} ({n_hits_3d_interactive:,} hits)")
    if not args.open_param_3d_interactive:
        print(f"Open 3D viewer: file://{args.param_3d_interactive_output.resolve()}")
    print(embedding_summary)
    print(rf_summary)
    print(lr_summary)
    print(dt_summary)
    print(dt_extreme_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
