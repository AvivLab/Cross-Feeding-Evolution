"""Runtime utilities shared by optimization GUI modules."""

from __future__ import annotations

import numpy as np
import warnings


class RunningStats:
    """Numerically-stable online mean/std accumulator (Welford)."""

    __slots__ = ("n", "mean", "M2")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, x: float) -> None:
        """Add one observation (ignores None/NaN)."""
        if x is None:
            return
        x = float(x)
        if np.isnan(x):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self) -> float:
        """Return running mean, or NaN when empty."""
        return self.mean if self.n > 0 else np.nan

    def get_std(self, ddof: int = 1) -> float:
        """Return sample/std estimate with configurable delta degrees of freedom."""
        if self.n <= ddof:
            return np.nan
        var = self.M2 / (self.n - ddof)
        return float(np.sqrt(var))


def configure_optimization_warnings(environ: dict) -> None:
    """Configure warning filters used by optimization workflows."""
    # Suppress sklearn warnings about disconnected graphs (common with sparse/disconnected data)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Graph is not fully connected.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.manifold._spectral_embedding")

    # Suppress UMAP warnings about small datasets (common when optimization just starts)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*n_neighbors is larger than the dataset size.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="umap")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*k >= N for N.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="umap.spectral")

    # Suppress multiprocessing resource_tracker shutdown warnings.
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
    warnings.filterwarnings("ignore", message=".*resource_tracker.*")
    warnings.filterwarnings("ignore", message=".*leaked semaphore.*")

    # Also propagate suppression to multiprocessing child processes (resource_tracker is separate process).
    pywarn = environ.get("PYTHONWARNINGS", "")
    extra = "ignore::UserWarning:multiprocessing.resource_tracker"
    if extra not in pywarn:
        environ["PYTHONWARNINGS"] = ",".join([p for p in [pywarn, extra] if p])


def build_run_summary_stats(all_results):
    """Build canonical run summary stats from loaded/saved results."""
    stats = {
        "total_runs": 0,
        "nan_runs": 0,
        "metric_stats": RunningStats(),
    }
    for result in all_results:
        mv = result.get("best_metric", result.get("final_metric", np.nan)) if isinstance(result, dict) else np.nan
        stats["total_runs"] += 1
        if np.isnan(mv):
            stats["nan_runs"] += 1
        else:
            stats["metric_stats"].add(mv)
    return stats


def reset_deleted_point_stats(deleted_point_stats: dict) -> None:
    """Reset deleted-point summary stats to canonical empty state."""
    deleted_point_stats["total_points"] = 0
    deleted_point_stats["nan_points"] = 0
    deleted_point_stats["metric_stats"] = RunningStats()
