"""Death/duplication probability curves (mirrors ``simulation/core.py`` + GUI display rules)."""

from __future__ import annotations

import numpy as np


def compute_death_probabilities(
    energy: np.ndarray,
    *,
    binary_death_at_zero_energy: bool,
    constant_death_probability: bool,
    death_probability: float,
    death_decay_rate: float,
) -> np.ndarray:
    """Per-organism death probability before the survival draw."""
    E = np.asarray(energy, dtype=float)
    if constant_death_probability:
        out = np.full_like(E, float(death_probability))
        if binary_death_at_zero_energy:
            out[E <= 0] = 1.0
    elif binary_death_at_zero_energy:
        out = np.zeros_like(E)
        out[E <= 0] = 1.0
    else:
        out = np.ones_like(E)
        positive = E > 0
        out[positive] = np.exp(-death_decay_rate * E[positive])
    return out


def compute_duplication_probabilities(
    energy: np.ndarray,
    *,
    constant_duplication_probability: bool,
    duplication_probability: float,
    dupk: float,
    dupx0: float,
) -> np.ndarray:
    """Per-organism duplication probability (core simulation, before binary display mask)."""
    E = np.asarray(energy, dtype=float)
    if constant_duplication_probability:
        return np.full_like(E, float(duplication_probability))
    out = np.zeros_like(E)
    above = E > 0
    out[above] = 1.0 / (1.0 + np.exp(-dupk * (E[above] - dupx0)))
    return out


def effective_duplication_probabilities_for_display(
    energy: np.ndarray,
    duplication_probs: np.ndarray,
    *,
    binary_death_at_zero_energy: bool,
) -> np.ndarray:
    """
    Effective duplication curve shown in GUIs.

    Binary death removes all E <= 0 organisms before duplication, so the displayed
    dup probability is zero at non-positive energy even when constant-dup is flat.
    """
    out = np.array(duplication_probs, dtype=float, copy=True)
    if binary_death_at_zero_energy:
        E = np.asarray(energy, dtype=float)
        out[E <= 0] = 0.0
    return out
