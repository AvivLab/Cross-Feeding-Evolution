"""Parameter applicability for assemble_figure_data (from paper hit-parameter specs)."""

from __future__ import annotations

from typing import FrozenSet, Tuple

from plot_primary_batch_violins import PAPER_CONFIG_ORDER

ALL_CONFIG_KEYS = frozenset(key for key, _, _ in PAPER_CONFIG_ORDER)

PARAM_SPECS: Tuple[Tuple[str, str, Tuple[float, float], FrozenSet[str]], ...] = (
    ("Constant Probability", r"Constant probability $p$", (0.0, 0.5), frozenset({"justDeath"})),
    ("Death Decay Rate", r"Death decay $\lambda$", (0.0, 20.0), frozenset({"justDeath", "Death+Dup"})),
    (
        "Duplication Sigmoid Intensity",
        r"Duplication steepness $\varepsilon$",
        (0.0, 10.0),
        frozenset({"justDup", "Death+Dup"}),
    ),
    (
        "Duplication Sigmoid Midpoint",
        r"Duplication midpoint $\omega$",
        (0.0, 5.0),
        frozenset({"justDup", "Death+Dup"}),
    ),
    ("Cost of Life", r"Cost of life $c_{m}$", (0.0, 0.05), ALL_CONFIG_KEYS),
    ("Flow Percentage", r"Flow fraction $\phi$ (%)", (0.0, 50.0), ALL_CONFIG_KEYS),
    ("Mutation Rate", r"Mutation rate $\mu_{m}$", (0.0, 0.005), ALL_CONFIG_KEYS),
    ("Initial A", r"Initial Task A $A_0$", (0.0, 1.0), ALL_CONFIG_KEYS),
    ("Initial Energy", r"Initial energy $E_{\mathrm{init}}$", (0.0, 20.0), ALL_CONFIG_KEYS),
    ("Mutation Scale", r"Mutation scale $\sigma_{m}$", (0.0, 1.0), ALL_CONFIG_KEYS),
)
