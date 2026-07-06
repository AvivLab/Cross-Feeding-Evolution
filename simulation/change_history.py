"""Shared helpers for simulation change_history rows.

Storage format is positional:
    [deaths, dups, accepted_mutations, flow_removed, accepted_aux_mutations]
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple


IDX_DEATHS = 0
IDX_DUPS = 1
IDX_ACCEPTED_MUTATIONS = 2
IDX_FLOW_REMOVED = 3
IDX_ACCEPTED_AUX_MUTATIONS = 4


def build_change_history_row(
    deaths: int,
    dups: int,
    accepted_mutations: int,
    flow_removed: int,
    accepted_aux_mutations: int,
) -> List[int]:
    """Build one canonical change_history row."""
    return [
        int(deaths),
        int(dups),
        int(accepted_mutations),
        int(flow_removed),
        int(accepted_aux_mutations),
    ]


def parse_change_history_row(row: Sequence[object]) -> Tuple[int, int, int, int, int]:
    """Parse one row defensively, tolerant to shorter row formats."""
    deaths = int(row[IDX_DEATHS]) if len(row) > IDX_DEATHS else 0
    dups = int(row[IDX_DUPS]) if len(row) > IDX_DUPS else 0
    accepted_mutations = int(row[IDX_ACCEPTED_MUTATIONS]) if len(row) > IDX_ACCEPTED_MUTATIONS else 0
    flow_removed = int(row[IDX_FLOW_REMOVED]) if len(row) > IDX_FLOW_REMOVED else 0
    accepted_aux_mutations = int(row[IDX_ACCEPTED_AUX_MUTATIONS]) if len(row) > IDX_ACCEPTED_AUX_MUTATIONS else 0
    return deaths, dups, accepted_mutations, flow_removed, accepted_aux_mutations


def parse_change_history_rows(rows: Iterable[Sequence[object]]) -> List[Tuple[int, int, int, int, int]]:
    """Parse many rows into canonical tuples."""
    return [parse_change_history_row(r) for r in rows]


def event_totals_from_change_history(
    change_history: Any,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Sum deaths, duplications, mutations, and outflows over all generations.

    Returns None when ``change_history`` is missing or empty.
    """
    if not isinstance(change_history, list) or not change_history:
        return None
    deaths = dups = mutations = outflows = 0
    n_parsed = 0
    for row in change_history:
        try:
            d, dup, acc, flow, acc_aux = parse_change_history_row(row)
        except Exception:
            continue
        n_parsed += 1
        deaths += int(d)
        dups += int(dup)
        mutations += int(acc) + int(acc_aux)
        outflows += int(flow)
    if n_parsed <= 0:
        return None
    return deaths, dups, mutations, outflows


def event_totals_from_metric_input(
    metric_input: Any,
) -> Optional[Tuple[int, int, int, int]]:
    """Event totals for one simulation from offload ``metric_input``."""
    if not isinstance(metric_input, dict) or bool(metric_input.get("collapsed")):
        return None
    return event_totals_from_change_history(metric_input.get("change_history"))

