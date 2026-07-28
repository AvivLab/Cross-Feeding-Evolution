"""Neutral trait drift replay and Monte Carlo percentile metrics."""

from __future__ import annotations

import numpy as np

from gui.metrics.definitions import (
    EXCHANGE_NEUTRAL_PERCENTILE_SIMS,
    NEUTRAL_PERCENTILE_MAX_SIMS,
    NEUTRAL_PERCENTILE_MIN_SIMS,
    NEUTRAL_PERCENTILE_TARGET_CI_PCT,
    T1_T2_RATIO_THRESHOLD_T2_HEAVY_METRICS,
    TRAIT_ENTROPY_BINS,
)
from simulation.change_history import parse_change_history_row
from simulation.helpers import investment_func, truncated_uniform_trait_mutation

# Last-call cache so sequential percentile requests for the same run reuse one sweep.
_LAST_NEUTRAL_PERCENTILE_KEY = None
_LAST_NEUTRAL_PERCENTILE_VALUE = None
_LAST_NEUTRAL_T2HEAVY_PERCENTILE_KEY = None
_LAST_NEUTRAL_T2HEAVY_PERCENTILE_VALUE = None


def parse_neutral_events(change_history):
    """Extract (deaths, dups, flow_removed, accepted_mutations) from change history."""
    events = []
    for row in change_history:
        try:
            deaths, dups, accepted_mutations, flow_removed, _ = parse_change_history_row(row)
        except Exception:
            continue
        events.append((deaths, dups, flow_removed, accepted_mutations))
    return events


def events_at_metabolism_time_of_last_generation(events):
    """Drop the last generation's demographic events.

    Real ``task1``/``task2`` are recorded at metabolism time (after inflow,
    before death/dup/flow/mutation) using post-inflow pools. Neutral synthetic
    tasks must use the same timing: traits after events ``0..G-2``, not after
    the final generation's demographic events ``G-1``.
    """
    if isinstance(events, tuple) and len(events) == 4:
        try:
            arrays = [np.asarray(a) for a in events]
            if all(getattr(a, "ndim", None) == 1 for a in arrays):
                sizes = {int(a.size) for a in arrays}
                if len(sizes) == 1:
                    if next(iter(sizes)) == 0:
                        return events
                    return tuple(a[:-1] for a in arrays)
        except Exception:
            pass
    if not events:
        return events
    return events[:-1]


def _events_to_compact_arrays(events):
    empty = np.empty(0, dtype=np.int64)
    if isinstance(events, tuple) and len(events) == 4:
        try:
            return tuple(np.asarray(events[i], dtype=np.int64) for i in range(4))
        except Exception:
            pass
    if not events:
        return empty, empty, empty, empty
    arr = np.asarray(events, dtype=np.int64)
    if arr.ndim == 2 and arr.shape[1] >= 4:
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    n = len(events)
    d0 = np.empty(n, dtype=np.int64)
    d1 = np.empty(n, dtype=np.int64)
    d2 = np.empty(n, dtype=np.int64)
    d3 = np.empty(n, dtype=np.int64)
    for i, row in enumerate(events):
        try:
            deaths, dups, accepted_mutations, flow_removed, _ = parse_change_history_row(row)
            d0[i], d1[i], d2[i], d3[i] = deaths, dups, flow_removed, accepted_mutations
        except Exception:
            try:
                d0[i] = int(row[0])
                d1[i] = int(row[1])
                d2[i] = int(row[2]) if len(row) > 2 else 0
                d3[i] = int(row[3]) if len(row) > 3 else 0
            except Exception:
                d0[i] = d1[i] = d2[i] = d3[i] = 0
    return d0, d1, d2, d3


def _remove_random_in_place(traits, n, k, rng):
    if n <= 0:
        return 0
    k = int(k)
    if k <= 0:
        return n
    if k >= n:
        return 0
    remove_idx = rng.choice(n, size=k, replace=False)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[remove_idx] = False
    survivors = traits[:n][keep_mask]
    n2 = survivors.size
    traits[:n2] = survivors
    return n2


def _apply_scheduled_dup_mutations(offspring, n_accepted_mutations, mutation_scale, rng):
    dups = int(offspring.size)
    n_mut = min(max(0, int(n_accepted_mutations)), dups)
    if n_mut <= 0 or float(mutation_scale) <= 0.0:
        return offspring
    mut_idx = rng.choice(dups, size=n_mut, replace=False)
    mut_mask = np.zeros(dups, dtype=bool)
    mut_mask[mut_idx] = True
    return truncated_uniform_trait_mutation(offspring, mut_mask, mutation_scale, rng)


def replay_neutral_coupled_traits(initial_traits, events, mutation_scale, seed):
    base_traits = np.asarray(initial_traits, dtype=float)
    start_n = int(base_traits.size)
    if start_n == 0:
        return np.asarray([], dtype=float)

    deaths_arr, dups_arr, flow_arr, muts_arr = _events_to_compact_arrays(events)
    n_proj = start_n
    capacity = start_n
    n_steps = int(deaths_arr.size)
    for i in range(n_steps):
        deaths = int(deaths_arr[i])
        dups = int(dups_arr[i])
        flow_removed = int(flow_arr[i])
        if deaths > 0 and n_proj > 0:
            n_proj = max(0, n_proj - deaths)
        if dups > 0:
            if n_proj == 0:
                break
            n_proj += dups
            if n_proj > capacity:
                capacity = n_proj
        if flow_removed > 0 and n_proj > 0:
            n_proj = max(0, n_proj - flow_removed)

    traits = np.empty(capacity, dtype=float)
    traits[:start_n] = base_traits
    n = start_n
    rng = np.random.default_rng(seed)
    mscale = float(mutation_scale)

    for i in range(n_steps):
        deaths = int(deaths_arr[i])
        dups = int(dups_arr[i])
        flow_removed = int(flow_arr[i])
        n_accepted = int(muts_arr[i]) if i < muts_arr.size else 0
        if deaths > 0 and n > 0:
            n = _remove_random_in_place(traits, n, deaths, rng)
        if dups > 0:
            if n == 0:
                break
            parent_idx = rng.integers(0, n, size=dups)
            offspring = traits[parent_idx].copy()
            offspring = _apply_scheduled_dup_mutations(offspring, n_accepted, mscale, rng)
            traits[n : n + dups] = offspring
            n += dups
        if flow_removed > 0 and n > 0:
            n = _remove_random_in_place(traits, n, flow_removed, rng)

    if n <= 0:
        return np.asarray([], dtype=float)
    return np.asarray(traits[:n].copy(), dtype=float)


def shannon_entropy_from_traits(traits, bins=TRAIT_ENTROPY_BINS):
    traits = np.asarray(traits, dtype=float)
    if traits.size == 0:
        return np.nan
    counts, _ = np.histogram(traits, bins=int(bins), range=(0.0, 1.0))
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.nan
    probs = counts / total
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _simulate_neutral_trait_stats(
    initial_traits, events, mutation_rate, mutation_scale, seed, bins=TRAIT_ENTROPY_BINS, which="both"
):
    _ = mutation_rate
    final_traits = replay_neutral_coupled_traits(initial_traits, events, mutation_scale, seed)
    if final_traits.size == 0:
        return np.nan, np.nan
    w = str(which or "both").lower()
    if w == "std":
        return float(np.std(final_traits)), np.nan
    if w == "entropy":
        return np.nan, shannon_entropy_from_traits(final_traits, bins=bins)
    return float(np.std(final_traits)), shannon_entropy_from_traits(final_traits, bins=bins)


def _simulate_neutral_final_A_traits_array(initial_traits, events, mutation_rate, mutation_scale, seed):
    _ = mutation_rate
    return replay_neutral_coupled_traits(initial_traits, events, mutation_scale, seed)


def synthetic_task1_task2_from_pooled_metabolites_coupled(A_final, metab1env, metab2env, inv_mod):
    Anumbers = np.asarray(A_final, dtype=float)
    if Anumbers.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    m1e = max(0.0, float(metab1env))
    m2e = max(0.0, float(metab2env))
    inv = float(inv_mod)
    Bnumbers = 1.0 - Anumbers
    Afunc = investment_func(inv, Anumbers)
    Bfunc = investment_func(inv, Bnumbers)

    change1 = min(float(np.sum(Afunc)), m1e)
    sum_a = float(np.sum(Afunc))
    if sum_a <= 1e-18:
        A_share = np.zeros_like(Afunc)
    else:
        A_share = (Afunc / sum_a) * change1

    m2_pool_for_b = m2e + float(np.sum(A_share))
    total_demand = float(np.sum(Bfunc))
    if total_demand > 1e-18 and m2_pool_for_b > 0.0:
        B_share = Bfunc * min(1.0, m2_pool_for_b / total_demand)
    else:
        B_share = np.zeros_like(Bfunc)
    return A_share, B_share


def neutral_percentile_pair(
    enzyme_A_final,
    initial_traits,
    events,
    mutation_rate,
    mutation_scale,
    events_cache_key=None,
    which="both",
):
    global _LAST_NEUTRAL_PERCENTILE_KEY, _LAST_NEUTRAL_PERCENTILE_VALUE

    w = str(which or "both").lower()
    if w not in ("both", "std", "entropy"):
        w = "both"

    real_traits = np.asarray(enzyme_A_final, dtype=float)
    real_std = float(np.std(real_traits)) if w in ("std", "both") else None
    real_entropy = (
        shannon_entropy_from_traits(real_traits, bins=TRAIT_ENTROPY_BINS)
        if w in ("entropy", "both")
        else None
    )
    if w in ("std", "both") and np.isnan(real_std):
        return np.nan, np.nan
    if w in ("entropy", "both") and np.isnan(real_entropy):
        return np.nan, np.nan

    initial_traits_arr = np.asarray(initial_traits, dtype=float)
    if initial_traits_arr.size == 0:
        return np.nan, np.nan

    mrate = float(mutation_rate)
    mscale = float(mutation_scale)
    key = (
        w,
        id(enzyme_A_final),
        id(initial_traits),
        events_cache_key if events_cache_key is not None else id(events),
        mrate,
        mscale,
        int(TRAIT_ENTROPY_BINS),
        int(NEUTRAL_PERCENTILE_MAX_SIMS),
        int(NEUTRAL_PERCENTILE_MIN_SIMS),
        float(NEUTRAL_PERCENTILE_TARGET_CI_PCT),
    )
    if _LAST_NEUTRAL_PERCENTILE_KEY == key and isinstance(_LAST_NEUTRAL_PERCENTILE_VALUE, tuple):
        return _LAST_NEUTRAL_PERCENTILE_VALUE

    n_valid_std = n_lt_real_std = 0
    n_valid_entropy = n_lt_real_entropy = 0
    compact_events = _events_to_compact_arrays(events)

    def _accumulate_one(neutral_std, neutral_entropy):
        nonlocal n_valid_std, n_lt_real_std, n_valid_entropy, n_lt_real_entropy
        if w in ("std", "both") and not np.isnan(neutral_std):
            n_valid_std += 1
            if neutral_std < real_std:
                n_lt_real_std += 1
        if w in ("entropy", "both") and not np.isnan(neutral_entropy):
            n_valid_entropy += 1
            if neutral_entropy < real_entropy:
                n_lt_real_entropy += 1

    def _adaptive_stop_ready():
        if not (NEUTRAL_PERCENTILE_TARGET_CI_PCT > 0.0):
            return False
        if w == "both":
            if n_valid_std < NEUTRAL_PERCENTILE_MIN_SIMS or n_valid_entropy < NEUTRAL_PERCENTILE_MIN_SIMS:
                return False
            p_std = n_lt_real_std / n_valid_std
            p_entropy = n_lt_real_entropy / n_valid_entropy
            ci_std = 100.0 * 1.96 * np.sqrt(p_std * (1.0 - p_std) / n_valid_std)
            ci_entropy = 100.0 * 1.96 * np.sqrt(p_entropy * (1.0 - p_entropy) / n_valid_entropy)
            return ci_std <= NEUTRAL_PERCENTILE_TARGET_CI_PCT and ci_entropy <= NEUTRAL_PERCENTILE_TARGET_CI_PCT
        if w == "std":
            if n_valid_std < NEUTRAL_PERCENTILE_MIN_SIMS:
                return False
            p_std = n_lt_real_std / n_valid_std
            return 100.0 * 1.96 * np.sqrt(p_std * (1.0 - p_std) / n_valid_std) <= NEUTRAL_PERCENTILE_TARGET_CI_PCT
        if w == "entropy":
            if n_valid_entropy < NEUTRAL_PERCENTILE_MIN_SIMS:
                return False
            p_entropy = n_lt_real_entropy / n_valid_entropy
            return (
                100.0 * 1.96 * np.sqrt(p_entropy * (1.0 - p_entropy) / n_valid_entropy)
                <= NEUTRAL_PERCENTILE_TARGET_CI_PCT
            )
        return False

    for sim_seed in range(NEUTRAL_PERCENTILE_MAX_SIMS):
        neutral_std, neutral_entropy = _simulate_neutral_trait_stats(
            initial_traits=initial_traits_arr,
            events=compact_events,
            mutation_rate=mrate,
            mutation_scale=mscale,
            seed=sim_seed,
            which=w,
        )
        _accumulate_one(neutral_std, neutral_entropy)
        if _adaptive_stop_ready():
            break

    std_percentile = float(100.0 * n_lt_real_std / n_valid_std) if n_valid_std > 0 else np.nan
    entropy_percentile = float(100.0 * n_lt_real_entropy / n_valid_entropy) if n_valid_entropy > 0 else np.nan
    if w == "std":
        entropy_percentile = np.nan
    elif w == "entropy":
        std_percentile = np.nan
    result = (std_percentile, entropy_percentile)
    _LAST_NEUTRAL_PERCENTILE_KEY = key
    _LAST_NEUTRAL_PERCENTILE_VALUE = result
    return result


def neutral_percentile_t2heavy_task2_share(
    _enzyme_A_final_cache_id,
    task1_real,
    task2_real,
    initial_traits,
    events,
    mutation_rate,
    mutation_scale,
    metab1env,
    metab2env,
    inv_mod,
    events_cache_key=None,
    *,
    t2_percent_fn,
):
    global _LAST_NEUTRAL_T2HEAVY_PERCENTILE_KEY, _LAST_NEUTRAL_T2HEAVY_PERCENTILE_VALUE

    t1r = np.asarray(task1_real, dtype=float)
    t2r = np.asarray(task2_real, dtype=float)
    if t1r.size == 0 or t2r.size == 0:
        return np.nan

    real_val = float(t2_percent_fn(t1r, t2r))
    if not np.isfinite(real_val):
        return np.nan

    initial_traits_arr = np.asarray(initial_traits, dtype=float)
    if initial_traits_arr.size == 0:
        return np.nan

    mrate = float(mutation_rate)
    mscale = float(mutation_scale)
    # Align with real task1/task2: metabolism-time traits + post-inflow pools.
    events_for_tasks = events_at_metabolism_time_of_last_generation(events)
    key = (
        id(_enzyme_A_final_cache_id),
        id(task1_real),
        id(task2_real),
        id(initial_traits),
        events_cache_key if events_cache_key is not None else id(events),
        "metab_time_traits_v1",
        mrate,
        mscale,
        float(metab1env),
        float(metab2env),
        float(inv_mod),
        id(t2_percent_fn),
        int(EXCHANGE_NEUTRAL_PERCENTILE_SIMS),
    )
    if _LAST_NEUTRAL_T2HEAVY_PERCENTILE_KEY == key and _LAST_NEUTRAL_T2HEAVY_PERCENTILE_VALUE is not None:
        return float(_LAST_NEUTRAL_T2HEAVY_PERCENTILE_VALUE)

    n_valid = n_lt_real = 0
    for sim_seed in range(EXCHANGE_NEUTRAL_PERCENTILE_SIMS):
        A_at_metab = _simulate_neutral_final_A_traits_array(
            initial_traits_arr, events_for_tasks, mrate, mscale, sim_seed
        )
        t1s, t2s = synthetic_task1_task2_from_pooled_metabolites_coupled(
            A_at_metab, metab1env, metab2env, inv_mod
        )
        nv = float(t2_percent_fn(t1s, t2s))
        if not np.isfinite(nv):
            continue
        n_valid += 1
        if nv < real_val:
            n_lt_real += 1

    out = float(100.0 * n_lt_real / n_valid) if n_valid > 0 else np.nan
    _LAST_NEUTRAL_T2HEAVY_PERCENTILE_KEY = key
    _LAST_NEUTRAL_T2HEAVY_PERCENTILE_VALUE = out
    return out
