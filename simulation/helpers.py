import numpy as np


def investment_func(modifier, x):
    """Map coupled-trait investment level to effective task return (exponential profile)."""
    return np.power(x, modifier)


def truncated_uniform_trait_mutation(
    parent_traits: np.ndarray,
    mut_mask: np.ndarray,
    mutation_scale: float,
    rng=None,
) -> np.ndarray:
    """
    Copy ``parent_traits``; where ``mut_mask`` is true, replace with a draw from
    ``Uniform([max(0, p-s), min(1, p+s)])`` on ``[0, 1]``.

    ``rng`` may be a ``numpy.random.Generator``; if ``None``, uses ``numpy.random.random``.
    """
    out = np.asarray(parent_traits, dtype=np.float64).copy()
    idx = np.flatnonzero(mut_mask)
    if idx.size == 0:
        return out
    p = np.asarray(parent_traits[idx], dtype=np.float64)
    s = float(mutation_scale)
    lo = np.maximum(0.0, p - s)
    hi = np.minimum(1.0, p + s)
    span = hi - lo
    if rng is None:
        u = np.random.random(idx.size)
    else:
        u = rng.random(idx.size)
    child = lo + u * span
    degenerate = span <= 1e-15
    child[degenerate] = p[degenerate]
    out[idx] = child
    return out
