"""Convex terminal payoffs as :class:`~superhedge.pwl.ConvexPWL` objects.

The method applies to *convex* payoffs. Any convex piecewise-linear payoff can be built
directly with ``ConvexPWL(breakpoints, values, left_slope, right_slope)``; the helpers
below cover the common cases.
"""
import numpy as np
from .pwl import ConvexPWL

__all__ = ["call", "put", "straddle", "strangle", "call_ladder", "max_of_calls",
           "pwl_interpolant", "affine", "is_convex"]


def call(K):
    """European call (x - K)^+."""
    return ConvexPWL([K], [0.0], 0.0, 1.0)


def put(K):
    """European put (K - x)^+."""
    return ConvexPWL([K], [0.0], -1.0, 0.0)


def straddle(K):
    """Straddle |x - K|."""
    return ConvexPWL([K], [0.0], -1.0, 1.0)


def strangle(K1, K2):
    """Strangle (K1 - x)^+ + (x - K2)^+, K1 <= K2."""
    return ConvexPWL([K1, K2], [0.0, 0.0], -1.0, 1.0)


def call_ladder(strikes, weights=None):
    """Long portfolio of calls sum_k w_k (x - K_k)^+ with w_k >= 0 (convex)."""
    ks = np.asarray(strikes, float)
    w = np.ones_like(ks) if weights is None else np.asarray(weights, float)
    if np.any(w < 0):
        raise ValueError("negative weights give a non-convex payoff")
    o = np.argsort(ks); ks, w = ks[o], w[o]
    ys = np.array([np.sum(w * np.maximum(k - ks, 0.0)) for k in ks])
    return ConvexPWL(ks, ys, 0.0, float(w.sum()))


def max_of_calls(K1, K2, w2=2.0):
    """max{(x - K1)^+, w2 (x - K2)^+}."""
    return ConvexPWL.max_of([call(K1), ConvexPWL([K2], [0.0], 0.0, w2)])


def pwl_interpolant(fun, lo, hi, q):
    """Convex PWL interpolant of a convex function with q kinks (interior nodes of an
    equispaced grid of [lo, hi]), extended linearly by the outer chord slopes."""
    grid = np.linspace(lo, hi, q + 2)
    ys = np.array([fun(v) for v in grid])
    s = np.diff(ys) / np.diff(grid)
    return ConvexPWL(grid[1:-1], ys[1:-1], s[0], s[-1], simplify=False)


def affine(a, b):
    """x -> a + b x."""
    return ConvexPWL([0.0], [a], b, b, simplify=False)


def is_convex(f, tol=1e-12):
    """True if the PWL function has non-decreasing slopes."""
    inner = np.diff(f.y) / np.diff(f.x) if len(f.x) > 1 else np.array([])
    s = np.concatenate([[f.sl_left], inner, [f.sl_right]])
    return bool(np.all(np.diff(s) >= -tol))
