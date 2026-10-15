"""Forward construction: admissible interval of hedge ratios and self-financing hedge.

At date t, with wealth p, previous position phi_prev and spot S, a position phi super-hedges
g_{t+1} over (t, t+1] if and only if, for every state k of g_{t+1}, r in {0, 1} and
u in {alpha_t, beta_t},

    z^{k,r}(u) S phi <= p - ghat^k(u S) - (-1)^r kappa phi_prev S,
    z^{k,r}(u) = 1 - (-1)^r kappa - u (1 + kappa nu^k).

The admissible set is the interval I_t(p) = [phi_l, phi_u]. The wealth then follows

    V_{t+1} = V_t + phi_t (S_{t+1} - S_t) - kappa S_t |phi_t - phi_{t-1}|.
"""
import numpy as np

__all__ = ["interval", "hedge", "width"]


def interval(res, t, phi_prev, p, S, zero_tol=1e-13):
    """Admissible interval [phi_l, phi_u] at date t (may be a half-line or empty if phi_l > phi_u)."""
    a, b, k = res.alphas[t], res.betas[t], res.kappa
    nu = np.asarray(res.nu[t + 1])
    ga = np.array([float(h.evaluate(a * S)[0]) for h in res.g[t + 1]])
    gb = np.array([float(h.evaluate(b * S)[0]) for h in res.g[t + 1]])
    lo, hi = -np.inf, np.inf
    for s in (1.0, -1.0):
        for u, gv in ((a, ga), (b, gb)):
            coef = 1.0 - s * k - u * (1.0 + k * nu)
            val = (p - gv - s * k * phi_prev * S) / (coef * S)
            if np.any(coef < -zero_tol):
                lo = max(lo, val[coef < -zero_tol].max())
            if np.any(coef > zero_tol):
                hi = min(hi, val[coef > zero_tol].min())
    return lo, hi


def _select(lo, hi, phi_prev, rule):
    if rule == "project":            # closest admissible position: minimal trade at this date
        return min(max(phi_prev, lo), hi)
    if rule == "lower":
        return lo
    if rule == "upper":
        return hi
    if rule in ("mid", "center"):          # farthest point from both boundaries of I_t
        if np.isfinite(lo) and np.isfinite(hi):
            return 0.5 * (lo + hi)
        return lo if np.isfinite(lo) else hi
    raise ValueError(f"unknown selection rule {rule!r}")


def hedge(res, path, V0=None, select="project", tol=1e-9, payoff=None):
    """Run the super-hedging strategy along a price path S_0, ..., S_M.

    Parameters
    ----------
    res : BackwardResult
    path : array of length M + 1
    V0 : initial capital (default: the super-hedging price g_0(0, S_0)).
    select : rule used to pick phi_t in I_t(V_t): "project" (default, minimal trade), "center"
        (midpoint of I_t, largest distance to its boundaries), "lower", "upper".
    payoff : callable giving the terminal payoff (default: g_M(0, .)).

    Returns a dict with positions, wealth, transaction costs, turnover, number of trades,
    number of empty intervals, number of support breaches and the terminal surplus
    V_T - g(S_T) (minus the liquidation cost kappa |phi_{M-1}| S_T if the recursion was run with
    liquidation=True). If the realised path leaves the support, the guarantee no longer holds;
    if I_t becomes empty the position is held.
    """
    M = res.M
    path = np.asarray(path, float)
    if len(path) != M + 1:
        raise ValueError(f"path must have length M + 1 = {M + 1}")
    V = res.price0(path[0]) if V0 is None else float(V0)
    phi_prev, cost, turnover, trades, empties, breaches = 0.0, 0.0, 0.0, 0, 0, 0
    phis, Vs, intervals = [], [V], []
    for t in range(M):
        S = path[t]
        lo, hi = interval(res, t, phi_prev, V, S)
        intervals.append((lo, hi))
        if lo <= hi + tol * max(1.0, abs(lo)):
            if hi < lo:                                   # singleton up to round-off
                lo = hi = 0.5 * (lo + hi)
            phi = _select(lo, hi, phi_prev, select)
        else:
            phi, empties = phi_prev, empties + 1
        r = path[t + 1] / S
        if r < res.alphas[t] - 1e-12 or r > res.betas[t] + 1e-12:
            breaches += 1
        c = res.kappa * S * abs(phi - phi_prev)
        cost += c
        turnover += abs(phi - phi_prev) * S
        trades += int(abs(phi - phi_prev) > 1e-8)
        V = V + phi * (path[t + 1] - S) - c
        phi_prev = phi
        phis.append(phi); Vs.append(V)
    liq = res.kappa * abs(phi_prev) * path[-1] if getattr(res, "liquidation", False) else 0.0
    gT = payoff(path[-1]) if payoff is not None else res.value(M, 0.0, path[-1])
    return dict(V0=Vs[0], VT=V, payoff=gT, liquidation_cost=liq, surplus=V - liq - gT, cost=cost, turnover=turnover,
                trades=trades, empties=empties, breaches=breaches, phis=phis, Vs=Vs, intervals=intervals)


def width(res, t, phi_prev, S, delta):
    """Width of I_t(pi_t + delta), pi_t = g_t(phi_prev, S): freedom bought by excess capital delta."""
    lo, hi = interval(res, t, phi_prev, res.value(t, phi_prev, S) + delta, S)
    return hi - lo, lo, hi
