"""Independent benchmark: global linear program of the super-hedging problem on a tree.

Shares no code with the backward recursion (useful to validate it). Variables: initial capital p and, for every
internal node n, a position phi_n and u_n >= |phi_n - phi_parent(n)| (phi_{-1} = 0).
Minimise p subject to V_leaf >= g(S_leaf) for every leaf, with V given by the self-financing equation. With ratios {alpha_t, beta_t} and a
convex payoff its value equals the super-hedging price of the interval model.
"""
__all__ = ["lp_superhedge", "tree_dp_superhedge", "nonrecombining_dp_superhedge"]

import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .pwl import ConvexPWL


def _clip(F, c):
    """psi -> min_phi F(phi) + c |phi - psi| (exact for convex PWL F)."""
    x, y = F.x, F.y
    sl = np.concatenate([[F.sl_left], np.diff(y) / np.diff(x) if len(x) > 1 else [], [F.sl_right]])
    if sl[0] > c or sl[-1] < -c:
        raise ArithmeticError("immediate profit: the price is -inf")
    # keep breakpoints i whose right slope sl[i+1] > -c and left slope sl[i] < c
    keep = (sl[1:] > -c) & (sl[:-1] < c)
    if not keep.any():                           # minimum region contains everything: flat
        k = int(np.argmin(y))
        return ConvexPWL([x[k]], [y[k]], -c, c, simplify=False)
    return ConvexPWL(x[keep], y[keep], max(sl[0], -c), min(sl[-1], c), simplify=False)


def lp_superhedge(S0, ratios, kappa, payoff, liquidation=False, path_payoff=False):
    """ratios: list (length M) of 1-D arrays of admissible one-step ratios S_{t+1}/S_t.
    liquidation=True charges kappa |phi_{M-1}| S_M at maturity. path_payoff=True: payoff is called with the whole
    price path (S_0, ..., S_M) as an array (path-dependent claims). Returns (price, phi_0)."""
    M = len(ratios)
    L = [len(r) for r in ratios]
    # internal nodes by depth; node = tuple of child indices
    levels = [[()]]
    for t in range(M - 1):
        levels.append([n + (k,) for n in levels[-1] for k in range(L[t])])
    internal = [n for lev in levels for n in lev]
    idx = {n: i for i, n in enumerate(internal)}
    spot = {(): S0}
    for lev in levels[1:]:
        for n in lev:
            spot[n] = spot[n[:-1]] * ratios[len(n) - 1][n[-1]]
    ni = len(internal)
    P, PHI, U = 0, 1, 1 + ni
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for n, j in idx.items():                       # s (phi_n - phi_par) - u_n <= 0
        for sg in (1.0, -1.0):
            rows += [r, r]; cols += [U + j, PHI + j]; vals += [-1.0, sg]
            if n:
                rows.append(r); cols.append(PHI + idx[n[:-1]]); vals.append(-sg)
            rhs.append(0.0); r += 1
    for leaf in itertools.product(*[range(l) for l in L]):
        s_par = S0
        path = [S0]
        rows.append(r); cols.append(P); vals.append(-1.0)
        for t in range(M):
            node = leaf[:t]
            s_ch = s_par * ratios[t][leaf[t]]
            j = idx[node]
            rows += [r, r]; cols += [PHI + j, U + j]; vals += [-(s_ch - s_par), kappa * s_par]
            s_par = s_ch
            path.append(s_ch)
        gT = payoff(np.array(path)) if path_payoff else payoff(s_par)
        if not liquidation:
            rhs.append(-gT); r += 1
        else:                                          # V_T - kappa S_T |phi_{M-1}| >= g(S_T): two rows
            row_cols, row_vals = cols[-(2 * M + 1):], vals[-(2 * M + 1):]
            jl = idx[leaf[:M - 1]]
            rows.append(r); cols.append(PHI + jl); vals.append(kappa * s_par)
            rhs.append(-gT); r += 1
            rows += [r] * len(row_cols); cols += list(row_cols); vals += list(row_vals)
            rows.append(r); cols.append(PHI + jl); vals.append(-kappa * s_par)
            rhs.append(-gT); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, 1 + 2 * ni)).tocsr()
    c = np.zeros(1 + 2 * ni); c[P] = 1.0
    bounds = [(None, None)] * (1 + ni) + [(0, None)] * ni
    sol = linprog(c, A_ub=A, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if sol.status != 0:
        return np.nan, np.nan
    return float(sol.fun), float(sol.x[PHI])


def tree_dp_superhedge(S0, alpha, beta, M, kappa, payoff, liquidation=False):
    """Polynomial dynamic programme on the recombining binomial tree {alpha, beta}^M (constant ratios).

    State at node (t, j): the previous position psi. W_{t,j}(psi) is the minimal mark-to-market wealth
    needed at the node; it is convex piecewise linear in psi and
        W_{t,j}(psi) = min_phi { kappa S |phi - psi| + max_c [ W_c(phi) - phi (S_c - S) ] },
    i.e. a pointwise maximum followed by an inf-convolution with kappa S |.| (slope clipping).
    O(M^2) nodes. By the endpoint reduction its value equals the interval-model price for convex payoffs.
    Returns (price, phi_0).
    """
    S = lambda t, j: S0 * alpha ** (t - j) * beta ** j
    W = []
    for j in range(M + 1):
        s = S(M, j); v = payoff(s)
        W.append(ConvexPWL([0.0], [v], -kappa * s if liquidation else 0.0, kappa * s if liquidation else 0.0,
                           simplify=False))
    for t in range(M - 1, -1, -1):
        Wn = []
        for j in range(t + 1):
            s = S(t, j)
            F = W[j + 1].add_linear(-(S(t + 1, j + 1) - s)).pointwise_max(W[j].add_linear(-(S(t + 1, j) - s)))
            if t == 0:
                F0 = F
            Wn.append(_clip(F, kappa * s))
        W = Wn
    price = float(W[0](0.0)[0])
    cand = np.concatenate([F0.x, [0.0]])
    phi0 = float(cand[np.argmin(F0(cand) + kappa * S0 * np.abs(cand))])
    return price, phi0


def nonrecombining_dp_superhedge(S0, alphas, betas, kappa, payoff, liquidation=False, path_payoff=False):
    """Exact dynamic programme on the non-recombining binomial tree prod_t {alpha_t, beta_t} (time-varying ranges).

    Same node recursion as tree_dp_superhedge, on 2^M leaves; the state is the previous position. With
    path_payoff=True the payoff is called with the whole path (S_0, ..., S_M) (path-dependent claims).
    By the endpoint reduction its value equals the interval-model price for convex payoffs. Returns the price.
    """
    alphas, betas = np.asarray(alphas, float), np.asarray(betas, float)
    M = len(alphas)
    if path_payoff and M > 18:
        raise ValueError("path_payoff stores all 2^M paths; use M <= 18")
    spots = [np.array([float(S0)])]
    for t in range(M):
        s = spots[-1]
        spots.append(np.stack([s * alphas[t], s * betas[t]], 1).ravel())
    if path_payoff:
        paths = np.empty((2 ** M, M + 1))
        for t in range(M + 1):
            paths[:, t] = np.repeat(spots[t], 2 ** (M - t))
        values = [payoff(p) for p in paths]
    else:
        values = [payoff(x) for x in spots[M]]
    W = [ConvexPWL([0.0], [v], -kappa * x if liquidation else 0.0, kappa * x if liquidation else 0.0, simplify=False)
         for v, x in zip(values, spots[M])]
    for t in range(M - 1, -1, -1):
        Wn = []
        for i, s in enumerate(spots[t]):
            sd, su = spots[t + 1][2 * i], spots[t + 1][2 * i + 1]
            F = W[2 * i].add_linear(-(sd - s)).pointwise_max(W[2 * i + 1].add_linear(-(su - s)))
            Wn.append(_clip(F, kappa * s))
        W = Wn
    return float(W[0](0.0)[0])
