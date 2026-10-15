"""Backward recursion for super-hedging under proportional transaction costs.

Model: one risky asset, cost rate kappa, conditional support of
S_{t+1}/S_t equal to [alpha_t, beta_t]. The minimal super-hedging price at date t, as a
function of the previous position phi and of the spot x, has the form

    g_t(phi, x) = max_k { ghat_t^k(x) - kappa * nu_t^k * phi * x },

with convex piecewise-linear ghat_t^k and normalised coefficients nu_t^k in [-1, 1]
(nu = +-1 are the two liquidation states). ``backward`` computes (nu_t, ghat_t) for
t = M-1, ..., 0 starting from the terminal payoff.

Numerical approximations (both optional):
* reduction of the interior coefficients to at most ``n_clusters`` representatives
  (``reduction="width"``: equal-width cells of (-1, 1), recommended; ``"count"``:
  blocks of equal cardinality);
* removal of breakpoints whose slope change is below ``pwl.SIMPLIFY_TOL`` (conservative).
"""
import time
import numpy as np
from . import pwl as _pwl
from .pwl import ConvexPWL

__all__ = ["AIPError", "BackwardResult", "backward", "backward_step", "active_components"]


class AIPError(RuntimeError):
    """Raised when the absence-of-immediate-profit condition fails (the price is -inf)."""


def _chunks(values, n_clusters, rule):
    """Contiguous partition of the sorted distinct values (indices into ``values``)."""
    n = len(values)
    if n == 0:
        return []
    order = np.argsort(values, kind="stable")
    if n_clusters is None or n_clusters >= n:          # no reduction: every distinct value is its own block
        return [order[i:i + 1] for i in range(n)]
    if rule == "count":
        return np.array_split(order, min(n, n_clusters))
    if rule == "width":
        cell = np.minimum(((values[order] + 1.0) / 2.0 * n_clusters).astype(int), n_clusters - 1)
        return np.split(order, np.flatnonzero(np.diff(cell)) + 1)
    raise ValueError(f"unknown reduction rule {rule!r}")


class BackwardResult:
    """Output of :func:`backward`: the functions g_t for t = 0..M (time-ordered)."""

    def __init__(self, nu, g, alphas, betas, kappa, stats):
        self.nu, self.g = nu, g
        self.alphas, self.betas, self.kappa = list(alphas), list(betas), kappa
        self.stats = stats          # one dict per backward step, from t = M-1 down to 0
        self.M = len(alphas)

    def value(self, t, phi, x):
        """g_t(phi, x): minimal super-hedging price at date t for previous position phi, spot x."""
        x = float(x)
        return max(float(h.evaluate(x)[0]) - self.kappa * n * phi * x for n, h in zip(self.nu[t], self.g[t]))

    def price0(self, S0, phi_prev=0.0):
        """Time-0 super-hedging price starting from cash (phi_{-1} = 0 by default)."""
        return self.value(0, phi_prev, S0)

    def n_states(self, t):
        return len(self.nu[t])


def backward_step(nu, g, alpha, beta, kappa, n_clusters=20, reduction="width", bound=None, domain=(0.0, np.inf),
                  zero_tol=1e-13, check_aip=True):
    """One step g_t = T_t g_{t+1}. Returns (nu_t, ghat_t, stats).

    bound=None   : approximate reduction (each block represented by its middle coefficient);
    bound="upper": certified upper reduction. A block B with coefficients in [nu_lo, nu_hi] is replaced by
                   the two components (G_B, nu_lo), (G_B, nu_hi), G_B = max_{k in B} ghat_k. Since
                   -kappa nu phi x is linear in nu, max_k (ghat_k - kappa nu_k phi x) <= max(G_B - kappa nu_lo phi x,
                   G_B - kappa nu_hi phi x): the result dominates the exact operator (<= 2 states per block);
    bound="lower": certified lower reduction. Each block keeps only its candidates at the middle coefficient
                   (a subset of the exact components), plus the extreme coefficients.
    domain       : spots at which the functions of this date are represented exactly (rays outside).
    """
    t0 = time.perf_counter()
    nu = np.asarray(nu, float)
    N = len(nu)
    s = np.array([1.0, -1.0])                                  # (-1)^r: r = 0 buy side, r = 1 sell side
    base = 1.0 - s[:, None] * kappa
    C = base - alpha * (1.0 + kappa * nu)[None, :]             # z^{k,r}(alpha)
    D = base - beta * (1.0 + kappa * nu)[None, :]              # z^{k,r}(beta)
    if check_aip:
        imin, imax = int(np.argmin(nu)), int(np.argmax(nu))
        if C[1, imin] < -zero_tol or D[0, imax] > zero_tol:
            raise AIPError(f"absence of immediate profit fails: z(1,1)={C[1, imin]:.3e}, "
                           f"z(N,0)={D[0, imax]:.3e}")
    # stacked coefficients Z[u, r, k]; every candidate is an oriented pair Z_a < 0 < Z_b
    Z = np.stack([C, D])
    zf = Z.ravel()
    uf = np.repeat([0, 1], 2 * N)
    sf = np.tile(np.repeat(s, N), 2)
    kf = np.tile(np.arange(N), 4)
    neg = np.where(zf < -zero_tol)[0]
    pos = np.where(zf > zero_tol)[0]
    zer = np.where(np.abs(zf) <= zero_tol)[0]
    za, zb = zf[neg][:, None], zf[pos][None, :]
    lam = (zb / (zb - za)).ravel()                             # convex weights in (0, 1)
    A = np.repeat(neg, len(pos))
    B = np.tile(pos, len(neg))
    nu_pair = -(lam * sf[A] + (1.0 - lam) * sf[B])             # new normalised coefficient
    cand_nu = np.concatenate([nu_pair, -sf[zer]])              # degenerate candidates: Z = 0
    cand_a = np.concatenate([A, zer])
    cand_b = np.concatenate([B, zer])
    cand_l = np.concatenate([lam, np.ones(len(zer))])
    cand_nu[np.abs(cand_nu - 1.0) < 1e-12] = 1.0
    cand_nu[np.abs(cand_nu + 1.0) < 1e-12] = -1.0
    t1 = time.perf_counter()

    # exact merge of equal coefficients; reduction of the interior ones; +-1 kept exactly
    uniq, inv = np.unique(cand_nu, return_inverse=True)
    members = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    interior_idx = np.where((uniq > -1.0) & (uniq < 1.0))[0]
    chunks = [interior_idx[c] for c in _chunks(uniq[interior_idx], n_clusters, reduction)]
    plan = []                                       # (coefficient, candidate indices defining the function)
    if uniq[0] == -1.0:
        plan.append((-1.0, members[0]))
    # the extreme coefficients are always kept exactly (they determine the AIP condition of the next step)
    if bound is None and chunks:
        if uniq[0] != -1.0 and len(chunks[0]) > 1:
            chunks = [chunks[0][:1], chunks[0][1:]] + chunks[1:]
        if uniq[-1] != 1.0 and len(chunks[-1]) > 1:
            chunks = chunks[:-1] + [chunks[-1][:-1], chunks[-1][-1:]]
    for j, ch in enumerate(chunks):
        if bound is None:
            plan.append((uniq[ch[len(ch) // 2]], np.concatenate([members[k] for k in ch])))
        elif bound == "upper":
            allm = np.concatenate([members[k] for k in ch])
            plan.append((uniq[ch[0]], allm))
            if len(ch) > 1:
                plan.append((uniq[ch[-1]], allm))
        elif bound == "lower":
            keep = {int(ch[len(ch) // 2])}
            if j == 0 and uniq[0] != -1.0:
                keep.add(int(ch[0]))
            if j == len(chunks) - 1 and uniq[-1] != 1.0:
                keep.add(int(ch[-1]))
            for k in sorted(keep):
                plan.append((uniq[k], members[k]))
        else:
            raise ValueError(f"unknown bound {bound!r}")
    if uniq[-1] == 1.0:
        plan.append((1.0, members[-1]))
    t2 = time.perf_counter()

    # exact PWL algebra: lambda ghat_a(u x) + (1 - lambda) ghat_b(v x), maximum within each group
    ab = (alpha, beta)
    lo, hi = domain
    cache = {}
    reps, new_g = [], []
    for coef, grp in plan:
        key = id(grp)
        if key not in cache:
            funcs = []
            for k in grp:
                a, b, l = cand_a[k], cand_b[k], cand_l[k]
                f = g[kf[a]].scaled(ab[uf[a]], l)
                if l < 1.0:
                    f = f + g[kf[b]].scaled(ab[uf[b]], 1.0 - l)
                funcs.append(f)
            cache[key] = ConvexPWL.max_of(funcs).restrict(lo, hi)
        reps.append(coef)
        new_g.append(cache[key])
    t3 = time.perf_counter()
    stats = dict(N_in=N, n_cand=len(cand_nu), n_dist=len(uniq), n_interior=len(interior_idx), n_comp=len(reps),
                 max_bkpts=max(h.n_breakpoints for h in new_g), tot_bkpts=sum(h.n_breakpoints for h in new_g),
                 t_tensor=t1 - t0, t_group=t2 - t1, t_pwl=t3 - t2)
    return np.array(reps), new_g, stats


def backward(alphas, betas, kappa, g_T, nu_T=(0.0,), n_clusters=20, reduction="width", bound=None,
             simplify_tol=None, spot_range=None, liquidation=False, **kw):
    """Backward recursion.

    Parameters
    ----------
    alphas, betas : sequences of length M
        Support [alphas[t], betas[t]] of S_{t+1}/S_t (0 < alpha <= beta).
    kappa : float
        Proportional transaction-cost rate.
    g_T : ConvexPWL or list of ConvexPWL
        Terminal payoff (convex). A list together with ``nu_T`` defines a payoff of the
        general form max_k { g_T^k(x) - kappa nu_T^k phi x }.
    n_clusters : int
        Maximal number of interior blocks per step (``10**9`` disables the reduction).
    reduction : {"width", "count", "extremes"}
        Partition rule of the reduction. "extremes" keeps, at every step, only the two components with the
        smallest and the largest coefficient (n_clusters is then ignored). Keeping a subset of the exact
        components gives a lower bound of the exact price; with constant ranges and the default terminal
        convention (liquidation=False) this two-component recursion reproduced the exact price in all our
        tests, at a cost linear in the number of dates. Check it with SuperHedger.check_against_tree.
    bound : {None, "upper", "lower"}
        None: approximate reduction (default). "upper"/"lower": certified reduction; the computed
        price is then an upper/lower bound of the exact super-hedging price (for "lower" the
        breakpoint simplification, which is upward, is switched off).
    simplify_tol : float, optional
        Breakpoint-simplification tolerance for this call (default ``pwl.SIMPLIFY_TOL``).
    spot_range : (lo, hi), optional
        Spots at t = 0 at which exactness is required. The functions at date t are then kept exact on
        the reachable range [lo * prod(alphas[:t]), hi * prod(betas[:t])] only; by default they are
        exact on x > 0.
    liquidation : bool
        If True, the terminal position is liquidated at cost kappa |phi| S_T (cash settlement):
        g_T(phi, x) = g(x) + kappa |phi| x = max(g(x) + kappa phi x, g(x) - kappa phi x).
        If False (default), the terminal position is valued at S_T without cost.
    """
    if len(alphas) != len(betas):
        raise ValueError("alphas and betas must have the same length")
    if any(a <= 0 or a > b for a, b in zip(alphas, betas)):
        raise ValueError("each support must satisfy 0 < alpha <= beta")
    if bound == "lower":
        simplify_tol = 0.0
    old_tol = _pwl.SIMPLIFY_TOL
    if simplify_tol is not None:
        _pwl.SIMPLIFY_TOL = simplify_tol
    try:
        M = len(alphas)
        if spot_range is None:
            doms = [(0.0, np.inf)] * (M + 1)
        else:
            lo, hi = spot_range
            doms = [(lo * float(np.prod(alphas[:t])), hi * float(np.prod(betas[:t]))) for t in range(M + 1)]
        nu = [None] * (M + 1)
        g = [None] * (M + 1)
        if isinstance(g_T, ConvexPWL):
            g_T = [g_T]
        if liquidation:
            if tuple(nu_T) != (0.0,) or len(g_T) != 1:
                raise ValueError("liquidation=True expects a single terminal payoff with nu_T = (0,)")
            g_T, nu_T = [g_T[0], g_T[0]], (-1.0, 1.0)
        nu[M], g[M] = np.asarray(nu_T, float), [h.restrict(*doms[M]) for h in g_T]
        stats = []
        if reduction == "extremes" and bound is not None:
            raise ValueError('reduction="extremes" cannot be combined with bound')
        for t in range(M - 1, -1, -1):
            if reduction == "extremes":
                nu[t], g[t], st = backward_step(nu[t + 1], g[t + 1], alphas[t], betas[t], kappa, n_clusters=None,
                                                reduction="width", domain=doms[t], **kw)
                keep = sorted({0, len(nu[t]) - 1})
                nu[t], g[t] = np.asarray(nu[t])[keep], [g[t][i] for i in keep]
                st["n_comp"] = len(keep)
            else:
                nu[t], g[t], st = backward_step(nu[t + 1], g[t + 1], alphas[t], betas[t], kappa,
                                                n_clusters=n_clusters, reduction=reduction, bound=bound,
                                                domain=doms[t], **kw)
            st["t"] = t
            stats.append(st)
    finally:
        _pwl.SIMPLIFY_TOL = old_tol
    res = BackwardResult(nu, g, alphas, betas, kappa, stats)
    res.bound, res.liquidation = bound, liquidation
    return res


def active_components(res, t, phi_grid, x_grid):
    """Number of components of g_t that attain the maximum (first argmax) somewhere on phi_grid x x_grid."""
    H = np.array([h.evaluate(x_grid) for h in res.g[t]])
    nu = np.asarray(res.nu[t])
    winners = set()
    for phi in phi_grid:
        V = H - res.kappa * nu[:, None] * phi * x_grid[None, :]
        winners.update(np.argmax(V, axis=0).tolist())
    return len(winners)
