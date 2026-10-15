"""High-level interface.

>>> from superhedge import SuperHedger, payoffs
>>> sh = SuperHedger(alphas=[0.98] * 4, betas=[1.02] * 4, kappa=0.002)
>>> sh.fit(payoffs.call(100.0))
>>> sh.price(100.0)            # initial capital, starting from cash
>>> sh.hedge_ratio(100.0)      # initial number of shares
>>> out = sh.hedge([100, 101, 99.5, 100.7, 102.0])   # run the strategy along a path
"""
import numpy as np
from .recursion import backward
from . import forward as _fw
from .benchmark import lp_superhedge, tree_dp_superhedge

__all__ = ["SuperHedger"]


class SuperHedger:
    """Super-hedging of a convex European claim under proportional transaction costs.

    Parameters
    ----------
    alphas, betas : sequences of length M
        Support of the one-period gross return S_{t+1}/S_t at each rebalancing date.
    kappa : float
        Proportional transaction-cost rate (e.g. 0.001 for 10 bp).
    n_clusters : int, default 20
        Maximal number of retained interior states per step (accuracy / speed trade-off).
        Accuracy improves with n_clusters; use price_bracket() to bound the reduction error.
    reduction : {"width", "count", "extremes"}, default "width"
        "extremes": two-component recursion (see superhedge.backward); exact in all our tests with constant
        ranges and liquidation=False, and always a lower bound.
    simplify_tol : float, default 1e-6
    liquidation : bool, default False
        True: the final position is closed at cost kappa |phi| S_T (cash settlement).
    spot_range : (lo, hi) or None
        Represent the functions exactly only on the spots reachable from [lo, hi].
    bound : {None, "upper"}, default None
        "upper": certified upper reduction. The price is then guaranteed to be at least the exact price and
        the hedge built from it is guaranteed to super-hedge on every path inside the ranges.
    """

    def __init__(self, alphas, betas, kappa, n_clusters=20, reduction="width", simplify_tol=1e-6,
                 liquidation=False, spot_range=None, bound=None):
        if bound not in (None, "upper"):
            raise ValueError('bound must be None or "upper" (the lower reduction is available via price_bracket)')
        self.alphas, self.betas = list(map(float, alphas)), list(map(float, betas))
        self.kappa = float(kappa)
        self.n_clusters, self.reduction, self.simplify_tol = n_clusters, reduction, simplify_tol
        self.liquidation, self.spot_range, self.bound = liquidation, spot_range, bound
        self.result_ = None
        self.payoff_ = None

    @property
    def M(self):
        return len(self.alphas)

    def fit(self, payoff):
        """Run the backward recursion for a convex terminal payoff (a ConvexPWL)."""
        self.payoff_ = payoff
        self.result_ = backward(self.alphas, self.betas, self.kappa, payoff, n_clusters=self.n_clusters,
                                reduction=self.reduction, simplify_tol=self.simplify_tol,
                                liquidation=self.liquidation, spot_range=self.spot_range, bound=self.bound)
        return self

    def price_bracket(self, S0):
        """Certified (lower, upper) bounds on the exact super-hedging price at S0 (same n_clusters)."""
        self._check()
        kw = dict(n_clusters=self.n_clusters, reduction=self.reduction, liquidation=self.liquidation,
                  spot_range=self.spot_range)
        lo = backward(self.alphas, self.betas, self.kappa, self.payoff_, bound="lower", **kw).price0(S0)
        up = backward(self.alphas, self.betas, self.kappa, self.payoff_, bound="upper",
                      simplify_tol=self.simplify_tol, **kw).price0(S0)
        return lo, up

    def check_against_tree(self, S0):
        """Exact price by the recombining-tree dynamic programme (constant ranges only; fast for large M)."""
        self._check()
        if len(set(self.alphas)) > 1 or len(set(self.betas)) > 1:
            raise ValueError("the recombining tree requires constant ranges")
        f = self.payoff_
        return tree_dp_superhedge(S0, self.alphas[0], self.betas[0], self.M, self.kappa, lambda x: float(f(x)[0]),
                                  liquidation=self.liquidation)[0]

    def _check(self):
        if self.result_ is None:
            raise RuntimeError("call fit(payoff) first")

    def price(self, S0, phi_prev=0.0):
        """Minimal super-hedging capital at t = 0 for spot S0 and initial position phi_prev."""
        self._check()
        return self.result_.price0(S0, phi_prev)

    def price_function(self, t, phi, x):
        """g_t(phi, x): minimal capital at date t given previous position phi and spot x."""
        self._check()
        return self.result_.value(t, phi, x)

    def admissible_interval(self, t, wealth, S, phi_prev=0.0):
        """Interval of positions that keep the super-hedge at date t for the given wealth."""
        self._check()
        return _fw.interval(self.result_, t, phi_prev, wealth, S)

    def hedge_ratio(self, S0, phi_prev=0.0):
        """Position at t = 0 when starting with exactly the super-hedging price."""
        lo, hi = self.admissible_interval(0, self.price(S0, phi_prev), S0, phi_prev)
        if hi <= lo:                      # singleton (up to round-off), the generic case
            return float(0.5 * (lo + hi))
        return float(min(max(phi_prev, lo), hi))

    def hedge(self, path, V0=None, select="project"):
        """Run the self-financing super-hedge along a realised path of length M + 1."""
        self._check()
        f = self.payoff_
        return _fw.hedge(self.result_, path, V0=V0, select=select, payoff=lambda x: float(f(x)[0]))

    def check_against_lp(self, S0):
        """Price of the same problem by the independent tree LP (exact; practical for M <= 12)."""
        self._check()
        f = self.payoff_
        q, _ = lp_superhedge(S0, [np.array([a, b]) for a, b in zip(self.alphas, self.betas)], self.kappa,
                             lambda x: float(f(x)[0]), liquidation=self.liquidation)
        return q

    def summary(self, S0):
        """Dict with price, hedge ratio and model dimensions."""
        self._check()
        return dict(price=float(self.price(S0)), hedge_ratio=float(self.hedge_ratio(S0)), M=self.M, kappa=self.kappa,
                    states_t0=self.result_.n_states(0),
                    max_states=max(s["n_comp"] for s in self.result_.stats))
