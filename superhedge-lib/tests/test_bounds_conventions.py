import itertools
import numpy as np
import pytest
from superhedge import backward, hedge, payoffs, lp_superhedge

S0 = 100.0
FUN = {"call": (payoffs.call(100.0), lambda x: max(x - 100.0, 0.0)),
       "straddle": (payoffs.straddle(100.0), lambda x: abs(x - 100.0))}


@pytest.mark.parametrize("name", ["call", "straddle"])
@pytest.mark.parametrize("kappa", [0.002, 0.01])
@pytest.mark.parametrize("ncl", [2, 5, 20])
@pytest.mark.parametrize("rule", ["width", "count"])
def test_certified_bracket(name, kappa, ncl, rule):
    g, f = FUN[name]
    M, a, b = 5, 0.98, 1.02
    q, _ = lp_superhedge(S0, [np.array([a, b])] * M, kappa, f)
    up = backward([a] * M, [b] * M, kappa, g, n_clusters=ncl, reduction=rule, bound="upper").price0(S0)
    lo = backward([a] * M, [b] * M, kappa, g, n_clusters=ncl, reduction=rule, bound="lower").price0(S0)
    assert lo <= q + 1e-9 <= up + 2e-9


def test_upper_bound_hedge_superhedges():
    g, f = FUN["straddle"]
    a, b, M = 0.98, 1.02, 5
    res = backward([a] * M, [b] * M, 0.01, g, n_clusters=2, bound="upper")
    paths = [S0 * np.cumprod([1.0] + list(r)) for r in itertools.product([a, b], repeat=M)]
    assert min(hedge(res, p, payoff=f)["surplus"] for p in paths) > -1e-9


def test_liquidation_convention():
    g, f = FUN["call"]
    a, b, M, k = 0.98, 1.02, 3, 0.005
    free = backward([a] * M, [b] * M, k, g, n_clusters=10**9)
    liq = backward([a] * M, [b] * M, k, g, n_clusters=10**9, liquidation=True)
    # LP with terminal liquidation: at maturity the position is closed, cost k |phi| S_T.
    # The call hedge ends long, so the liquidated price is higher.
    assert liq.price0(S0) > free.price0(S0)
    paths = [S0 * np.cumprod([1.0] + list(r)) for r in itertools.product([a, b], repeat=M)]
    for p in paths:
        out = hedge(liq, p, payoff=f)
        assert out["surplus"] > -1e-9 and out["liquidation_cost"] >= 0


def test_spot_range_is_exact_on_reachable_spots():
    g, f = FUN["straddle"]
    a, b, M = 0.97, 1.03, 4
    full = backward([a] * M, [b] * M, 0.005, g, n_clusters=10**9 if M < 4 else 50)
    loc = backward([a] * M, [b] * M, 0.005, g, n_clusters=10**9 if M < 4 else 50, spot_range=(90.0, 110.0))
    for x in np.linspace(90, 110, 11):
        assert abs(full.value(0, 0.3, x) - loc.value(0, 0.3, x)) < 1e-8


def test_superhedger_upper_bound_option():
    from superhedge import SuperHedger, payoffs
    sh = SuperHedger([0.98] * 4, [1.02] * 4, 0.005, n_clusters=3, bound="upper").fit(payoffs.straddle(100.0))
    exact = sh.check_against_lp(100.0)
    assert sh.price(100.0) >= exact - 1e-9
    rng = np.random.default_rng(1)
    for _ in range(50):
        path = 100.0 * np.cumprod(np.concatenate([[1.0], rng.choice([0.98, 1.02], 4)]))
        assert sh.hedge(path)["surplus"] >= -1e-9


def test_extremes_reduction_matches_tree_dp_and_is_lower_bound():
    from superhedge import tree_dp_superhedge, nonrecombining_dp_superhedge
    f = lambda x: max(x - 100.0, 0.0)
    for kappa in (0.0005, 0.005, 0.02):
        for M in (3, 10, 21):
            a, b = 0.95 ** (1 / M), 1.05 ** (1 / M)
            q, _ = tree_dp_superhedge(100.0, a, b, M, kappa, f)
            p = backward([a] * M, [b] * M, kappa, payoffs.call(100.0), reduction="extremes").price0(100.0)
            assert abs(p - q) <= 1e-8 * max(q, 1.0)
    rng = np.random.default_rng(0)                        # time-varying ranges: lower bound (up to round-off)
    a = 1 - rng.uniform(0.002, 0.03, 8); b = 1 + rng.uniform(0.002, 0.03, 8)
    q = nonrecombining_dp_superhedge(100.0, a, b, 0.005, f)
    p = backward(list(a), list(b), 0.005, payoffs.call(100.0), reduction="extremes", simplify_tol=0.0).price0(100.0)
    assert p <= q + 1e-9
