"""Price behaviour as the number of trading dates grows, against the exact tree benchmarks."""
import numpy as np
import pytest

from superhedge import SuperHedger, payoffs, tree_dp_superhedge, lp_superhedge

S0 = 100.0
CALL = (payoffs.call(100.0), lambda x: max(x - 100.0, 0.0))
STRADDLE = (payoffs.straddle(100.0), lambda x: abs(x - 100.0))


def model(M, aT=0.95, bT=1.05):
    return aT ** (1 / M), bT ** (1 / M)


@pytest.mark.parametrize("kappa", [0.0, 0.002, 0.01])
@pytest.mark.parametrize("claim", [CALL, STRADDLE])
def test_refining_the_trading_grid_never_raises_the_price(kappa, claim):
    g, f = claim
    prices = {}
    for M in (1, 2, 3, 4, 6, 8, 12, 24):
        a, b = model(M)
        prices[M] = SuperHedger([a] * M, [b] * M, kappa, reduction="extremes").fit(g).price(S0)
    for m in prices:
        for mm in prices:
            if mm > m and mm % m == 0:
                assert prices[mm] <= prices[m] + 1e-9


@pytest.mark.parametrize("kappa", [0.002, 0.01])
def test_costs_slow_the_benefit_of_frequency(kappa):
    _, f = CALL
    p = {M: tree_dp_superhedge(S0, *model(M), M, kappa, f)[0] for M in (1, 10, 100)}
    q = {M: tree_dp_superhedge(S0, *model(M), M, 0.0, f)[0] for M in (1, 10, 100)}
    assert p[1] > p[10] > p[100] and q[1] > q[10] > q[100]
    # the cost premium P(kappa) - P(0) grows with the number of dates
    assert p[1] - q[1] < p[10] - q[10] < p[100] - q[100]


@pytest.mark.parametrize("M", [1, 3, 5, 8])
@pytest.mark.parametrize("kappa", [0.0, 0.005])
def test_library_against_tree_lp_and_tree_dp(M, kappa):
    g, f = STRADDLE
    a, b = model(M)
    lp, _ = lp_superhedge(S0, [np.array([a, b])] * M, kappa, f)
    dp, _ = tree_dp_superhedge(S0, a, b, M, kappa, f)
    lib = SuperHedger([a] * M, [b] * M, kappa, reduction="extremes").fit(g).price(S0)
    assert abs(lp - dp) < 1e-6
    assert abs(lib - dp) < 1e-8 * max(dp, 1.0)


def test_extremes_mode_up_to_one_hundred_dates():
    g, f = CALL
    for M in (21, 50, 100):
        a, b = model(M)
        dp, _ = tree_dp_superhedge(S0, a, b, M, 0.005, f)
        lib = SuperHedger([a] * M, [b] * M, 0.005, reduction="extremes").fit(g).price(S0)
        assert abs(lib - dp) < 1e-7 * dp
