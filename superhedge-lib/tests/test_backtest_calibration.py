import numpy as np
from superhedge import backward, hedge, payoffs, tree_dp_superhedge, lp_superhedge
from superhedge.calibration import ewma_quantile_support
from superhedge.backtest import discounted_path, superhedge_contract, bs_price, leland_delta_hedge


def test_ewma_support_uses_past_only():
    p = np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 800))) * 100
    a, b = ewma_quantile_support(p, 500, window=252, h=5, lower=0.01, upper=0.99)
    q = p.copy(); q[501:] *= 3.0
    assert (a, b) == ewma_quantile_support(q, 500, window=252, h=5, lower=0.01, upper=0.99)
    assert a < 1.0 < b


def test_tree_dp_matches_lp():
    for M in (2, 4, 6):
        for liq in (False, True):
            f = lambda x: abs(x - 100.0)
            d, _ = tree_dp_superhedge(100.0, 0.98, 1.02, M, 0.005, f, liquidation=liq)
            q, _ = lp_superhedge(100.0, [np.array([0.98, 1.02])] * M, 0.005, f, liquidation=liq)
            assert abs(d - q) < 1e-9


def test_backtest_without_rates_equals_hedge():
    S = np.array([100.0, 100.5, 101.0, 100.2, 99.8, 100.9, 101.3, 100.7, 101.1])
    D = np.zeros_like(S)
    idx = [0, 2, 4, 6, 8]
    X, div = discounted_path(S, D, idx, r=0.0)
    res, out = superhedge_contract(X, div, 100.0, [0.98] * 4, [1.02] * 4, 0.002, "call")
    ref = hedge(res, S[idx], payoff=lambda x: max(x - 100.0, 0.0))
    assert abs(out["surplus"] - ref["surplus"]) < 1e-12


def test_bs_put_call_parity_and_leland():
    c = bs_price(100, 100, 1 / 12, 0.2, 0.03, 0.01); p = bs_price(100, 100, 1 / 12, 0.2, 0.03, 0.01, "put")
    assert abs(c - p - (100 * np.exp(-0.01 / 12) - 100 * np.exp(-0.03 / 12))) < 1e-10
    X = np.array([100.0, 101.0, 99.0, 100.5, 102.0]); out = leland_delta_hedge(X, np.zeros(4), 100.0, 4 / 52, 0.2, 0.001, 1 / 52)
    assert out["sigma_leland"] > 0.2 and np.isfinite(out["surplus"])


def test_nonrecombining_dp_matches_lp_and_recursion_time_varying():
    from superhedge import nonrecombining_dp_superhedge, lp_superhedge, backward, payoffs
    rng = np.random.default_rng(3)
    for M in (2, 3, 5):
        a = 1 - rng.uniform(0.005, 0.03, M); b = 1 + rng.uniform(0.005, 0.03, M)
        f = lambda x: abs(x - 100.0)
        dp = nonrecombining_dp_superhedge(100.0, a, b, 0.004, f)
        lp, _ = lp_superhedge(100.0, [np.array([u, v]) for u, v in zip(a, b)], 0.004, f)
        assert abs(dp - lp) < 1e-7
        if M <= 3:
            ex = backward(list(a), list(b), 0.004, payoffs.straddle(100.0), n_clusters=None).price0(100.0)
            assert abs(dp - ex) < 1e-8


def test_nonrecombining_dp_constant_ranges_equals_tree_dp():
    from superhedge import nonrecombining_dp_superhedge, tree_dp_superhedge
    f = lambda x: max(x - 100.0, 0.0)
    for liq in (False, True):
        q, _ = tree_dp_superhedge(100.0, 0.98, 1.02, 6, 0.005, f, liquidation=liq)
        assert abs(nonrecombining_dp_superhedge(100.0, [0.98] * 6, [1.02] * 6, 0.005, f, liquidation=liq) - q) < 1e-10


def test_path_payoff_lp_and_dp_agree():
    from superhedge import nonrecombining_dp_superhedge, lp_superhedge
    asian = lambda p: max(p[1:].mean() - 100.0, 0.0)
    lookback = lambda p: p.max() - p[-1]
    for f in (asian, lookback):
        lp, _ = lp_superhedge(100.0, [np.array([0.98, 1.02])] * 4, 0.005, f, path_payoff=True)
        dp = nonrecombining_dp_superhedge(100.0, [0.98] * 4, [1.02] * 4, 0.005, f, path_payoff=True)
        assert abs(lp - dp) < 1e-7
    # European payoff through the path interface gives the usual price
    lp1, _ = lp_superhedge(100.0, [np.array([0.98, 1.02])] * 3, 0.005, lambda p: max(p[-1] - 100.0, 0.0), path_payoff=True)
    lp2, _ = lp_superhedge(100.0, [np.array([0.98, 1.02])] * 3, 0.005, lambda x: max(x - 100.0, 0.0))
    assert abs(lp1 - lp2) < 1e-9
