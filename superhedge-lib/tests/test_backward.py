import numpy as np
import pytest
from superhedge import backward, payoffs, lp_superhedge, AIPError

S0 = 100.0
FUN = {"call": lambda K: (payoffs.call(K), lambda x: max(x - K, 0.0)),
       "put": lambda K: (payoffs.put(K), lambda x: max(K - x, 0.0)),
       "straddle": lambda K: (payoffs.straddle(K), lambda x: abs(x - K))}


@pytest.mark.parametrize("name", ["call", "put", "straddle"])
@pytest.mark.parametrize("K", [95.0, 100.0, 105.0])
@pytest.mark.parametrize("kappa", [0.0, 0.002, 0.01])
@pytest.mark.parametrize("M", [1, 2, 3])
def test_exact_against_lp(name, K, kappa, M):
    g, f = FUN[name](K)
    res = backward([0.98] * M, [1.02] * M, kappa, g, n_clusters=10**9)
    q, _ = lp_superhedge(S0, [np.array([0.98, 1.02])] * M, kappa, f)
    assert abs(res.price0(S0) - q) < 1e-9


def test_time_varying_supports():
    rng = np.random.default_rng(0)
    for _ in range(5):
        al, be = rng.uniform(0.94, 0.99, 3), rng.uniform(1.01, 1.06, 3)
        g, f = FUN["straddle"](100.0)
        res = backward(al, be, 0.005, g, n_clusters=10**9)
        q, _ = lp_superhedge(S0, [np.array([a, b]) for a, b in zip(al, be)], 0.005, f)
        assert abs(res.price0(S0) - q) < 1e-9


def test_degenerate_coefficients():
    g, f = FUN["call"](100.0)
    res = backward([0.75] * 3, [1.25] * 3, 0.25, g, n_clusters=10**9)      # 1 - kappa - alpha = 0
    q, _ = lp_superhedge(S0, [np.array([0.75, 1.25])] * 3, 0.25, f)
    assert abs(res.price0(S0) - q) < 1e-9


def test_reduced_recursion_is_accurate():
    g, f = FUN["call"](100.0)
    res = backward([0.98] * 5, [1.02] * 5, 0.005, g, n_clusters=20)
    q, _ = lp_superhedge(S0, [np.array([0.98, 1.02])] * 5, 0.005, f)
    assert abs(res.price0(S0) - q) / q < 1e-3


def test_aip_violation_is_detected():
    g, _ = FUN["call"](100.0)
    with pytest.raises(AIPError):
        backward([0.97] * 4, [0.99] * 4, 0.001, g)                            # support below 1
