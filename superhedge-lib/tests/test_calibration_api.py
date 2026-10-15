import numpy as np
import pytest
from superhedge import SuperHedger, payoffs
from superhedge.calibration import quantile_support, horizon_support


def test_quantile_support_uses_past_only():
    p = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 600))) * 100
    a1, b1 = quantile_support(p, 400, window=252, h=5, lower=0.01, upper=0.99)
    q = p.copy(); q[401:] *= 10.0                       # changing the future must not matter
    assert (a1, b1) == quantile_support(q, 400, window=252, h=5, lower=0.01, upper=0.99)
    assert a1 < 1.0 < b1


def test_horizon_support():
    al, be = horizon_support(0.95, 1.05, 4)
    assert np.isclose(np.prod(al), 0.95) and np.isclose(np.prod(be), 1.05)


def test_api_price_and_lp():
    sh = SuperHedger([0.98] * 4, [1.02] * 4, kappa=0.002).fit(payoffs.call(100.0))
    assert sh.price(100.0) == pytest.approx(1.7604627, abs=1e-6)
    assert sh.price(100.0) == pytest.approx(sh.check_against_lp(100.0), abs=1e-9)
    assert 0.0 < sh.hedge_ratio(100.0) < 1.0
