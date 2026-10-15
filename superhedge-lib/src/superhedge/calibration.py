"""Calibration of the conditional support [alpha, beta] from historical prices.

The support of the h-period gross return S_{t+h}/S_t is estimated at a date t0 from the
overlapping h-period returns observed in the window [t0 - W, t0] only (no look-ahead):
alpha = q_lower(R), beta = q_upper(R). ``lower=0, upper=1`` gives the historical min/max.
"""
import numpy as np

__all__ = ["quantile_support", "ewma_quantile_support", "ewma_vol", "rolling_supports", "horizon_support"]


def quantile_support(prices, t0, window=252, h=1, lower=0.0, upper=1.0):
    """Quantile support of h-period gross returns using prices[t0 - window : t0 + 1] only.

    Parameters
    ----------
    prices : 1-D array of positive prices (daily closes, say)
    t0 : index of the calibration date (the last price that may be used)
    window : number of past observations (returns S_{s+h}/S_s with t0 - window <= s, s + h <= t0)
    h : return horizon, in observations (e.g. 5 for weekly rebalancing on daily data)
    lower, upper : quantile levels, 0 <= lower < upper <= 1

    Returns
    -------
    (alpha, beta)
    """
    prices = np.asarray(prices, float)
    if t0 - window < 0:
        raise ValueError("not enough history before t0")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("need 0 <= lower < upper <= 1")
    s = np.arange(t0 - window, t0 - h + 1)
    R = prices[s + h] / prices[s]
    return float(np.quantile(R, lower)), float(np.quantile(R, upper))


def rolling_supports(prices, dates_idx, **kw):
    """quantile_support evaluated at each index of dates_idx."""
    return [quantile_support(prices, t0, **kw) for t0 in dates_idx]


def horizon_support(alpha_T, beta_T, M):
    """Spread a horizon support [alpha_T, beta_T] evenly over M steps: alpha_t = alpha_T**(1/M)."""
    return [alpha_T ** (1.0 / M)] * M, [beta_T ** (1.0 / M)] * M


def ewma_vol(log_returns, lam=0.94, init=None):
    """RiskMetrics EWMA volatility; sigma[s] uses returns up to and including s."""
    r = np.asarray(log_returns, float)
    v = np.empty_like(r)
    var = np.var(r[:20]) if init is None else init ** 2
    for i, x in enumerate(r):
        var = lam * var + (1.0 - lam) * x * x
        v[i] = var
    return np.sqrt(v)


def ewma_quantile_support(prices, t0, window=252, h=1, lower=0.0, upper=1.0, lam=0.94):
    """Volatility-scaled quantile support (no look-ahead).

    Past h-period log returns are standardised by the EWMA volatility known at their start,
    z_s = log(S_{s+h}/S_s) / (sigma_s sqrt(h)); the support at t0 rescales the quantiles of z by the
    current volatility: log alpha = q_lower(z) sigma_{t0} sqrt(h), log beta = q_upper(z) sigma_{t0} sqrt(h).
    """
    prices = np.asarray(prices, float)
    if t0 - window < 1:
        raise ValueError("not enough history before t0")
    lr = np.diff(np.log(prices[: t0 + 1]))          # lr[i] = log(S_{i+1}/S_i), uses prices up to t0
    sig = ewma_vol(lr, lam)                           # sig[i]: EWMA vol after observing lr[i] (time i+1)
    s = np.arange(t0 - window, t0 - h + 1)
    z = np.log(prices[s + h] / prices[s]) / (sig[s - 1] * np.sqrt(h))
    k = np.sqrt(h) * sig[t0 - 1]
    return float(np.exp(np.quantile(z, lower) * k)), float(np.exp(np.quantile(z, upper) * k))
