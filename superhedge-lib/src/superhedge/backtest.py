"""Out-of-sample backtest of one European contract with market conventions.

Conventions:
* the claim is written on the (unadjusted) price S; the hedger holds shares and receives their
  dividends as cash on ex-dates;
* cash earns the risk-free rate r, fixed at the T-bill yield observed at inception and constant over
  the contract; everything is expressed in discounted units (numeraire B_t = exp(r tau_t)), so that the
  model of the package applies to the discounted price S_t / B_t with the discounted strike K / B_T;
* transaction costs kappa S_t |Delta phi| at each rebalancing date; optionally the terminal position is
  liquidated at cost kappa |phi| S_T (cash settlement).
"""
import numpy as np
from math import erf, exp, log, sqrt

from .recursion import backward
from .forward import hedge
from .pwl import ConvexPWL

__all__ = ["discounted_path", "superhedge_contract", "bs_price", "bs_delta", "leland_delta_hedge"]

_N = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))


def discounted_path(S, D, idx, r, days_per_year=252.0):
    """Discounted prices at the rebalancing indices and discounted dividends paid in each period.

    S, D: daily prices and dividends; idx: rebalancing indices t0 = idx[0] < ... < idx[-1] = maturity.
    Returns (X, div) with X[k] = S[idx[k]] exp(-r tau_k) and div[k] = sum of discounted dividends
    with ex-date in (idx[k], idx[k+1]].
    """
    t0 = idx[0]
    X = np.array([S[i] * exp(-r * (i - t0) / days_per_year) for i in idx])
    div = np.array([sum(D[d] * exp(-r * (d - t0) / days_per_year) for d in range(idx[k] + 1, idx[k + 1] + 1))
                    for k in range(len(idx) - 1)])
    return X, div


def superhedge_contract(X, div, K_disc, alphas, betas, kappa, payoff="call", n_clusters=20, liquidation=False,
                        select="project", res=None):
    """Price at inception and run the super-hedge along the discounted path X (len M + 1).

    Returns (backward result, hedge dict); the surplus includes dividends received on the position.
    """
    g = {"call": ConvexPWL([K_disc], [0.0], 0.0, 1.0), "put": ConvexPWL([K_disc], [0.0], -1.0, 0.0),
         "straddle": ConvexPWL([K_disc], [0.0], -1.0, 1.0)}[payoff]
    if res is None:
        res = backward(alphas, betas, kappa, g, n_clusters=n_clusters, liquidation=liquidation)
    out = hedge(res, X, select=select, payoff=lambda x: float(g(x)[0]))
    dividends = float(np.dot(out["phis"], div))
    out["dividends"] = dividends
    out["surplus"] += dividends
    return res, out


def bs_price(S, K, T, sigma, r=0.0, q=0.0, kind="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T)); d2 = d1 - sigma * sqrt(T)
    c = S * exp(-q * T) * _N(d1) - K * exp(-r * T) * _N(d2)
    return c if kind == "call" else c - S * exp(-q * T) + K * exp(-r * T)


def bs_delta(S, K, T, sigma, r=0.0, q=0.0, kind="call"):
    if T <= 0 or sigma <= 0:
        return float(S > K) if kind == "call" else -float(S < K)
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    return exp(-q * T) * (_N(d1) if kind == "call" else _N(d1) - 1.0)


def leland_delta_hedge(X, div, K_disc, T_years, sigma, kappa, dt_years, kind="call", liquidation=False):
    """Leland (1985) delta hedge in discounted units: charge the BS price at the Leland volatility
    sigma_L^2 = sigma^2 (1 + sqrt(2/pi) 2 kappa / (sigma sqrt(dt))), rebalance to the Leland delta."""
    sL = sigma * sqrt(1.0 + sqrt(2.0 / np.pi) * 2.0 * kappa / (sigma * sqrt(dt_years)))
    M = len(X) - 1
    kinds = ["call", "put"] if kind == "straddle" else [kind]
    price = sum(bs_price(X[0], K_disc, T_years, sL, kind=k) for k in kinds)
    V, phi_prev, cost = price, 0.0, 0.0
    for t in range(M):
        tau = T_years - t * dt_years
        phi = sum(bs_delta(X[t], K_disc, tau, sL, kind=k) for k in kinds)
        c = kappa * X[t] * abs(phi - phi_prev); cost += c
        V += phi * (X[t + 1] - X[t]) - c + phi * div[t]
        phi_prev = phi
    liq = kappa * abs(phi_prev) * X[-1] if liquidation else 0.0
    pay = sum(max(X[-1] - K_disc, 0.0) if k == "call" else max(K_disc - X[-1], 0.0) for k in kinds)
    return dict(V0=price, VT=V, surplus=V - liq - pay, cost=cost, sigma_leland=sL)
