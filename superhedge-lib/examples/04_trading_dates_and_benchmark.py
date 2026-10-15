"""How does the super-hedging price behave as the number of trading dates grows, and how does the library compare
with exact benchmarks?

Setting: S0 = K = 100, at-the-money call (or straddle). The total return over the life of the option is assumed
to lie in [0.95, 1.05]; this range is split evenly over M rebalancing periods (alpha_t = 0.95**(1/M),
beta_t = 1.05**(1/M)). Refining the trading grid of a fixed model can only lower the price: with M' a multiple
of M, an M'-date trader can copy any M-date strategy.

Methods compared:
  * library, reduction="extremes" -- two-component recursion (fast, linear in M);
  * library, default reduction    -- equal-width reduction with n_clusters = 20;
  * tree DP -- exact dynamic programme on the recombining binomial tree (benchmark, any M);
  * tree LP -- exact linear program on the full scenario tree (benchmark, exponential in M; M <= 10 here).

Run:  python examples/04_trading_dates_and_benchmark.py [call|straddle]
Writes examples/output/trading_dates.csv and, if matplotlib is installed, examples/output/trading_dates.png.
"""
import csv
import os
import sys
import time

import numpy as np

from superhedge import SuperHedger, payoffs, tree_dp_superhedge, lp_superhedge

S0, K = 100.0, 100.0
A_T, B_T = 0.95, 1.05
KAPPAS = [0.0, 0.002, 0.005, 0.01]
M_ALL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 21, 30, 50, 75, 100]
M_DEFAULT_MAX = 21      # the default reduction is slower; run it up to daily rebalancing of a one-month option
M_LP_MAX = 10
N_CLUSTERS = 20

kind = sys.argv[1] if len(sys.argv) > 1 else "call"
g = payoffs.call(K) if kind == "call" else payoffs.straddle(K)
f = (lambda x: max(x - K, 0.0)) if kind == "call" else (lambda x: abs(x - K))
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(out_dir, exist_ok=True)


def timed(fn):
    t = time.perf_counter()
    v = fn()
    return v, time.perf_counter() - t


rows = []
print(f"{kind}, horizon range [{A_T}, {B_T}], prices in % of S0")
print(f"{'kappa':>6} {'M':>4} {'tree DP':>9} {'extremes':>9} {'err':>8} {'default':>9} {'err':>8} {'tree LP':>9} "
      f"{'t_ext':>7} {'t_def':>7} {'t_DP':>7} {'t_LP':>7}")
for kappa in KAPPAS:
    for M in M_ALL:
        a, b = A_T ** (1 / M), B_T ** (1 / M)
        (dp, _), t_dp = timed(lambda: tree_dp_superhedge(S0, a, b, M, kappa, f))
        ext, t_ext = timed(lambda: SuperHedger([a] * M, [b] * M, kappa, reduction="extremes").fit(g).price(S0))
        dflt = t_dflt = lp = t_lp = np.nan
        if M <= M_DEFAULT_MAX:
            dflt, t_dflt = timed(lambda: SuperHedger([a] * M, [b] * M, kappa, n_clusters=N_CLUSTERS).fit(g).price(S0))
        if M <= M_LP_MAX:
            (lp, _), t_lp = timed(lambda: lp_superhedge(S0, [np.array([a, b])] * M, kappa, f))
        e_ext, e_def = (ext - dp) / dp, (dflt - dp) / dp
        rows.append(dict(kappa=kappa, M=M, price_tree_dp=dp, price_extremes=ext, err_extremes=e_ext,
                         price_default=dflt, err_default=e_def, price_tree_lp=lp, time_extremes=t_ext,
                         time_default=t_dflt, time_tree_dp=t_dp, time_tree_lp=t_lp))
        print(f"{kappa:6.3f} {M:4d} {dp:9.4f} {ext:9.4f} {e_ext:8.1e} {dflt:9.4f} {e_def:8.1e} {lp:9.4f} "
              f"{t_ext:7.3f} {t_dflt:7.3f} {t_dp:7.4f} {t_lp:7.3f}")
    print()

with open(os.path.join(out_dir, "trading_dates.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# Refining a fixed model never raises the price (nested grids).
for kappa in KAPPAS:
    p = {r["M"]: r["price_tree_dp"] for r in rows if r["kappa"] == kappa}
    assert all(p[mm] <= p[m] + 1e-9 for m in p for mm in p if mm > m and mm % m == 0), "nested refinement raised the price"
fin = lambda key: [abs(r[key]) for r in rows if np.isfinite(r[key])]
lp_gap = max(abs(r["price_tree_lp"] - r["price_tree_dp"]) for r in rows if np.isfinite(r["price_tree_lp"]))
print("nested refinements never raise the price: OK")
print(f"largest relative error vs tree DP: extremes {max(fin('err_extremes')):.1e}, "
      f"default (n_clusters={N_CLUSTERS}) {max(fin('err_default')):.1e}")
print(f"largest |tree LP - tree DP|: {lp_gap:.1e}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
for kappa in KAPPAS:
    r = [x for x in rows if x["kappa"] == kappa]
    line, = ax[0].plot([x["M"] for x in r], [x["price_tree_dp"] for x in r], label=f"kappa = {kappa:g}")
    c = line.get_color()
    rd = [x for x in r if np.isfinite(x["price_default"])]
    ax[0].plot([x["M"] for x in r], [x["price_extremes"] for x in r], "o", mfc="none", color=c, ms=5)
    ax[0].plot([x["M"] for x in rd], [x["price_default"] for x in rd], "x", color=c, ms=5)
    ax[1].plot([x["M"] for x in r], [max(abs(x["err_extremes"]), 1e-16) for x in r], "o-", mfc="none", color=c, lw=0.8)
    ax[1].plot([x["M"] for x in rd], [max(abs(x["err_default"]), 1e-16) for x in rd], "x--", color=c, lw=0.8)
r = [x for x in rows if x["kappa"] == 0.005]
for key, lab, mk in (("time_extremes", "library, extremes", "o-"), ("time_default", "library, default", "x--"),
                     ("time_tree_dp", "tree DP", "s-"), ("time_tree_lp", "tree LP", "^-")):
    rr = [x for x in r if np.isfinite(x[key])]
    ax[2].plot([x["M"] for x in rr], [x[key] for x in rr], mk, label=lab, ms=4)
ax[0].set(xscale="log", xlabel="trading dates M", ylabel="price (% of S0)",
          title="Price: tree DP (lines), extremes (o), default (x)")
ax[1].set(xscale="log", yscale="log", xlabel="trading dates M", ylabel="|relative error|",
          title="Library vs exact tree DP (o extremes, x default)")
ax[2].set(xscale="log", yscale="log", xlabel="trading dates M", ylabel="seconds", title="Run time (kappa = 0.005)")
ax[0].legend(fontsize=8)
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "trading_dates.png"), dpi=150)
print("figure written to", os.path.join(out_dir, "trading_dates.png"))
