# superhedge-lib

Jointly created, developed, and maintained by **Amal Omrani** and **Wenbin Yan**.

## Co-authors

- [Amal Omrani](https://github.com/Amal-Omrani)
- [Wenbin Yan](https://github.com/Yan-Wenbin)

**Super-hedging prices and hedging strategies for options under proportional transaction costs.**

You sell an option and hedge it by trading the underlying. Every trade costs a fraction `kappa` of the traded
value, and you do not trust any probability model for the returns. You only assume that each period's return
stays in a known range:

```
alpha_t  <=  S[t+1] / S[t]  <=  beta_t
```

`superhedge` computes the **smallest initial capital** from which a self-financing strategy covers the option
payoff in **every** scenario compatible with these ranges, net of transaction costs, together with the hedge
that achieves it. It implements the explicit construction of Lepinette and Omrani for convex European claims,
in which the minimal capital is a closed-form function of the spot price **and of the position you currently
hold**, and it returns the interval of *all* hedge ratios that keep the guarantee.

## Installation

```bash
git clone https://github.com/<user>/superhedge.git
cd superhedge
pip install -e .            # numpy and scipy only
pip install -e ".[examples,dev]"   # + matplotlib for the figures, pytest for the tests
```

## Quick start

```python
from superhedge import SuperHedger, payoffs

# 4 weekly rebalancing dates, weekly returns within [-2%, +2%], 20 bp transaction costs
sh = SuperHedger(alphas=[0.98] * 4, betas=[1.02] * 4, kappa=0.002)
sh.fit(payoffs.call(100.0))

sh.price(100.0)               # 1.7605 : minimal capital, starting from cash
sh.hedge_ratio(100.0)         # 0.5080 : shares to hold after trading at t = 0
sh.check_against_tree(100.0)  # 1.7605 : exact benchmark (binomial-tree dynamic programme)

out = sh.hedge([100, 101.5, 100.2, 98.9, 100.4])   # run the strategy along a price path
out["surplus"], out["cost"], out["phis"]
```

## Price versus number of trading dates, and comparison with benchmarks

[`examples/04_trading_dates_and_benchmark.py`](examples/04_trading_dates_and_benchmark.py) prices an at-the-money
call whose total return over the option's life lies in [0.95, 1.05], with the range split over M rebalancing
dates, for M = 1 to 100 and four cost rates. It compares the library with two exact benchmarks: a dynamic
programme on the binomial tree and a linear program on the full scenario tree.

![price versus trading dates](docs/trading_dates.png)

* **More trading dates lower the price**, since a trader with more dates can always copy a strategy with fewer
  dates. Without costs the price keeps falling (2.50% of the spot at M = 1, 0.63% at M = 10, 0.20% at M = 100).
  **With costs the gain flattens** once the move per period is smaller than the cost rate
  (at 1%: 3.00%, 1.92%, 1.79%). Rebalancing more often than that adds cost but hardly lowers the price.
* With `reduction="extremes"` the library **matches the exact benchmarks** (relative error below 10⁻⁷ up to
  M = 100) at a cost that grows linearly in M (0.13 s at M = 100); it becomes faster than the tree benchmark between M = 50 and M = 75.
* The default reduction (`n_clusters=20`) is exact for small M but **overprices** when there are many dates and
  the per-period move is below the cost rate (up to 22% at M = 21 with 1% costs). In that regime use
  `reduction="extremes"` or check with `check_against_tree`.

The same behaviour is checked automatically in [`tests/test_trading_dates.py`](tests/test_trading_dates.py).

## What you get

| Method | Returns |
|---|---|
| `price(S0, phi_prev=0)` | minimal super-hedging capital at the initial spot (and current position) |
| `hedge_ratio(S0)` | position to hold after trading at t = 0 |
| `admissible_interval(t, wealth, S, phi_prev)` | all positions that keep the guarantee for the given wealth |
| `price_function(t, phi, x)` | minimal capital at date t for any position phi and spot x |
| `hedge(path, select="project"/"center")` | positions, wealth, costs, turnover, trades and terminal surplus along a path |
| `price_bracket(S0)` | certified lower and upper bounds on the exact price |
| `check_against_tree(S0)`, `check_against_lp(S0)` | exact benchmarks (binomial-tree DP, scenario-tree LP) |

Options of `SuperHedger(alphas, betas, kappa, ...)`:

| Option | Meaning |
|---|---|
| `reduction="width"` (default), `n_clusters=20` | reduces the number of internal states to at most `n_clusters` per date |
| `reduction="extremes"` | keeps only the two extreme states: fast and, in all our tests with constant ranges and `liquidation=False`, exact; always a lower bound |
| `bound="upper"` | certified upper reduction: price and hedge are guaranteed to super-hedge |
| `liquidation=True` | the final position is closed at cost `kappa * |phi| * S_T` (cash settlement) |
| `spot_range=(lo, hi)` | keep the functions exact only on the spots reachable from `[lo, hi]` (faster) |

Payoffs must be **convex**: calls, puts, straddles, strangles, call ladders, maxima of calls, or any convex
piecewise-linear function `ConvexPWL(x, y, left_slope, right_slope)`. Ranges may differ from one period to the
next. Ranges can be estimated from a price history with `superhedge.calibration.quantile_support`.

## Which method for which task

* **One price, constant ranges:** `reduction="extremes"` or the tree benchmark `tree_dp_superhedge`; both exact,
  a fraction of a second even for M = 100.
* **Time-varying ranges:** the library (with `bound="upper"` for a guaranteed price), or
  `nonrecombining_dp_superhedge` (exact, cost doubles with each date; practical up to about 20 dates).
* **Hedging along a real path, intervals of admissible hedges, price as a function of spot and position:**
  the library (`hedge`, `admissible_interval`, `price_function`).

## Repository layout

```
src/superhedge/   the package: recursion (backward), forward (hedge), pwl (exact piecewise-linear algebra),
                  payoffs, calibration, benchmark (tree DP, scenario-tree LP), backtest (discounting,
                  dividends, Leland delta hedge), api (SuperHedger)
examples/         01 quick start, 02 hedge along a path, 03 custom payoff and validation,
                  04 price versus trading dates and benchmarks
tests/            pytest suite (run: pytest)
docs/             user guide and figure
```

## Licence

MIT. If you use the package, please cite it (see `CITATION.cff`).
