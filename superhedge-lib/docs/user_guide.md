# User guide

## 1. What the library computes

Consider one risky asset with price `S`, rebalancing dates `t = 0, …, M` and a European
claim paying `g(S_M)` at `M`. Every change of position from `φ_{t-1}` to `φ_t` shares costs
`κ · S_t · |φ_t − φ_{t-1}|`. The only assumption on prices is that each period's gross
return lies in a known range:

```
alpha_t <= S[t+1] / S[t] <= beta_t .
```

The **super-hedging price** is the smallest initial capital `V_0` for which there is a
self-financing strategy with `V_M >= g(S_M)` for *every* price path compatible with these
ranges. The library returns

* `price(S0)` — this capital (starting from cash);
* `hedge_ratio(S0)` — the number of shares to hold after trading at `t = 0`;
* `admissible_interval(t, wealth, S, phi_prev)` — all hedge ratios that keep the
  guarantee at date `t` for a given wealth (the interval is a single point when the wealth
  equals the minimal price, and widens with any excess capital);
* `hedge(path)` — the strategy run along a path: positions, wealth, costs, turnover, trades,
  and the terminal surplus `V_M − g(S_M)`.

Because the guarantee holds for all paths inside the ranges, the price is conservative by
construction. It is an upper bound for any arbitrage-free price of a model whose returns
stay in the ranges, and a natural *reserve* for an option sold under frictions.

## 2. Inputs

| Input | Meaning | Typical choice |
|---|---|---|
| `alphas`, `betas` | range of each period's gross return | calibrated from history (below) or stress scenarios |
| `kappa` | proportional cost rate (half spread + fees, as a fraction of traded value) | 0.0005–0.002 for liquid ETFs |
| payoff | convex piecewise-linear function | `payoffs.call(K)`, `payoffs.put(K)`, … or `ConvexPWL(x, y, left_slope, right_slope)` |
| `n_clusters` | number of retained internal states per step | 20 (fast) to 50 (accurate) |
| `reduction` | partition rule for the states | `"width"` (recommended) |
| `bound` | `None` (approximate) or `"upper"` (certified: price and hedge guaranteed) | `None` for pricing studies, `"upper"` for a guaranteed hedge |

**Payoffs must be convex.** Calls, puts, straddles, strangles, long ladders of calls/puts
and maxima of such payoffs are convex. Spreads and butterflies are *not*; for them the
method is not applicable as such. A conservative bound is obtained by super-hedging any
convex payoff above them on the reachable price range.

### Calibrating the ranges

```python
from superhedge.calibration import quantile_support, horizon_support

# weekly (h = 5 days) return range estimated on the last 252 days, 1%-99% quantiles
alpha, beta = quantile_support(daily_prices, t0, window=252, h=5, lower=0.01, upper=0.99)

# or spread a horizon range [alpha_T, beta_T] evenly over M periods
alphas, betas = horizon_support(0.95, 1.05, M=4)
```
`quantile_support` uses only prices up to index `t0`. `lower=0, upper=1` gives the
historical min/max (most conservative). Tighter quantiles give lower prices but the
guarantee then fails whenever a future return leaves the range (a *breach*).

The model requires that the range contains returns on both sides of 1 in a suitable sense
(absence of immediate profit). If a support lies entirely above or below 1 the price is
`-inf` and the library raises `AIPError`.

## 3. Accuracy and speed

Without state reduction the computation is exact up to floating point, but the number of states explodes after
three or four dates. Two ways to keep it small:

* `reduction="extremes"`: keeps only the two extreme states at each date. Keeping a subset of the exact states
  always gives a lower bound; with constant ranges and `liquidation=False` it reproduced the exact price in all
  our tests (up to M = 100), at a cost linear in M. Recommended for constant ranges.
* `reduction="width"` (default) with `n_clusters` states: exact for a few dates; with many dates and per-period
  moves smaller than the cost rate it overprices (several percent at M = 21). Use `price_bracket` to bound the
  error, or `bound="upper"` when the hedge itself must carry the guarantee.

Always validate a configuration once: `check_against_tree(S0)` (exact, constant ranges, any M) or
`check_against_lp(S0)` (exact, any ranges, M <= 12). See `examples/04_trading_dates_and_benchmark.py`.

## 4. Certified bounds, exact references, conventions

* `price_bracket(S0)` runs two certified reductions and returns `(lower, upper)`: the exact super-hedging price
  lies in between, for any `n_clusters`. The upper reduction replaces each block of states by two states
  (the block's extreme coefficients with the maximum of the block's functions); the lower reduction keeps a subset
  of the exact states. In our tests the lower bound was exact.
* `check_against_tree(S0)`: exact price by dynamic programming on the recombining binomial tree
  (constant ranges only; fast for large `M`). `check_against_lp(S0)`: exact tree LP (small `M`).
* `liquidation=True`: the final position is closed at cost `kappa |phi| S_T` (cash settlement). Default: the
  final position is valued at `S_T` without cost (physical delivery / position kept).
* `spot_range=(lo, hi)`: represent the functions exactly only on the spots reachable from `[lo, hi]`.
* `superhedge.backtest`: discounting at a constant rate, dividends paid to the hedge position, Leland
  delta-hedge benchmark (`leland_delta_hedge`).

## 5. Reading the output of `hedge`

| Key | Meaning |
|---|---|
| `V0`, `VT` | initial capital and terminal wealth |
| `phis`, `Vs` | position chosen at each date, wealth at each date |
| `intervals` | admissible interval at each date |
| `surplus` | `VT − payoff(S_M)`; non-negative whenever the path stays in the ranges |
| `cost`, `turnover`, `trades` | transaction costs paid, traded value, number of trades |
| `breaches` | number of periods whose realised return left the range |
| `empties` | number of dates where no admissible position existed (only after a breach) |

The default selection rule `select="project"` keeps the current position whenever it is
admissible and otherwise moves to the nearest admissible position. This minimises trading
at each date.

## 6. Lower-level API

```python
from superhedge import backward, hedge, interval, lp_superhedge, payoffs

res = backward(alphas, betas, kappa, payoffs.put(100.0), n_clusters=20)
res.price0(100.0)                 # price
res.value(t, phi, x)              # g_t(phi, x): minimal capital at date t, previous position phi, spot x
interval(res, t, phi_prev, wealth, S)
hedge(res, path)
res.stats                         # per-step statistics: states, candidates, breakpoints, timings
```

## 7. Limitations

* one risky asset, deterministic return ranges (possibly different at each date);
* convex payoffs only;
* European exercise;
* the approximate reduction has no proven error sign; use `price_bracket` (certified), `bound="upper"` or the exact references.
