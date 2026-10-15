# Examples

| Script | What it shows |
|---|---|
| `01_quickstart.py` | price and hedge ratio of an at-the-money call; effect of the cost rate |
| `02_hedge_along_a_path.py` | the self-financing hedge along a price path; surplus on 1,000 random paths |
| `03_custom_payoff_and_validation.py` | a custom convex payoff, checked against the exact LP benchmark |
| `04_trading_dates_and_benchmark.py` | price versus number of trading dates (M = 1 to 100), library versus exact tree DP and LP, run times; writes `output/trading_dates.csv` and `output/trading_dates.png` |

Run from the repository root, e.g. `python examples/04_trading_dates_and_benchmark.py call` (or `straddle`).
