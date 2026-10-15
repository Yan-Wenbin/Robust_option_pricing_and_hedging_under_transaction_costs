"""Run the self-financing super-hedge along price paths and inspect the result."""
import numpy as np
from superhedge import SuperHedger, payoffs

S0, K, M = 100.0, 100.0, 4
sh = SuperHedger([0.98] * M, [1.02] * M, kappa=0.002).fit(payoffs.straddle(K))

path = [100.0, 101.5, 100.2, 98.9, 100.4]
out = sh.hedge(path)
print("date  spot     position  wealth")
for t in range(M):
    print(f"{t:>4}  {path[t]:7.2f}  {out['phis'][t]:8.4f}  {out['Vs'][t]:7.4f}")
print(f"{M:>4}  {path[M]:7.2f}            {out['VT']:7.4f}")
print(f"payoff {out['payoff']:.4f}, surplus {out['surplus']:.4f}, costs paid {out['cost']:.4f}, trades {out['trades']}")

# Worst case over random paths inside the support: the surplus is never negative
rng = np.random.default_rng(0)
surplus = [sh.hedge(S0 * np.cumprod(np.r_[1.0, rng.uniform(0.98, 1.02, M)]))["surplus"] for _ in range(1000)]
print(f"1000 random paths: min surplus {min(surplus):.2e}, mean surplus {np.mean(surplus):.4f}")
