"""Price and hedge an at-the-money call under transaction costs.

Model: 4 weekly rebalancing dates, each weekly return within [-2%, +2%], cost rate 20 bp.
"""
from superhedge import SuperHedger, payoffs

S0, K = 100.0, 100.0
sh = SuperHedger(alphas=[0.98] * 4, betas=[1.02] * 4, kappa=0.002)
sh.fit(payoffs.call(K))

print(f"super-hedging price      : {sh.price(S0):.4f}")
print(f"initial hedge ratio      : {sh.hedge_ratio(S0):.4f} shares")
print(f"check (tree LP)          : {sh.check_against_lp(S0):.4f}")

# Effect of transaction costs on the price
for kappa in [0.0, 0.001, 0.002, 0.005, 0.01]:
    p = SuperHedger([0.98] * 4, [1.02] * 4, kappa).fit(payoffs.call(K)).price(S0)
    print(f"kappa = {kappa:6.3f}  price = {p:.4f}  ({100 * p / S0:.2f}% of spot)")
