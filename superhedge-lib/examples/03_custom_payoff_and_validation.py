"""Any convex piecewise-linear payoff, time-varying supports, and validation against the LP."""
import numpy as np
from superhedge import ConvexPWL, SuperHedger, payoffs

# Convex payoff given by breakpoints, values and the two outer slopes:
# 0 below 95, slope 0.5 up to 105, slope 1.5 above  ->  a "call spread ladder" with increasing slopes
g = ConvexPWL(x=[95.0, 105.0], y=[0.0, 5.0], sl_left=0.0, sl_right=1.5)
print("convex:", payoffs.is_convex(g))

alphas = [0.97, 0.99, 0.985]        # e.g. a volatile first period (event), calmer afterwards
betas = [1.03, 1.01, 1.015]
sh = SuperHedger(alphas, betas, kappa=0.003, n_clusters=50).fit(g)
print(f"price {sh.price(100.0):.6f}   LP {sh.check_against_lp(100.0):.6f}")
print(sh.summary(100.0))
