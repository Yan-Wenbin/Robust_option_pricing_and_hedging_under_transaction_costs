"""superhedge: super-hedging prices and strategies under proportional transaction costs.

Quick start
-----------
>>> from superhedge import SuperHedger, payoffs
>>> sh = SuperHedger(alphas=[0.98] * 4, betas=[1.02] * 4, kappa=0.002).fit(payoffs.call(100.0))
>>> round(sh.price(100.0), 4)
1.7605
"""
from .pwl import ConvexPWL
from . import payoffs, calibration, backtest
from .recursion import backward, BackwardResult, AIPError
from .forward import interval, hedge
from .benchmark import lp_superhedge, tree_dp_superhedge, nonrecombining_dp_superhedge
from .api import SuperHedger

__version__ = "1.2.0"
__all__ = ["ConvexPWL", "payoffs", "calibration", "backward", "BackwardResult", "AIPError",
           "interval", "hedge", "lp_superhedge", "tree_dp_superhedge", "nonrecombining_dp_superhedge", "backtest", "SuperHedger", "__version__"]
