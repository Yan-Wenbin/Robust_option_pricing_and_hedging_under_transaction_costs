import itertools
import numpy as np
from superhedge import backward, hedge, payoffs


def test_superhedge_on_all_paths():
    a, b, kappa, M = 0.98, 1.02, 0.005, 4
    res = backward([a] * M, [b] * M, kappa, payoffs.straddle(100.0), n_clusters=20)
    rng = np.random.default_rng(3)
    paths = [100 * np.cumprod([1.0] + list(r)) for r in itertools.product([a, b], repeat=M)]
    paths += [100 * np.cumprod(np.concatenate([[1.0], rng.uniform(a, b, M)])) for _ in range(200)]
    for p in paths:
        out = hedge(res, p, payoff=lambda x: abs(x - 100.0))
        assert out["surplus"] > -1e-9 and out["empties"] == 0 and out["breaches"] == 0


def test_self_financing_equation():
    res = backward([0.98] * 3, [1.02] * 3, 0.005, payoffs.call(100.0), n_clusters=20)
    path = np.array([100.0, 102.0, 99.96, 101.9592])
    out = hedge(res, path)
    V, prev = out["V0"], 0.0
    for t, phi in enumerate(out["phis"]):
        V = V + phi * (path[t + 1] - path[t]) - 0.005 * path[t] * abs(phi - prev)
        prev = phi
    assert abs(V - out["VT"]) < 1e-12
