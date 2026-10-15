import numpy as np
from superhedge import ConvexPWL, payoffs

XS = np.linspace(-20, 20, 4001)


def random_convex(rng):
    n = rng.integers(1, 5)
    x = np.sort(rng.uniform(-10, 10, n))
    s = np.sort(rng.uniform(-3, 3, n + 1))
    y = [rng.uniform(-5, 5)]
    for i in range(1, n):
        y.append(y[-1] + s[i] * (x[i] - x[i - 1]))
    return ConvexPWL(x, y, s[0], s[-1], simplify=False)


def test_operations_are_exact():
    rng = np.random.default_rng(1)
    for _ in range(200):
        f, h = random_convex(rng), random_convex(rng)
        assert np.allclose(f.pointwise_max(h)(XS), np.maximum(f(XS), h(XS)), atol=1e-9)
        assert np.allclose((f + h)(XS), f(XS) + h(XS), atol=1e-9)
        assert np.allclose(f.scaled(1.3, 0.4)(XS), 0.4 * f(1.3 * XS), atol=1e-9)


def test_payoffs():
    x = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
    assert np.allclose(payoffs.call(100)(x), np.maximum(x - 100, 0))
    assert np.allclose(payoffs.put(100)(x), np.maximum(100 - x, 0))
    assert np.allclose(payoffs.straddle(100)(x), np.abs(x - 100))
    assert np.allclose(payoffs.strangle(95, 105)(x), np.maximum(95 - x, 0) + np.maximum(x - 105, 0))
    assert np.allclose(payoffs.call_ladder([95, 100])(x), np.maximum(x - 95, 0) + np.maximum(x - 100, 0))
    assert all(payoffs.is_convex(p) for p in (payoffs.call(1), payoffs.put(1), payoffs.max_of_calls(95, 102.5)))


def test_simplify_is_conservative():
    rng = np.random.default_rng(2)
    for _ in range(100):
        f = random_convex(rng)
        g = ConvexPWL(f.x, f.y, f.sl_left, f.sl_right, simplify=False)
        g.simplify(tol=0.5)
        assert np.all(g(XS) >= f(XS) - 1e-9)
