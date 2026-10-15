"""Exact algebra of convex piecewise-linear (PWL) functions on the real line.

A function is stored as breakpoints x_1 < ... < x_n, values y_i = f(x_i) and the slopes of
the two rays (x < x_1 and x > x_n). Scaling composition, addition and pointwise maximum are
exact on this representation.
"""
import numpy as np

# Tolerance of the (conservative) breakpoint simplification. 0 disables it.
SIMPLIFY_TOL = 1e-6


class ConvexPWL:
    __slots__ = ("x", "y", "sl_left", "sl_right")

    def __init__(self, x, y, sl_left, sl_right, simplify=True):
        self.x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        self.y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        self.sl_left = float(sl_left)
        self.sl_right = float(sl_right)
        if len(self.x) > 1:
            # merge breakpoints that coincide up to round-off (they would create zero-length segments)
            gap = np.diff(self.x)
            dup = gap <= 1e-12 * np.maximum(1.0, np.abs(self.x[1:]))
            if np.any(dup):
                keep = np.concatenate([[True], ~dup])
                self.x, self.y = self.x[keep], self.y[keep]
        if simplify:
            self.simplify()

    # ------------------------------------------------------------------
    def simplify(self, tol=None):
        """Drop breakpoints where the slope changes by at most tol.

        For a convex function the result lies above the original (chord or ray
        replacement), so this is a conservative approximation.
        """
        tol = SIMPLIFY_TOL if tol is None else tol
        if len(self.x) == 0:
            return
        inner = np.diff(self.y) / np.diff(self.x) if len(self.x) > 1 else np.array([])
        slopes = np.concatenate([[self.sl_left], inner, [self.sl_right]])
        keep = np.abs(np.diff(slopes)) > tol
        if np.any(keep):
            self.x, self.y = self.x[keep], self.y[keep]
        else:  # globally affine: keep one anchor point
            self.x, self.y = self.x[:1], self.y[:1]

    def evaluate(self, tx):
        tx = np.atleast_1d(np.asarray(tx, dtype=np.float64))
        res = np.empty_like(tx)
        x, y = self.x, self.y
        lm, rm = tx < x[0], tx > x[-1]
        mm = ~(lm | rm)
        res[lm] = y[0] + self.sl_left * (tx[lm] - x[0])
        res[rm] = y[-1] + self.sl_right * (tx[rm] - x[-1])
        if np.any(mm):
            if len(x) > 1:
                idx = np.clip(np.searchsorted(x, tx[mm]) - 1, 0, len(x) - 2)
                sl = np.diff(y) / np.diff(x)
                res[mm] = y[idx] + sl[idx] * (tx[mm] - x[idx])
            else:
                res[mm] = y[0]
        return res

    def __call__(self, tx):
        return self.evaluate(tx)

    # ------------------------------------------------------------------
    def scaled(self, u, lam):
        """x -> lam * f(u x), u > 0, lam >= 0 (exact)."""
        return ConvexPWL(self.x / u, self.y * lam, self.sl_left * u * lam, self.sl_right * u * lam,
                         simplify=False)

    def add_linear(self, a):
        """x -> f(x) + a x (exact)."""
        return ConvexPWL(self.x, self.y + a * self.x, self.sl_left + a, self.sl_right + a, simplify=False)

    def __add__(self, other):
        xs = np.unique(np.concatenate([self.x, other.x]))
        return ConvexPWL(xs, self.evaluate(xs) + other.evaluate(xs),
                         self.sl_left + other.sl_left, self.sl_right + other.sl_right)

    def pointwise_max(self, other):
        """Exact max of two PWL functions: union of breakpoints, crossings, ray intersections."""
        xu = np.unique(np.concatenate([self.x, other.x]))
        yf, yg = self.evaluate(xu), other.evaluate(xu)
        extra = []
        for side in ("left", "right"):
            if side == "left":
                s1, s2, p1, p2 = self.sl_left, other.sl_left, (self.x[0], self.y[0]), (other.x[0], other.y[0])
            else:
                s1, s2, p1, p2 = self.sl_right, other.sl_right, (self.x[-1], self.y[-1]), (other.x[-1], other.y[-1])
            if abs(s1 - s2) > 1e-12:
                xi = (p2[1] - s2 * p2[0] - p1[1] + s1 * p1[0]) / (s1 - s2)
                if (side == "left" and xi < xu[0]) or (side == "right" and xi > xu[-1]):
                    extra.append(xi)
        d = yf - yg
        for i in np.where(d[:-1] * d[1:] < 0)[0]:
            extra.append(xu[i] - d[i] * (xu[i + 1] - xu[i]) / (d[i + 1] - d[i]))
        xs = np.unique(np.concatenate([xu, extra])) if extra else xu
        return ConvexPWL(xs, np.maximum(self.evaluate(xs), other.evaluate(xs)),
                         min(self.sl_left, other.sl_left), max(self.sl_right, other.sl_right))

    @staticmethod
    def max_of(funcs):
        """Divide-and-conquer pointwise maximum."""
        if len(funcs) == 1:
            return funcs[0]
        mid = len(funcs) // 2
        return ConvexPWL.max_of(funcs[:mid]).pointwise_max(ConvexPWL.max_of(funcs[mid:]))

    def slope_at(self, x0, side="right"):
        """One-sided derivative f'(x0+) (side='right') or f'(x0-) (side='left')."""
        x, y = self.x, self.y
        if len(x) == 1:
            return self.sl_right if (x0 > x[0] or (x0 == x[0] and side == "right")) else self.sl_left
        if x0 < x[0] or (x0 == x[0] and side == "left"):
            return self.sl_left
        if x0 > x[-1] or (x0 == x[-1] and side == "right"):
            return self.sl_right
        sl = np.diff(y) / np.diff(x)
        i = np.searchsorted(x, x0, side="right" if side == "right" else "left") - 1
        return float(sl[min(max(i, 0), len(sl) - 1)])

    def restrict(self, lo=0.0, hi=np.inf):
        """Function equal to f on [lo, hi] and extended linearly outside (exact on [lo, hi]).

        Breakpoints outside [lo, hi] are removed; the rays continue the slopes of f at lo+ and hi-.
        Used to discard structure at spots that can never occur (x <= 0 or outside the reachable range).
        """
        inside = (self.x > lo) & (self.x < hi)
        if inside.all():
            return self
        if not inside.any():
            x0 = lo if np.isfinite(lo) else hi
            if not np.isfinite(x0):
                return self
            sl_l, sl_r = self.slope_at(x0, "right"), self.slope_at(hi, "left") if np.isfinite(hi) else self.sl_right
            return ConvexPWL([x0], self.evaluate(x0), sl_l if np.isfinite(lo) else self.sl_left, sl_r, simplify=False)
        sl_l = self.slope_at(lo, "right") if np.isfinite(lo) else self.sl_left
        sl_r = self.slope_at(hi, "left") if np.isfinite(hi) else self.sl_right
        return ConvexPWL(self.x[inside], self.y[inside], sl_l, sl_r, simplify=False)

    @property
    def n_breakpoints(self):
        return len(self.x)
