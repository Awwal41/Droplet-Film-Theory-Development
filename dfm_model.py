"""Droplet-Film Model: physics Qcr + log-space residual, fit from well status only."""
import numpy as np
from scipy.optimize import differential_evolution, minimize

SEED = 42
FEATURES = [
    "Dia", "Dev(deg)", "Area (m2)", "z", "GasDens", "LiquidDens",
    "g (m/s2)", "P/T", "friction_factor", "critical_film_thickness",
]
LABEL = {"Loaded": -1, "Near L.U": 0, "Unloaded": 1}
NAMES = {v: k for k, v in LABEL.items()}
NEAR_BAND = 0.05
PB = [(-20, 20), (0, 3), (0, 3), (-10, 10), (0.1, 3), (-6, 6)]


def classify(m, band=NEAR_BAND):
    return np.where(m > band, 1, np.where(m < -band, -1, 0))


def as_loaded(s):
    """Near L.U (0 or 'Near L.U') counts as Loaded for every accuracy."""
    a = np.asarray(s)
    if a.dtype.kind in "SUO":
        return np.where(a == "Near L.U", "Loaded", a)
    return np.where(a == 0, -1, a)


def phys_terms(X, p):
    Dia, t, Ar, z, G, L, g, PT, ff, film = X.T
    lp1, p2, p3, lp4, p5 = p[:5]
    r = np.clip(Dia / film, 1e-8, 1e6)
    t1 = 2 * g * Dia * (1 - 3 * r + 4 * r * r) * (L - G) * np.cos(np.radians(t)) / (ff * G) * np.exp(lp4)
    t2 = np.abs(np.sin(np.radians(p5 * t))) ** p3 * np.abs(L - G) ** p2 / G ** 2
    return np.exp(lp1), t1, t2, PT / np.abs(z)


def qcr_phys(X, p, alpha=None):
    c, t1, t2, s = phys_terms(X, p)
    A = 1 / (1 + np.exp(-p[5])) if alpha is None else alpha
    return c * np.sqrt(np.maximum(np.abs(A * t1 + (1 - A) * t2) * s, 1e-12))


def film_share(X, p):
    c, t1, t2, s = phys_terms(X, p)
    A = 1 / (1 + np.exp(-p[5]))
    return np.abs(A * t1) / (np.abs(A * t1) + np.abs((1 - A) * t2) + 1e-300)


def logistic_loss(m, y):
    t = np.where(y == 1, 1.0, np.where(y == -1, 0.0, 0.5))
    return np.mean(np.logaddexp(0, m) - t * m)


def resid_feats(X):
    dev = np.radians(X[:, 1])
    return np.column_stack([
        np.log(np.abs(X[:, [0, 3, 4, 5, 7, 8, 9]]) + 1e-12),
        np.cos(dev), np.sin(dev),
    ])


class PhysicsOnly:
    """Physics Qcr only. alpha=None: DFM mix; 0: LDM; 1: LFM."""

    def __init__(self, alpha=None):
        self.alpha = alpha

    def fit(self, X, Q, y):
        B = PB if self.alpha is None else PB[:5]
        full = (lambda p: p) if self.alpha is None else (lambda p: np.r_[p, 0.0])
        self.p = full(differential_evolution(
            lambda p: logistic_loss(4 * np.log(Q / qcr_phys(X, full(p), self.alpha)), y),
            B, maxiter=200, popsize=15, seed=SEED, tol=1e-8,
        ).x)
        return self

    def qcr(self, X, Q=None):
        return qcr_phys(X, self.p, self.alpha)


class DFM:
    """Physics Qcr + learned log-space residual. Fit from well status, not from Qcr."""

    def __init__(self, lam=0.1, alpha=None):
        self.lam, self.alpha = lam, alpha

    def fit(self, X, Q, y):
        self.p = PhysicsOnly(self.alpha).fit(X, Q, y).p
        F = resid_feats(X)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-12
        F = (F - self.mu) / self.sd
        m0 = np.log(Q / qcr_phys(X, self.p, self.alpha))

        def obj(th):
            return logistic_loss(np.exp(th[0]) * (m0 - th[1] - F @ th[2:]), y) + self.lam * np.sum(th[2:] ** 2) / len(y)

        th = minimize(obj, np.r_[np.log(4), 0, np.zeros(F.shape[1])], method="L-BFGS-B").x
        self.b, self.w, self.k = th[1], th[2:], np.exp(th[0])
        return self

    def qcr(self, X, Q=None):
        return qcr_phys(X, self.p, self.alpha) * np.exp(
            self.b + (resid_feats(X) - self.mu) / self.sd @ self.w
        )

    def p_unloaded(self, X, Q):
        return 1 / (1 + np.exp(-self.k * np.log(Q / self.qcr(X))))
