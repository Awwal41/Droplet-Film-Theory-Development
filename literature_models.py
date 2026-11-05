"""Published critical-rate models, evaluated with the same wells and scoring as the DFM.

Velocities are in m/s. Standard rates use the paper conversion
Q [m3/day] = 2.5e8 * v * A * (P/T) / z, which matches Gasflowrate / Test Vg.

Surface tension is 0.06 N/m when water is present, otherwise 0.02 N/m (condensate).
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from dfm_model import SEED, logistic_loss

CONV = 2.5e8
SIG_WATER, SIG_COND = 0.06, 0.02
# Turner 20% (We=30, Cd=0.44); Li flattened-droplet prefactor 0.724 vs 1.593
C_TURNER = 1.2 * (4 * 30 / (3 * 0.44)) ** 0.25  # 3.557
C_LI = (0.724 / 1.593) * (4 * 30 / (3 * 0.44)) ** 0.25  # 1.405


def _sigma(df):
    water = df["Water presence"].to_numpy(float)
    return np.where(water > 0, SIG_WATER, SIG_COND)


def _q_from_v(v, df):
    return CONV * np.asarray(v) * df["Area (m2)"].to_numpy() * df["P/T"].to_numpy() / df["z"].to_numpy()


def _droplet_v(df, c):
    g = df["g (m/s2)"].to_numpy()
    rg, rl = df["GasDens"].to_numpy(), df["LiquidDens"].to_numpy()
    sig = _sigma(df)
    return c * (g * sig * np.maximum(rl - rg, 1e-12) / np.maximum(rg, 1e-12) ** 2) ** 0.25


def _film_v(df, delta):
    """Chen / Barnea film-reversal velocity. delta is film thickness (m)."""
    g = df["g (m/s2)"].to_numpy()
    D = df["Dia"].to_numpy()
    rg, rl = df["GasDens"].to_numpy(), df["LiquidDens"].to_numpy()
    fi = np.maximum(df["friction_factor"].to_numpy(), 1e-12)
    th = np.radians(df["Dev(deg)"].to_numpy())
    dbar = np.clip(delta / D, 0.0, 0.49)
    inner = 2 * g * delta * (1 - 3 * dbar + 3 * dbar ** 2) * np.maximum(rl - rg, 0) * np.cos(th) / (fi * np.maximum(rg, 1e-12))
    return np.sqrt(np.maximum(inner, 0.0))


def qcr_turner(df):
    return _q_from_v(_droplet_v(df, C_TURNER), df)


def qcr_li(df):
    return _q_from_v(_droplet_v(df, C_LI), df)


def qcr_chen(df):
    return _q_from_v(_film_v(df, df["film thickness"].to_numpy()), df)


def qcr_shekhar(df):
    """Shekhar-style film reversal using the thicker (bottom-of-pipe) film."""
    return _q_from_v(_film_v(df, df["critical_film_thickness"].to_numpy()), df)


def qcr_liu(df):
    """Liu et al.: film velocity with the Belfroid inclination correction."""
    v = _film_v(df, df["film thickness"].to_numpy())
    # Belfroid: theta from horizontal, degrees
    phi = 90.0 - df["Dev(deg)"].to_numpy()
    corr = np.sin(np.radians(1.7 * np.clip(phi, 1e-3, 90))) ** 0.38
    return _q_from_v(v * corr, df)


def _turner_beta_v(df, b1, b2):
    """Abhulimen/Turner simplified field form, returned in m/s. P from gas density (psia)."""
    rho_field = df["GasDens"].to_numpy() / 16.0185
    P = np.maximum(rho_field / 0.0031, 1.0)
    vw = b1 * np.maximum(67 - 0.0031 * P, 1e-6) ** 0.25 / np.sqrt(0.0031 * P)
    vc = b2 * np.maximum(45 - 0.0031 * P, 1e-6) ** 0.25 / np.sqrt(0.0031 * P)
    water = df["Water presence"].to_numpy(float) > 0
    return np.where(water, vw, vc) * 0.3048


def qcr_abhulimen(df, b1=None, b2=None):
    if b1 is None or b2 is None:
        b1, b2 = 5.321, 4.02  # Turner 20% water / condensate, before any refit
    return _q_from_v(_turner_beta_v(df, b1, b2), df)


def fit_abhulimen(df_train, Q_train, y_train):
    """Fit Abhulimen β1, β2 on training wells from status only (logistic loss)."""

    def obj(b):
        qcr = qcr_abhulimen(df_train, b[0], b[1])
        return logistic_loss(np.log(np.maximum(Q_train, 1e-12) / np.maximum(qcr, 1e-12)), y_train)

    b = differential_evolution(obj, [(0.5, 20.0), (0.5, 20.0)], seed=SEED, popsize=12, maxiter=80).x
    return float(b[0]), float(b[1])


MODELS = [
    ("Turner et al. (1969)", qcr_turner),
    ("Li et al. (2001)", qcr_li),
    ("Chen et al. (2016)", qcr_chen),
    ("Shekhar et al. (2017)", qcr_shekhar),
    ("Liu et al. (2018)", qcr_liu),
    ("Abhulimen et al. (2022)", qcr_abhulimen),
]


def literature_qcr(df, abhulimen_beta=None):
    out = pd.DataFrame({"Well no": df["Well no"].values})
    for name, fn in MODELS:
        if name.startswith("Abhulimen") and abhulimen_beta is not None:
            out[name] = fn(df, *abhulimen_beta)
        else:
            out[name] = fn(df)
    return out
