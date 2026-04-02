import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List, Callable, Dict, Any


class DFT():
    """
    Droplet-Film Model (DFM) cast as an ordinal classifier.

    Instead of regressing on Qcr_true (which may be unavailable), this model:
      1. Predicts Qcr via the physics equation (_eq).
      2. Computes a log-ratio score:  ratio = log(Qcr_pred / Qg)
         • ratio >> 0  →  well is loaded   (Qcr >> Qg)
         • ratio ~  0  →  near loading condition
         • ratio << 0  →  well is unloaded (Qcr << Qg)
      3. Maps the continuous score to three ordinal classes
         (0=Unloaded, 1=Near L.U., 2=Loaded) via two learned
         threshold parameters τ₁ < τ₂:
             P(y=0 | ratio) = σ(τ₁ − ratio)
             P(y=1 | ratio) = σ(τ₂ − ratio) − σ(τ₁ − ratio)
             P(y=2 | ratio) = 1 − σ(τ₂ − ratio)
      4. Optimises all DFM parameters (p1–p5, α per sample) together
         with τ₁ and τ₂ by minimising the ordinal cross-entropy loss.

    Optimised parameter layout (self.opt_params):
        [p1, p2, p3, p4, p5, τ₁, τ₂, α_0, α_1, ..., α_{n-1}]

    Args
    ----
    seed : int
    feature_tol : float  – distance threshold for full-feature α matching
    dev_tol     : float  – tolerance for Dev(deg) matching
    multiple_dev_policy : str – 'max' | 'min' | 'mean' | 'median'
    """

    def __init__(
            self,
            seed: int = 42,
            feature_tol: float = 1.0,
            dev_tol: float = 1e-3,
            multiple_dev_policy: str = "max",
    ):
        self.seed = seed
        self.feature_tol = feature_tol
        self.dev_tol = dev_tol
        self.multiple_dev_policy = multiple_dev_policy

    # ------------------------------------------------------------------
    # Physics equation  (unchanged)
    # ------------------------------------------------------------------
    def _eq(self, params, X):
        """
        Compute predicted Qcr from physics parameters.

        Parameters
        ----------
        params : 1-D array  [p1, p2, p3, p4, p5, α_0, …, α_{n-1}]
        X      : (n, 10) feature matrix
        """
        (
            Dia, Dev_deg, Area_m2, z,
            GasDens, LiquidDens, g_m_s2,
            P_T, friction_factor, critical_film_thickness,
        ) = X.T

        z                       = np.maximum(z,                       1e-8)
        GasDens                 = np.maximum(GasDens,                 1e-8)
        friction_factor         = np.maximum(friction_factor,         1e-8)
        critical_film_thickness = np.maximum(critical_film_thickness, 1e-8)

        p1, p2, p3, p4, p5 = params[:5]
        alpha = params[5:]

        term1 = (
            2 * g_m_s2 * Dia
            * (1 - 3 * (Dia / critical_film_thickness)
               + 3 * (Dia / critical_film_thickness) ** 2)
            * (LiquidDens - GasDens) * np.cos(Dev_deg)
            / (friction_factor * GasDens)
        ) * p4

        term2 = (
            np.abs(np.sin(p5 * Dev_deg)) ** p3
            * (LiquidDens - GasDens) ** p2
            / GasDens ** 2
        )

        Qcr = p1 * np.sqrt(
            np.abs((term1 * alpha + (1 - alpha) * term2))
            * ((1 / z) * P_T)
        )
        return Qcr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _class_probs(self, ratio, tau1, tau2):
        """
        Ordinal logistic probabilities for 3 classes.

        Returns (P0, P1, P2) each of shape (n,).
        """
        eps  = 1e-15
        P_le0 = self._sigmoid(tau1 - ratio)          # P(y ≤ 0)
        P_le1 = self._sigmoid(tau2 - ratio)          # P(y ≤ 1)
        P0 = np.clip(P_le0,          eps, 1 - eps)
        P1 = np.clip(P_le1 - P_le0, eps, 1 - eps)
        P2 = np.clip(1.0 - P_le1,   eps, 1 - eps)
        return P0, P1, P2

    # ------------------------------------------------------------------
    # Ordinal cross-entropy loss
    # ------------------------------------------------------------------
    def _loss(self, params) -> float:
        """
        Ordinal cross-entropy loss (no Qcr_true required).

        params layout: [p1, p2, p3, p4, p5, τ₁, τ₂, α_0, …, α_{n-1}]
        """
        p_phys = params[:5]
        tau1, tau2 = params[5], params[6]
        alpha  = params[7:]

        # --- physics prediction ---
        Qcr_pred = self._eq(
            params=np.concatenate([p_phys, alpha]),
            X=self.X_train,
        )
        Qcr_pred = np.maximum(Qcr_pred, 1e-8)
        gsflow   = np.maximum(self.gsflow_train, 1e-8)

        ratio = np.log(Qcr_pred / gsflow)

        # --- ordinal class probabilities ---
        P0, P1, P2 = self._class_probs(ratio, tau1, tau2)

        # --- cross-entropy (summed over correct class per sample) ---
        mask0 = (self.loading_train == 0)
        mask1 = (self.loading_train == 1)
        mask2 = (self.loading_train == 2)

        log_lik = (
            np.sum(np.log(P0[mask0]))
            + np.sum(np.log(P1[mask1]))
            + np.sum(np.log(P2[mask2]))
        )

        # --- ordering penalty: enforce τ₂ > τ₁ ---
        penalty = max(0.0, (tau1 - tau2) + 0.1) * 1e3

        return -log_lik / len(self.loading_train) + penalty

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
            self,
            X: np.ndarray,
            gsflow: np.ndarray,
            loading: np.ndarray,
    ):
        """
        Fit DFM parameters + ordinal thresholds via cross-entropy minimisation.

        Parameters
        ----------
        X       : (n, 10) feature matrix (same column order as _eq)
        gsflow  : (n,)   actual gas flow rates  [Qg]
        loading : (n,)   integer class labels  0=Unloaded, 1=Near L.U., 2=Loaded
        """
        self.X_train       = X
        self.gsflow_train  = np.asarray(gsflow, dtype=float)
        self.loading_train = np.asarray(loading, dtype=int)

        n_train = len(loading)

        # Initial guess: [p1–p5, τ₁, τ₂, α×n]
        # τ₁=-0.5 (below 0 in log-ratio → boundary between unloaded and near),
        # τ₂= 0.5 (above 0 → boundary between near and loaded)
        x0 = np.concatenate([
            [1.0, 1.0, 0.5, 1.0, 1.0],   # p1–p5
            [-0.5, 0.5],                   # τ₁, τ₂
            np.full(n_train, 0.5),         # α per sample
        ])

        bounds = (
            [(None, None)] * 5    # p1–p5: unconstrained
            + [(None, None)] * 2  # τ₁, τ₂: unconstrained (penalty enforces order)
            + [(0.0, 1.0)] * n_train  # α ∈ [0, 1]
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                self._loss,
                x0=x0,
                bounds=bounds,
                method="Powell",
                options={"disp": True, "ftol": 1e-8, "xtol": 1e-8, "maxfev": 100_000},
            )

        self.opt_params = result.x
        return self

    # ------------------------------------------------------------------
    # predict helpers (α assignment for new samples – unchanged logic)
    # ------------------------------------------------------------------
    def _assign_alpha(self, X: np.ndarray) -> np.ndarray:
        """
        Assign per-sample α for new data using the same dev-based strategy
        as the original implementation.
        """
        dev_train   = self.X_train[:, 1]
        alpha_train = self.opt_params[7:]          # now offset by 2 (τ₁, τ₂)
        alpha_used  = []

        for i in range(len(X)):
            d_new = X[i, 1]

            if d_new < 10:
                match_idx = np.where(np.abs(dev_train - d_new) <= self.dev_tol)[0]
                if len(match_idx) == 0:
                    alpha_used.append(np.mean(alpha_train))
                elif len(match_idx) == 1:
                    alpha_used.append(alpha_train[match_idx[0]])
                else:
                    alphas = alpha_train[match_idx]
                    policy = self.multiple_dev_policy
                    alpha_used.append(
                        np.max(alphas)    if policy == 'max'    else
                        np.min(alphas)    if policy == 'min'    else
                        np.mean(alphas)   if policy == 'mean'   else
                        np.median(alphas) if policy == 'median' else
                        (_ for _ in ()).throw(ValueError("Invalid multiple_dev_policy."))
                    )
            elif 10 <= d_new < 20:
                match_idx = np.where(np.abs(dev_train - d_new) <= self.dev_tol)[0]
                if len(match_idx) == 0:
                    alpha_used.append(np.mean(alpha_train))
                else:
                    alpha_used.append(np.min(alpha_train[match_idx]))
            else:
                distances = np.linalg.norm(self.X_train - X[i, :], axis=1)
                if np.min(distances) < self.feature_tol:
                    alpha_used.append(alpha_train[np.argmin(distances)])
                else:
                    alpha_used.append(np.mean(alpha_train))

        return np.array(alpha_used)

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------
    def predict_proba(
            self,
            X: np.ndarray,
            gsflow: np.ndarray,
    ) -> np.ndarray:
        """
        Return class probabilities (n, 3) for classes [0, 1, 2].

        Parameters
        ----------
        X      : (n, 10) feature matrix
        gsflow : (n,)   actual gas flow rates for the new samples
        """
        p_phys     = self.opt_params[:5]
        tau1, tau2 = self.opt_params[5], self.opt_params[6]
        alpha_used = self._assign_alpha(X)

        Qcr_pred = self._eq(
            params=np.concatenate([p_phys, alpha_used]),
            X=X,
        )
        Qcr_pred = np.maximum(Qcr_pred, 1e-8)
        gsflow   = np.maximum(np.asarray(gsflow, dtype=float), 1e-8)

        ratio = np.log(Qcr_pred / gsflow)
        P0, P1, P2 = self._class_probs(ratio, tau1, tau2)
        return np.column_stack([P0, P1, P2])

    # ------------------------------------------------------------------
    # predict  (hard class labels)
    # ------------------------------------------------------------------
    def predict(
            self,
            X: np.ndarray,
            gsflow: np.ndarray,
    ) -> np.ndarray:
        """
        Predict loading class labels (0=Unloaded, 1=Near L.U., 2=Loaded).

        Parameters
        ----------
        X      : (n, 10) feature matrix
        gsflow : (n,)   actual gas flow rates for the new samples
        """
        proba = self.predict_proba(X, gsflow)
        return np.argmax(proba, axis=1)

    # ------------------------------------------------------------------
    # predict_qcr  (retain access to raw physics output)
    # ------------------------------------------------------------------
    def predict_qcr(self, X: np.ndarray) -> np.ndarray:
        """
        Return the raw physics-predicted Qcr values (no classification).
        Useful for diagnostics and plotting.
        """
        p_phys     = self.opt_params[:5]
        alpha_used = self._assign_alpha(X)
        return self._eq(
            params=np.concatenate([p_phys, alpha_used]),
            X=X,
        )