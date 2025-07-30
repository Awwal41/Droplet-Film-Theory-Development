import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List, Callable, Dict, Any, List

class DFT():
    """
    This class defines the Droplet-Film Theory model for predicitng well flow rates 

    Args 
    ____
    """
    def __init__(
            self,
            seed: int=42,
            feature_tol: float=1.0, 
            dev_tol: float=1e-3,
            multiple_dev_policy: str="max", 
    ):
        self.seed = seed
        self.feature_tol = feature_tol
        self.dev_tol = dev_tol
        self.multiple_dev_policy = multiple_dev_policy
    
    def _eq(
            self,
            params,
            X,
    ):
        """
        Computes the predicted Y using the custom equation.
        
        Parameters:
        - X: array with shape (n_samples, 10); features in order:
            [Dia, Dev(deg), Area (m2), z, GasDens, LiquidDens, g (m/s2), P/T, friction_factor, critical_film_thickness]
        - params: array-like parameters where the first 5 values are global parameters (p1-p5)
                    and the remaining values correspond to α (one per sample or predicted sample)
                    
        Returns:
        - Y_pred: predicted Y values (array)
        """

        (
            Dia,
            Dev_deg,
            Area_m2,
            z,
            GasDens,
            LiquidDens,
            g_m_s2,
            P_T,
            friction_factor, 
            critical_film_thickness,
        ) = X.T

        p1, p2, p3, p4, p5 = params[:5]
        alpha = params[5:]
      
        # Guard against division errors
        z = np.maximum(z, 1e-8)
        GasDens = np.maximum(GasDens, 1e-8)
        friction_factor = np.maximum(friction_factor, 1e-8)
        critical_film_thickness = np.maximum(critical_film_thickness, 1e-8)


        term1 = (2 * g_m_s2 * Dia * (1 - 3 * (Dia / critical_film_thickness) +
                3 * ((Dia / critical_film_thickness) ** 2)) *
                (LiquidDens - GasDens) * np.cos(Dev_deg) /
                (friction_factor * GasDens)) * p4

        term2 = np.abs(np.sin(p5 * Dev_deg)) ** p3 * ((LiquidDens - GasDens) ** p2 / (GasDens ** 2))

        Qcr = p1 * np.sqrt(np.abs((term1 * alpha + (1 - alpha) * term2)) * ((1 / z) * (P_T)))
        
        return Qcr

    def _loss(
            self,
            params,
    ) -> float:
        y_pred = self._eq(params=params, X=self.X_train)
        self.loss = np.mean((self.y_train - y_pred) ** 2)
        return self.loss

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ):
        self.X_train = X
        self.y_train = y

        n_train = len(self.y_train)
        x0 = np.concatenate(([1.0, 1.0, 0.5, 1.0, 1.0], np.full(n_train, 0.5)))
        bounds = [(None, None)] * 5 + [(0.0, 1.0)] * n_train
        result = minimize(self._loss, x0=x0, bounds=bounds, method="Powell",
                      options={'maxiter': 5000, 'maxfun': 10000, 'disp': True}) #(f, x0)
        
        if result.success:
            self.opt_params = result.x
        else:
            raise RuntimeError("Optimization failed: " + result.message)
        
        return self 

    def predict(
            self, 
            X, 
            dev_train: Optional[np.ndarray] = None,
            alpha_strategy: str='enhanced_dev_based',
    ):
        """
        Predicts Y for new input samples using learned parameters and an enhanced dev-based α assignment.
        
        Parameters:
        - X_new: New data array (n_samples, 10)
        - learned_params: Array containing p1-p5 and training α (length 5 + len(X_train))
        - dev_train: The Dev(deg) values from training data (array)
        - X_train: The full training feature array (needed for full-feature matching)
        - alpha_strategy: Must be 'enhanced_dev_based' in this implementation.
        - multiple_dev_policy: Policy to choose among multiple matching training samples when dev < 20
            (e.g., 'max', 'min', 'mean', or 'median')
        - dev_tol: Tolerance for matching Dev(deg) values.
        - feature_tol: Threshold distance for full-feature matching when Dev(deg) >= 30.
        
        For each new sample:
            - If Dev(deg) < 20:
                Uses regular dev-based matching: find training samples with similar Dev(deg)
                and, if multiple are found, applies the given policy.
            - If 20 <= Dev(deg) < 30:
                Uses training α from matching samples (within dev_tol) but selects the minimum α.
            - If Dev(deg) >= 30:
                Computes the Euclidean distance between the new sample and every training sample.
                If the closest distance is below feature_tol, uses that training sample's α;
                otherwise, falls back to the global mean of training α.
        
        Returns:
        - Y_pred: Predicted Y values (array)
        """
        dev_train: np.ndarray=self.X_train[:, 1]
        p_opt = self.opt_params[:5]
        alpha_train = self.opt_params[5:]
        alpha_used = []

        for i in range(len(X)):
            d_new = X[i, 1]  # Dev(deg) of new sample

            if d_new < 10:
                # For dev < 20: use matching based on Dev(deg) with specified policy
                match_idx = np.where(np.abs(dev_train - d_new) <= self.dev_tol)[0]
                if len(match_idx) == 0:
                    alpha_used.append(np.mean(alpha_train))
                elif len(match_idx) == 1:
                    alpha_used.append(alpha_train[match_idx[0]])
                else:
                    alphas = alpha_train[match_idx]
                    if self.multiple_dev_policy == 'max':
                        alpha_used.append(np.max(alphas))
                    elif self.multiple_dev_policy == 'min':
                        alpha_used.append(np.min(alphas))
                    elif self.multiple_dev_policy == 'mean':
                        alpha_used.append(np.mean(alphas))
                    elif self.multiple_dev_policy == 'median':
                        alpha_used.append(np.median(alphas))
                    else:
                        raise ValueError("Invalid multiple_dev_policy.")
            elif 10 <= d_new < 20:
                # For 20 <= dev < 30: use the minimum α among matching samples
                match_idx = np.where(np.abs(dev_train - d_new) <= self.dev_tol)[0]
                if len(match_idx) == 0:
                    alpha_used.append(np.mean(alpha_train))
                else:
                    alphas = alpha_train[match_idx]
                    alpha_used.append(np.min(alphas))
            else:  # d_new >= 30
                # For dev >= 30: use full-feature matching
                distances = np.linalg.norm(self.X_train - X[i, :], axis=1)
                min_dist = np.min(distances)
                if min_dist < self.feature_tol:
                    closest_idx = np.argmin(distances)
                    alpha_used.append(alpha_train[closest_idx])
                else:
                    alpha_used.append(np.mean(alpha_train))
        alpha_used = np.array(alpha_used)
        self.opt_params = np.concatenate((p_opt, alpha_used))

        return self._eq(self.opt_params, X)
        