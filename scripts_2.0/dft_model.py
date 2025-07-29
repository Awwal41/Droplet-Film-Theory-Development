import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List, Callable, Dict, Any, List

class DFT():
    def __init__(
            self,
            seed: int=42,
    ):
        self.seed = seed
    
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
        return np.mean((self.y - y_pred) ** 2)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            x0: np.ndarray, #initial guess 
    ):
        self.X_train = X
        self.y_train = y

        bounds = [(None, None)] * 5 + [(0.0, 1.0)] * X.shape[0]
        self.opt_params = minimize(self._loss, x0=x0, bounds=bounds) #(f, x0)
        self.opt_params = self.opt_params.x
        
        return self 

    def predict(
            self, 
            X, 
            dev_train, 
            alpha_strategy: str='enhances_dev_based',
            multiple_dev_policy: str='max',
            dev_tol: float=1e-3,
            feature_tol: float=1.0,
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
        p_opt = self.opt_params[:5]
        alpha_train = self.opt_params[5:]
        alpha_used = []

        for i in range(len(X)):
            d_new = X[i, 1]  # Dev(deg) of new sample

            if d_new < 10:
                # For dev < 20: use matching based on Dev(deg) with specified policy
                match_idx = np.where(np.abs(dev_train - d_new) <= dev_tol)[0]
                if len(match_idx) == 0:
                    alpha_used.append(np.mean(alpha_train))
                elif len(match_idx) == 1:
                    alpha_used.append(alpha_train[match_idx[0]])
                else:
                    alphas = alpha_train[match_idx]
                    if multiple_dev_policy == 'max':
                        alpha_used.append(np.max(alphas))
                    elif multiple_dev_policy == 'min':
                        alpha_used.append(np.min(alphas))
                    elif multiple_dev_policy == 'mean':
                        alpha_used.append(np.mean(alphas))
                    elif multiple_dev_policy == 'median':
                        alpha_used.append(np.median(alphas))
                    else:
                        raise ValueError("Invalid multiple_dev_policy.")
            elif 10 <= d_new < 20:
                # For 20 <= dev < 30: use the minimum α among matching samples
                match_idx = np.where(np.abs(dev_train - d_new) <= dev_tol)[0]
                if len(match_idx) == 0:
                    alpha_used.append(np.mean(alpha_train))
                else:
                    alphas = alpha_train[match_idx]
                    alpha_used.append(np.min(alphas))
            else:  # d_new >= 30
                # For dev >= 30: use full-feature matching
                distances = np.linalg.norm(self.X_train - X[i, :], axis=1)
                min_dist = np.min(distances)
                if min_dist < feature_tol:
                    closest_idx = np.argmin(distances)
                    alpha_used.append(alpha_train[closest_idx])
                else:
                    alpha_used.append(np.mean(alpha_train))
        alpha_used = np.array(alpha_used)
        self.opt_params = np.concatenate((p_opt, alpha_used))

        return self._eq(self.opt_params, X)
        