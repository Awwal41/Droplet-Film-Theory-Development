import pandas as pd
import numpy as np
import sys

from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Callable, Dict, Any, List
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from pysindy import SINDy
import feyn
import pandas as pd

        
class QLatticeWrapper():
    def __init__(
            self, 
            feature_tags: List,
            output_tag: int="Qcr",
            seed: int=42, 
            max_complexity: int=10, 
            n_epochs: int=10,
            criterion: str="bic"
    ): 
        super().__init__()

        self.seed = seed
        self.feature_tags = feature_tags
        self.output_tag = output_tag
        self.max_complexity = max_complexity
        self.n_epochs = n_epochs
        self.criterion = criterion

        # connect QLattice
        self.ql = feyn.connect_qlattice()
        self.opt_model = None
        self.y_pred = None

    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray,
    ) -> None:
        
        # QLattice requires python dataframes 
        data = pd.DataFrame(X, columns=self.feature_tags)
        data[self.output_tag] = y

        models = self.ql.auto_run(
            data=data, 
            output_name=self.output_tag,
            criterion=self.criterion, 
            max_complexity=self.max_complexity,
            n_epochs=self.n_epochs,
        )

        self.opt_model = models[0]

        return self
    
    def predict(
            self, 
            X: np.ndarray, 
    ) -> np.ndarray:
        # QLattice requires python dataframes 
        data = pd.DataFrame(X, columns=self.feature_tags)

        self.y_pred = self.opt_model.predict(data[self.feature_tags])

        return self.y_pred
    
    def express(
            self,
    ):
        return self.opt_model.sympify()
    



class Helm:
    """
    Handles dataset splitting, scaling, k-fold CV,
    hyperparameter tuning, regression modeling,
    and physics-based well-status classification.
    """

    def __init__(
        self,
        path: str,
        seed: int = 42,
        drop_cols: Optional[List[str]] = None,
        includ_cols: Optional[List[str]] = None,
        test_size: float = 0.20,
        scale: bool = True,
    ):
        self.path = path
        self.seed = seed
        self.scale = scale

        # =========================
        # LOAD DATA
        # =========================
        df = pd.read_csv(self.path)

        if drop_cols is None:
            drop_cols = []
            self.X = df[includ_cols]
        else:
            self.X = df.drop(columns=drop_cols)

        self.feature_names = self.X.columns.tolist()

        self.y = df["Qcr"]
        self.gsflow = df["Gasflowrate"]
        self.df = df

        # Encode loading state
        self.loading = df["Test status"].apply(
            lambda x: -1 if x == "Unloaded" else (0 if x == "Near L.U" else 1)
        ).to_numpy()

        # =========================
        # TRAIN / TEST SPLIT
        # =========================
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.gsflow_train,
            self.gsflow_test,
            self.loading_train,
            self.loading_test,
        ) = train_test_split(
            self.X,
            self.y,
            self.gsflow,
            self.loading,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.loading,
        )

        # =========================
        # SCALING
        # =========================
        if self.scale:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

            self.X_train_rdy = self.scaler_X.fit_transform(self.X_train)
            self.X_test_rdy = self.scaler_X.transform(self.X_test)

            self.y_train_rdy = self.scaler_y.fit_transform(
                self.y_train.values.reshape(-1, 1)
            )
            self.y_test_rdy = self.scaler_y.transform(
                self.y_test.values.reshape(-1, 1)
            )
        else:
            self.X_train_rdy = np.array(self.X_train)
            self.X_test_rdy = np.array(self.X_test)
            self.y_train_rdy = self.y_train.values
            self.y_test_rdy = self.y_test.values

        # Convert to numpy
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train).flatten()
        self.y_test = np.array(self.y_test).flatten()
        self.loading_train = np.array(self.loading_train)
        self.loading_test = np.array(self.loading_test)
        self.gsflow_train = np.array(self.gsflow_train)
        self.gsflow_test = np.array(self.gsflow_test)

    # =====================================================
    # REGRESSION METRICS
    # =====================================================
    def regression_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================
    def classification_scores(
        self,
        y_pred: np.ndarray,
        gsflow: np.ndarray,
        loading: np.ndarray,
        interval: float = 0.01,
    ):
        loading_pred = np.where(
            y_pred > gsflow + interval,
            1,
            np.where(y_pred < gsflow - interval, -1, 0),
        )

        self.acc = accuracy_score(loading, loading_pred)
        self.cm = confusion_matrix(loading, loading_pred, labels=[-1, 0, 1])

        return self.acc

    # =====================================================
    # CROSS VALIDATION
    # =====================================================
    def _cross_val(self, model):
        kf = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=42
        )

        acc_scores = []

        for tr_idx, val_idx in kf.split(self.X_train, self.loading_train):
            X_tr, X_val = self.X_train[tr_idx], self.X_train[val_idx]
            y_tr, y_val = self.y_train[tr_idx], self.y_train[val_idx]
            gs_val = self.gsflow_train[val_idx]
            load_val = self.loading_train[val_idx]

            if self.scale:
                sx = StandardScaler()
                sy = StandardScaler()
                X_tr = sx.fit_transform(X_tr)
                X_val = sx.transform(X_val)
                y_tr_rdy = sy.fit_transform(y_tr.reshape(-1, 1))
            else:
                y_tr_rdy = y_tr

            if isinstance(model, SINDy):
                model.fit(X_tr, x_dot=y_tr_rdy)
            else:
                model.fit(X_tr, y_tr_rdy)

            y_val_pred = model.predict(X_val)
            if self.scale:
                y_val_pred = sy.inverse_transform(
                    y_val_pred.reshape(-1, 1)
                ).flatten()

            acc_scores.append(
                self.classification_scores(y_val_pred, gs_val, load_val)
            )

        return float(np.mean(acc_scores)), float(np.std(acc_scores))

    # =====================================================
    # TRAIN + EVALUATE MODEL
    # =====================================================
    def evolv_model(
        self,
        build_model: Callable[[Dict[str, Any]], Any],
        hparam_grid: Dict[str, List[Any]],
        k_folds: int = 5,
    ):
        # =========================
        # HYPERPARAMETER SEARCH
        # =========================
        if k_folds > 0:
            self.best_score = -np.inf
            self.best_params = None
            self.k_folds = k_folds

            print(
                "Training model and optimizing hyperparameters via k-fold CV...",
                file=sys.stderr,
            )

            keys = list(hparam_grid.keys())
            values = [hparam_grid[k] for k in keys]

            for combo in itertools.product(*values):
                hparams = dict(zip(keys, combo))
                model = build_model(hparams=hparams)
                score, score_std = self._cross_val(model)

                if score > self.best_score:
                    self.best_score = score
                    self.best_score_std = score_std
                    self.best_params = hparams

            print(
                "Retraining optimized model on full training set",
                file=sys.stderr,
            )
            self.model = build_model(hparams=self.best_params)
        else:
            self.model = build_model(hparams=hparam_grid)

        # =========================
        # FINAL TRAINING
        # =========================
        if isinstance(self.model, SINDy):
            self.model.fit(self.X_train_rdy, x_dot=self.y_train_rdy)
        else:
            self.model.fit(self.X_train_rdy, self.y_train_rdy)

        # =========================
        # PREDICTIONS
        # =========================
        self.y_train_pred = self.model.predict(self.X_train_rdy)
        self.y_test_pred = self.model.predict(self.X_test_rdy)

        if self.scale:
            self.y_train_pred = self.scaler_y.inverse_transform(
                self.y_train_pred.reshape(-1, 1)
            ).flatten()

            self.y_test_pred = self.scaler_y.inverse_transform(
                self.y_test_pred.reshape(-1, 1)
            ).flatten()

        # =========================
        # METRICS
        # =========================
        train_reg = self.regression_scores(self.y_train, self.y_train_pred)
        test_reg = self.regression_scores(self.y_test, self.y_test_pred)

        test_class_acc = self.classification_scores(
            self.y_test_pred,
            self.gsflow_test,
            self.loading_test,
        )

        # =========================
        # OUTPUT
        # =========================
        print(
            f"Best CV Classification Accuracy = {self.best_score:.4f} ± {self.best_score_std:.4f}",
            file=sys.stderr,
        )
        print("Best Hyperparameters:", self.best_params, file=sys.stderr)

        print(
            "Training Regression Metrics: "
            f"RMSE={train_reg['RMSE']:.4f}, "
            f"MAE={train_reg['MAE']:.4f}, "
            f"R2={train_reg['R2']:.4f}",
            file=sys.stderr,
        )

        print(
            "Test Regression Metrics: "
            f"RMSE={test_reg['RMSE']:.4f}, "
            f"MAE={test_reg['MAE']:.4f}, "
            f"R2={test_reg['R2']:.4f}",
            file=sys.stderr,
        )

        print(
            f"Test Classification Accuracy = {test_class_acc:.4f}",
            file=sys.stderr,
        )

        return self.model


    def plot_model_results(
        self, 
        trained_model,
        X_scaled, 
        df, 
        status_col,
        model_name="Model",
        output_file="scatter_plot.pdf"):
        """
        Plot predicted vs. measured well flow rates with categorical coloring.

        Parameters
        ----------
        trained_model : sklearn-like estimator
            Fitted model with .predict() method.
        X_scaled : array-like
            Scaled features for prediction.
        df : pandas.DataFrame
            DataFrame containing status labels for coloring.
        status_col : str
            Column in df that contains status labels.
        model_name : str, optional
            Title for the plot. Default is "Model".
        output_file : str, optional
            File name for saving the figure. Default is "scatter_plot.pdf".
        """

        color_map = {
            'Loaded': '#FF6363',  # Light red
            'Unloaded': '#3674B5',  # Light purple
            'Questionable': '#D3D3D3',  # Light orange
            'Near L.U': '#FFCC99'  # Light blue
        }

        # Get gasflow data from the dataframe
        gasflow_subset = df['Gasflowrate']

        # Predictions
        y_pred_subset = trained_model.predict(X_scaled)

        # Color assignment
        colors = df[status_col].map(color_map).fillna('#D3D3D3')  # Light gray fallback

        # Plot setup
        plt.figure(figsize=(8, 6), dpi=300)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.3, font='Arial')

        # Scatter plot
        plt.scatter(gasflow_subset, y_pred_subset, c=colors, alpha=1, s=100,
                    edgecolors="gray", linewidth=0.5)

        # Reference line
        plt.plot([0, 350000], [0, 350000], '--', color='#FF6666', linewidth=1.5)

        # Labels & Title
        plt.title(f"{model_name} Results", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Measured Well Flow Rate (m³/day)", fontsize=14, fontweight='bold')
        plt.ylabel("Predicted Critical Rate (m³/day)", fontsize=14, fontweight='bold')

        # Grid & Limits
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 350000)
        plt.ylim(0, 350000)

        # Legend
        legend_patches = [mpatches.Patch(color=color, label=status) 
                          for status, color in color_map.items()]
        plt.legend(handles=legend_patches, title='Actual Label', fontsize=12,
                   title_fontsize=14, loc='best', frameon=True, edgecolor='gray')

        # Save & Show
        plt.tight_layout()
        plt.savefig(output_file, format="pdf", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
    def plot_results(self, trained_model, X_scaled, model_name="Model", output_file="scatter_plot.pdf"):
            """
            Convenience method to plot model results using the stored dataframe.
            
            Parameters
            ----------
            trained_model : sklearn-like estimator
                Fitted model with .predict() method.
            X_scaled : array-like
                Scaled features for prediction.
            model_name : str, optional
                Title for the plot. Default is "Model".
            output_file : str, optional
                File name for saving the figure. Default is "scatter_plot.pdf".
            """
            return self.plot_model_results(
                trained_model=trained_model,
                X_scaled=X_scaled,
                df=self.df,
                status_col='Test status',
                model_name=model_name,
                output_file=output_file
            )
    

