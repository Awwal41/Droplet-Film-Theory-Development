import pandas as pd
import numpy as np
import sys

from sklearn.metrics import accuracy_score, confusion_matrix
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
            output_tag: int="loading",
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
    hyperparameter tuning, and direct classification
    of well loading status (-1 = Unloaded, 0 = Near L.U., 1 = Loaded).
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

        self.gsflow = df["Gasflowrate"]
        self.df = df

        # Encode loading state — this is now the classification target
        # 0 = Unloaded, 1 = Near L.U., 2 = Loaded
        self.loading = df["Test status"].apply(
            lambda x: 0 if x == "Unloaded" else (1 if x == "Near L.U" else 2)
        ).to_numpy()

        # =========================
        # TRAIN / TEST SPLIT
        # =========================
        (
            self.X_train,
            self.X_test,
            self.gsflow_train,
            self.gsflow_test,
            self.loading_train,
            self.loading_test,
        ) = train_test_split(
            self.X,
            self.gsflow,
            self.loading,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.loading,
        )

        # =========================
        # SCALING (features only)
        # =========================
        if self.scale:
            self.scaler_X = StandardScaler()
            self.X_train_rdy = self.scaler_X.fit_transform(self.X_train)
            self.X_test_rdy = self.scaler_X.transform(self.X_test)
        else:
            self.X_train_rdy = np.array(self.X_train)
            self.X_test_rdy = np.array(self.X_test)

        # Convert to numpy
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.loading_train = np.array(self.loading_train)
        self.loading_test = np.array(self.loading_test)
        self.gsflow_train = np.array(self.gsflow_train)
        self.gsflow_test = np.array(self.gsflow_test)

    # =====================================================
    # CLASSIFICATION METRICS
    # =====================================================
    def classification_scores(
        self,
        y_pred: np.ndarray,
        loading: np.ndarray,
    ):
        """Compute accuracy and confusion matrix for direct loading-status classification."""
        self.acc = accuracy_score(loading, y_pred)
        self.cm = confusion_matrix(loading, y_pred, labels=[0, 1, 2])
        return self.acc

    # =====================================================
    # CROSS VALIDATION
    # =====================================================
    def _cross_val(self, model):
        kf = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=42
        )

        acc_scores = []
        cm_total = np.zeros((3, 3), dtype=int)

        for tr_idx, val_idx in kf.split(self.X_train, self.loading_train):
            X_tr, X_val = self.X_train[tr_idx], self.X_train[val_idx]
            load_tr = self.loading_train[tr_idx]
            load_val = self.loading_train[val_idx]

            if self.scale:
                sx = StandardScaler()
                X_tr = sx.fit_transform(X_tr)
                X_val = sx.transform(X_val)

            model.fit(X_tr, load_tr)
            load_val_pred = model.predict(X_val)

            acc_scores.append(
                self.classification_scores(load_val_pred, load_val)
            )
            cm_total += self.cm

        row_sums = cm_total.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm_total.diagonal() / row_sums, 0.0)

        return float(np.mean(acc_scores)), float(np.std(acc_scores)), per_class_acc

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
        self.best_score = -np.inf
        self.best_params = None
        self.best_search_params = None
        self.k_folds = k_folds

        print(
            "Training model and optimizing hyperparameters via k-fold CV...",
            file=sys.stderr,
        )

        keys = list(hparam_grid.keys())
        values = [hparam_grid[k] for k in keys]

        for combo in itertools.product(*values):
            search_hparams = dict(zip(keys, combo))
            model_hparams = dict(search_hparams)

            model = build_model(hparams=model_hparams)
            score, score_std, per_class_acc = self._cross_val(model)

            if score > self.best_score:
                self.best_score = score
                self.best_score_std = score_std
                self.best_cv_per_class_acc = per_class_acc
                self.best_params = model_hparams
                self.best_search_params = search_hparams

        print(
            "Retraining optimized model on full training set...",
            file=sys.stderr,
        )
        self.model = build_model(hparams=self.best_params)

        # =========================
        # FINAL TRAINING
        # =========================
        self.model.fit(self.X_train_rdy, self.loading_train)

        # =========================
        # PREDICTIONS
        # =========================
        self.loading_train_pred = self.model.predict(self.X_train_rdy)
        self.loading_test_pred = self.model.predict(self.X_test_rdy)

        # =========================
        # OUTPUT
        # =========================
        class_labels = ["Unloaded (0)", "Near L.U. (1)", "Loaded (2)"]

        if k_folds > 0:
            print(
                f"\nBest CV Classification Accuracy: \n>>> {self.best_score:.4f} ± {self.best_score_std:.4f}",
                file=sys.stderr,
            )
            per_class_str = ", ".join(
                f"{label}: {acc:.4f}"
                for label, acc in zip(class_labels, self.best_cv_per_class_acc)
            )
            print(
                f"\nBest CV Per-Class Accuracy:\n>>> {per_class_str}",
                file=sys.stderr,
            )
            print(
                "\nBest Hyperparameters:\n>>>",
                self.best_search_params,
                file=sys.stderr,
            )

        # Training accuracy
        train_acc = self.classification_scores(self.loading_train_pred, self.loading_train)
        print(
            f"\nTraining Classification Accuracy:\n>>> {train_acc:.4f}",
            file=sys.stderr,
        )
        cm = self.cm
        row_sums = cm.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)
        per_class_str = ", ".join(
            f"{label}: {acc:.4f}"
            for label, acc in zip(class_labels, per_class_acc)
        )
        print(
            f"\nPer-Class Training Accuracy:\n>>> {per_class_str}",
            file=sys.stderr,
        )

        # Test accuracy
        test_acc = self.classification_scores(self.loading_test_pred, self.loading_test)
        print(
            f"\nTest Classification Accuracy:\n>>> {test_acc:.4f}",
            file=sys.stderr,
        )
        cm = self.cm
        row_sums = cm.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)
        per_class_str = ", ".join(
            f"{label}: {acc:.4f}"
            for label, acc in zip(class_labels, per_class_acc)
        )
        print(
            f"\nPer-Class Test Accuracy:\n>>> {per_class_str}",
            file=sys.stderr,
        )

        return self.model

    # =====================================================
    # DFM-SPECIFIC CROSS VALIDATION
    # =====================================================
    def _cross_val_dfm(self, model):
        """
        K-fold CV for DFT models whose interface is:
            fit(X, gsflow, loading)
            predict(X, gsflow)

        Features are NOT scaled here — DFT operates on raw physical
        units and its own internal parameter bounds handle scale.
        """
        kf = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=42
        )

        acc_scores = []
        cm_total   = np.zeros((3, 3), dtype=int)

        for tr_idx, val_idx in kf.split(self.X_train, self.loading_train):
            X_tr,   X_val   = self.X_train[tr_idx],    self.X_train[val_idx]
            gs_tr,  gs_val  = self.gsflow_train[tr_idx], self.gsflow_train[val_idx]
            load_tr         = self.loading_train[tr_idx]
            load_val        = self.loading_train[val_idx]

            model.fit(X_tr, gs_tr, load_tr)
            load_val_pred = model.predict(X_val, gs_val)

            acc_scores.append(
                self.classification_scores(load_val_pred, load_val)
            )
            cm_total += self.cm

        row_sums      = cm_total.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm_total.diagonal() / row_sums, 0.0)

        return float(np.mean(acc_scores)), float(np.std(acc_scores)), per_class_acc

    # =====================================================
    # DFM TRAIN + EVALUATE
    # =====================================================
    def evolv_dfm_model(
        self,
        build_model: Callable[[Dict[str, Any]], Any],
        hparam_grid: Dict[str, List[Any]],
        k_folds: int = 5,
    ):
        """
        Hyperparameter search + evaluation for DFT-style models.

        Mirrors evolv_model but passes gsflow to fit() and predict()
        at every stage. No sklearn scaling is applied to X — the DFT
        equation uses raw physical feature values.

        Parameters
        ----------
        build_model : callable  (hparams) -> DFT instance
        hparam_grid : dict of hyperparameter lists
        k_folds     : number of stratified CV folds
        """
        self.best_score        = -np.inf
        self.best_params       = None
        self.best_search_params = None
        self.k_folds           = k_folds

        print(
            "Training DFM model and optimising hyperparameters via k-fold CV...",
            file=sys.stderr,
        )

        keys   = list(hparam_grid.keys())
        values = [hparam_grid[k] for k in keys]

        for combo in itertools.product(*values):
            search_hparams = dict(zip(keys, combo))
            model_hparams  = dict(search_hparams)

            model = build_model(hparams=model_hparams)
            score, score_std, per_class_acc = self._cross_val_dfm(model)

            if score > self.best_score:
                self.best_score          = score
                self.best_score_std      = score_std
                self.best_cv_per_class_acc = per_class_acc
                self.best_params         = model_hparams
                self.best_search_params  = search_hparams

        print(
            "Retraining optimised DFM model on full training set...",
            file=sys.stderr,
        )
        self.model = build_model(hparams=self.best_params)

        # =========================
        # FINAL TRAINING  (raw X — no scaling)
        # =========================
        self.model.fit(self.X_train, self.gsflow_train, self.loading_train)

        # =========================
        # PREDICTIONS
        # =========================
        self.loading_train_pred = self.model.predict(self.X_train, self.gsflow_train)
        self.loading_test_pred  = self.model.predict(self.X_test,  self.gsflow_test)

        # =========================
        # OUTPUT
        # =========================
        class_labels = ["Unloaded (0)", "Near L.U. (1)", "Loaded (2)"]

        if k_folds > 0:
            print(
                f"\nBest CV Classification Accuracy: \n>>> {self.best_score:.4f} ± {self.best_score_std:.4f}",
                file=sys.stderr,
            )
            per_class_str = ", ".join(
                f"{label}: {acc:.4f}"
                for label, acc in zip(class_labels, self.best_cv_per_class_acc)
            )
            print(
                f"\nBest CV Per-Class Accuracy:\n>>> {per_class_str}",
                file=sys.stderr,
            )
            print(
                "\nBest Hyperparameters:\n>>>",
                self.best_search_params,
                file=sys.stderr,
            )

        # Training accuracy
        train_acc = self.classification_scores(self.loading_train_pred, self.loading_train)
        print(
            f"\nTraining Classification Accuracy:\n>>> {train_acc:.4f}",
            file=sys.stderr,
        )
        cm            = self.cm
        row_sums      = cm.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)
        per_class_str = ", ".join(
            f"{label}: {acc:.4f}"
            for label, acc in zip(class_labels, per_class_acc)
        )
        print(
            f"\nPer-Class Training Accuracy:\n>>> {per_class_str}",
            file=sys.stderr,
        )

        # Test accuracy
        test_acc = self.classification_scores(self.loading_test_pred, self.loading_test)
        print(
            f"\nTest Classification Accuracy:\n>>> {test_acc:.4f}",
            file=sys.stderr,
        )
        cm            = self.cm
        row_sums      = cm.sum(axis=1)
        per_class_acc = np.where(row_sums > 0, cm.diagonal() / row_sums, 0.0)
        per_class_str = ", ".join(
            f"{label}: {acc:.4f}"
            for label, acc in zip(class_labels, per_class_acc)
        )
        print(
            f"\nPer-Class Test Accuracy:\n>>> {per_class_str}",
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
        Plot predicted loading class vs. actual loading class with categorical coloring.

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
            'Loaded': '#FF6363',      # Light red
            'Unloaded': '#3674B5',    # Light blue
            'Questionable': '#D3D3D3',# Light gray
            'Near L.U': '#FFCC99'     # Light orange
        }

        # Predictions (class labels: -1, 0, 1)
        y_pred_subset = trained_model.predict(X_scaled)

        # Color assignment from actual status
        colors = df[status_col].map(color_map).fillna('#D3D3D3')

        # Actual class labels for x-axis
        actual_loading = df["Test status"].apply(
            lambda x: 0 if x == "Unloaded" else (1 if x == "Near L.U" else 2)
        )

        # Plot setup
        plt.figure(figsize=(8, 6), dpi=300)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.3, font='Arial')

        # Scatter plot: actual class vs. predicted class
        plt.scatter(actual_loading, y_pred_subset, c=colors, alpha=1, s=100,
                    edgecolors="gray", linewidth=0.5)

        # Perfect-prediction reference line
        plt.plot([0, 1, 2], [0, 1, 2], '--', color='#FF6666', linewidth=1.5)

        # Labels & Title
        plt.title(f"{model_name} Results", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Actual Loading Status", fontsize=14, fontweight='bold')
        plt.ylabel("Predicted Loading Status", fontsize=14, fontweight='bold')
        plt.xticks([0, 1, 2], ["Unloaded\n(0)", "Near L.U.\n(1)", "Loaded\n(2)"])
        plt.yticks([0, 1, 2], ["Unloaded\n(0)", "Near L.U.\n(1)", "Loaded\n(2)"])

        # Grid & Limits
        plt.grid(True, linestyle='--', alpha=0.7)

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
    
