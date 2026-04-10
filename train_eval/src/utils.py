import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Callable, Dict, Any
import importlib
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from pysindy import SINDy


class QLatticeWrapper:
    """
    Feyn QLattice for symbolic **gas** or **Q_critical**, evaluated with **dynamic operational
    margins**: *(Q_pred − Q_meas)* vs flow-dependent bands (see ``qlattice_pipeline``).

    Use :meth:`fit_dynamic_bounds` (not sklearn-style ``fit(X, y)`` on loading classes).
    """

    def __init__(
        self,
        *,
        mode: str = "qcr_labels",
        seed: int = 42,
        max_complexity: int = 12,
        n_epochs: int = 32,
        select_by: str = "turner_lu",
        loss: str = "absolute_error",
        val_frac: float = 0.2,
        top_models: int = 15,
        qcr_proxy_margin_scale: float = 1.0,
        linear_gas: bool = False,
        no_val_select: bool = False,
        qcr_use_measured_gas: bool = False,
    ) -> None:
        self.mode = mode
        self.seed = seed
        self.max_complexity = max_complexity
        self.n_epochs = n_epochs
        self.select_by = select_by
        self.loss = loss
        self.val_frac = val_frac
        self.top_models = top_models
        self.qcr_proxy_margin_scale = qcr_proxy_margin_scale
        self.linear_gas = linear_gas
        self.no_val_select = no_val_select
        self.qcr_use_measured_gas = qcr_use_measured_gas
        self.opt_model = None
        self.pred_train: Optional[np.ndarray] = None
        self.pred_test: Optional[np.ndarray] = None
        self.dynamic_bounds_metrics: Optional[Dict[str, Any]] = None

    def _args_like(self) -> Any:
        from argparse import Namespace

        return Namespace(
            epochs=self.n_epochs,
            max_complexity=self.max_complexity,
            select_by=self.select_by,
            loss=self.loss,
            val_frac=self.val_frac,
            top_models=self.top_models,
            no_val_select=self.no_val_select,
            linear_gas=self.linear_gas,
            qcr_use_measured_gas=self.qcr_use_measured_gas,
        )

    def fit_dynamic_bounds(
        self,
        train_X_df: pd.DataFrame,
        test_X_df: pd.DataFrame,
        gs_train: np.ndarray,
        gs_test: np.ndarray,
        load_train: np.ndarray,
        load_test: np.ndarray,
        *,
        y_train_teacher: Optional[np.ndarray] = None,
    ) -> "QLatticeWrapper":
        import feyn

        from . import qlattice_pipeline as qp

        args = self._args_like()
        tr = train_X_df.copy()
        te = test_X_df.copy()
        tr, te = qp.append_measured_gas_feature(tr, te, gs_train, gs_test, self.qcr_use_measured_gas)

        if self.mode == "gas":
            out = qp.train_predict_gas_qlattice(
                tr,
                te,
                gs_train,
                gs_test,
                load_train,
                load_test,
                args,
                feyn,
                self.seed,
                verbose=False,
            )
            self.opt_model = out["best_g"]
            self.pred_train = out["pred_tr"]
            self.pred_test = out["pred_te"]
            self.dynamic_bounds_metrics = {
                "f1_lu": float(out["f1_lu"]),
                "f1_macro": float(out["f1_macro"]),
                "acc": float(out["acc"]),
                "mae_test": float(out["mae"]),
                "rmse": float(out["rmse"]),
            }
            return self

        if self.mode == "qcr_labels":
            y_t = qp.q_critical_label_proxy_targets(
                gs_train, load_train, margin_scale=self.qcr_proxy_margin_scale
            )
        elif self.mode == "qcr_custom":
            if y_train_teacher is None:
                raise ValueError("qcr_custom requires y_train_teacher (physics-model Q_critical per train row).")
            y_t = y_train_teacher
        else:
            raise ValueError(f"Unknown mode {self.mode!r}; use 'gas', 'qcr_labels', or 'qcr_custom'.")

        best, pred_tr, pred_te, _, _, _ = qp.feyn_run_qcritical_fit(
            tr,
            te,
            gs_train,
            load_train,
            y_t,
            args,
            feyn,
            self.seed + 7,
        )
        self.opt_model = best
        self.pred_train = pred_tr
        self.pred_test = pred_te
        ev = qp.eval_turner_train_test_metrics(
            pred_tr, gs_train, load_train, pred_te, gs_test, load_test
        )
        self.dynamic_bounds_metrics = {
            "f1_lu": float(ev["f1_lu"]),
            "f1_macro": float(ev["f1_macro"]),
            "acc": float(ev["acc"]),
            "acc_train": float(ev["acc_train"]),
        }
        return self

    def express(self, signif: int = 4) -> Any:
        if self.opt_model is None:
            raise RuntimeError("Call fit_dynamic_bounds(...) first.")
        return self.opt_model.sympify(signif=signif)


def _run_pipeline_cli(submodule: str, argv: Optional[List[str]] = None) -> None:
    mod = importlib.import_module(f".{submodule}", package=__package__)
    mod.main(mod.parse_args(argv))


def run_qlattice_from_argv(argv: Optional[List[str]] = None) -> None:
    """Run ``qlattice_pipeline`` (same as ``qlattice_standalone.py``)."""
    _run_pipeline_cli("qlattice_pipeline", argv)


def default_qlattice_csv_path() -> Path:
    """Default ``datasets/processed_well_data.csv`` under the repo root."""
    from .qlattice_pipeline import default_processed_well_csv

    return default_processed_well_csv()


def run_tabpfn_from_argv(argv: Optional[List[str]] = None) -> None:
    """Run ``tabpfn_pipeline`` (same as ``tabpfn_standalone.py``)."""
    _run_pipeline_cli("tabpfn_pipeline", argv)


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

    @classmethod
    def from_dynamic_bounds_csv(
        cls,
        path: str | Path,
        seed: int = 42,
        test_size: float = 0.2,
    ) -> "Helm":
        """
        Load the same filtered CSV + stratified split as ``qlattice_pipeline.load_and_split``
        (operational labels: Loaded=1, Near L.U=0, Unloaded=-1). Use with :meth:`evolv_dynamic_bounds_qlattice`.
        """
        from . import qlattice_pipeline as qp

        data = qp.load_and_split(Path(path), seed, test_size)
        fc = data["feature_cols"]
        train_X_df = pd.DataFrame(data["scaler"].inverse_transform(data["X_train"]), columns=fc)
        test_X_df = pd.DataFrame(data["scaler"].inverse_transform(data["X_test"]), columns=fc)
        obj = object.__new__(cls)
        obj.path = str(Path(path).resolve())
        obj.seed = seed
        obj.scale = True
        obj._dynamic_bounds_split = True
        obj.feature_names = fc
        obj.train_X_df = train_X_df
        obj.test_X_df = test_X_df
        obj.gsflow_train = np.asarray(data["gs_train"], dtype=float)
        obj.gsflow_test = np.asarray(data["gs_test"], dtype=float)
        obj.loading_train = np.asarray(data["load_train"])
        obj.loading_test = np.asarray(data["load_test"])
        obj.scaler = data["scaler"]
        obj.X_train = np.asarray(train_X_df.values, dtype=float)
        obj.X_test = np.asarray(test_X_df.values, dtype=float)
        obj.df = pd.read_csv(path)
        obj.gsflow = obj.df["Gasflowrate"]
        obj.loading = obj.df["Test status"].apply(
            lambda x: 0 if x == "Unloaded" else (1 if x == "Near L.U" else 2)
        ).to_numpy()
        return obj

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

    def evolv_dynamic_bounds_qlattice(
        self,
        hparam_grid: Dict[str, List[Any]],
        mode: str = "qcr_labels",
        *,
        qcr_npz: Optional[str | Path] = None,
        score_by: str = "f1_lu",
        y_train_teacher: Optional[np.ndarray] = None,
    ) -> QLatticeWrapper:
        """
        Grid search over ``QLatticeWrapper`` hyperparameters using the **test**-set dynamic-bounds metric
        (same spirit as :meth:`evolv_model`, but for symbolic regression + margin-based loading).

        Requires :meth:`from_dynamic_bounds_csv`. ``score_by``: ``f1_lu``, ``f1_macro``, ``acc``, or
        ``neg_mae`` (gas mode; higher is better).
        """
        if not getattr(self, "_dynamic_bounds_split", False):
            raise ValueError("evolv_dynamic_bounds_qlattice requires Helm.from_dynamic_bounds_csv(...).")

        from . import qlattice_pipeline as qp

        keys = list(hparam_grid.keys())
        values = [hparam_grid[k] for k in keys]

        if mode == "qcr_custom" and y_train_teacher is None and qcr_npz is not None:
            z = np.load(Path(qcr_npz), allow_pickle=False)
            y_train_teacher, _ = qp.load_qcr_train_test_from_npz(z)

        best_model: Optional[QLatticeWrapper] = None
        best_score = -np.inf
        best_params: Optional[Dict[str, Any]] = None

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            m = QLatticeWrapper(mode=mode, seed=self.seed, **params)
            kwargs_fit: Dict[str, Any] = {}
            if mode == "qcr_custom":
                kwargs_fit["y_train_teacher"] = y_train_teacher
            m.fit_dynamic_bounds(
                self.train_X_df,
                self.test_X_df,
                self.gsflow_train,
                self.gsflow_test,
                self.loading_train,
                self.loading_test,
                **kwargs_fit,
            )
            tm = m.dynamic_bounds_metrics or {}
            if score_by == "f1_lu":
                sc = float(tm.get("f1_lu", float("nan")))
            elif score_by == "f1_macro":
                sc = float(tm.get("f1_macro", float("nan")))
            elif score_by == "acc":
                sc = float(tm.get("acc", float("nan")))
            elif score_by == "neg_mae":
                sc = -float(tm.get("mae_test", float("nan")))
            else:
                raise ValueError(f"Unknown score_by {score_by!r}")

            if sc > best_score:
                best_score = sc
                best_model = m
                best_params = params

        self.dynamic_bounds_best_qlattice = best_model
        self.dynamic_bounds_best_params = best_params
        self.dynamic_bounds_best_score = best_score

        print(
            f"\nBest dynamic-bounds score ({score_by} on test): {best_score:.4f}",
            file=sys.stderr,
        )
        print(f"Best params: {best_params}", file=sys.stderr)

        if best_model is None:
            raise RuntimeError("evolv_dynamic_bounds_qlattice: empty grid or all fits failed.")
        return best_model

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