import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Callable, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import itertools
from pysindy import SINDy
import feyn

class QLatticeWrapper:
    def __init__(
            self,
            feature_tags: List[str],
            output_tag: str = "Qcr",
            seed: int = 42,
            max_complexity: int = 10,
            n_epochs: int = 10,
            criterion: str = "bic"
    ):
        self.seed = seed
        self.feature_tags = feature_tags
        self.output_tag = output_tag
        self.max_complexity = max_complexity
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.ql = feyn.connect_qlattice()
        self.opt_model = None
        self.y_pred = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        data = pd.DataFrame(X, columns=self.feature_tags)
        data[self.output_tag] = y
        models = self.ql.auto_run(
            data=data,
            output_name=self.output_tag,
            criterion=self.criterion,
            max_complexity=self.max_complexity,
            n_epochs=self.n_epochs,
            verbose=False
        )
        self.opt_model = models[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        data = pd.DataFrame(X, columns=self.feature_tags)
        self.y_pred = self.opt_model.predict(data[self.feature_tags])
        return self.y_pred

    def express(self):
        return self.opt_model.sympify()

class ChiefBldr:
    def __init__(
            self,
            path: str,
            seed: int = 42,
            drop_cols: Optional[List[str]] = None,
            includ_cols: Optional[List[str]] = None,
            test_size: float = 0.20,
            scale: bool = True
    ):
        self.path = path
        self.seed = seed
        self.scale = scale

        # Load data
        try:
            self.df = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.path}")

        # Select features
        if drop_cols is None:
            drop_cols = []
            self.X = self.df[includ_cols]
        else:
            includ_cols = []
            self.X = self.df.drop(columns=drop_cols)

        self.feature_names = self.X.columns.tolist()
        self.y = self.df['Qcr']
        self.gsflow = self.df['Gasflowrate']
        self.loading = self.df['Test status'].apply(
            lambda x: -1 if x == 'Unloaded' else (0 if x == 'Near L.U' else 1)).to_numpy()

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test, self.gsflow_train, self.gsflow_test, \
        self.loading_train, self.loading_test = train_test_split(
            self.X, self.y, self.gsflow, self.loading,
            test_size=test_size, random_state=self.seed, stratify=self.loading
        )

        # Scale features and target
        if self.scale:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            self.X_train_rdy = self.scaler_X.fit_transform(self.X_train)
            self.X_test_rdy = self.scaler_X.transform(self.X_test)
            self.y_train_rdy = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
            self.y_test_rdy = self.scaler_y.transform(self.y_test.values.reshape(-1, 1))
        else:
            self.X_train_rdy = np.array(self.X_train)
            self.X_test_rdy = np.array(self.X_test)
            self.y_train_rdy = self.y_train.values
            self.y_test_rdy = self.y_test.values

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train).flatten()
        self.y_test = np.array(self.y_test).flatten()
        self.loading_train = np.array(self.loading_train)
        self.loading_test = np.array(self.loading_test)
        self.gsflow_train = np.array(self.gsflow_train)
        self.gsflow_test = np.array(self.gsflow_test)

    def evolv_model(
            self,
            build_model: Callable[[Dict[str, Any]], float],
            hparam_grid: Dict[str, List[Any]],
            k_folds: int = 5
    ):
        if k_folds > 0:
            self.best_score = -float("inf")
            self.best_params = None
            self.k_folds = k_folds
            print("Training model and optimizing hyperparameters via k-fold CV...", file=sys.stderr)

            keys = list(hparam_grid.keys())
            values = [hparam_grid[k] for k in keys]
            for combo in itertools.product(*values):
                hparams = dict(zip(keys, combo))
                model = build_model(hparams=hparams)
                self.score = self._cross_val(model=model)
                if self.score > self.best_score:
                    self.best_score = self.score
                    self.best_params = hparams

            print(f"Done. Best score = {self.best_score}", file=sys.stderr)
            print(f"Best hyperparameters: {self.best_params}", file=sys.stderr)
            print("Retraining optimized model on full training set", file=sys.stderr)
            self.model = build_model(hparams=self.best_params)
        else:
            self.model = build_model(hparams=hparam_grid)

        if isinstance(self.model, SINDy):
            self.model.fit(self.X_train_rdy, x_dot=self.y_train_rdy)
        else:
            self.model.fit(self.X_train_rdy, self.y_train_rdy)

        self.y_train_pred = self.model.predict(self.X_train_rdy)
        self.y_test_pred = self.model.predict(self.X_test_rdy)
        if self.scale:
            self.y_train_pred = self.scaler_y.inverse_transform(self.y_train_pred.reshape(-1, 1)).flatten()
            self.y_test_pred = self.scaler_y.inverse_transform(self.y_test_pred.reshape(-1, 1)).flatten()

        print(f"Training set score: {self.classification_scores(self.y_train_pred, self.gsflow_train, self.loading_train)}", file=sys.stderr)
        print(f"Test set score: {self.classification_scores(self.y_test_pred, self.gsflow_test, self.loading_test)}", file=sys.stderr)

        return self.model

    def _cross_val(self, model: Callable[[Dict[str, Any]], float]) -> float:
        kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        acc_scores = []
        for train_idx, val_idx in kf.split(self.X_train, self.loading_train):
            X_train_cv, X_val_cv = self.X_train[train_idx], self.X_train[val_idx]
            y_train_cv, y_val_cv = self.y_train[train_idx], self.y_train[val_idx]
            gsflow_val_cv = self.gsflow_train[val_idx]
            loading_val_cv = self.loading_train[val_idx]

            if self.scale:
                scaler_X = StandardScaler()
                X_train_cv_rdy = scaler_X.fit_transform(X_train_cv)
                X_val_rdy = scaler_X.transform(X_val_cv)
                scaler_y = StandardScaler()
                y_train_cv_rdy = scaler_y.fit_transform(y_train_cv.reshape(-1, 1))
            else:
                X_train_cv_rdy = X_train_cv
                X_val_rdy = X_val_cv
                y_train_cv_rdy = y_train_cv

            if isinstance(model, SINDy):
                model.fit(X_train_cv_rdy, x_dot=y_train_cv_rdy)
            else:
                model.fit(X_train_cv_rdy, y_train_cv_rdy)

            y_val_pred_cv = model.predict(X_val_rdy)
            if self.scale:
                y_val_pred_cv = scaler_y.inverse_transform(y_val_pred_cv.reshape(-1, 1)).flatten()

            acc = self.classification_scores(y_pred=y_val_pred_cv, gsflow=gsflow_val_cv, loading=loading_val_cv)
            acc_scores.append(acc)

        self.acc_cv_scores = acc_scores
        self.mean_acc = float(np.mean(self.acc_cv_scores))
        return self.mean_acc

    def classification_scores(self, y_pred: np.ndarray, gsflow: np.ndarray, loading: np.ndarray, interval: float = 0.01):
        loading_pred = np.where(y_pred > gsflow + interval, 1,
                                np.where(y_pred < gsflow - interval, -1, 0))
        self.acc = accuracy_score(loading, loading_pred)
        self.cm = confusion_matrix(loading, loading_pred, labels=[-1, 0, 1])
        return self.acc

    def plot_model_results(
            self,
            trained_model,
            X_scaled,
            df,
            status_col,
            model_name="Model",
            output_file="scatter_plot.pdf"
    ):
        color_map = {
            'Loaded': '#FF6363',
            'Unloaded': '#3674B5',
            'Questionable': '#D3D3D3',
            'Near L.U': '#FFCC99'
        }

        gasflow_subset = df['Gasflowrate']
        y_pred_subset = trained_model.predict(X_scaled)
        colors = df[status_col].map(color_map).fillna('#D3D3D3')

        plt.figure(figsize=(8, 6), dpi=300)
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.3, font='Arial')
        plt.scatter(gasflow_subset, y_pred_subset, c=colors, alpha=1, s=100,
                    edgecolors="gray", linewidth=0.5)
        plt.plot([0, 350000], [0, 350000], '--', color='#FF6666', linewidth=1.5)
        plt.title(f"{model_name} Results", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Measured Well Flow Rate (m³/day)", fontsize=14, fontweight='bold')
        plt.ylabel("Predicted Critical Rate (m³/day)", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(0, 350000)
        plt.ylim(0, 350000)

        legend_patches = [mpatches.Patch(color=color, label=status)
                          for status, color in color_map.items()]
        plt.legend(handles=legend_patches, title='Actual Label', fontsize=12,
                   title_fontsize=14, loc='best', frameon=True, edgecolor='gray')

        try:
            plt.tight_layout()
            plt.savefig(output_file, format="pdf", dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error saving or displaying plot: {e}", file=sys.stderr)
        finally:
            plt.close()

    def plot_results(self, trained_model, X_scaled, model_name="Model", output_file="scatter_plot.pdf"):
        return self.plot_model_results(
            trained_model=trained_model,
            X_scaled=X_scaled,
            df=self.df,
            status_col='Test status',
            model_name=model_name,
            output_file=output_file
        )