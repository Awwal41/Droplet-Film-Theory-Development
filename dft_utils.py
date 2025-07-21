import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Callable, Dict, Any, List
import itertools

class DataMstr:
    def __init__(
            self, 
            path: str,
            seed: int=42,
            drop_cols: Optional[List[str]]=None,
            includ_cols: Optional[List[str]]=None,
            test_size: float=0.20,
    ):
        super().__init__()
        self.path = path
        self.seed = seed

        df = pd.read_csv(self.path)
        
        if drop_cols is None: 
            drop_cols=[]
            self.X = df[includ_cols]
        else: 
            self.X = df.drop(columns=drop_cols)

        self.feature_names = self.X.columns.tolist()
        self.y = df['Qcr']
        self.gsflow = df['Gasflowrate']  # additional target for classification metrics

        # load class labels: loaded/unloaded/near loaded
        self.loading = df['Test status'].apply(
            lambda x: -1 if x == 'Unloaded' else (0 if x == 'Near L.U' else 1)).to_numpy()
        
        # perform the train-test split 
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

        # scale features and continuous target (Qcr)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)

        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test_scaled = self.scaler_y.transform(self.y_test.values.reshape(-1, 1))

        # convert to a numpy array and store test data 

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
        k_folds: int=5,
    ):

        self.best_score  = -float("inf")
        self.best_params = None
        self.k_folds = k_folds
        print("Training model and optimizing hyperparameters via k-fold CV...")

        # grab the keys and corresponding lists
        keys   = list(hparam_grid.keys())
        values = [hparam_grid[k] for k in keys]

        # iterate over every combination
        for combo in itertools.product(*values):
            hparams = dict(zip(keys, combo))

            # evaluate model on these params
            model = build_model(hparams=hparams)
            self.score = self._cross_val(model=model)

            # track the best
            if self.score > self.best_score:
                self.best_score  = self.score
                self.best_params = hparams
        
        print("Done. Best score =", self.best_score)
        print("Best hyperparameters:", self.best_params)

        print("Retraining optimized model on full training set")
        self.model = build_model(hparams=self.best_params)
        self.model.fit(self.X_train_scaled, self.y_train_scaled)
        self.y_train_scaled_pred = self.model.predict(self.X_train_scaled)
        self.y_train_pred = self.scaler_y.inverse_transform(self.y_train_scaled_pred.reshape(-1, 1)).flatten()
        print(f"Training set score: {self.classification_scores(self.y_train_pred, self.gsflow_train, self.loading_train)}")
        self.y_test_scaled_pred = self.model.predict(self.X_test_scaled)
        self.y_test_pred = self.scaler_y.inverse_transform(self.y_test_scaled_pred.reshape(-1, 1)).flatten()
        print(f"Test set score: {self.classification_scores(self.y_test_pred, self.gsflow_test, self.loading_test)}")

        return self.model 

    def _cross_val(
            self, 
            model,
    ) -> float:
        # Use StratifiedKFold to preserve class balance in splits
        kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        acc_scores = []
        for train_idx, val_idx in kf.split(self.X_train, self.loading_train):  # Using validation split from training data

            # divide into trianing/validation sets
            X_train_cv, X_val_cv = self.X_train[train_idx], self.X_train[val_idx]
            y_train_cv, y_val_cv = self.y_train[train_idx], self.y_train[val_idx]
            gsflow_val_cv = self.gsflow_train[val_idx]
            loading_val_cv = self.loading_train[val_idx]
            
            scaler_X = StandardScaler()
            X_train_cv_scaled = scaler_X.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler_X.transform(X_val_cv)
            
            scaler_y = StandardScaler()
            y_train_cv_scaled = scaler_y.fit_transform(y_train_cv.reshape(-1, 1))

            model.fit(X_train_cv_scaled, y_train_cv_scaled)

            # evalaute the performance on the validation set 
            y_val_pred_scaled = model.predict(X_val_cv_scaled)
            y_val_pred_cv = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
            acc = self.classification_scores(y_pred=y_val_pred_cv, gsflow=gsflow_val_cv, loading=loading_val_cv)
            acc_scores.append(acc)
            
        self.acc_cv_scores = acc_scores
        self.mean_acc = float(np.mean(self.acc_cv_scores))

        return self.mean_acc
    
    def classification_scores(
        self, 
        y_pred: np.ndarray,
        gsflow: np.ndarray, 
        loading: np.ndarray, 
        interval: float=0,
    ):
        loading_pred = np.where(y_pred > gsflow + interval, 1, 
                        np.where(y_pred < gsflow - interval, -1, 0))

        self.acc = accuracy_score(loading, loading_pred) 
        self.cm = confusion_matrix(loading, loading_pred,  labels=[-1, 0, 1])

        return self.acc

    

