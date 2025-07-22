import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Callable, Dict, Any, List
import itertools
from sklearn.pipeline import Pipeline
from pysindy import SINDy

class ChiefBldr:
    """
    This module handles dataset splititng, k-fold cross validation, hyperparmater tuning, and model testing. 

    Args
    ____
    path: str
        File path for well dataset
    seed: int
        Random seed for model trianing 
    drop_cols: List[str] 
        List of strings specifying which data columns should be dropped, the rest included 
    incl_cols: List[str]
        If len(drop_cols)==0, list of strings specifiying which data columns should be included, the rest dropped 
    test_size: float 
        Test set size => Default: 0.20 (20% of dataset) 
    """
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

        # build pandas data object from the file path 
        df = pd.read_csv(self.path)
        
        if drop_cols is None: 
            drop_cols=[] 
            self.X = df[includ_cols] # include only these columns 
        else: 
            includ_cols=[]
            self.X = df.drop(columns=drop_cols) # exclude these colums

        self.feature_names = self.X.columns.tolist() # store feature namaes
        self.y = df['Qcr']
        self.gsflow = df['Gasflowrate']  # additional target for classification metrics

        # lone hot encode loaded/unloaded/near loaded class labels
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
        """
        User calls this function to build and train the model, tune hyperparameters, and test the final model performance 

        Args
        ____
        build_model: function 
            Pre-specified function to build the model based on inputted hyperparameters 
        hparam_grid: Dict 
            Dictionary of hyperparameters and ranges for grid search 
        k_folds: int
            Number of folds in cross validation => Default: 5-fold CV
        
        """

        self.best_score  = -float("inf") # store the best score for hyperparameter opt 
        self.best_params = None
        self.k_folds = k_folds
        print("Training model and optimizing hyperparameters via k-fold CV...")

        # grab the keys and corresponding lists
        keys   = list(hparam_grid.keys()) 
        values = [hparam_grid[k] for k in keys]

        # iterate over every combination in hyperparameter set 
        for combo in itertools.product(*values): 
            hparams = dict(zip(keys, combo))

            # evaluate model on these params
            model = build_model(hparams=hparams) # model defined and passed by user
            self.score = self._cross_val(model=model) # perform k-fold CV for each set of hparams 

            # collect the best set of hyperparameters 
            if self.score > self.best_score:
                self.best_score  = self.score
                self.best_params = hparams

        # print best score hparams from hyperparameter tuning 
        print("Done. Best score =", self.best_score)
        print("Best hyperparameters:", self.best_params)

        # retrain the model with the full training set and evalute test set performance 
        print("Retraining optimized model on full training set")
        self.model = build_model(hparams=self.best_params)

        if isinstance(model, SINDy): # PySINDy models require unconventional format
            
            self.model.fit(self.X_train_scaled, x_dot=self.y_train_scaled)
        else:
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
            model: Callable[[Dict[str, Any]], float],
    ) -> float:
        
        """
        This function performs k-fold cross validation based on the well classification accuracy 

        Args
        ____
        model: Callable[[Dict[str, Any]], float],
            Model to be trained 
        """
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

            if isinstance(model, SINDy): # PySINDy models require unconventional format
                
                model.fit(X_train_cv_scaled, x_dot=y_train_cv_scaled)
            else:
                model.fit(X_train_cv_scaled, y_train_cv_scaled)

            # evalaute the performance on the validation set 
            y_val_pred_scaled = model.predict(X_val_cv_scaled)
            y_val_pred_cv = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
            acc = self.classification_scores(y_pred=y_val_pred_cv, gsflow=gsflow_val_cv, loading=loading_val_cv)
            acc_scores.append(acc) # compute classification score based on well status 
            
        self.acc_cv_scores = acc_scores
        self.mean_acc = float(np.mean(self.acc_cv_scores))

        return self.mean_acc 
    
    def classification_scores(
        self, 
        y_pred: np.ndarray,
        gsflow: np.ndarray, 
        loading: np.ndarray, 
        interval: float=0.01,
    ):
        """
        This functiion converts model regression values to well status classificaitons and computes accuracy 

        Args
        ____
        y_pred: np.ndarray 
            Regressed gas velocities 
        gsflow: np.ndarray 
            Gas flow rate from dataset
        loading: np.ndarray 
            Well status: Loaded, Unloaded, Near Loader
        Interval: float 
            Interval for Near Loaded wells => Default: 0.01

        """
        loading_pred = np.where(y_pred > gsflow + interval, 1, 
                        np.where(y_pred < gsflow - interval, -1, 0)) # classificy loading status based on y_pred and gs_flow

        self.acc = accuracy_score(loading, loading_pred) # compute classificaiton accuracy 
        self.cm = confusion_matrix(loading, loading_pred,  labels=[-1, 0, 1]) # compute confusion matrix 

        return self.acc

    

