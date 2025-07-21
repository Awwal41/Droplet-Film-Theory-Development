import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold 
from sklearn.preprocessing import StandardScaler
from typing import Optional, List 

class DataMstr:
    def __init__(
            self, 
            path: str,
            seed: int=42,
            drop_cols: Optional[List[str]]=None,
            includ_cols: Optional[List[str]]=None,
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
        self.y = df['Qcr']
        self.gsflow = df['Gasflowrate']  # additional target for classification metrics

        # load class labels: loaded/unloaded/near loaded
        self.loading = df['Test status'].apply(
            lambda x: -1 if x == 'Unloaded' else (0 if x == 'Near L.U' else 1)).to_numpy()
        

    def split_data(
            self,
            test_size: int=0.20, 
    ) -> None:
        
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
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        self.X_train_scaled = scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = scaler_X.transform(self.X_test)

        self.y_train_scaled = scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test_scaled = scaler_y.transform(self.y_test.values.reshape(-1, 1))

        # t_train is just an index array for plotting
        t_train = np.arange(len(self.y_train_scaled))
        t = np.arange(len(self.y))

        # convert to a numpy array and store test data 
        self.X = np.array(self.X)
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)

        self.y = np.array(self.y)
        self.y_train = np.array(self.y_train).flatten()
        self.y_test = np.array(self.y_test).flatten()
        
        self.loading = np.array(self.loading)
        self.loading_train = np.array(self.loading_train)
        self.loading_test = np.array(self.loading_test)

        self.gsflow = np.array(self.gsflow)
        self.gsflow_train = np.array(self.gsflow_train)
        self.gsflow_test = np.array(self.gsflow_test)
        
        return 

    def custom_CV(
            self, 
            model, 
            k_folds: int=5, 

    ) -> List:
        # Use StratifiedKFold to preserve class balance in splits
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
            acc = self._classification_scores(y_pred=y_val_pred_cv, gsflow=gsflow_val_cv, loading=loading_val_cv)
            acc_scores.append(acc)
        
        self.acc_cv_scores = acc_scores
        self.mean_acc = np.mean(self.acc_cv_scores)

        return self.mean_acc


    def _classification_scores(
            self, 
            y_pred: np.ndarray,
            gsflow: np.ndarray, 
            loading: np.ndarray, 
            interval: float=0,
    ):
        self.loading_pred = np.where(y_pred > gsflow + interval, 1, 
                        np.where(y_pred < gsflow - interval, -1, 0))
        
        self.acc = accuracy_score(loading, self.loading_pred) 
        self.cm = confusion_matrix(loading, self.loading_pred,  labels=[-1, 0, 1])

        return self.acc
    
    

