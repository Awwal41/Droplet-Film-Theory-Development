import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pysindy import SINDy
from sklearn.preprocessing import StandardScaler
from pysindy.optimizers import STLSQ
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GridSearchCV
from pysindy.feature_library import PolynomialLibrary, FourierLibrary, GeneralizedLibrary
from boruta import BorutaPy
from typing import Optional, List, Tuple
from functools import partial

class SINDy:
    def __init__(
            self, 
            model, 
    ):
        super().__init__()

class DFT:
    def __init__(
            self, 
            path: str,
            seed: int=42,
            drop_cols: Optional[List[str]]=None,
            interval: float=0,
    ):
        super().__init__()
        self.path = path
        self.seed = seed
        self.interval = interval 

        df = pd.read_csv(self.path)
        
        if drop_cols is None: 
            drop_cols=[]

        self.X = df.drop(columns=drop_cols)
        self.y = df['Qcr']
        self.gsflow = df['Gasflowrate']  # additional target for classification metrics

        # load class labels: loaded/unloaded/near loaded
        self.loading_class = df['Test status'].apply(
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
            self.loading_class, 
            test_size=test_size, 
            random_state=self.seed, 
            stratify=self.loading_class,
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
        self.loading_train = np.array(self.loading_train)
        self.loading_test = np.array(self.loading_test)
        self.loading = np.array(self.loading_class)
        self.gsflow_test = np.array(self.gsflow_test)
        self.gsflow = np.array(self.gsflow)
        self.y_test = np.array(self.y_test)
        self.y = np.array(self.y)

        return 

    def _to_well_status(
            self, 
            y_pred: np.ndarray,
            gsflow: np.ndarray, 
    ):
        
        loading_pred = np.where(y_pred > gsflow + self.interval, 1, 
                        np.where(y_pred < gsflow - self.interval, -1, 0))
    
    def grid_searchCV(
            self, 
            model, # trained model 
            param_grid,
            k: int=5,
    ) -> GridSearchCV: 
        
        scorer = make_scorer(self._scorer, greater_is_better=True)
        gs = GridSearchCV(
            estimator=model, 
            param_grid=param_grid,
            scoring='accuracy', 
            cv=k,
        )

        gs.fit(self.X_train_scaled, self.y_train_scaled)

        self.gs_cv = gs 
        self.best_params = gs.best_params_
        self.best_score = gs. best_score_

        return gs

    def test_model(
            self, 
            model,
    ) -> float:
        x = 0 