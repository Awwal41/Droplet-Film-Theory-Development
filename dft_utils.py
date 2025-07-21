import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Optional, List 

class DataMstr:
    def __init__(
            self, 
            path: str,
            seed: int=42,
            drop_cols: Optional[List[str]]=None,
    ):
        super().__init__()
        self.path = path
        self.seed = seed

        df = pd.read_csv(self.path)
        
        if drop_cols is None: 
            drop_cols=[]

        self.X = df.drop(columns=drop_cols)
        self.y = df['Qcr']
        self.gsflow = df['Gasflowrate']  # additional target for classification metrics

        # load class labels: loaded/unloaded/near loaded
        self.loading_pred_class = df['Test status'].apply(
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
            self.loading_pred_train, 
            self.loading_pred_test, 
        ) = train_test_split(
            self.X, 
            self.y, 
            self.gsflow, 
            self.loading_pred_class, 
            test_size=test_size, 
            random_state=self.seed, 
            stratify=self.loading_pred_class,
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
        self.loading_pred_train = np.array(self.loading_pred_train)
        self.loading_pred_test = np.array(self.loading_pred_test)
        self.loading_pred = np.array(self.loading_pred_class)
        self.gsflow_test = np.array(self.gsflow_test)
        self.gsflow = np.array(self.gsflow)
        self.y_test = np.array(self.y_test)
        self.y = np.array(self.y)

        return 

    def classification_scores(
            self, 
            y_pred: np.ndarray,
            gsflow: np.ndarray, 
            loading_pred_actual: np.ndarray, 
    ):
        
        self.loading_pred = np.where(y_pred > gsflow + self.interval, 1, 
                        np.where(y_pred < gsflow - self.interval, -1, 0))
        
        self.acc = accuracy_score(loading_pred_actual, self.loading_pred) 
        self.cm = confusion_matrix(loading_pred_actual, self.loading_pred,  labels=[-1, 0, 1])

        return self.acc
    
    

