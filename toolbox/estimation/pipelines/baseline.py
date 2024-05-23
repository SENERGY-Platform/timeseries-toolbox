from darts.models import NaiveMean
from .helper import create_darts_encoder_based_on_freq, convert_df_to_ts

import mlflow 

class Baseline(mlflow.pyfunc.PythonModel):
    def __init__(self, freq, **kwargs) -> None:
        super().__init__()
        
        self.model = NaiveMean()

    def fit(self, train_ts):
        self.model.fit(train_ts)
        
    def predict(self, number_steps):
        return self.model.predict(number_steps)

    @staticmethod
    def get_hyperparams(freq):
        return {}