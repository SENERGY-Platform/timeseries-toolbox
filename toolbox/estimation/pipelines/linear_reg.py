from darts.models import LinearRegressionModel
import mlflow 

from .helper import create_darts_encoder_based_on_freq, convert_df_to_ts

class LinearReg(mlflow.pyfunc.PythonModel):
    def __init__(self, freq, add_time_covariates, **kwargs) -> None:
        super().__init__()
        
        if add_time_covariates:
            encoders = create_darts_encoder_based_on_freq(freq)
            kwargs['add_encoders'] = encoders

        if add_time_covariates:
            kwargs['lags_future_covariates'] = (0, kwargs['lags'])

        self.model = LinearRegressionModel(**kwargs)

    def fit(self, train_ts):
        self.model.fit(train_ts)
        
    def predict(self, number_steps):
        return self.model.predict(number_steps)

    @staticmethod
    def get_hyperparams(freq, train_ts):
        # TODO depending on dataset freq 
        lags_per_freq = {
            "H": [1],
            "D": [1]
        }
        n_train_samples = len(train_ts)
        if n_train_samples >= 5:
            lags_per_freq['H'].append(5)
        if n_train_samples >= 10:
            lags_per_freq['H'].append(10)
            lags_per_freq['D'].append(10)
        if n_train_samples >= 30:
            lags_per_freq['D'].append(30)

        hyperparams = {
            "add_time_covariates": [True, False],
            "lags": lags_per_freq[freq]
        }
        return hyperparams