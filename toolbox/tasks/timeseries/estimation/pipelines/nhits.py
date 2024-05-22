from darts.models import NHiTSModel
from .helper import create_darts_encoder_based_on_freq, convert_df_to_ts
import mlflow 

class DartNHITS(mlflow.pyfunc.PythonModel):
    def __init__(self, freq, add_time_covariates, **kwargs) -> None:
        super().__init__()

        if add_time_covariates:
            encoders = create_darts_encoder_based_on_freq(freq)
            kwargs['add_encoders'] = encoders

        self.model = NHiTSModel(**kwargs)

        
    def fit(self, train_ts):
        self.model.fit(train_ts)
        
    def predict(self, number_steps):
        return self.model.predict(number_steps)

    @staticmethod
    def get_hyperparams(freq, train_ts):
        # TODO depending on dataset freq 
        hyperparams = {
            "add_time_covariates": [True, False],
            "output_chunk_length": [1, 10],
            "input_chunk_length": [1,10],
            "num_stacks": [1,2,3],
            "num_blocks": [1,2,3],
            "num_layers": [1,2,3]
        }
        return hyperparams
 