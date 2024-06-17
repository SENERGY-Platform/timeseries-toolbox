from toolbox.tasks.fit_model.use_cases.anomaly.anomaly import Anomaly
from toolbox.model_registry import store_model
import pandas as pd 
import numpy as np 
import unittest
import mlflow

def random_dates(start, end, n=10):
    start_u = start.value//10**9
    end_u = end.value//10**9

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

def random_series():
    start = pd.to_datetime('2015-01-01')
    end = pd.to_datetime('2015-01-03')
    n = 50 
    ts = random_dates(start, end, n)
    data = pd.Series(np.random.randint(0, 10, n), index=ts)
    return data

def random_df():
    start = pd.to_datetime('2015-01-01')
    end = pd.to_datetime('2015-01-03')
    n = 50 
    ts = random_dates(start, end, n)
    data = pd.DataFrame({"value": np.random.randint(0, 10, n), "time": ts})
    return data

class TestAnomalyPipeline(unittest.TestCase):

    def test_simple_run(self):
        data = random_series()
        task = Anomaly()
        model_name = "cnn"
        config = {
                    "window_length": 205,
                    "batch_size": 1,
                    "lr": 0.01,
                    "num_epochs": 1,
                    "loss": "MSE",
                    "op": "SGD",
                    "latent_dims": 32,
                    "early_stopping_patience": 10,
                    "early_stopping_delta": 0,
                    "kernel_size": 7
        }
        pipeline, _, _ = task.fit(data, config, model_name)

        pipeline.predict(_, random_df(), _)
        pipeline.predict(_, random_df(), _)
        mlflow.set_tracking_uri("http://localhost:5003")
        # TODO remove
        #store_model(pipeline, "", {}, "job", "", {}, {}, task.get_model_signature())
