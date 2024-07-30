from toolbox.anomaly.cnn.pipeline import CNNAnomalyPipeline
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
        model_parameter = {
                    "window_length": 205,
                    "batch_size": 1,
                    "lr": 0.01,
                    "num_epochs": 1,
                    "loss": "MSE",
                    "op": "SGD",
                    "latent_dims": 32,
                    "early_stopping_patience": 10,
                    "early_stopping_delta": 0,
                    "kernel_size": 7,
                    "out_dir": ".",
                    "plot_enabled": False
        }
        
        pipeline = CNNAnomalyPipeline(**model_parameter)
        pipeline.fit(random_series(), random_series())

        pipeline.predict(_, random_df(), _)
        pipeline.predict(_, random_df(), _)
