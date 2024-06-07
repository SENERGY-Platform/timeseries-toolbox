import mlflow 
import numpy as np 
import pandas as pd 
from utils import random_series

mlflow.set_tracking_uri("http://localhost:5000")
reg_model_name= "test88970a3b4ed04edaa591224935a3564d"
model_uri = f"models:/{reg_model_name}@production"
loaded_model = mlflow.pyfunc.load_model(model_uri)

timeseries = random_series()
reconstructions, anomaly_indices, normal_indices, test_losses = loaded_model.predict(timeseries)

print(reconstructions)
print(normal_indices)
print(anomaly_indices)
print(test_losses)