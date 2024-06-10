import mlflow 
from utils import random_df
from toolbox.anomaly.pipelines.utils import Reconstruction
import pandas as pd 

mlflow.set_tracking_uri("http://localhost:5003")
reg_model_name= "job"
model_uri = f"models:/{reg_model_name}/1"
loaded_model = mlflow.pyfunc.load_model(model_uri)

timeseries = random_df()
model = loaded_model.unwrap_python_model()
reconstruction_errors = [Reconstruction(pd.Timestamp.now(), 30), Reconstruction(pd.Timestamp.now(), 20)]
model.set_all_reconstruction_errors(reconstruction_errors)

reconstructions, anomaly_indices, normal_indices, test_losses, _ = loaded_model.predict(timeseries)

print(reconstructions)
print(normal_indices)
print(anomaly_indices)
print(test_losses)