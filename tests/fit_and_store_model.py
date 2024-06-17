from toolbox.tasks.fit_model.fit import Fit
from toolbox.ml_config import MLSettings, KafkaTopicConfiguration
from utils import random_df

task_settings = MLSettings(**{
    "model_parameter": {
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
    },
    "model_name": "cnn",
    "use_case": "peak_shaving"
})
data_settings = KafkaTopicConfiguration(**{})
timeseries = random_df()

task = Fit(
    task_settings=task_settings, 
    mlflow_url="http://localhost:5003", 
    job_name="job",
    userid="user",
    tool_box_version="",
    data_settings=data_settings
)

task.run(timeseries)