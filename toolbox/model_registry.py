import json

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

def store_model(model_artifact, userid, config, job_name, tool_box_version, metrics, data_settings):
    # This will store the model at MLFlow model registry
    # If it does not exist, it will be created with version 1
    # All following models will increment the version by 1
    # The latest version gets the alias `production`

    print('store model')

    mlflow.end_run()
    mlflow.set_experiment(job_name)
    run_relative_artifcat_path = 'models'

    # Create a new model version and save model
    with mlflow.start_run(run_name="store-best-model") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params(config)

        mlflow.pyfunc.log_model(
            artifact_path=run_relative_artifcat_path,
            python_model=model_artifact
        )
    
    model_uri = f"runs:/{run.info.run_id}/{run_relative_artifcat_path}"
    tags = {
        "userid": userid,
        "tool_box_version": tool_box_version,
        "data_settings": json.dumps(data_settings)
    }
    tags.update(config)
    created_model_version = mlflow.register_model(model_uri, job_name, tags=tags)
    client = MlflowClient()
    client.set_registered_model_alias(job_name, "production", created_model_version.version)


def load_model(model_name,version, mlflow_url):
    mlflow.set_tracking_uri(mlflow_url)
    model_uri = f"models:/{model_name}/{version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model