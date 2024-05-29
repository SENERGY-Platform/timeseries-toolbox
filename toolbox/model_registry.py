import mlflow
import mlflow.pyfunc

def store_model(model_artifact, userid, config, job_name, commit, metrics):
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
        "commit": commit
    }
    tags.update(config)
    mlflow.register_model(model_uri, job_name, tags=tags)

def load_model(model_name,version, mlflow_url):
    mlflow.set_tracking_uri(mlflow_url)
    model_uri = f"models:/{model_name}/{version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model