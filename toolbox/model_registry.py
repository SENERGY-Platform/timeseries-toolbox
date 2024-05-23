import mlflow
import mlflow.pyfunc

def store_model(checkpoint, userid, config, experiment, model_artifact_name, commit):
    print('store model')
    model_artifact = checkpoint.to_dict()['model']
    print(model_artifact.quantil)

    mlflow.end_run()
    mlflow.set_experiment(experiment)
    run_relative_artifcat_path = 'models'

    # Create a new model version and save model
    with mlflow.start_run(run_name="store-best-model") as run:
        mlflow.log_metrics({})
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
    mlflow.register_model(model_uri, model_artifact_name, tags=tags)