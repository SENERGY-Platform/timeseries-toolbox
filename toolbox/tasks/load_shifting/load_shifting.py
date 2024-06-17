
import mlflow

from toolbox.load_shifting.load_shifting import LoadShifting
from toolbox.data.loaders.s3.s3 import S3DataLoader
from toolbox.tasks.task import Task

TASK_NAME = "load_shifting"

class LoadShifting(Task):
    def __init__(
        self, 
        task_settings,
        data_settings,
        mlflow_url, 
    ) -> None:
        super().__init__()
        self.task_settings = task_settings
        self.mlflow_url = mlflow_url
        self.data_settings = data_settings

    def store_shifted_loads(self, optimal_shifted_loads):
        print('Store optimal shifted loads')
        s3 = S3DataLoader(self.data_settings.s3_url, self.data_settings.aws_access, self.data_settings.aws_secret)
        s3.put_data("results", self.data_settings.file_name, optimal_shifted_loads)
        
        #with tempfile.TemporaryDirectory() as tmp_dir:
        #    path = Path(tmp_dir, "features.pickle")
        #    with open(path, 'wb') as f:
        #        pickle.dump(optimal_shifted_loads, f)
        #    run_name = f"with optimized hyperparameters"
        #    mlflow.end_run()
        #    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        #        mlflow.log_artifact(path)

    def run(self, data):
        mlflow.set_tracking_uri(self.mlflow_url)
        optimal_shifted_loads = LoadShifting(self.mlflow_url).run_load_shifting(data)        
        self.store_shifted_loads(optimal_shifted_loads)
