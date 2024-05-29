from toolbox.tasks.task import Task
from toolbox.anomaly.anomaly import AnomalyTask
from toolbox.model_registry import store_model

import mlflow

class Fit(Task):
    def __init__(
        self, 
        task_settings, 
        mlflow_url, 
        job_name,
        userid,
        commit,
    ) -> None:
        super().__init__()
        self.task_settings = task_settings
        self.job_name = job_name
        self.commit = commit
        self.userid = userid
        mlflow.set_tracking_uri(mlflow_url)

    def run(self, data):
        task = AnomalyTask()
        # TODO maybe do data splitting independtly of the task, just provide single split, growing window or moving window
        train_config = self.task_settings.model_parameter
        model_name = self.task_settings.model_name
        pipeline, metrics, plots = task.fit(data, train_config, model_name)
        
        # Log to MLFLOW
        store_model(pipeline, self.userid, train_config, self.job_name, self.commit, metrics)
