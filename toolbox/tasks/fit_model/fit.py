from toolbox.tasks.task import Task
from toolbox.tasks.timeseries.anomaly.anomaly import AnomalyTask
from toolbox.model_registry import store_model

class Fit(Task):
    def __init__(self, task_settings) -> None:
        super().__init__()
        self.task_settings = task_settings

    def run(self, data):
        task = AnomalyTask()
        train_data, test_data = task.split_data(data)
        train_config = self.task_settings.model_parameter
        model_name = self.task_settings.model_name
        pipeline = task.fit_and_evaluate_model(train_data, test_data, train_config, model_name)
        
        # Log to MLFLOW
        model_artifact_name = "TODO"
        userid = "TODO"
        experiment = "TODO"
        commit = "TODO"
        store_model(pipeline, userid, train_config, experiment, model_artifact_name, "anomaly", commit)
