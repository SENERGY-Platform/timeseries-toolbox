import mlflow

from parameter_tuning.tune import run_hyperparameter_tuning_for_each_model
from model_selection.selection import train_best_models_and_test

from model_registry import store_model


class SelectBestModel(Task):
    def __init__(
        self, 
        task_settings, 
        mlflow_url, 
        job_name,
        userid,
        tool_box_version,
        data_settings
    ) -> None:
        super().__init__()
        self.task_settings = task_settings
        self.job_name = job_name
        self.tool_box_version = tool_box_version
        self.userid = userid
        self.data_settings = data_settings
        mlflow.set_tracking_uri(mlflow_url)

    def run(self, data):    
        task = AnomalyTask()
        train_config = self.task_settings.model_parameter
        model_name = self.task_settings.model_name
        pipeline, metrics, plots = task.fit(data, train_config, model_name)

        best_config_per_model = run_hyperparameter_tuning_for_each_model(models, experiment_name, selection_metric, train_data, metric_direction, task, task_settings.frequency)
        print(f'Best configs per model: {best_config_per_model}')

        best_metric_value, best_checkpoint, best_config = train_best_models_and_test(models, best_config_per_model, train_data, test_data, metric_direction, selection_metric, experiment_id, config.MLFLOW_URL, task)
        print(f'Best value: {best_metric_value}, Best config: {best_config}')

        # Store model with best metric
        store_model(pipeline, self.userid, train_config, self.job_name, self.tool_box_version, metrics, self.data_settings.__dict__, task.get_model_signature())
