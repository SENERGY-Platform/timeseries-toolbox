import mlflow

from toolbox.tasks.task import Task
from toolbox.anomaly.anomaly import AnomalyTask
from toolbox.peak_shaving.use_case import PeakShavingUseCase
from toolbox.model_registry import store_model


class Fit(Task):
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
        use_case_name = self.task_settings.use_case
        use_case = ""
        if use_case_name == "anomaly":
            use_case = AnomalyTask()
        elif use_case_name == "peak_shaving":
            use_case = PeakShavingUseCase()
        
        train_config = self.task_settings.model_parameter
        model_name = self.task_settings.model_name
        pipeline, metrics, plots = use_case.fit(data, train_config, model_name)
        store_model(pipeline, self.userid, train_config, self.job_name, self.tool_box_version, metrics, self.data_settings.__dict__, task.get_model_signature())
