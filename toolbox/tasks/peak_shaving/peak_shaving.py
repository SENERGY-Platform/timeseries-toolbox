from toolbox.peak_shaving.pipeline import PeakShavingPipeline
from toolbox.tasks.task import Task
from toolbox.model_registry import store_model

import mlflow 

class PeakShaving(Task):
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
        pipeline = PeakShavingPipeline()
        pipeline.fit(data)
        train_config = {} # Could be a parameter in the future if needed
        metrics = {}
        store_model(pipeline, self.userid, train_config, self.job_name, self.tool_box_version, metrics, self.data_settings, None)
    
