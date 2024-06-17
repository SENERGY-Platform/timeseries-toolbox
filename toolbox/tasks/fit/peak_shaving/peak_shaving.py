from toolbox.peak_shaving.pipeline import PeakShavingPipeline
from toolbox.tasks.fit.fit import Fit 

TASK_NAME = "peak_shaving"

class PeakShaving(Fit):
    def __init__(self, task_settings, mlflow_url, job_name, userid, tool_box_version, data_settings) -> None:
        super().__init__(task_settings, mlflow_url, job_name, userid, tool_box_version, data_settings)
        
    def fit(self, data, train_config, model_name):
        pipeline = PeakShavingPipeline()
        data = list(data)
        pipeline.fit(data)
        return pipeline, {}, None

    def get_model_signature(self):
        return None