from toolbox.tasks.timeseries.anomaly.pipelines.cnn.pipeline import CNNAnomalyPipeline
from toolbox.tasks.timeseries.anomaly.pipelines.trf.pipeline import TRFAnomalyPipeline

def get_pipeline(pipeline_name):
    if pipeline_name == "transformer":
        return TRFAnomalyPipeline
    elif pipeline_name == "cnn":
        return CNNAnomalyPipeline