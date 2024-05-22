from toolbox.estimation import pipelines
from toolbox.estimation.pipelines import DartNHITS, DartProphet, LinearReg, Baseline

def get_pipeline(pipeline_name):
    return getattr(pipelines, pipeline_name)

def get_all_pipelines():
    return [DartProphet, LinearReg, DartNHITS, Baseline]