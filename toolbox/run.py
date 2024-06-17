from toolbox.ml_config import Config
from toolbox.data.loaders.load import get_data_loader
import logging 
import sys 

from toolbox.ml_config import Config
from toolbox.tasks.fit.anomaly import anomaly
from toolbox.tasks.fit.peak_shaving import peak_shaving
from toolbox.tasks.load_shifting import load_shifting


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def get_data(data_loader_name, data_settings):
    dataloader = get_data_loader(data_loader_name, data_settings)
    return dataloader.get_data()

def get_task(task_name, config: Config):
    task_settings = config.TASK_SETTINGS
    mlflow_url = config.MLFLOW_URL
    job_name = config.EXPERIMENT_NAME
    userid = config.USER_ID 
    toolbox_version = config.TOOLBOX_VERSION 
    data_settings = config.DATA_SETTINGS

    if task_name == anomaly.TASK_NAME:
        return anomaly.Anomaly(task_settings, mlflow_url, job_name, userid, toolbox_version, data_settings)
    elif task_name == load_shifting.TASK_NAME:
        return load_shifting.LoadShifting(task_settings)
    elif task_name == peak_shaving.TASK_NAME:
        return peak_shaving.PeakShaving(task_settings, mlflow_url, job_name, userid, toolbox_version, data_settings)

def run():    
    config = Config()
    task_name = config.TASK
    data_settings = config.DATA_SETTINGS
    data_loader_name = config.DATA_SOURCE

    task = get_task(task_name, config)
    data_df = get_data(data_loader_name, data_settings)

    task.run(data_df)