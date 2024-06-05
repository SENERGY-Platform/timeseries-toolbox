from toolbox.ml_config import Config
from toolbox.data.loaders.load import get_data_loader
from toolbox.tasks.load import get_task
import logging 
import sys 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def get_data(data_loader_name, data_settings):
    dataloader = get_data_loader(data_loader_name, data_settings)
    return dataloader.get_data()

def run():    
    config = Config()
    task_name = config.TASK
    data_settings = config.DATA_SETTINGS
    data_loader_name = config.DATA_SOURCE

    task = get_task(task_name, config)
    data_df = get_data(data_loader_name, data_settings)

    task.run(data_df)