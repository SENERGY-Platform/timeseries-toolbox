from ml_config import Config
from data.loaders.load import get_data_loader
from tasks.load import get_task

def get_data(data_loader_name, data_settings):
    dataloader = get_data_loader(data_loader_name, data_settings)
    return dataloader.get_data()

def run():    
    config = Config()
    task_name = config.TASK
    task_settings = config.TASK_SETTINGS
    data_settings = config.DATA_SETTINGS
    data_loader_name = config.DATA_SOURCE

    task = get_task(task_name, task_settings)
    data_df = get_data(data_loader_name, data_settings)

    task.run(data_df, task_settings)