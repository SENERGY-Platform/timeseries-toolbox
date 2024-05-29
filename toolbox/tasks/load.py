from toolbox.tasks.fit_model.fit import Fit
from toolbox.tasks.load_shifting.run import LoadShifting
from toolbox.ml_config import Config

def get_task(task_name, config: Config):
    if task_name == 'ml_fit':
        return Fit(config.task_settings, config.MLFLOW_URL, config.EXPERIMENT_NAME, config.USER_ID, config.COMMIT)
    elif task_name == 'load_shifting':
        return LoadShifting(config.task_settings)