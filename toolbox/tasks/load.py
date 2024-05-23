from toolbox.tasks.fit_model.fit import Fit
from toolbox.tasks.load_shifting.run import LoadShifting

def get_task(task_name, task_settings):
    if task_name == 'ml_fit':
        return Fit(task_settings)
    elif task_name == 'load_shifting':
        return LoadShifting(task_settings)