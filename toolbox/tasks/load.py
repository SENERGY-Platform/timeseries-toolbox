from tasks.timeseries.estimation.estimation import ConsumptionEstimationTask
from tasks.timeseries.anomaly.anomaly import AnomalyTask
from tasks.load_shifting.run import LoadShifting

def get_task(task_name, task_settings):
    if task_name == 'anomaly':
        return AnomalyTask(task_settings)
    elif task_name == 'estimation':
        return ConsumptionEstimationTask(task_settings)
    elif task_name == 'load_shifting':
        return LoadShifting(task_settings)