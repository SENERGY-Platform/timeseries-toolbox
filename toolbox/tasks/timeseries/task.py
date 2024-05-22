from toolbox.tasks.task import Task

class TimeSeriesTask(Task):
    def __init__(self, frequency) -> None:
        super().__init__()
        self.frequency = frequency