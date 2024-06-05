from toolbox.anomaly.plots import plot_losses, plot_reconstructions 
from sklearn.model_selection import train_test_split
from toolbox.anomaly.pipelines.cnn.pipeline import CNNAnomalyPipeline
from toolbox.anomaly.pipelines.trf.pipeline import TRFAnomalyPipeline
from toolbox.data.preprocessors.sorting import Sorter



class AnomalyTask():
    def __init__(self) -> None:
        super().__init__()

    def _get_pipeline(self, pipeline_name):
        if pipeline_name == "transformer":
            return TRFAnomalyPipeline
        elif pipeline_name == "cnn":
            return CNNAnomalyPipeline

    def fit(self, train_data, config, model_name):
        config['plot_enabled'] = False
        config['out_dir'] = '.'
        pipeline = self._get_pipeline(model_name)(**config)
        sorter = Sorter()
        train_data = sorter.run(train_data)
        train_data, validation_data = self.split_data(train_data)
        pipeline.fit(train_data, validation_data)
        return pipeline, {}, None

    def split_data(self, data):
        return train_test_split(data, shuffle=False, test_size=0.25)

    def get_pipeline_hyperparams(self, pipeline_name, train_ts):
        return self._get_pipeline(pipeline_name).get_hyperparams(self.frequency, train_ts, self.window_size)
