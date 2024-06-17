from .pipeline import PeakShavingPipeline
from toolbox.timeseries.series import convert_df_to_series

class PeakShavingUseCase():
    def fit(self, data, train_config, model_name):
        pipeline = PeakShavingPipeline()
        data = list(convert_df_to_series(data))
        pipeline.fit(data)
        return pipeline, {}, None

    def get_model_signature(self):
        return None