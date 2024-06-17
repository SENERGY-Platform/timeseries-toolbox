from .pipeline import PeakShavingPipeline

class PeakShavingUseCase():
    def fit(self, data, train_config, model_name):
        pipeline = PeakShavingPipeline()
        data = list(data)
        pipeline.fit(data)
        return pipeline, {}, None

    def get_model_signature(self):
        return None