from .pipeline import PeakShavingPipeline

class PeakShavingUseCase():
    def fit(self, data, train_config, model_name):
        pipeline = PeakShavingPipeline()
        pipeline.fit(data)
        return pipeline, None, None