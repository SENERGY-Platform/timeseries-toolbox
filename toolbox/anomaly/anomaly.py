from toolbox.anomaly.plots import plot_losses, plot_reconstructions 
from sklearn.model_selection import train_test_split
from toolbox.anomaly.pipelines.cnn.pipeline import CNNAnomalyPipeline
from toolbox.anomaly.pipelines.trf.pipeline import TRFAnomalyPipeline
from toolbox.data.preprocessors.normalization import Normalizer
from toolbox.data.preprocessors.resampling import Resampler
from toolbox.data.preprocessors.smoothing import Smoothing
from toolbox.data.preprocessors.duplicates import Duplicates

QUANTILS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

class AnomalyTask():
    def __init__(self) -> None:
        super().__init__()

    def _get_pipeline(self, pipeline_name):
        if pipeline_name == "transformer":
            return TRFAnomalyPipeline
        elif pipeline_name == "cnn":
            return CNNAnomalyPipeline

    def fit(self, train_data, config, model_name):
        # data: numpy array [NUMBER_SAMPLE x WINDOW_SIZE] 

        config['plot_enabled'] = False
        config['out_dir'] = '.'
        pipeline = self._get_pipeline(model_name)(**config)
        training_max_value = train_data.max()
        train_data = self._preprocess_df(train_data, training_max_value)
        train_data, validation_data = self.split_data(train_data)
        pipeline.fit(train_data, validation_data)
        return pipeline, {}, None

    def _find_best_quantil(self, pipeline, test_data):
        # TODO: not needed really
        # Quantils are also parameters but not for training
        best_quantil = None
        results_per_quantil = {}
        best_loss = None
        for quantil in QUANTILS:
            reconstructions, anomaly_indices, normal_indices, test_losses = pipeline.predict_with_quantil(test_data, quantil)
            results_per_quantil[quantil] = {
                "reconstructions": reconstructions,
                "anomaly_indices": anomaly_indices,
                "normal_indices": normal_indices,
                "test_losses": test_losses
            }

            loss = test_losses.sum().item()

            if best_loss == None:
                best_loss = loss 
                best_quantil = quantil
            elif loss < best_loss:
                best_loss = loss
                best_quantil = quantil

        metrics = {
            "loss": best_loss
        }

        pipeline.set_quantil(best_quantil)

        # Generate plots
        plots = []
        reconstructions_of_best_quantil = results_per_quantil[best_quantil]['reconstructions']
        normal_indices_of_best_quantil =  results_per_quantil[best_quantil]['normal_indices']
        anomaly_indices_of_best_quantil =  results_per_quantil[best_quantil]['anomaly_indices']
        
        if len(reconstructions) > 0:
            if len(normal_indices_of_best_quantil) > 0:
                normal_recons_plot = plot_reconstructions(reconstructions_of_best_quantil, normal_indices_of_best_quantil, test_data, "Normal")
                plots.append(normal_recons_plot)

            if len(anomaly_indices_of_best_quantil) > 0:
                anomaly_recons_plot = plot_reconstructions(reconstructions_of_best_quantil, anomaly_indices_of_best_quantil, test_data, "Anomaly")
                plots.append(anomaly_recons_plot)

        losses_of_best_quantil = results_per_quantil[best_quantil]['test_losses']
        losses_hist = plot_losses(losses_of_best_quantil)
        plots.append(losses_hist)

        return pipeline, metrics, plots

    def split_data(self, data):
        return train_test_split(data, shuffle=True, test_size=0.25)

    def get_pipeline_hyperparams(self, pipeline_name, train_ts):
        return self._get_pipeline(pipeline_name).get_hyperparams(self.frequency, train_ts, self.window_size)

    def _preprocess_df(self, data, training_max_value):
        dup = Duplicates()
        data = dup.run(data)

        norm = Normalizer()
        data = norm.run(data, training_max_value)

        re = Resampler()
        data = re.run(data)

        smooter = Smoothing()
        data = smooter.run(data)

        return data
