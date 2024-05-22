from toolbox.tasks.timeseries.task import TimeSeriesTask
from toolbox.tasks.timeseries.anomaly.plots import plot_losses, plot_reconstructions
from toolbox.tasks.timeseries.anomaly.load import get_pipeline
from toolbox.model_registry import store_model
import numpy as np 
from sklearn.model_selection import train_test_split
import ray

QUANTILS = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]

class AnomalyTask(TimeSeriesTask):
    def __init__(self, task_settings) -> None:
        super().__init__(task_settings.frequency)
        self.window_size = task_settings.window_size
        self.stride = task_settings.stride
        self.task_settings = task_settings

    def fit_and_evaluate_model(self, train_data, test_data, config):
        # data: numpy array [NUMBER_SAMPLE x WINDOW_SIZE] 

        pipeline_name = config['pipeline']

        # Remove pipeline name to pass remaining configs as model parameters
        del config['pipeline']
        del config['freq']
        config['window_length'] = self.window_size
        config['plot_enabled'] = False
        
        pipeline = get_pipeline(pipeline_name)(**config)

        train_data, validation_data = self.split_data(train_data)
        pipeline.fit(train_data, validation_data)

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

    def convert_data(self, data_df):
        values = list(data_df['value'])
        windows = []

        start = 0
        end = self.window_size

        while end < len(values):
            window = values[start:end]
            windows.append(window)
            start += self.stride
            end = start + self.window_size

        return np.asarray(windows)

    def split_data(self, data):
        return train_test_split(data, shuffle=True, test_size=0.25)

    def get_pipeline_hyperparams(self, pipeline_name, train_ts):
        return get_pipeline(pipeline_name).get_hyperparams(self.frequency, train_ts, self.window_size)

    def fit(self, window_data):
        # Fit single model with specific parameters
        train_config = self.task_settings.model_parameter
        train_config['plot_enabled'] = False
        train_config['out_dir'] = '.'
        model_name = train_config['model_name']
        pipeline = get_pipeline(model_name)(**train_config)
        train_data, validation_data = self.split_data(window_data)
        pipeline.fit(train_data, validation_data) 

        # Log to MLFLOW
        model_artifact_name = "TODO"
        userid = "TODO"
        experiment = "TODO"
        commit = "TODO"
        store_model(pipeline, userid, train_config, experiment, model_artifact_name, "anomaly", commit)

    def tune(self):
        # TODO: Parameter Tuning for a specfic model -> see parameter_tuning.py
        pass 

    def select_best(self):
        # TODO: Train and evaluate multiple models + parameter tuning -> see select_best_model.py
        pass

    def run(self, data_df):
        window_data = self.convert_data(data_df)
        self.fit(window_data)