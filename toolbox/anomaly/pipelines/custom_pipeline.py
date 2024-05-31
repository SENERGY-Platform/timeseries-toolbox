from toolbox.general_pipelines.train.pipeline import TrainPipeline
from toolbox.general_pipelines.inference.pipeline import InferencePipeline
from toolbox.data.preprocessors.normalization import Normalizer
from toolbox.data.preprocessors.resampling import Resampler
from toolbox.data.preprocessors.smoothing import Smoothing
from toolbox.data.preprocessors.duplicates import Duplicates
from toolbox.anomaly.pipelines.quantil import Quantil
from toolbox.anomaly.pipelines.isolation import Isolation 

from torch.utils.data import DataLoader
import mlflow
import numpy as np

class AnomalyPipeline(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        batch_size,
        lr,
        num_epochs,
        loss,
        op,
        out_dir,
        early_stopping_patience,
        early_stopping_delta,
        plot_enabled,
        window_length
    ):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr 
        self.num_epochs = num_epochs
        self.loss = loss
        self.op = op
        self.out_dir = out_dir
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.plot_enabled = plot_enabled
        self.window_length = window_length

    def fit(self, train_data, val_data):
        print("Start model fit")
        print(f"Training Raw Data: {train_data[:5]}")
        print(f"Val Raw Data: {val_data[:5]}")
        self.training_max_value = train_data.max()
        
        train_data = self._preprocess_df(train_data)
        print(f"Preprocessed Train Data: {train_data[:5]}")
        val_data = self._preprocess_df(val_data)
        print(f"Preprocessed Val Data: {val_data[:5]}")

        train_data = self.convert_data(train_data)
        print(f"Train: Model Input/Windows: {train_data.shape}: {train_data[:5]}")
        val_data = self.convert_data(val_data)
        print(f"Val: Model Input/Windows: {val_data.shape}: {val_data[:5]}")
        
        # TODO: check enough data in both datasets 

        train_dataset = self.create_dataset(train_data)
        val_dataset = self.create_dataset(val_data)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        pipeline = TrainPipeline(self.model, train_dataloader, self.num_epochs, self.lr, val_dataloader, self.loss, self.op, self.out_dir, self.early_stopping_patience, self.early_stopping_delta, self.plot_enabled)
        trained_model, n_epochs = pipeline.train()
        self.model = trained_model

        # Calculate final train sample losses for threshold 
        pipeline = InferencePipeline(self.model, train_dataloader, self.loss)
        all_losses, _ = pipeline.run()
        self.train_losses = all_losses
       
        return n_epochs, self.train_losses

    def setup_anomaly_scorer(self, strategy):
        if strategy == 'quantil':
            self.quantil = Quantil()
           
        elif strategy == 'isolation':
            self.isolation = Isolation()

    def get_anomalies(self, reconstruction_errors):
        if self.strategy == 'quantil':
            anomaly_indices, _ = self.quantil.check(reconstruction_errors)
           
        elif self.strategy == 'isolation':
            anomaly_indices = self.isolation.check(reconstruction_errors, 0.005)
            # TODO: Cut of previous reconstruction error list self.test_losses
            
        return anomaly_indices
            
    def predict_with_quantil(self, data, quantil):
        if quantil:
            self.quantil = quantil
        
        return self._predict(data)

    def _predict(self, raw_data):
        preprocessed_data = self._preprocess_without_smoothing(raw_data)
        smoothed_data = self._preprocess_df(raw_data)

        data = self.convert_data(smoothed_data)
        test_dataset = self.create_dataset(data)
        dataloader = DataLoader(test_dataset, batch_size=64)
        pipeline = InferencePipeline(self.model, dataloader, self.loss)
        self.test_losses, self.test_recons = pipeline.run()

        anomaly_indices, normal_indices = self.get_anomalies(self.test_losses)
        reconstructions = self.convert_to_numpy(self.test_recons) 
        
        if reconstructions.shape[0] > 0:
            anomalous_time_window = preprocessed_data[-self.window_length:]
            anomalous_time_window_smooth = smoothed_data[-self.window_length:]
    
            return reconstructions, anomaly_indices, normal_indices, self.test_losses, anomalous_time_window, anomalous_time_window_smooth

    def _preprocess_without_smoothing(self, data):
        # TODO: timestamp.replace(microsecond=0)
        dup = Duplicates()
        data = dup.run(data)

        norm = Normalizer()
        data = norm.run(data, self.training_max_value)

        re = Resampler()
        data = re.run(data)

        return data 

    def _preprocess_df(self, data):
        data = self._preprocess_without_smoothing(data)

        smooter = Smoothing()
        data = smooter.run(data)

        return data

    def set_quantil(self, quantil):
        self.quantil = quantil
    
    def predict(self, context, data, params=None):
        return self._predict(data)    

    def convert_data(self, data_series):
        print(data_series)
        stride = 1 # TODO in task settings
        values = list(data_series)
        windows = []

        start = 0
        end = self.window_length

        while end < len(values):
            window = values[start:end]
            windows.append(window)
            start += stride
            end = start + self.window_length

        return np.asarray(windows)