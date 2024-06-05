import logging 
import sys 

from toolbox.general_pipelines.train.pipeline import TrainPipeline
from toolbox.general_pipelines.inference.pipeline import InferencePipeline
from toolbox.data.preprocessors.normalization import Normalizer
from toolbox.data.preprocessors.resampling import Resampler
from toolbox.data.preprocessors.smoothing import Smoothing
from toolbox.data.preprocessors.duplicates import Duplicates
from toolbox.anomaly.pipelines.quantil import Quantil
from toolbox.anomaly.pipelines.isolation import Isolation 
from toolbox.data.preprocessors.drop_ms import DropMs

from torch.utils.data import DataLoader
import mlflow
import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_data_summary(prefix, series):
    logger.debug("###")
    logger.debug(prefix)
    logger.debug(series.describe())
    if series.size > 0:
        logger.debug(f"First: {series.index[0]} - {series.iloc[0]}")
        logger.debug(f"Last: {series.index[-1]} - {series.iloc[-1]}")
    logger.debug("###")

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
        self.strategy = "isolation" # TODO configurable
        self.setup_anomaly_scorer()

    def __check_input_data_size(self, data, data_title):
        assert data.size >= self.window_length, f"Not enough {data_title} data to build at least one window"

    def fit(self, train_data, val_data):
        logger.debug("Start model fitting")
        log_data_summary("Raw Train Data", train_data)
        log_data_summary("Raw Val Data", val_data)
        self.training_max_value = train_data.max()
        
        train_data = self._preprocess_df(train_data)
        log_data_summary("Preprocessed Train Data", train_data)
        val_data = self._preprocess_df(val_data)
        log_data_summary("Preprocessed Val Data", val_data)

        self.__check_input_data_size(train_data, "Train")
        self.__check_input_data_size(val_data, "Val")

        train_data = self.convert_data(train_data)
        logger.debug(f"Train: Model Input/Windows: {train_data.size}: {train_data[:5]}")
        val_data = self.convert_data(val_data)
        logger.debug(f"Val: Model Input/Windows: {val_data.size}: {val_data[:5]}")
        
        train_dataset = self.create_dataset(train_data)
        logger.debug(f"Train Dataset Length: {len(train_dataset)}")
        val_dataset = self.create_dataset(val_data)
        logger.debug(f"Val Dataset Length: {len(val_data)}")

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

    def setup_anomaly_scorer(self):
        if self.strategy == 'quantil':
            self.quantil = Quantil()
           
        elif self.strategy == 'isolation':
            self.isolation = Isolation()

    def get_anomalies(self, reconstruction_errors):
        if self.strategy == 'quantil':
            anomaly_indices, _ = self.quantil.check(reconstruction_errors)
           
        elif self.strategy == 'isolation':
            anomaly_indices = self.isolation.check(reconstruction_errors, 0.005)
            # TODO: Cut of previous reconstruction error list self.test_losses
            
        return anomaly_indices
            
    def _predict(self, raw_data):
        # Will only run prediction on last window
        preprocessed_data = self._preprocess_without_smoothing(raw_data)
        smoothed_data = self._preprocess_df(raw_data)

        # Cut beginning of data
        inference_data = smoothed_data[-self.window_length:]
        log_data_summary("Preprocessed Inference Data", inference_data)

        self.__check_input_data_size(inference_data, "Inference")

        window_data = self.convert_data(smoothed_data)
        test_dataset = self.create_dataset(window_data)
        logger.debug(f"Inference Dataset Length: {len(test_dataset)}")
        dataloader = DataLoader(test_dataset, batch_size=1)
        pipeline = InferencePipeline(self.model, dataloader, self.loss)
        self.test_losses, self.test_recons = pipeline.run()

        anomaly_indices = self.get_anomalies(self.test_losses)
        reconstructions = self.convert_to_numpy(self.test_recons) 
        logger.debug(f"Reconstructions: {reconstructions.shape}")

        reconstructions = reconstructions[0]
        anomalous_time_window = preprocessed_data[-self.window_length:]
        anomalous_time_window_smooth = smoothed_data[-self.window_length:]
    
        return reconstructions, anomaly_indices, self.test_losses, anomalous_time_window, anomalous_time_window_smooth, self.isolation.get_all_reconstruction_errors()

    def _preprocess_without_smoothing(self, data):
        drop = DropMs()
        data = drop.run(data)
        log_data_summary("After: Drop ms", data)

        dup = Duplicates()
        data = dup.run(data)
        log_data_summary("After: Deduplication", data)

        norm = Normalizer()
        data = norm.run(data, self.training_max_value)
        log_data_summary("After: Normalization", data)

        re = Resampler()
        data = re.run(data)
        log_data_summary("After: Resampling", data)

        return data 

    def _preprocess_df(self, data):
        data = self._preprocess_without_smoothing(data)

        smooter = Smoothing()
        data = smooter.run(data)
        log_data_summary("After: Smoothing", data)

        return data
    
    def predict(self, context, data, params=None):
        return self._predict(data)    

    def convert_data(self, data_series):
        stride = 1 # TODO in task settings
        values = list(data_series)
        windows = []

        start = 0
        end = self.window_length

        while end <= len(values):
            window = values[start:end]
            windows.append(window)
            start += stride
            end = start + self.window_length

        return np.asarray(windows)