import logging 
import sys 

from toolbox.general_pipelines.train.pipeline import TrainPipeline
from toolbox.general_pipelines.inference.pipeline import InferencePipeline
from toolbox.data.preprocessors.normalization import Normalizer
from toolbox.data.preprocessors.resampling import Resampler
from toolbox.data.preprocessors.smoothing import Smoothing
from toolbox.data.preprocessors.duplicates import Duplicates
from toolbox.anomaly.quantil import Quantil
from toolbox.anomaly.isolation import Isolation 
from toolbox.data.preprocessors.drop_ms import DropMs
from toolbox.anomaly.utils import Reconstruction

from torch.utils.data import DataLoader
import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

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

        train_window_data = self.convert_data(train_data)
        logger.debug(f"Train: Model Input/Windows: {train_window_data.size}: {train_window_data[:5]}")
        val_window_data = self.convert_data(val_data)
        logger.debug(f"Val: Model Input/Windows: {val_window_data.size}: {val_window_data[:5]}")
        
        train_dataset = self.create_dataset(train_window_data)
        logger.debug(f"Train Dataset Length: {len(train_dataset)}")
        val_dataset = self.create_dataset(val_window_data)
        logger.debug(f"Val Dataset Length: {len(val_dataset)}")

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        pipeline = TrainPipeline(self.model, train_dataloader, self.num_epochs, self.lr, val_dataloader, self.loss, self.op, self.out_dir, self.early_stopping_patience, self.early_stopping_delta, self.plot_enabled)
        trained_model, n_epochs = pipeline.train()
        self.model = trained_model

        # Calculate final train sample losses for threshold 
        pipeline = InferencePipeline(self.model, train_dataloader, self.loss)
        reconstruction_errors, _ = pipeline.run()

        self.set_train_reconstruction_errors(reconstruction_errors.numpy(), train_data)

    def set_train_reconstruction_errors(self, reconstruction_errors, data):
        # reconstruction_errors: Numpy Array [NUMBER_WINDOWS, 1]
        logger.debug(f"All Train Reconstruction Errors Shape: {reconstruction_errors.shape}") 
        reconstruction_errors = []
        for window_index in range(len(reconstruction_errors)):
            window_start_index = window_index*self.window_length
            window_end_index = window_start_index + self.window_length
            last_timestamp_of_window = data[window_start_index:window_end_index].index[-1]
            reconstruction_error = Reconstruction(last_timestamp_of_window, reconstruction_errors[window_index])
            reconstruction_errors.append(reconstruction_error)
            
        if self.strategy == 'isolation':
            self.isolation.set_all_reconstruction_errors(reconstruction_errors)
    
    def setup_anomaly_scorer(self):
        if self.strategy == 'quantil':
            self.quantil = Quantil()
           
        elif self.strategy == 'isolation':
            self.isolation = Isolation(pd.Timedelta(14, "d"), logger)

    def is_reconstruction_error_anomalous(self, reconstruction_error):
        if self.strategy == 'quantil':
            #TODO: old quantil check is based on batches, here single recon error will be passed. 
            # anomaly_indices, _ = self.quantil.check(reconstruction_error)
            pass 

        elif self.strategy == 'isolation':
            return self.isolation.check(reconstruction_error, 0.005)
            

    def get_all_reconstruction_errors(self):
        if self.strategy == 'isolation':
            return self.isolation.get_all_reconstruction_errors()

    def set_all_reconstruction_errors(self, all_reconstruction_errors):
        if self.strategy == 'isolation':
            return self.isolation.set_all_reconstruction_errors(all_reconstruction_errors)

    def convert_df_to_series(self, df: pd.DataFrame):
        df = df.set_index("time")
        return df["value"]

    def _predict(self, raw_data_df):
        # We cut the inference data to only keep the last window.
        # Previous data is only used for better smoothing
        # Output will therefore only be valid for one window

        raw_data_series = self.convert_df_to_series(raw_data_df)
        preprocessed_data = self._preprocess_without_smoothing(raw_data_series)
        smoothed_data = self._preprocess_df(raw_data_series)

        inference_data = smoothed_data[-self.window_length:]
        log_data_summary("Preprocessed Inference Data", inference_data)

        self.__check_input_data_size(inference_data, "Inference")

        window_data = self.convert_data(inference_data)
        test_dataset = self.create_dataset(window_data)
        logger.debug(f"Inference Dataset Length: {len(test_dataset)}")
        dataloader = DataLoader(test_dataset, batch_size=1)
        pipeline = InferencePipeline(self.model, dataloader, self.loss)
        reconstruction_errors, reconstructions = pipeline.run()

        reconstruction_error = Reconstruction(inference_data.index[-1], reconstruction_errors.numpy()[0])
        reconstruction_error_is_anomalous = self.is_reconstruction_error_anomalous(reconstruction_error)
        logger.debug(f"All Reconstruction Outputs: {reconstructions.shape}")

        reconstruction = reconstructions.numpy()[0]
        anomalous_time_window = preprocessed_data[-self.window_length:]
        anomalous_time_window_smooth = smoothed_data[-self.window_length:]
    
        return reconstruction, reconstruction_error_is_anomalous, anomalous_time_window, anomalous_time_window_smooth, self.isolation.get_all_reconstruction_errors()

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