from toolbox.general_pipelines.train.pipeline import TrainPipeline
from toolbox.general_pipelines.inference.pipeline import InferencePipeline

from torch.utils.data import DataLoader
import torch 
import mlflow

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
        plot_enabled
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

    def fit(self, train_data, val_data):
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

    def calc_threshold(self, losses, quantil):
        quantiles = torch.tensor([quantil], dtype=torch.float32)
        quants = torch.quantile(losses, quantiles)
        threshold = quants[0]
        return threshold

    def get_anomalies(self, strategy='quantil', quantil='95'):
        if strategy == 'quantil':
            threshold = self.calc_threshold(self.train_losses, quantil)
            anomaly_indices = torch.where(self.test_losses > threshold)[0]
            normal_indices = torch.where(self.test_losses < threshold)[0]
            
            return anomaly_indices, normal_indices

    def predict_with_quantil(self, data, quantil):
        if quantil:
            self.quantil = quantil
        
        return self._predict(data)

    def _predict(self, data):
        test_dataset = self.create_dataset(data)
        dataloader = DataLoader(test_dataset, batch_size=64)
        pipeline = InferencePipeline(self.model, dataloader, self.loss)
        self.test_losses, self.test_recons = pipeline.run()

        anomaly_indices, normal_indices = self.get_anomalies("quantil", self.quantil)
        reconstructions = self.convert_to_numpy(self.test_recons) 
        return reconstructions, anomaly_indices, normal_indices, self.test_losses

    def set_quantil(self, quantil):
        self.quantil = quantil
    
    def predict(self, context, data, params=None):
        return self._predict(data)    