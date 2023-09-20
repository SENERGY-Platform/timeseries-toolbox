from toolbox.anomaly_detection.pipelines.custom_pipeline import AnomalyPipeline
from toolbox.anomaly_detection.pipelines.cnn.cnn_autoencoder import Autoencoder
from toolbox.anomaly_detection.pipelines.cnn.dataset import DataSet

class CNNAnomalyPipeline(AnomalyPipeline):
    def __init__(
        self, 
        window_length,
        batch_size,
        lr,
        num_epochs,
        loss,
        op,
        out_dir,
        early_stopping_patience,
        early_stopping_delta,
        latent_dims,
        plot_enabled
    ):
        super().__init__(
            batch_size,
            lr,
            num_epochs,
            loss,
            op,
            out_dir,
            early_stopping_patience,
            early_stopping_delta,
            plot_enabled
        )
        self.window_length = window_length
        self.model = Autoencoder(latent_dims)

    def create_dataset(self, data):
        # 2D Numpy Array to Torch Dataset
        assert data.ndim == 2
        assert data.shape[1] == self.window_length
        return DataSet(data)

    @staticmethod
    def get_hyperparams(freq, train_ts):
        return {
            'batch_size':  [8, 16, 32, 64, 128],
            # TODO keine Hyperparameter fuers Training :/ 'quantil': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98],
            'op': ['ADAM', 'SGD'],
            'lr': [0.01, 0.001, 0.0001],
            'loss': ['L1', 'MSE'],
            'latent_dims': [32],
            'num_epochs': [20],
            'early_stopping_patience': [10],
            'early_stopping_delta': [0],
            'out_dir': ["."]
        }