from toolbox.anomaly.pipelines.custom_pipeline import AnomalyPipeline
from toolbox.anomaly.pipelines.cnn.cnn_autoencoder import Autoencoder
from toolbox.anomaly.pipelines.cnn.dataset import DataSet

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
        plot_enabled,
        kernel_size
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
            plot_enabled,
            window_length
        )
        self.model = Autoencoder(latent_dims, window_length, kernel_size)

    def create_dataset(self, data):
        # 2D Numpy Array to Torch Dataset
        print(data)
        assert data.ndim == 2, "Window data must be 2d"
        assert data.shape[1] == self.window_length, "Window length does not match configured window_length"
        return DataSet(data)

    @staticmethod
    def get_hyperparams(freq, train_ts, window_length):
        kernel_sizes = [2,5,7,10]
        kernel_sizes = filter(lambda kernel: kernel < window_length, kernel_sizes)
        return {
            'batch_size':  [8, 16, 32, 64, 128],
            'op': ['ADAM', 'SGD'],
            'lr': [0.01, 0.001, 0.0001],
            'loss': ['L1', 'MSE'],
            'latent_dims': [32],
            'num_epochs': [20],
            'early_stopping_patience': [10],
            'early_stopping_delta': [0],
            'out_dir': ["."],
            'kernel_size': kernel_sizes
        }