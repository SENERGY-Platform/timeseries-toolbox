from toolbox.tasks.timeseries.anomaly.pipelines.custom_pipeline import AnomalyPipeline
from toolbox.tasks.timeseries.anomaly.pipelines.trf.transformer import TransformerTimeSeriesEncoder
from toolbox.tasks.timeseries.anomaly.pipelines.trf.dataset import WindowDataset

class TRFAnomalyPipeline(AnomalyPipeline):
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
        num_enc_layers,
        num_heads,
        emb_dim,
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
            plot_enabled,
            window_length
        )

        self.sequence_length = max(window_length * 0.1, 1) # 10th of the complete window as tokens - minimum one step
        self.token_emb_dim = int(window_length / self.sequence_length)

        self.model = TransformerTimeSeriesEncoder(number_encoder_layers=num_enc_layers, 
                                                        number_heads=num_heads, 
                                                        embedding_dimension=emb_dim, 
                                                        sequence_length=self.sequence_length, 
                                                        token_length=self.token_emb_dim)

    def create_dataset(self, data):
        # 2D Numpy Array to Torch Dataset
        assert data.ndim == 2
        assert data.shape[1] == self.window_length
        return WindowDataset(data, self.sequence_length, self.token_emb_dim)

    def convert_to_numpy(self, tensor):
        # Convert Model Output back to 2D Numpy Array
        return tensor.flatten(1)

    @staticmethod
    def get_hyperparams(freq, train_ts, window_length):
        return {
            'batch_size':  [8],
            'op': ['ADAM'],
            'lr': [0.01],
            'loss': ['L1'],
            'num_epochs': [5],
            'early_stopping_patience': [10],
            'early_stopping_delta': [0],
            'out_dir': ["."],
            'num_enc_layers': [5],
            'num_heads': [5],
            'emb_dim': [50]
        }
        return {
            'batch_size':  [8, 16, 32, 64, 128],
            'op': ['ADAM', 'SGD'],
            'lr': [0.01, 0.001, 0.0001],
            'loss': ['L1', 'MSE'],
            'num_epochs': [20],
            'early_stopping_patience': [10],
            'early_stopping_delta': [0],
            'out_dir': ["."],
            'num_enc_layers': [5],
            'num_heads': [5],
            'emb_dim': [50]
        }