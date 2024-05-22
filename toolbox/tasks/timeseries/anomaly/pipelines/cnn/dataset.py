from torch.utils.data import Dataset
import torch 

class DataSet(Dataset):
    def __init__(self, data):
        # data: [NUMBER_SAMPLES x WINDOW_SIZE]
        # sequence_length: Input Size of Transformer model == Number of Input Tokens
        # token_emb_dim: Length of Token Embedding 

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)