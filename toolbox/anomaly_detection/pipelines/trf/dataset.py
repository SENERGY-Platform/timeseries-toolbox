from torch.utils.data import Dataset
import torch

class WindowDataset(Dataset):
    def __init__(self, data, sequence_length, token_emb_dim):
        # data: [NUMBER_SAMPLES x WINDOW_SIZE]
        # sequence_length: Input Size of Transformer model == Number of Input Tokens
        # token_emb_dim: Length of Token Embedding 

        # BATCH x SEQUENCE_LENGTH x TOKEN_DIM
        self.data = data
        self.sequence_length = sequence_length
        self.token_emb_dim = token_emb_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.reshape(self.sequence_length, self.token_emb_dim) 

        return torch.tensor(sample, dtype=torch.float32)
