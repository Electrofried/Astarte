# astarte/dataset.py
import torch
from torch.utils.data import Dataset

class RollingTextDataset(Dataset):
    def __init__(self, token_ids, chunk_length):
        self.token_ids = token_ids
        self.chunk_length = chunk_length

    def __len__(self):
        # The number of samples is the total tokens minus the chunk length.
        return max(0, len(self.token_ids) - self.chunk_length)

    def __getitem__(self, idx):
        # The input is a chunk of tokens, and the target is the token immediately following it.
        chunk = self.token_ids[idx: idx + self.chunk_length]
        target = self.token_ids[idx + self.chunk_length]
        return torch.tensor(chunk, dtype=torch.long), target
