# astarte/dataset.py
import torch
from torch.utils.data import Dataset

class RollingTextDataset(Dataset):
    """
    Splits tokenized text into sequential chunks with periodic rest intervals.

    Every nth sample is a "pause" sample that returns a zeroed input and a flag.
    """
    def __init__(self, token_ids, chunk_length=512, pause_interval=2):
        self.token_ids = token_ids
        self.chunk_length = chunk_length
        self.pause_interval = pause_interval
        self.n_normal = len(token_ids) // chunk_length
        self.n_pause = self.n_normal // (pause_interval - 1)
        self.total_samples = self.n_normal + self.n_pause

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx % self.pause_interval == (self.pause_interval - 1):
            # Pause sample: no training target
            input_ids = torch.zeros(self.chunk_length, dtype=torch.long)
            target = -100
            pause = True
        else:
            k = idx - (idx // self.pause_interval)
            start = k * self.chunk_length
            end = start + self.chunk_length
            if end >= len(self.token_ids):
                chunk = self.token_ids[start:]
                padding = [0] * (self.chunk_length - len(chunk))
                input_ids = torch.tensor(chunk + padding, dtype=torch.long)
                target = 0
            else:
                chunk = self.token_ids[start:end]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                target = self.token_ids[end] if end < len(self.token_ids) else 0
            pause = False
        return input_ids, target, pause
