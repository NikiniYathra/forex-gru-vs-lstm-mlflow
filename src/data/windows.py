from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler


def get_device(preference="auto"):
    """Use CUDA when available, otherwise CPU."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


class WindowDataset:
    """Builds look-back windows on demand from a 2-D table of scaled features.

    ends[k] is the row where window k ends and where its label sits;
    the window covers rows ends[k]-lookback+1 ... ends[k].
    """

    def __init__(self, X, y, ends, lookback):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.ends = torch.from_numpy(np.ascontiguousarray(ends, dtype=np.int64))
        self.offsets = torch.arange(-(lookback - 1), 1)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, idx):
        """idx is a LIST of window numbers (a whole batch), so one indexing call builds the batch."""
        e = self.ends[idx]
        return self.X[e[:, None] + self.offsets], self.y[e]


def make_loader(dataset, batch_size, shuffle, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    base = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(base, batch_size, drop_last=False), batch_size=None)


def load_split(folder, split, lookback):
    data = np.load(Path(folder) / "data.npz")
    return WindowDataset(data[f"{split}_X"], data[f"{split}_y"], data[f"{split}_ends"], lookback)