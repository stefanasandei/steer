"""
The dataset class used to access elements.
"""

import torch
from torch.utils.data import Dataset


class CommaDataset(Dataset):

    def __init__(self, path: str):
        super().__init__()

        self.path = path

    def __len__(self):
        pass

    def __getitem__(self, idx) -> torch.Tensor:
        pass
