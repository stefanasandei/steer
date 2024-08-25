"""
The dataset class used to access elements.
"""

import torch
from torch.utils.data import Dataset

import data.fetch
from config import cfg


class CommaDataset(Dataset):

    def __init__(self, path: str, chunk_num: int, train: bool):
        super().__init__()

        if not data.fetch.dataset_present(cfg["data"]["path"]):
            data.fetch.download_dataset(cfg["data"]["path"])

        self.path = f"{path}/Chunk_{chunk_num}"

    def __len__(self):
        # sample values to figure out input & output shapes
        return 128

    def __getitem__(self, idx) -> torch.Tensor:
        seq_len = 20

        past_frames = torch.zeros((seq_len, 3, 32, 32))  # (T, C, W, H)
        past_path = torch.zeros((seq_len, 3))

        future_path = torch.zeros((seq_len, 3))

        steering_angle = torch.zeros((1))
        speed = torch.zeros((1))

        return {
            "past_frames": past_frames,
            "past_path": past_path,
        }, {
            "future_path": future_path,
            "steering_angle": steering_angle,
            "speed": speed
        }
