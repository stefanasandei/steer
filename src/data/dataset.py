"""
The dataset class used to access elements.
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pickle
import numpy as np
from PIL import Image

import data.fetch
from config import cfg
import lib.paths as paths

import matplotlib.pyplot as plt


class CommaDataset(Dataset):

    def __init__(self, path: str, chunk_num: int, train: bool, device: str):
        super().__init__()

        self.path = f"{path}/Chunk_{chunk_num}"
        self.frame_paths = []
        self.device = device

        # make sure we have the dataset processed
        if not data.fetch.dataset_present(cfg["data"]["path"]):
            data.fetch.download_dataset(cfg["data"]["path"])

        # load the train/val dataset, contains a list of valid frames
        # from one frame we can access past&future info
        split_path = f"{self.path}/train" if train else f"{self.path}/train"
        with open(split_path, "rb") as f:
            self.frame_paths = pickle.load(f)

        # the parent dir of the parent dir
        self.get_route_path = (
            lambda file_path: "/".join(file_path.rsplit("/", 2)[:-2]) + "/"
        )
        self.get_id = lambda file_path: file_path.rsplit(
            "/", 1)[-1].split(".")[0]

        # just convert to tensor
        self.frame_transform = ToTensor()

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        past_seq_len = cfg["model"]["past_steps"]
        future_seq_len = cfg["model"]["future_steps"]

        # find the current frame and its coresponding
        # data (timestamp)
        main_frame_path = self.frame_paths[idx]
        route_path = self.get_route_path(main_frame_path)
        frame_id = int(self.get_id(main_frame_path))

        # 1. load frame orientation & position data
        frame_data = np.load(f"{route_path}frame.npz")
        orientations = frame_data["orientation"]
        positions = frame_data["position"]

        # convert positions to reference frame
        local_path = paths.get_local_path(positions, orientations, frame_id)
        local_path = torch.tensor(
            local_path, dtype=torch.float32, device=self.device)

        # divide data into previous and future arrays
        future_path = local_path[frame_id + 1: frame_id + 1 + future_seq_len]
        past_path = local_path[
            frame_id - past_seq_len: frame_id + 1
        ]  # also include the current path

        # 2. load past frames (including the current one as last)
        past_frames = []
        for f_id in range(frame_id - past_seq_len, frame_id + 1):
            filename = str(f_id).zfill(6) + ".jpeg"
            frame = Image.open(f"{route_path}video/{filename}")
            past_frames.append(self.frame_transform(frame))

        # Stack frames into single array, (T, C, H, W)
        past_frames = torch.stack(past_frames).to(self.device).float()

        # 3. load can data (speed & steering angle)
        # todo: maybe return past_seq_len can data?
        can_data = np.load(f"{route_path}can_telemetry.npz")
        speed = torch.tensor(
            can_data["speed"][frame_id], device=self.device, dtype=torch.float32)
        steering_angle = torch.tensor(
            can_data["angle"][frame_id], device=self.device, dtype=torch.float32)

        return {
            "past_frames": past_frames,  # (T, C, W, H)
            "past_path": past_path,  # (T, 3)
        }, {
            "future_path": future_path,  # (T, 3)
            "steering_angle": steering_angle,  # (1,)
            "speed": speed,  # (1,)
        }
