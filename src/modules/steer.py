"""
Used for reference in the benchmarks.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class SteerNet(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, past_frames, past_xyz):
        """
        past_frames: (B, T, C, W, H)
        past_xyz: (B, T, 3)
        """
        B = past_frames.shape[0]
        return torch.randn(B, 128)


# let's test the model
if __name__ == "__main__":
    B, T = 2, 3
    past_frames = torch.randn((B, T, 3, 1164 // 2, 874 // 2))
    past_xyz = torch.randn((B, T, 3))

    model = SteerNet()
    print(model(past_frames, past_xyz).shape)
