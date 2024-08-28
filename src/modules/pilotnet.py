import torch
import torch.nn as nn
from torch.nn import functional as F

from config import cfg

# Used for reference in the benchmarks


class PilotNet(nn.Module):
    """
    roughly following Nvidia's paper 'End to End Learning for Self-Driving Cars'
    modified to take in multiple frames (using concatenation) and also output future path
    """

    def __init__(
        self,
        num_past_frames=cfg["model"]["past_steps"] + 1,
        num_future_steps=cfg["model"]["future_steps"],
    ):
        super().__init__()

        self.num_future_steps = num_future_steps

        # image processing
        self.conv1 = nn.Conv2d(
            in_channels=num_past_frames * 3, out_channels=24, kernel_size=5, stride=2
        )
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # merge image features with paths
        # also "steering controller"
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        # output layers
        self.fc_future_path = nn.Linear(
            10, num_future_steps * 3
        )  # predicting (x, y, z) for N future steps
        self.fc_steering = nn.Linear(10, 1)  # steering angle in radians
        self.fc_speed = nn.Linear(10, 1)  # speed in m/s

    def forward(self, past_frames, past_xyz):
        """
        past_frames: (B, T, C, W, H)
        past_xyz: (B, T, 3)
        """

        past_frames = past_frames.view(
            past_frames.shape[0], -1, past_frames.shape[3], past_frames.shape[4]
        )  # (B, T*C, W, H)

        # process the past frames with CNN layers
        x = F.relu(self.conv1(past_frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # flatten
        x = x.view(x.size(0), -1)

        # concatenate CNN features with past xyz coordinates
        x = torch.cat((x, past_xyz.view(past_xyz.size(0), -1)), dim=1).float()

        # controller, give it more
        # time to "think" ;)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # putputs
        future_path = self.fc_future_path(x).view(
            -1, self.num_future_steps, 3
        )  # reshape to (N, 3)
        steering = self.fc_steering(x)
        speed = self.fc_speed(x)

        return {
            "future_path": future_path,
            "steering_angle": steering.squeeze(),
            "speed": speed.squeeze(),
        }
