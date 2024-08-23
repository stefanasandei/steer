import torch
import torch.nn as nn
from torch.nn import functional as F


# Used for reference in the benchmarks
class PilotNet(nn.Module):
    """an end-to-end network to predict steering angles"""

    def __init__(self, num_past_frames=5, num_future_steps=10):
        super().__init__()

        self.num_future_steps = num_future_steps

        # CNN layers for image processing
        self.conv1 = nn.Conv2d(
            in_channels=num_past_frames * 3, out_channels=24, kernel_size=5, stride=2
        )
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Fully connected layers for combining features and predicting outputs
        self.fc1 = nn.Linear(64 * 1 * 18 + num_past_frames * 3, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        # Output layers
        self.fc_future_path = nn.Linear(
            10, num_future_steps * 3
        )  # Predicting (x, y, z) for N future steps
        self.fc_steering = nn.Linear(10, 1)  # Predicting steering angle
        self.fc_speed = nn.Linear(10, 1)  # Predicting speed

    def forward(self, past_frames, past_xyz):
        # Process the past frames with CNN layers
        x = F.relu(self.conv1(past_frames))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten the CNN output
        x = x.view(x.size(0), -1)

        # Concatenate CNN features with past xyz coordinates
        x = torch.cat((x, past_xyz.view(past_xyz.size(0), -1)), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Outputs
        future_path = self.fc_future_path(x).view(
            -1, self.num_future_steps, 3
        )  # Reshape to (N, 3)
        steering = self.fc_steering(x)
        speed = self.fc_speed(x)

        return future_path, steering, speed


# ex shapes: past_frames: (8, past_frames*3, 66, 200); past_cyz: (8, past_frames, 3)
