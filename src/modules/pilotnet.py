import torch
import torch.nn as nn
from torch.nn import functional as F

# Used for reference in the benchmarks
class PilotNet(nn.Module):
    """an end-to-end network to predict steering angles"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, 
                      kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=36,
                      kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=48,
                      kernel_size=5, stride=2),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.controller = nn.Sequential(
            nn.LazyLinear(1164),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
        )

        self.out = nn.Linear(10, 1)

    def forward(self, frame):
        x = self.conv1(frame)
        x = self.conv2(x)

        x = self.controller(x)
        return self.out(x)
