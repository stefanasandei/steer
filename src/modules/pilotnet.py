import torch
import torch.nn as nn
from torch.nn import functional as F


class PilotNet(nn.Module):
    """
    roughly following Nvidia's paper 'End to End Learning for Self-Driving Cars'
    modified to take in multiple frames (using concatenation) and also output future path
    """

    def __init__(
        self,
    ):
        super().__init__()

        # image processing
        self.conv1 = nn.LazyConv2d(out_channels=24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # merge image features with paths
        # will output the hidden state
        self.fc = nn.Sequential(
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

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
        h = self.fc(x)

        # hidden state
        # will be taken care of by the AV wrapper
        return h


# let's test the model
if __name__ == "__main__":
    B, T = 2, 3
    past_frames = torch.randn((B, T, 3, 1164 // 2, 874 // 2))
    past_xyz = torch.randn((B, T, 3))

    model = PilotNet()
    print(model(past_frames, past_xyz).shape)
