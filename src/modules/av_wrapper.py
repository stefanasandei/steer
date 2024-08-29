import torch.nn as nn

from config import cfg


class AVWrapper(nn.Module):
    """
    a wrapper module for autonomous vehicle NNs,
    takes care of outputs, outputs a dict with the data;
    can't compile a model returning a dict, so we wrap it (returning hidden)
    and then compile the inner model
    """

    def __init__(
        self,
        net: nn.Module,
        num_future_steps=cfg["model"]["future_steps"],
    ):
        super().__init__()

        self.net = net
        self.num_future_steps = num_future_steps

        # Output layers
        self.out = nn.ModuleDict(
            {
                "future_path": nn.Linear(
                    num_future_steps * 3
                ),  # (x, y, z) for T future steps
                "steering_angle": nn.LazyLinear(1),  # in radians
                "speed": nn.LazyLinear(1),  # in meters per second
            }
        )

    def forward(self, past_frames, past_xyz):
        # (B, H)
        h = self.net(past_frames, past_frames)  # hidden state

        # (B, T, 3)
        future_path = self.out["future_path"](
            h).view(-1, self.num_future_steps, 3)

        # (B, 1)
        steering = self.out["steering_angle"](h).squeeze()

        # (B, 1)
        speed = self.out["speed"](h).squeeze()

        return {
            "future_path": future_path,
            "steering_angle": steering.squeeze(),
            "speed": speed.squeeze(),
        }
