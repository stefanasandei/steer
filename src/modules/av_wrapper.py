import torch
import torch.nn as nn
from typing import Optional, TypedDict

from config import cfg

ModelOutputDict = TypedDict(
    "ModelOutput",
    {
        "future_path": torch.Tensor,
        "steering_angle": torch.Tensor,
        "speed": torch.Tensor,
    },
)


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
        return_dict=True,
    ):
        super().__init__()

        self.net = net
        self.num_future_steps = num_future_steps
        self.return_dict = return_dict

        # Output layers
        self.out = nn.ModuleDict(
            {
                "future_path": nn.LazyLinear(
                    num_future_steps * 3
                ),  # (x, y, z) for T future steps
                # in radians
                "steering_angle": nn.LazyLinear(1),
                # in meters per second
                "speed": nn.LazyLinear(1),
            }
        )

    def forward(
        self,
        past_frames,
        past_xyz,
        targets: Optional[torch.Tensor | ModelOutputDict] = None,
    ) -> tuple[torch.Tensor | ModelOutputDict, Optional[torch.Tensor]]:
        """
        Optionally takes in targets to compute loss. Will return
        a tuple with the predictions (either a tensor or a dict) and
        optionally the loss.
        """

        # (B, H)
        h = self.net(past_frames, past_xyz)  # hidden state

        # (B, T, 3)
        future_path = self.out["future_path"](h)
        future_path = future_path.view(-1, self.num_future_steps, 3)

        # (B, 1)
        steering = self.out["steering_angle"](h).squeeze()

        # (B, 1)
        speed = self.out["speed"](h).squeeze()

        # now compute loss
        loss = None
        if targets is not None:
            # todo
            loss = torch.tensor([0.0])

        # create result
        if self.return_dict:
            result = {
                "future_path": future_path,
                "steering_angle": steering,
                "speed": speed,
            }
        else:
            new_row = torch.tensor([steering.item(), speed.item(), 0.0])
            result = torch.cat((result, new_row), dim=1)

        # final return
        return result, loss
