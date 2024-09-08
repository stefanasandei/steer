import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, TypedDict

from config import cfg

# allow .item() for speed and angles in the loss eq.
torch._dynamo.config.capture_scalar_outputs = True

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
        device="cuda"
    ):
        super().__init__()

        self.net = net
        self.num_future_steps = num_future_steps
        self.return_dict = return_dict
        self.device = device

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

        # create result
        if self.return_dict:
            pred = {
                "future_path": future_path,
                "steering_angle": steering,
                "speed": speed,
            }
        else:
            empty_column = torch.zeros(
                (future_path.shape[0], 1, 1), device=self.device)
            steering = steering.view(-1, 1, 1)
            speed = speed.view(-1, 1, 1)

            new_row = torch.cat([steering, speed, empty_column],
                                dim=2)  # (B, 1, 3)

            # pred.shape = (B, T+1, 3)
            pred = torch.cat((future_path, new_row), dim=1)

        # now compute loss
        loss = None if targets is None else self._get_loss(pred, targets)

        # final return
        return pred, loss

    def _get_loss(self, y_hat, targets):
        loss = None

        if self.return_dict:
            loss_future_path = F.mse_loss(
                y_hat["future_path"], targets["future_path"])
            loss_steering_angle = F.mse_loss(
                y_hat["steering_angle"], targets["steering_angle"])
            loss_speed = F.mse_loss(
                y_hat["speed"], targets["speed"])
        else:
            loss_future_path = F.mse_loss(
                y_hat[:, :-1, :], targets["future_path"])
            loss_steering_angle = F.mse_loss(
                y_hat[:, -1, 0],  targets["steering_angle"])
            loss_speed = F.mse_loss(
                y_hat[:, -1, 1], targets["speed"])

        loss = loss_future_path + loss_steering_angle + loss_speed

        return loss
