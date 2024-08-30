import torch.nn as nn

from config import cfg
from modules.pilotnet import PilotNet
from modules.av_wrapper import AVWrapper


def PilotNetWrapped(device: str) -> nn.Module:
    pilot = PilotNet(num_past_frames=cfg["model"]["past_steps"] + 1,
                     num_future_steps=cfg["model"]["future_steps"]).to(device)

    pilot = pilot.compile()
    model = AVWrapper(pilot).to(device)
    # can't compile a model returning a dict

    return model
