import torch.nn as nn

from modules.pilotnet import PilotNet
from modules.seq2seq import Seq2Seq
from modules.steer import SteerNet
from modules.av_wrapper import AVWrapper
from config import cfg


def PilotNetWrapped(device: str, return_dict: bool = True) -> nn.Module:
    model = PilotNet().to(device)

    if return_dict:
        model.compile()
        model = AVWrapper(model, return_dict=True).to(device)
        # can't compile a model returning a dict
    else:
        model = AVWrapper(model, return_dict=False).to(device)
        model.compile()

    return model


def Seq2SeqWrapped(device: str) -> nn.Module:
    model = Seq2Seq().to(device)

    model.compile()
    model = AVWrapper(model).to(device)
    # can't compile a model returning a dict

    return model


def SteerNetWrapped(device: str, return_dict: bool = True) -> nn.Module:
    # add one more frame, the current one
    model = SteerNet(cfg["model"]["past_steps"]+1).to(device)

    if return_dict:
        model.compile()
        model = AVWrapper(model, return_dict=True).to(device)
        # can't compile a model returning a dict
    else:
        model = AVWrapper(model, return_dict=False).to(device)
        model.compile()

    return model
