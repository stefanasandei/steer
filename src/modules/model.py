import torch.nn as nn

from modules.pilotnet import PilotNet
from modules.seq2seq import Seq2Seq
from modules.av_wrapper import AVWrapper


def PilotNetWrapped(device: str) -> nn.Module:
    pilot = PilotNet().to(device)

    pilot.compile()
    model = AVWrapper(pilot).to(device)
    # can't compile a model returning a dict

    return model


def Seq2SeqWrapped(device: str) -> nn.Module:
    model = Seq2Seq().to(device)

    model.compile()
    model = AVWrapper(model).to(device)
    # can't compile a model returning a dict

    return model
