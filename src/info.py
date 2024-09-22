from modules.model import SteerNetWrapped, Seq2SeqWrapped, PilotNetWrapped
import torch
import torch.nn as nn
import argparse
from torchinfo import summary
from thop import profile


def compute_summary(model: nn.Module):
    B, T, HW = 1, 11, 224
    sample_frames = torch.randn((B, T, 3, HW, HW), device="cuda")
    sample_paths = torch.randn((B, T, 3), device="cuda")

    flops, _ = profile(model, inputs=(
        sample_frames, sample_paths), verbose=False)
    print("flops: " + str(flops))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="name of the model (pilotnet, seq2seq or steer)",
                        required=True)
    args = parser.parse_args()

    depth = 3
    model = None

    if args.model == "steer":
        depth = 6  # for detailed Mamba layers overview
        model = SteerNetWrapped("cuda", compile=False)
    elif args.model == "pilotnet":
        model = PilotNetWrapped("cuda", compile=False)
    elif args.model == "seq2seq":
        model = Seq2SeqWrapped("cuda", compile=False)

    # param count & layers
    summary(model, depth=depth)

    # tlfops & compute time
    compute_summary(model)

# steer: 6.3M
# seq2seq 5.9M (about ~5M is from RegNet)
# pilotnet: 0.8M
