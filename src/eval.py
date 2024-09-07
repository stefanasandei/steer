import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.amp as amp
import argparse

from data.dataset import CommaDataset
from config import cfg
from modules.model import Seq2SeqWrapped, PilotNetWrapped


def get_val_loss(model: nn.Module, val_dataloader: DataLoader, device: str, eval_iters=10):
    val_loss = 0

    model.eval()

    for idx, (val_features, val_labels) in enumerate(val_dataloader):
        if idx == eval_iters:
            break

        with amp.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(val_features["past_frames"],
                            val_features["past_path"], val_labels)

        val_loss += loss.item()

    model.train()
    return val_loss / eval_iters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path of the saved model checkpoint",
        required=True,
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    val_dataset = CommaDataset(
        cfg["data"]["path"], chunk_num=1, train=False, device=device
    )
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    model = Seq2SeqWrapped(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    print(f"val_loss={get_val_loss(model, val_dataloader, device):.2f}")
