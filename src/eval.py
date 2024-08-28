import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.amp as amp

from data.dataset import CommaDataset
from config import cfg

# hyperparameters
seed = 42
batch_size = 16
eval_iters = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

# data
val_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=False, device=device)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

torch.set_float32_matmul_precision("high")

# model
model = torch.load("ckpt.pt")["model"]


@torch.no_grad()
def estimate_loss():
    val_loss = 0

    model.eval()

    for idx, (train_features, train_labels) in enumerate(val_dataloader):
        if idx == eval_iters:
            break

        with amp.autocast(device_type=device):
            y_hat = model(train_features["past_frames"],
                          train_features["past_path"])

            loss_path = F.mse_loss(
                y_hat["future_path"], train_labels["future_path"])
            loss_angle = F.mse_loss(
                y_hat["steering_angle"], train_labels["steering_angle"]
            )
            loss_speed = F.mse_loss(y_hat["speed"], train_labels["speed"])
            loss = loss_path + loss_angle + loss_speed

        val_loss += loss.item()

    model.train()
    return val_loss / eval_iters


print(estimate_loss())
