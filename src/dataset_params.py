import torch
from torch.utils.data import DataLoader
from tabulate import tabulate

from data.dataset import CommaDataset
from config import cfg

# params
batch_size = 4
dataset_percentage = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

# data
train_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=True, device=device, dataset_percentage=dataset_percentage
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)


def iters_to_time(iters: int, per_batch: float = 0.177) -> float:
    # per_batch = 0.089 # pilotnet
    sec_per_iter = per_batch*batch_size
    sec = iters * sec_per_iter
    hours = sec / 3600.0
    return hours


one_epoch = len(train_dataloader)
epochs_num = 20
total_iters = one_epoch * epochs_num

# print results
print(tabulate([
    [f"{dataset_percentage}% of dataset", len(train_dataloader)],
    ["batch size", batch_size],
    [f"total epochs", epochs_num],
    ["time (hours)", iters_to_time(total_iters)],
], floatfmt=".2f"))
