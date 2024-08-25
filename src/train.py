import torch
from torch.utils.data import DataLoader

from data.dataset import CommaDataset
from config import cfg

# hyperparameters
seed = 42
batch_size = 16

torch.manual_seed(seed)

# data
if __name__ == "__main__":
    train_dataset = CommaDataset(cfg["data"]["path"], chunk_num=1, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))

    print(f"Feature batch shape: {[train_features[a].shape for a in train_features]}")
    print(f"Labels batch shape: {[train_labels[a].shape for a in train_labels]}")
