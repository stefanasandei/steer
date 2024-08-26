import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
import time

from data.dataset import CommaDataset
from config import cfg
from modules.pilotnet import PilotNet

# hyperparameters
seed = 42
batch_size = 16
epochs = 10
learning_rate = 1e-3
log_interval = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(seed)

# data
train_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=True, device=device)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

torch.set_float32_matmul_precision('high')

# model
model = PilotNet().to(device)

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=2)

lossi = []
t0 = time.time()

for epoch in range(epochs+1):
    model.train()

    epoch_loss = 0
    epoch_time = 0

    for train_features, train_labels in train_dataloader:
        # forward pass
        with torch.cuda.amp.autocast():
            y_hat = model(train_features["past_frames"],
                          train_features["past_path"])

            loss_path = F.mse_loss(
                y_hat["future_path"], train_labels["future_path"])
            loss_angle = F.mse_loss(y_hat["steering_angle"],
                                    train_labels["steering_angle"])
            loss_speed = F.mse_loss(y_hat["speed"], train_labels["speed"])

            loss = loss_path + loss_angle + loss_speed

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # timing and stats
        lossi.append(loss.item())
        epoch_loss += loss.item()
        t1 = time.time()
        epoch_time += (t1-t0)
        t0 = t1

    scheduler.step(avg_loss)

    # logging
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"epoch {epoch}; loss={avg_loss:.4f}; time={epoch_time*1000:.2f}ms")


plt.plot(lossi)
plt.show()
