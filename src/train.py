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
max_steps = 20
learning_rate = 1e-3
log_interval = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

# data
train_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=True, device=device)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

# model
model = PilotNet().to(device)

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# overfit a single batch, for testing
train_features, train_labels = next(iter(train_dataloader))

lossi = []
t0 = time.time()

for step in range(max_steps+1):
    # forward pass
    y_hat = model(train_features["past_frames"], train_features["past_path"])

    loss_path = F.mse_loss(y_hat["future_path"], train_labels["future_path"])
    loss_angle = F.mse_loss(y_hat["steering_angle"],
                            train_labels["steering_angle"])
    loss_speed = F.mse_loss(y_hat["speed"], train_labels["speed"])

    loss = loss_path + loss_angle + loss_speed

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # timing & logging
    lossi.append(loss.item())

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if step % log_interval == 0:
        print(f"step {step}; loss={loss.item():.2f}; time={dt*1000:.2f}ms")


plt.plot(lossi)
plt.show()
