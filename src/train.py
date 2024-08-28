import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.amp as amp

from data.stats import Stats
from data.dataset import CommaDataset
from config import cfg
from modules.pilotnet import PilotNet
from eval import get_val_loss

# hyperparameters
seed = 42
batch_size = 16
learning_rate = 1e-3
epochs = 10
eval_iters = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
run_name = "debug0"

torch.manual_seed(seed)

# data
train_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=True, device=device
)
val_dataset = CommaDataset(cfg["data"]["path"], chunk_num=1, train=False, device=device)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

torch.set_float32_matmul_precision("high")

# model
model = PilotNet().to(device)

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = amp.GradScaler(device=device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

stats = Stats(run_name, epochs)


def save_checkpoint(val_loss: float, epoch: int):
    print(f"Saving a checkpoint with val_loss={val_loss:.2f} on epoch {epoch}")

    # will store whole objects, used to resume training
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch_num": epoch,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, "ckpt.pt")


# todo: refactor to iters instead of epochs
for epoch in range(epochs + 1):
    model.train()

    epoch_loss = 0
    avg_loss = 0

    for train_features, train_labels in train_dataloader:
        # forward pass
        with amp.autocast(device_type=device):
            y_hat = model(train_features["past_frames"], train_features["past_path"])

            loss_path = F.mse_loss(y_hat["future_path"], train_labels["future_path"])
            loss_angle = F.mse_loss(
                y_hat["steering_angle"], train_labels["steering_angle"]
            )
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
        stats.track_step(loss=loss.item())
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    scheduler.step(avg_loss)

    val_loss = get_val_loss(model, val_dataloader)
    if val_loss < stats.best_loss:
        save_checkpoint(val_loss, epoch)

    # logging
    print(f"epoch {epoch}; loss={avg_loss:.4f}")
    stats.track_epoch(loss=val_loss, lr=scheduler.get_last_lr()[0])

print(f"Finished training. Saving to {stats.architecture}.pt")

# save the model state dict
torch.save(model.state_dict(), f"{stats.architecture}.pt")
