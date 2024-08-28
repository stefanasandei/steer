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
max_iters = 1000
eval_iters = 10
eval_interval = 100

device = "cuda" if torch.cuda.is_available() else "cpu"

run_name = "debug0"
out_dir = ".."

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

stats = Stats(run_name, epochs=int(len(train_dataset) / max_iters))


def save_checkpoint(val_loss: float, iter: int):
    print(f"Saving a checkpoint with val_loss={val_loss:.2f} on iter {iter}")

    # will store whole objects, used to resume training
    checkpoint = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": iter,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, f"{out_dir}/ckpt.pt")


model.train()
for iter, (train_features, train_labels) in enumerate(train_dataloader):
    if iter == max_iters:
        break

    # forward pass
    with amp.autocast(device_type=device):
        y_hat = model(train_features["past_frames"], train_features["past_path"])

        loss_path = F.mse_loss(y_hat["future_path"], train_labels["future_path"])
        loss_angle = F.mse_loss(y_hat["steering_angle"], train_labels["steering_angle"])
        loss_speed = F.mse_loss(y_hat["speed"], train_labels["speed"])

        loss = loss_path + loss_angle + loss_speed

    # backward pass
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    scheduler.step(loss)

    # timing and stats
    if iter % eval_interval == 0:
        val_loss = get_val_loss(model, val_dataloader)
        if val_loss < stats.best_loss:
            save_checkpoint(val_loss, iter)

        print(f"iter {iter}; train_loss={loss.item():.4f}; val_loss={val_loss:.4f}")
        stats.track_iter(loss=loss.item(), val_loss=val_loss)
    else:
        stats.track_iter(loss=loss.item())


print(f"Finished training. Saving to {out_dir}/{stats.architecture}.pt")
torch.save(model.state_dict(), f"{out_dir}/{stats.architecture}.pt")
