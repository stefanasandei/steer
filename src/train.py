import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.amp as amp

from data.stats import Stats
from data.dataset import CommaDataset, cycle
from config import cfg
from modules.model import SteerNetWrapped, PilotNetWrapped, Seq2SeqWrapped
from eval import get_val_loss
from lib.lr import get_lr

# hyperparameters
seed = 42
batch_size = 64

max_lr = 3e-3
min_lr = max_lr * 0.02
warmup_iters = 100
learning_rate = min_lr

eval_iters = 5
eval_interval = 100
max_iters = 900  # 4 epochs
# dataset len is 147389, with batch size 16 it takes ~9000 iters
# 20% of the dataset is 29477, with a batch size of 64 it takes ~ 500 iters for an epoch

device = "cuda" if torch.cuda.is_available() else "cpu"

run_name = "steer"
out_dir = f"../runs"  # save models in /workspace/runs/
# repo in /workspace/steer; dataset in /workspace/comma2k19

torch.manual_seed(seed)

# data
train_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=True, device=device, dataset_percentage=10
)
val_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=False, device=device, dataset_percentage=100
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

torch.set_float32_matmul_precision("high")

# model
model = SteerNetWrapped(device, return_dict=False)

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr)
scaler = amp.GradScaler(device=device)

stats = Stats(run_name, epochs=int(
    len(train_dataset) / max_iters), enabled=True)


def save_checkpoint(val_loss: float, iter: int):
    print(f"Saving a checkpoint with val_loss={val_loss:.2f} on iter {iter}")

    # will store whole objects, used to resume training
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, f"{out_dir}/{stats.architecture}-ckpt.pt")


model.train()
for iter, (train_features, train_labels) in enumerate(cycle(train_dataloader)):
    if iter == max_iters + 1:
        break

    # forward pass
    with amp.autocast(device_type=device, dtype=torch.bfloat16):
        y_hat, loss = model(
            train_features["past_frames"], train_features["past_path"], train_labels
        )

    # backward pass
    optimizer.param_groups[0]["lr"] = learning_rate
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    learning_rate = get_lr(
        iter, max_lr, min_lr, warmup_iters, max_iters)

    # timing and stats
    if iter % eval_interval == 0:
        val_loss = get_val_loss(model, val_dataloader, device)
        if val_loss < stats.best_loss and iter > 0:
            save_checkpoint(val_loss, iter)

        print(
            f"iter {iter}; train_loss={loss.item():.4f}; val_loss={val_loss:.4f}")
        stats.track_iter(loss=loss.item(), lr=learning_rate, val_loss=val_loss)
    else:
        stats.track_iter(loss=loss.item(), lr=learning_rate)


print(f"Finished training. Saving to {out_dir}/{stats.architecture}.pt")
torch.save(model.state_dict(), f"{out_dir}/{stats.architecture}.pt")

# stats.plot_loss()
