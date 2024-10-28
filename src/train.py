import torch
import torch.nn.functional as F
import torch.amp as amp

from data.stats import Stats
from data.dataset import cycle, get_train_data, get_valid_data
from config import cfg
from modules.model import SteerNetWrapped, PilotNetWrapped, Seq2SeqWrapped
from eval import get_val_loss
from lib.lr import get_lr

# hyperparameters
seed = 42
batch_size = 4

steps_per_epoch = 800
epochs = 10

max_lr = 5e-3
min_lr = 1e-5
warmup_iters = steps_per_epoch * epochs * 3 // 100  # 3% of total iters
learning_rate = min_lr

eval_iters = 5
eval_interval = 300
max_iters = steps_per_epoch * epochs

# python3 ./src/prepare.py --split

# dataset size: ~147389; use only 20%: 29477
# with batch size 64 => ~450 per epoch
# if it takes 10s/iter => 5 hours for 4 epochs (~2.5$)

# use 4%: 5895 and batch size 32
# results in 184 iters/epoch
# aka ~10 epochs for 1800 iters (0.39$)

# or 2% of dataset with batch of 32 => 19 epochs

device = "cuda" if torch.cuda.is_available() else "cpu"

run_name = "steer-curv"
out_dir = f"/mnt/e/steer/runs"  # save models in /workspace/runs/
# repo in /workspace/steer; dataset in /workspace/comma2k19

torch.manual_seed(seed)

# data
train_dataloader, train_dataset = get_train_data(device, batch_size)
val_dataloader, _ = get_valid_data(device, batch_size)

torch.set_float32_matmul_precision("high")

print(f"total iters: {max_iters}\ndataset size: {len(train_dataset)}")
print(
    f"iters per epoch: {len(train_dataset)//batch_size}\ntotal epochs: {max_iters//(len(train_dataset)//batch_size)}")

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
    if iter > 100:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
