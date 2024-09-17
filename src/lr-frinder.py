import torch
from torch.utils.data import DataLoader
import torch.amp as amp
import matplotlib.pyplot as plt

from data.dataset import CommaDataset, cycle
from config import cfg
from modules.model import SteerNetWrapped, PilotNetWrapped, Seq2SeqWrapped


# hyperparameters
seed = 42
batch_size = 4

# learning rate stats
max_iters = 500
lr_exponents = torch.linspace(-6, -1.5, max_iters)
lr_values = 10**lr_exponents
lri, lossi = [], []

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

# data
train_dataset = CommaDataset(
    cfg["data"]["path"], chunk_num=1, train=True, device=device, dataset_percentage=2
)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

torch.set_float32_matmul_precision("high")

# model
model = SteerNetWrapped(device, return_dict=False)

# training
optimizer = torch.optim.AdamW(model.parameters())
scaler = amp.GradScaler(device=device)

model.train()
for iter, (train_features, train_labels) in enumerate(cycle(train_dataloader)):
    if iter == max_iters:
        break

    # forward pass
    with amp.autocast(device_type=device, dtype=torch.bfloat16):
        y_hat, loss = model(
            train_features["past_frames"], train_features["past_path"], train_labels
        )

    # backward pass
    optimizer.param_groups[0]["lr"] = lr_values[iter]
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    # track stats
    lri.append(lr_exponents[iter])
    if not torch.isnan(loss).all():
        lossi.append(min(loss.item(), 1000.0))
    else:
        lossi.append(1000.0)
    print(f"iter {iter}: loss={loss.item():.2f}, lr={lr_values[iter]}")

plt.figure(figsize=(10, 5))
plt.xlabel("learning rate exponent")
plt.ylabel("loss value")
plt.plot(lri, lossi)
plt.savefig("lr.pdf", bbox_inches='tight')
plt.show()
