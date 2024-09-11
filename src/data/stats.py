"""
Utility class to log stats to wandb. Used in training.
"""

import matplotlib.pyplot as plt
import wandb
import time
import sys

from config import cfg


class Stats:
    """
    Used to log data to WanDB. A run will have the following values:
    - train/loss, val/loss, time/step, learning_rate
    """

    def __init__(self, architecture: str, epochs: int, enabled=True):
        self.conf_wandb = cfg["data"]["stats"]
        self.enabled = enabled
        self.best_loss = sys.float_info.max
        self.architecture = architecture
        self.iter = 0
        self.losses = []

        if self.enabled:
            # will request api key from stdin
            wandb.login()

            wandb.init(
                project=self.conf_wandb["project"],
                config={
                    "dataset": self.conf_wandb["dataset"],
                    "architecture": architecture,
                    "epochs": epochs,
                },
            )

        # won't be accurate for the first step yolo
        self.t0 = time.time()

    def track_iter(self, loss: float, lr: float, val_loss=0.0):
        self.iter += 1
        self.losses.append(loss)

        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1

        self.best_loss = max(self.best_loss, val_loss)

        data = {"loss/train": loss, "time": dt *
                1000, "learning_rate": lr}  # ms

        if val_loss != 0.0:
            data["loss/val"] = val_loss

        if self.enabled:
            # send data to wandb
            wandb.log(data)
        else:
            # log to stdout
            print(
                f"iter {self.iter}; loss={loss:.2f}; lr={lr}; time={dt*1000:.1f}ms")

    def plot_loss(self):
        plt.plot(self.losses)
        plt.show()
