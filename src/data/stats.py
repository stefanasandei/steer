"""
Utility class to log stats to wandb. Used in training.
"""

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

    def track_iter(self, loss: float, val_loss=0.0):
        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1

        data = {"loss/train": loss, "time/step": dt * 1000}  # ms

        if val_loss != 0.0:
            data["loss/val"] = val_loss

        if self.enabled:
            # send data to wandb
            wandb.log(data)
