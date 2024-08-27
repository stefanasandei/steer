"""
Utility class to log stats to wandb. Used in training.
"""

import wandb
import time

from config import cfg


class Stats:

    def __init__(self, architecture: str, epochs: int, enabled=True):
        self.conf_wandb = cfg["data"]["stats"]
        self.enabled = enabled

        if self.enabled:
            # will request api key from stdin
            wandb.login()

            wandb.init(project=self.conf_wandb["project"], config={
                "dataset": self.conf_wandb["dataset"],
                "architecture": architecture,
                "epochs": epochs
            })

        # won't be accurate for the first step yolo
        self.t0 = time.time()

    # to be called every step
    def track(self, loss: float):
        t1 = time.time()
        dt = t1-self.t0
        self.t0 = t1

        data = {
            "loss": loss,
            "time": dt*1000  # ms
        }

        if self.enabled:
            # send data to wandb
            wandb.log(data)
