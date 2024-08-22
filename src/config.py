"""
Parses the configuration files.
"""

import json
import os

cfg_path = "./config"
cfg = {}

for conf in os.listdir(cfg_path):
    conf_name = conf.split(".")[0]
    cfg[conf_name] = {}

    with open(f"{cfg_path}/{conf}", "r") as f:
        cfg[conf_name] = json.load(f)
