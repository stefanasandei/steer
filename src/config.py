"""
Parses the configuration files.
"""

import json
import os

cfg_path = "./config"
cfg = {}


class JSONWithCommentsDecoder(json.JSONDecoder):
    def __init__(self, **kw):
        super().__init__(**kw)

    def decode(self, s: str) -> any:
        s = '\n'.join(l for l in s.split('\n')
                      if not l.lstrip(' ').startswith('//'))
        return super().decode(s)


for conf in os.listdir(cfg_path):
    conf_name = conf.split(".")[0]
    cfg[conf_name] = {}

    with open(f"{cfg_path}/{conf}", "r") as f:
        # todo: may not work
        cfg[conf_name] = json.load(f, cls=JSONWithCommentsDecoder)
