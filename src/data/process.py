"""
Preprocesses the dataset. For each chunk it parses the routes and segments.
Separates each video into 1200 frames (1min @ 20fps). Parses the collected data
into numpy archives.
"""

import torch
import os


def process_chunk(path: str):
    print(path)
