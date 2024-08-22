"""
Utility functions to fetch chunks from the dataset. Each chunk is about 9GB. 
Downloads from HuggingFace CDN: https://huggingface.co/datasets/commaai/comma2k19
"""

import os

from data.process import process_chunk
from config import cfg


def dataset_present(root_path: str) -> bool:
    return False


def download_dataset(root_path: str, num_chunks=cfg["data"]["num_chunks"]):
    for i in range(num_chunks):
        chunk_dir = download_chunk(root_path, chunk_num=i)
        process_chunk(chunk_dir)


def download_chunk(root_path: str, chunk_num: int) -> str:
    CHUNK_URL = f"https://huggingface.co/datasets/commaai/comma2k19/resolve/main/Chunk_{chunk_num}.zip"

    chunk_dir = f"{root_path}/{chunk_num}"

    return chunk_dir
