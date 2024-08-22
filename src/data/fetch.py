"""
Utility functions to fetch chunks from the dataset. Each chunk is about 9GB. 
Downloads from HuggingFace CDN: https://huggingface.co/datasets/commaai/comma2k19
"""

import os
import requests
import shutil

from data.process import process_chunk
from config import cfg


def dataset_present(root_path: str) -> bool:
    if not os.path.isdir(root_path):
        return False

    return len(os.listdir(root_path)) > 0


def download_dataset(root_path: str, num_chunks=cfg["data"]["num_chunks"]):
    for i in range(1, num_chunks + 1):
        if cfg["data"]["log"]:
            print(f"Started downloading chunk {i}.")

        chunk_dir = download_chunk(root_path, chunk_num=i)
        process_chunk(chunk_dir)

    if cfg["data"]["log"]:
        print(f"Dataset download finished.")


def download_chunk(root_path: str, chunk_num: int) -> str:
    zip_name = f"Chunk_{chunk_num}.zip"

    data_debug = True
    url_root = (
        "https://huggingface.co/datasets/commaai/comma2k19/resolve/main"
        if not data_debug
        else "http://127.0.0.1:5000/download"
    )

    CHUNK_URL = f"{url_root}/{zip_name}"
    chunk_dir = f"{root_path}/{zip_name.split('.')[0]}"

    # download the zip
    response = requests.get(CHUNK_URL, stream=True)
    with open(zip_name, "wb") as f:
        for zip_chunk in response.iter_content(chunk_size=512):
            if not zip_chunk:
                continue
            f.write(zip_chunk)

    # extract it
    shutil.unpack_archive(zip_name, root_path)

    # delete the original zip
    os.remove(zip_name)

    return chunk_dir
