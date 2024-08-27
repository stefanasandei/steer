"""
Utility functions to fetch chunks from the dataset. Each chunk is about 9GB. 
Downloads from HuggingFace CDN: https://huggingface.co/datasets/commaai/comma2k19
"""

from concurrent.futures import ProcessPoolExecutor
import itertools
import requests
import shutil
import pickle
import random
import os

from tqdm import tqdm

from data.process import process_chunk
from config import cfg


def dataset_present(root_path: str) -> bool:
    if not os.path.isdir(root_path):
        return False

    return len(os.listdir(root_path)) > 0


def download_dataset(
    root_path: str, num_chunks=cfg["data"]["num_chunks"], skip_download=False
):
    for i in range(1, num_chunks + 1):
        if cfg["data"]["log"]:
            print(f"Started downloading chunk {i}.")

        if not skip_download:
            chunk_dir = download_chunk(root_path, chunk_num=i)
        else:
            zip_name = f"Chunk_{i}.zip"
            chunk_dir = f"{root_path}/{zip_name.split('.')[0]}"

        process_chunk(chunk_dir)
        split_chunk(chunk_dir)

    if cfg["data"]["log"]:
        print(f"Dataset download finished.")


def download_chunk(root_path: str, chunk_num: int) -> str:
    zip_name = f"Chunk_{chunk_num}.zip"

    url_root = (
        "https://huggingface.co/datasets/commaai/comma2k19/resolve/main"
        if not cfg["data"]["debug"]
        else "http://127.0.0.1:5000/download"
    )

    CHUNK_URL = f"{url_root}/{zip_name}"
    chunk_dir = f"{root_path}/{zip_name.split('.')[0]}"

    # download the zip
    response = requests.get(CHUNK_URL, stream=True)
    total_size = int(response.headers.get("Content-Length", 0))
    downloaded_size = 0

    with open(zip_name, "wb") as f, tqdm(
        unit="B",
        total=total_size,
        desc=f"Chunk {chunk_num}",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for zip_chunk in response.iter_content(chunk_size=1024 * 1024):
            if not zip_chunk:
                continue
            size = f.write(zip_chunk)

            bar.update(size)
            downloaded_size += size

    if downloaded_size < total_size:
        print("Failed to download the chunk. Check if the URL is correct.")
        exit(0)

    # extract it
    shutil.unpack_archive(zip_name, root_path)

    # delete the original zip
    os.remove(zip_name)

    return chunk_dir


def split_chunk(chunk_dir: str):
    """
    1. Grab all routes (randomized)
    2. For each route choose the valid images
    3. Concat all valid image frames into an array
    4. Save the array with the image paths to a train file (\w pickle)

    In the dataset, we can get the the other data based on the frame path,
    by going into the parent dir. Each frame is ordered, so we know the
    timestamp and what are the past and future frames.
    """

    # grab all routes
    routes = [
        f"{chunk_dir}/processed/{f}" for f in os.listdir(f"{chunk_dir}/processed")
    ]
    random.seed(42)
    random.shuffle(routes)

    # split into train/val
    split = int(len(routes) * cfg["data"]["split"])
    train_routes = routes[:split]
    val_routes = routes[split:]

    if cfg["data"]["log"]:
        print(f"train: {len(train_routes)} routes; val: {len(val_routes)} routes")

    train_files = select_frames(train_routes)
    val_files = select_frames(val_routes)

    # save the array with the frame path
    with open(f"{chunk_dir}/train", "wb") as f:
        pickle.dump(train_files, f)

    with open(f"{chunk_dir}/val", "wb") as f:
        pickle.dump(val_files, f)


def select_frames(routes: str) -> list[str]:
    frame_results = []

    # for each route choose valid images
    if cfg["data"]["multiprocess_routes"]:
        with ProcessPoolExecutor() as executor:
            frame_paths = executor.map(
                good_frames_from_route,
                routes,
                itertools.repeat(cfg["model"]["future_steps"]),
                itertools.repeat(cfg["model"]["past_steps"]),
            )
    else:
        frame_paths = []
        for route in routes:
            frame_paths.append(
                good_frames_from_route(
                    route, cfg["model"]["future_steps"], cfg["model"]["past_steps"]
                )
            )

    # merge all the frame paths from all routes
    for res in frame_paths:
        frame_results.extend(res)

    # an array with all valid captured frames
    # the dataset can find the past&future paths
    # based on the frame index, along with the
    # can data thanks to the timestamp
    return frame_results


def good_frames_from_route(
    route_path: str, future_steps: int, past_steps: int
) -> list[str]:
    all_frames = sorted(
        [int(f[: len(".jpeg") + 1]) for f in os.listdir(f"{route_path}/video")]
    )

    # skip if not enough total steps
    if len(all_frames) < (past_steps + future_steps + 1):
        return []

    # remove frames with not enough previous & future info
    all_frames = all_frames[past_steps:-future_steps]

    frame_paths = [
        f"{route_path}/video/{(str(frame).zfill(6) + '.jpeg')}" for frame in all_frames
    ]

    return frame_paths
