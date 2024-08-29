"""
Create a video using reference data or predictions.
"""

import cv2
import numpy as np
from tqdm import tqdm

from lib.drawing import draw_debug_frame, draw_frame
from modules.pilotnet import PilotNet
from config import cfg


def create_debug_video(route_path: str, output_path: str):
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1164, 874))

    frames = np.load(f"{route_path}/frame.npz")
    can_data = np.load(f"{route_path}/can_telemetry.npz")

    for i in tqdm(range(0, 1170)):
        img = draw_debug_frame(frames, can_data, route_path, index=i, duration=30)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        out.write(img)

    out.release()


def create_video(route_path: str, output_path: str):
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1164, 874))

    # model
    model = PilotNet(
        num_past_frames=cfg["model"]["past_steps"] + 1,
        num_future_steps=cfg["model"]["future_steps"],
    )

    # data
    frames = np.load(f"{route_path}/frame.npz")

    for i in tqdm(
        range(cfg["model"]["past_steps"] + 1, 1200 - cfg["model"]["future_steps"])
    ):
        curr_frame = cv2.imread(f"{route_path}/video/{str(i).zfill(6)}.jpeg")

        # todo
        y_hat = model(past_frames=None, past_xyz=None)

        img = draw_frame(
            curr_frame, y_hat["future_path"], y_hat["speed"], y_hat["steering_angle"]
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        out.write(img)

    out.release()


if __name__ == "__main__":
    create_debug_video(
        "./comma2k19/Chunk_1/processed/2018-08-02--08-34-47", "ref_video1.mp4"
    )
