"""
Preprocesses the dataset. For each chunk it parses the routes and segments.
Separates each video into 1200 frames (1min @ 20fps). Parses the collected data
into numpy archives.
"""

import numpy as np
import os

from config import cfg


def process_chunk(chunk_path: str):
    route_paths = os.listdir(chunk_path)
    os.mkdir(f"{chunk_path}/processed")

    for route_path in route_paths:
        route = Route(f"{chunk_path}/{route_path}")
        route.save(f"{chunk_path}/processed/{route.get_name()}")


class Route:
    """
    A route is a top-level subdirectory of a chunk. It is composed of multiple 1 minute segments.
    Directory name is of form '{vehicle_id}|{time}', a processed route will be saved under 'processed/{time}'.
    Each segment has the following (relevant) structure: video.hevc, processed_log/CAN/{speed/value, steering_angle/value},
    global_pose/{frame_times, frame_positions, frame_velocities, frame_orientations}. The video file will be split
    into multiple JPEGs. Two numpy archives will be created: can_telemetry.npz (speed and steering angle) and frame.npz
    (frame pos, velocity, tme and orientations). The dataset class will take care of converting the global frame
    data into local paths.
    """

    def __init__(self, route_path):
        self.path = route_path

        self._load()
        self._sync()

    def _load(self):
        if cfg["data"]["log"]:
            print(f"Processing route {0}.")

        print(f"segments: {os.listdir(self.path)}")

        # todo
        pass

    def _sync(self):
        pass

    def get_name(self):
        # b0c9d2329ad1606b_2018-08-02--08-34-47 -> 2018-08-02--08-34-47
        chr = '|' if '|' in self.path else '_'
        return self.path.split(chr)[-1]

    def save(self, path: str):
        """Under the dir processed/{name}, it will save the following: can_telemetry.npz, frame.npz 
        and video/frame_{0, 1200}.jpeg"""

        print(f"Saving to {path}")

        with open(path, "w") as f:
            f.write("lol")
