"""
Preprocesses the dataset. For each chunk it parses the routes and segments.
Separates each video into 1200 frames (1min @ 20fps). Parses the collected data
into numpy archives.
"""

import numpy as np
import cv2
import os

from config import cfg


def process_chunk(chunk_path: str):
    route_paths = os.listdir(chunk_path)

    try:
        os.mkdir(f"{chunk_path}/processed")
    except:
        print("Chunk already processed. Delete the processed dir to retry.")
        return

    for route_path in route_paths:
        if route_path.find("processed") != -1:
            continue

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

        self.segments = sorted(os.listdir(self.path), key=lambda seg: int(seg))

        self.CAN_speeds = {"value": [], "t": []}
        self.CAN_angles = {"value": [], "t": []}

        self.frame_t = []
        self.frame_pos = []
        self.frame_vel = []
        self.frame_ori = []

        self._load()
        self._sync()

    def _load(self):
        if cfg["data"]["log"]:
            print(f"Processing route {0}.")

        for segment in self.segments:
            seg_dir = f"{self.path}/{segment}"

            # read CAN data
            speed_dir = f"{seg_dir}/processed_log/CAN/speed"
            speed_t = np.load(f"{speed_dir}/t")
            self.CAN_speeds["t"].append(speed_t)
            speed_value = np.load(f"{speed_dir}/value")
            self.CAN_speeds["value"].append(np.squeeze(speed_value))

            angle_dir = f"{seg_dir}/processed_log/CAN/steering_angle"
            angle_t = np.load(f"{angle_dir}/t")
            self.CAN_angles["t"].append(angle_t)
            angle_value = np.load(f"{angle_dir}/value")
            self.CAN_angles["value"].append(np.squeeze(angle_value))

            # read frame data
            frame_dir = f"{seg_dir}/global_pose"

            frame_t_array = np.load(f"{frame_dir}/frame_gps_times")
            self.frame_t.append(frame_t_array)

            frame_pos_array = np.load(f"{frame_dir}/frame_positions")
            self.frame_pos.append(frame_pos_array)

            frame_vel_array = np.load(f"{frame_dir}/frame_velocities")
            self.frame_vel.append(frame_vel_array)

            frame_ori_array = np.load(f"{frame_dir}/frame_orientations")
            self.frame_ori.append(frame_ori_array)

        # concatenate all segments together
        self.CAN_speeds["t"] = np.concatenate(self.CAN_speeds["t"])
        self.CAN_speeds["value"] = np.concatenate(self.CAN_speeds["value"])

        self.CAN_angles["t"] = np.concatenate(self.CAN_angles["t"])
        self.CAN_angles["value"] = np.concatenate(self.CAN_angles["value"])

        self.frame_t = np.concatenate(self.frame_t)
        self.frame_pos = np.concatenate(self.frame_pos)
        self.frame_vel = np.concatenate(self.frame_vel)
        self.frame_ori = np.concatenate(self.frame_ori)

    def _sync(self):
        def find_nearest(arr, target):
            idx = np.searchsorted(arr, target)
            idx = np.clip(idx, 1, len(arr) - 1)
            left = arr[idx - 1]
            right = arr[idx]
            idx -= target - left < right - target
            return idx

        speed_idx = find_nearest(self.CAN_speeds["t"], self.frame_t)
        self.synced_speed_value = self.CAN_speeds["value"][speed_idx]
        angle_idx = find_nearest(self.CAN_angles["t"], self.frame_t)
        self.synced_angle_value = self.CAN_angles["value"][angle_idx]

    def get_name(self):
        # b0c9d2329ad1606b_2018-08-02--08-34-47 -> 2018-08-02--08-34-47
        chr = "|" if "|" in self.path else "_"
        return self.path.split(chr)[-1]

    def save(self, path: str):
        """Under the dir processed/{name}, it will save the following: can_telemetry.npz, frame.npz
        and video/frame_{0, 1200}.jpeg"""

        if cfg["data"]["log"]:
            print(f"Saving to {path}")

        os.mkdir(path)

        # save CAN data
        np.savez_compressed(
            f"{path}/can_telemetry.npz",
            {"speed": self.synced_speed_value, "angle": self.synced_angle_value},
        )

        # save frame data
        np.savez_compressed(
            f"{path}/frame.npz",
            {
                "t": self.frame_t,
                "position": self.frame_pos,
                "velocity": self.frame_vel,
                "orientation": self.frame_ori,
            },
        )

        # save video frames
        os.mkdir(f"{path}/video")

        frame_count = 0
        # For each segment, load in the frames and save each one to the images dir
        for segment in self.segments:
            video_path = f"{self.path}/{segment}/video.hevc"

            # for debug
            max_frames = 1200 if not cfg["data"]["log"] else 20
            curr_frame = 0

            cap = cv2.VideoCapture(str(video_path))
            while cap.isOpened():
                ret, frame = cap.read()

                if ret and curr_frame <= max_frames:
                    # Zero pad frame_count and save frame
                    img_path = f"{path}/video/{(str(frame_count).zfill(6) + '.jpg')}"
                    cv2.imwrite(str(img_path), frame)

                    frame_count += 1
                    curr_frame += 1
                else:
                    break

            cap.release()
