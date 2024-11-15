"""
Functions to draw stuff onto frames to visualize model outputs
"""

import cv2
import numpy as np
from typing import TypedDict
import PIL

from lib.camera import img_from_device, denormalize, FULL_FRAME_SIZE
import lib.orientation as orient

FrameData = TypedDict('FrameData', {
                      "t": np.ndarray, "position": np.ndarray, "orientation": np.ndarray})

CANData = TypedDict("CANData", {
    "speed": np.ndarray, "angle": np.ndarray
})


def adjust_gamma(image: PIL.Image, gamma: float) -> PIL.Image:
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def draw_path(img: np.ndarray, path: np.ndarray, shape_props={
    "width": 1, "height": 1, "fill_color": (0, 128, 255), "line_color": (0, 255, 0)
}):
    device_path_l = path + np.array([0, 0, shape_props["height"]])
    device_path_r = path + np.array([0, 0, shape_props["height"]])
    device_path_l[:, 1] -= shape_props["width"]
    device_path_r[:, 1] += shape_props["width"]

    img_points_norm_l = img_from_device(device_path_l)
    img_points_norm_r = img_from_device(device_path_r)
    img_pts_l = denormalize(img_points_norm_l)
    img_pts_r = denormalize(img_points_norm_r)

    # filter out things rejected along the way
    valid = np.logical_and(np.isfinite(img_pts_l).all(
        axis=1), np.isfinite(img_pts_r).all(axis=1))
    img_pts_l = img_pts_l[valid].astype(int)
    img_pts_r = img_pts_r[valid].astype(int)

    for i in range(1, len(img_pts_l)):
        # Scale image points from original image size to current size
        w1, h1 = FULL_FRAME_SIZE
        h2, w2, _ = img.shape
        u1, v1 = img_pts_l[i-1]
        u2, v2 = img_pts_r[i-1]
        u3, v3 = img_pts_l[i]
        u4, v4 = img_pts_r[i]

        pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.float64)
        pts[:, 0] *= w2/w1
        pts[:, 1] *= h2/h1
        pts = pts.astype(np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(img, [pts], shape_props["fill_color"])
        cv2.polylines(img, [pts], True, shape_props["line_color"])


def draw_text(img: np.ndarray, text: str, origin: tuple[int, int]):
    color = (255, 255, 255)
    thickness = 6
    font_scale = 2

    length = len(text)*40
    height = 30

    # top-left & bottom-right
    img = cv2.rectangle(
        img, (origin[0]-20, origin[1]+height), (origin[0]+length, origin[1]-60), (0, 0, 0), -1)

    # bottom-left
    img = cv2.putText(img, text, origin,
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return img


def draw_frame(img: np.ndarray, path: np.ndarray, speed: float, steering_angle: float) -> np.ndarray:
    img = cv2.resize(img, FULL_FRAME_SIZE,
                     interpolation=cv2.INTER_LINEAR)
    img = adjust_gamma(img, gamma=1.2)

    # draw future path
    draw_path(img, path)

    # speed
    img = draw_text(img, f"{speed:.1f} km/h", (50, 75))

    # steering angle
    img = draw_text(img, f"{steering_angle:.2f}", (950, 75))

    return img


def draw_debug_frame(frame_data: FrameData, can_data: CANData, route_path: str, index: int, duration: int) -> np.ndarray:
    """
    meant to index to data from the comma dataset
    """

    offset = 4

    ecef_from_local = orient.rot_from_quat(frame_data["orientation"][index])

    local_from_ecef = ecef_from_local.T
    frame_positions_local = np.einsum(
        'ij,kj->ki', local_from_ecef, frame_data["position"] - frame_data["position"][index])

    target_frames = frame_positions_local[index+offset:index+duration+offset]

    speed = can_data["speed"][index] * 3.6  # from m/s to km/h
    angle = can_data["angle"][index]

    img = cv2.imread(f'{route_path}/video/{str(index).zfill(6)}.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = adjust_gamma(img, gamma=1.2)

    img = draw_frame(img, target_frames, speed, angle)

    return img
