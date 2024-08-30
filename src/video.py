"""
Create a video using reference data or predictions.
"""

from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from lib.paths import get_local_path
from lib.drawing import draw_debug_frame, draw_frame
from modules.model import PilotNetWrapped
from config import cfg


def create_debug_video(route_path: str, output_path: str):
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1164, 874))

    frames = np.load(f"{route_path}/frame.npz")
    can_data = np.load(f"{route_path}/can_telemetry.npz")

    for i in tqdm(range(0, 1170)):
        img = draw_debug_frame(
            frames, can_data, route_path, index=i, duration=30)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        out.write(img)

    out.release()


@torch.no_grad
def create_video(route_path: str, output_path: str, model_path: str):
    # video writer
    fourcc = cv2.VideoWriter_fourcc(*"h264")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1164, 874))

    # model
    device = "cuda" if torch.cuda.is_available() else "mps"

    model = PilotNetWrapped(device)
    model.load_state_dict(torch.load(model_path))

    # data
    frames = np.load(f"{route_path}/frame.npz")
    positions, orientations = frames["position"], frames["orientation"]
    trans = transforms.Compose([transforms.ToTensor()])

    for i in tqdm(
        range(cfg["model"]["past_steps"] + 1,
              1200 - cfg["model"]["future_steps"])
    ):
        curr_frame = cv2.imread(f"{route_path}/video/{str(i).zfill(6)}.jpeg")
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)

        # prepare past path
        local_path = get_local_path(positions, orientations, i)
        previous_path = local_path[i - cfg["model"]["past_steps"]: i + 1]
        prev_path = torch.from_numpy(previous_path)

        # prepare past frames
        frames = []
        for f_id in range(i - cfg["model"]["past_steps"], i + 1):
            frame = Image.open(f"{route_path}/video/{str(f_id).zfill(6)}.jpeg")
            frames.append(trans(frame))
        frames = torch.stack(frames)

        frames = torch.unsqueeze(frames, 0).to(device)
        prev_path = torch.unsqueeze(prev_path, 0).float().to(device)

        y_hat = model(past_frames=frames, past_xyz=prev_path)

        future_path = y_hat["future_path"].squeeze().detach().cpu().numpy()
        speed = y_hat["speed"].squeeze().detach().cpu().numpy()
        steering_angle = y_hat["steering_angle"].squeeze(
        ).detach().cpu().numpy()

        img = draw_frame(curr_frame, future_path, speed, steering_angle)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)

    out.release()


if __name__ == "__main__":
    # todo bug, video not saved
    create_video(
        "../comma2k19/Chunk_1/processed/2018-08-02--08-34-47", "ref_video.mp4", "pilotnet2.pt")
