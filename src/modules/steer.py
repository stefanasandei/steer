"""
The main presented model. Based on the VideoMamba architecture, it levereges
SSM's capability to process long sequence data to *steer* a vehicle.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import transforms
from einops import rearrange


class SteerNet(nn.Module):
    """ """

    def __init__(
        self,
        img_size: int,
        n_frames: int
    ):
        super().__init__()

        self.img_size = img_size
        self.n_frames = n_frames

        self.patch_embd = VideoPatchEmbedding(
            n_frames=n_frames, img_size=img_size)

    def forward(self, past_frames, past_xyz):
        """
        past_frames: (B, C, T, W, H)
        past_xyz: (B, T, 3)
        """

        # (B, T', C) -> flatten sequence of token embeddings
        patches = self.patch_embd(past_frames)

        # for debug, todo
        B = past_frames.shape[0]
        return torch.randn(B, 128)


class VideoPatchEmbedding(nn.Module):
    """
    Creates 3D patches from a video sequence. Returns "tubelet embedding",
    by cutting "tubes" from the video. After getting the embeddings, it
    adds a classification token at the end, a positional token embed to all
    embeddings and a temporal token embed (these three being learnable params).
    The additional tokens help the SSM better learn the ordered features.
    """

    def __init__(self, patch_size=16,
                 img_size=224, img_channels=3, n_frames=3,
                 embd_dim=192,  # from Table 1: model sizes (tiny)
                 kernel_size=1,
                 drop_rate=0.1  # from Table 6: training settings
                 ):
        super().__init__()

        self.img_size = (img_size, img_size)
        self.patches_size = (patch_size, patch_size)
        self.n_patches = (img_size // patch_size) ** 2
        self.tubelet_size = kernel_size
        self.embd_dim = embd_dim

        self.proj = nn.Conv3d(
            in_channels=img_channels,
            out_channels=embd_dim,
            kernel_size=(kernel_size, patch_size, patch_size),
            stride=(kernel_size, patch_size, patch_size)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
        self.pos_embd = nn.Parameter(
            torch.zeros(self.n_patches+1, self.embd_dim))
        self.temp_embd = nn.Parameter(torch.zeros(
            1, n_frames // kernel_size, self.embd_dim))

        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        # x.shape = (B, C, T, H, W)

        # stays the same relative shape
        # aka (B, 192, T, 14, 14) for img of (3, 224, 224)
        x = self.proj(x)

        # convert to other shape to allow for concat
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # B, T, H, W, C
        x = x.reshape(B*T, H*W, C)  # merge dims

        # 1. classification token
        # from "lucidrains/vit-pytorch" (great repo :D)
        cls_token = self.cls_token.expand(B*T, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # basically (B*T, H*W+1, C); add a prefix token

        # 2. positional embedding
        # same shape: (B*T, H*W+1, C)
        x = x + self.pos_embd  # just add the spatial info

        # 3. temporal embedding
        cls_tokens = x[:B, :1, :]  # grab first token from first frame
        x = x[:, 1:]  # exclude cls token from all frames (h*w+1 to h*w)
        x = rearrange(x, '(b t) hw c -> (b hw) t c', b=B, t=T)
        x = x + self.temp_embd  # remains same shape
        x = rearrange(x, '(b hw) t c -> b (t hw) c', b=B, t=T)
        # re-add the classification token
        x = torch.cat((cls_tokens, x), dim=1)
        # (B, T', C); new linear sequence of embedding tokens

        x = self.drop(x)
        return x


# includes resizing to the specified target
SteerTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224, 224]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# let's test the model
if __name__ == "__main__":
    B, T, HW = 8, 30, 224
    past_frames = torch.randn((B, 3, T, HW, HW))
    past_xyz = torch.randn((B, T, 3))

    model = SteerNet(n_frames=T, img_size=HW)
    model(past_frames, past_xyz)
