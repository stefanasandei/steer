"""
The main presented model. Based on the VideoMamba architecture, it levereges
SSM's capability to process long sequence data to *steer* a vehicle.
"""

from typing import Optional
from einops import rearrange
from torchvision.transforms import transforms
from torch.nn import functional as F
import torch.nn as nn
import torch

# mamba pkg raises FutureWarnings from torch usage
import warnings
warnings.filterwarnings("ignore")


_is_mamba_installed = False
try:
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
    from mamba_ssm import Mamba

    _is_mamba_installed = True
except ImportError:
    _is_mamba_installed = False


class SteerNet(nn.Module):
    """ 
    Based on the VideoMamba (Kunchang Li et al.). Takes in a video sequence,
    extracts 3D paches, forwards those through bidirectional Mamba blocks,
    and returns the computed hidden state features. 
    """

    def __init__(
        self,
        img_size: int,
        n_frames: int,

        depth=24,  # from Table 1: model sizes (tiny)
        embd_dim=192,  # from Table 1: model sizes (tiny)
    ):
        super().__init__()

        self.img_size = img_size
        self.n_frames = n_frames

        self.patch_embd = VideoPatchEmbedding(
            n_frames=n_frames, img_size=img_size, embd_dim=embd_dim)

        self.blocks = nn.ModuleList(
            [Block(n_embd=embd_dim, layer_idx=i) for i in range(depth)])
        self.norm = nn.LayerNorm(embd_dim)

    def forward(self, past_frames, past_xyz):
        """
        past_frames: (B, C, T, W, H)
        past_xyz: (B, T, 3) (todo)
        """

        # (B, T', C) -> flatten sequence of token embeddings
        # get video patches
        patches = self.patch_embd(past_frames)

        # forward through the mamba encoder
        # same shape
        residual = None
        hidden_states = patches
        for block in self.blocks:
            hidden_states, residual = block(
                hidden_states, residual
            )

        # layer norm
        hidden_states = layer_norm_fn(
            hidden_states,
            # use the nn.LayerNorm state, but forward
            # it using a triton kernel
            self.norm.weight, self.norm.bias,
            residual_in_fp32=True
        )

        # only the classification token
        # to be passed to the av_wrapper to extract outputs
        features = hidden_states[:, 0, :]

        # (B, embd_dim)
        return features


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


class Block(nn.Module):
    """
    A single block wrapping a mamba class, layer normalization and a skip connection.
    Takes in hidden features and  returns other features (along with a residual) 
    of the same shape. 
    """

    def __init__(self, n_embd: int, layer_idx: Optional[int] = None, norm_epsilon=1e-5):
        super().__init__()

        assert _is_mamba_installed, "Mamba is not installed. Please install it using `pip install mamba-ssm`."

        self.d_model = n_embd
        self.norm_eps = norm_epsilon
        self.layer_idx = layer_idx

        self.mamba = Mamba(d_model=n_embd, layer_idx=layer_idx)
        self.ln = nn.LayerNorm(n_embd, eps=norm_epsilon)

    def forward(self, h: torch.Tensor,
                residual: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes in a hidden state (the sequence encoded from the
        previous block) and a residual value.
        """

        # triton operation, for performance
        h, residual = layer_norm_fn(h,
                                    self.ln.weight, self.ln.bias,
                                    residual,
                                    prenorm=True, eps=self.norm_eps,
                                    residual_in_fp32=True)

        # (B, C)
        h = self.mamba(h)

        # (B, C)
        return h, residual


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
    past_frames = torch.randn((B, 3, T, HW, HW), device="cuda")
    past_xyz = torch.randn((B, T, 3), device="cuda")

    model = SteerNet(n_frames=T, img_size=HW).to("cuda")
    print(model(past_frames, past_xyz).shape)
