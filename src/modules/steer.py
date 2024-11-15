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
        n_frames: int,
        img_size=224,
        depth=24,  # from Table 1: model sizes (tiny) - 24
        embd_dim=192,  # from Table 1: model sizes (tiny)
        drop_rate=0.1,  # from Table 6: training settings - 0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.n_frames = n_frames
        self.embd_dim = embd_dim

        # from video sequence to embeddings
        self.patch_embd = VideoPatchEmbedding(
            n_frames=n_frames, img_size=img_size, embd_dim=embd_dim, drop_rate=drop_rate
        )

        # process embedding sequence using mamba blocks
        self.video_encoder = BlockList(n_layers=depth, n_hidden=embd_dim)

        # process past path using mamba block list
        self.path_encoder = nn.Sequential(
            # since each path only has 3 components (x,y,z)
            # use a much smaller mamba encoder block, and use
            # two linear layers two project into right dims
            # note: first merge last two dims of past_xyz
            nn.Linear(n_frames * 3, embd_dim // 6, bias=False),
            BlockList(n_layers=depth // 2, n_hidden=embd_dim // 6),
            nn.Linear(embd_dim // 6, embd_dim, bias=False),
        )

        # final head to output hidden features
        # to be later passed to an av_wrapper to
        # get proper outputs
        self.head = nn.Sequential(
            # slowly downsize the embeddings
            nn.Linear(embd_dim, embd_dim // 3 * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embd_dim // 3 * 2, embd_dim // 2),
        )

    def forward(self, past_frames, past_xyz):
        """
        past_frames: (B, C, T, W, H)
        past_xyz: (B, T, 3)

        returns hidden features of shape (B, embd_dim/2)
        """

        # get batch size
        B = past_frames.shape[0]

        # (B, T', C) -> flatten sequence of token embeddings
        # get video patches
        patches = self.patch_embd(past_frames)

        # (B, T', C); C = embd_dim
        # pass through a list of mamba blocks
        video_features = self.video_encoder(patches)

        # go from (B,T,3) to (B,T*3) and to (B, 1, C)
        past_xyz = past_xyz.view(B, 1, -1)
        path_features = self.path_encoder(past_xyz)

        # add the two sets of features
        # (B,1,C) + (B,T',C) = (B,T',C)
        hidden = video_features + path_features

        # only the classification token
        # (B, C)
        hidden = hidden[:, 0, :]

        # (B, embd_dim/2)
        features = self.head(hidden)

        # to be passed to the av_wrapper to extract outputs
        return features


class VideoPatchEmbedding(nn.Module):
    """
    Creates 3D patches from a video sequence. Returns "tubelet embedding",
    by cutting "tubes" from the video. After getting the embeddings, it
    adds a classification token at the end, a positional token embed to all
    embeddings and a temporal token embed (these three being learnable params).
    The additional tokens help the SSM better learn the ordered features.
    """

    def __init__(
        self,
        patch_size=16,
        img_size=224,
        img_channels=3,
        n_frames=3,
        embd_dim=192,  # from Table 1: model sizes (tiny)
        kernel_size=1,
        drop_rate=0.1,  # from Table 6: training settings
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
            stride=(kernel_size, patch_size, patch_size),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
        self.pos_embd = nn.Parameter(
            torch.zeros(self.n_patches + 1, self.embd_dim))
        self.temp_embd = nn.Parameter(
            torch.zeros(1, n_frames // kernel_size, self.embd_dim)
        )

        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        # x.shape = (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # stays the same relative shape
        # aka (B, 192, T, 14, 14) for img of (3, 224, 224)
        x = self.proj(x)

        # convert to other shape to allow for concat
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # B, T, H, W, C
        x = x.reshape(B * T, H * W, C)  # merge dims

        # 1. classification token
        # from "lucidrains/vit-pytorch" (great repo :D)
        cls_token = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # basically (B*T, H*W+1, C); add a prefix token

        # (120, 197, 192); (197, 192); (1, 30, 192)
        # (B*T, W*H+1, C); (W*H+1, C); (1, T, C)
        # (x.shape, self.pos_embd.shape, self.temp_embd.shape)

        # 2. positional embedding
        # same shape: (B*T, H*W+1, C)
        x = x + self.pos_embd  # just add the spatial info

        # 3. temporal embedding
        cls_tokens = x[:B, :1, :]  # grab first token from first frame
        x = x[:, 1:]  # exclude cls token from all frames (h*w+1 to h*w)
        x = rearrange(x, "(b t) hw c -> (b hw) t c", b=B, t=T)
        x = x + self.temp_embd  # remains same shape
        x = rearrange(x, "(b hw) t c -> b (t hw) c", b=B, t=T)
        # re-add the classification token
        x = torch.cat((cls_tokens, x), dim=1)
        # (B, T', C); new linear sequence of embedding tokens

        # T' = 5881 = 30 * (224/16) * (224/16) + 1 (cls token)
        # = T * (H/P) * (W/P); and C = 192; embedding size

        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    A single block wrapping a mamba class, layer normalization and a skip connection.
    Takes in hidden features and  returns other features (along with a residual)
    of the same shape.
    """

    def __init__(self, n_embd: int, layer_idx: Optional[int] = None, norm_epsilon=1e-5, use_triton_kernel=False):
        super().__init__()

        assert (
            _is_mamba_installed
        ), "Mamba is not installed. Please install it using `pip install mamba-ssm`."

        self.d_model = n_embd
        self.norm_eps = norm_epsilon
        self.layer_idx = layer_idx
        self.use_triton_kernel = use_triton_kernel

        self.mamba = Mamba(d_model=n_embd, layer_idx=layer_idx)
        self.ln = nn.LayerNorm(n_embd, eps=norm_epsilon)

    def forward(
        self, h: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes in a hidden state (the sequence encoded from the
        previous block) and a residual value.
        """

        # triton operation, for performance
        if self.use_triton_kernel:
            # https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
            # doesn't work with torch.compile! atm: 09.09.2024
            h, residual = layer_norm_fn(
                h,
                self.ln.weight,
                self.ln.bias,
                residual,
                prenorm=True,
                eps=self.norm_eps,
                residual_in_fp32=True,
            )
        else:
            # skip connection
            residual = (h + residual) if residual is not None else h
            h = self.ln(residual.to(dtype=self.ln.weight.dtype))

        # (B, T', C)
        h = self.mamba(h)

        # (B, T', C)
        return h, residual


class BlockList(nn.Module):
    """
    Helper class to abstract a list of Mamba blocks. Also
    manages residual connections.
    """

    def __init__(self, n_layers: int = 24, n_hidden: int = 192, use_triton_kernel: bool = False):
        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.use_triton_kernel = use_triton_kernel

        self.blocks = nn.ModuleList(
            [Block(n_embd=self.n_hidden, layer_idx=i, use_triton_kernel=use_triton_kernel)
             for i in range(self.n_layers)]
        )
        self.ln = nn.LayerNorm(self.n_hidden)

    def forward(self, hidden):
        # forward through the mamba encoder
        # same shape
        residual = None
        for block in self.blocks:
            hidden, residual = block(hidden, residual)

        # layer norm
        if self.use_triton_kernel:
            # doesn't work with torch.compile! atm: 09.09.2024
            hidden = layer_norm_fn(
                hidden,
                # use the nn.LayerNorm state, but forward
                # it using a triton kernel
                self.norm.weight,
                self.norm.bias,
                residual_in_fp32=True,
            )
        else:
            hidden = self.ln(hidden.to(dtype=self.ln.weight.dtype))

        # same shape: (B, T', C)
        return hidden


# includes resizing to the specified target
SteerTransform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)

# let's test the model
if __name__ == "__main__":
    B, T, HW = 4, 30, 224
    past_frames = torch.randn((B, T, 3, HW, HW), device="cuda")
    past_xyz = torch.randn((B, T, 3), device="cuda")

    model = SteerNet(n_frames=T, img_size=HW).to("cuda")
    print(model(past_frames, past_xyz).shape)
