import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class hMLP_decoder(nn.Module):
    """Patch to Image De-bedding"""

    def __init__(
        self,
        patch_size=(16, 16),
        out_chans=3,
        hidden_dim=768,
        spatial_dims=2,
        groups=12,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        if spatial_dims == 1:
            conv = nn.ConvTranspose1d
            self.conv_func = F.conv_transpose1d
        elif spatial_dims == 2:
            conv = nn.ConvTranspose2d
            self.conv_func = F.conv_transpose2d
        elif spatial_dims == 3:
            conv = nn.ConvTranspose3d
            self.conv_func = F.conv_transpose3d
        else:
            conv = nn.Linear
        kernel_size_ = int(np.sqrt(self.patch_size))
        self.kernel_size_ = kernel_size_
        self.out_proj = nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[
                        conv(
                            hidden_dim,
                            hidden_dim // 4,
                            kernel_size=kernel_size_,
                            stride=kernel_size_,
                            bias=False,
                        ),
                        nn.GroupNorm(groups, hidden_dim // 4, affine=True),
                        nn.GELU(),
                    ]
                ),
                # nn.ConvTranspose2d(embed_dim//4, out_chans, kernel_size=4, stride=4),
            ]
        )
        out_head = conv(hidden_dim // 4, out_chans, kernel_size=kernel_size_, stride=kernel_size_)
        self.out_kernel = nn.Parameter(out_head.weight)
        self.out_bias = nn.Parameter(out_head.bias)

    def forward(self, x, state_labels, stage_info=None, metadata=None, **kwargs):
        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D
        x = x.squeeze((-2, -1))  # (T B) C H W D -> (T B) C H [W] [D]
        for i, proj in enumerate(self.out_proj):
            x = proj(x)
        # Project specific dims...
        x = self.conv_func(
            x, self.out_kernel[:, state_labels], self.out_bias[state_labels], stride=self.kernel_size_
        )
        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        return x
