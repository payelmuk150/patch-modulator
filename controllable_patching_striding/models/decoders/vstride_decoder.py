from __future__ import annotations

from typing import Tuple, Union, cast

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class VstrideDecoder(nn.Module):
    def __init__(
        self,
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        out_chans: int = 3,
        hidden_dim: int = 768,
        spatial_dims: int = 2,
        groups: int = 12,
        learned_pad: bool = True,
    ) -> None:
        super().__init__()

        self.learned_pad = learned_pad
        self.hidden_dim = hidden_dim
        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.spatial_dims = spatial_dims
        self.norm_layer1 = nn.GroupNorm

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        if self.spatial_dims == 1:
            conv: type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d] = (
                nn.ConvTranspose1d
            )
            self.conv_func = F.conv_transpose1d
        elif self.spatial_dims == 2:
            conv = nn.ConvTranspose2d
            self.conv_func = F.conv_transpose2d
        elif self.spatial_dims == 3:
            conv = nn.ConvTranspose3d
            self.conv_func = F.conv_transpose3d

        self.proj1 = conv(
            hidden_dim,
            hidden_dim // 4,
            kernel_size=self.base_kernel1,  # type: ignore
            bias=False,
        )

        # Normalization layer after the first convolutional layer
        self.norm1 = self.norm_layer1(groups, hidden_dim // 4, affine=True)
        self.act1 = nn.GELU()

        self.proj2 = conv(
            hidden_dim // 4,
            out_chans,
            kernel_size=self.base_kernel2,  # type: ignore
        )

    def forward(
        self,
        x: Tensor,
        state_labels,
        stage_info=None,
        metadata=None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        embed_kernel = stage_info["random_kernel"]
        debed_kernel = tuple((b, a) for (a, b) in embed_kernel)

        stride1 = tuple([debed_kernel[i][0] for i in range(self.spatial_dims)])
        stride2 = tuple([debed_kernel[i][1] for i in range(self.spatial_dims)])

        if self.learned_pad:
            # learned padding is taken care of in patch jitterer
            padding1, padding2 = 0, 0
        else:
            padding1 = tuple(
                [
                    int(np.ceil((self.stride1[i] - stride) / 2.0))
                    for i, stride in enumerate(stride1)
                ]
            )  # type: ignore
            padding2 = tuple(
                [
                    int(np.ceil((self.stride2[i] - stride) / 2.0))
                    for i, stride in enumerate(stride2)
                ]
            )  # type: ignore

        padding1 = cast(Tuple[int, ...], padding1)  # type: ignore
        padding2 = cast(Tuple[int, ...], padding2)  # type: ignore

        weight1 = self.proj1.weight
        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D
        x = x.squeeze((-2, -1))  # (T B) C H W D -> (T B) C H [W] [D]

        x = self.conv_func(
            x, weight1, bias=self.proj1.bias, stride=stride1#, padding=padding1
        )
        x = self.norm1(x)  # Apply normalization
        x = self.act1(x)  # Apply GELU activation

        weight2 = self.proj2.weight
        x = self.conv_func(
            x,
            weight2[:, state_labels],
            bias=self.proj2.bias[state_labels],  # type: ignore
            stride=stride2,
            #padding=padding2,
        )

        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)

        return x
