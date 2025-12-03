import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, to_3tuple
from functools import partial
import numpy as np
from typing import Sequence, Tuple
from torch import Tensor
from einops import rearrange
from the_well.data.datasets import BoundaryCondition

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def to_ntuple(n):
    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple([x] * n)
    return parse

def window_partition(x, window_size):
    """
    Partitions the input tensor into windows of the specified size.
    Supports both 2D and 3D inputs.

    Args:
        x (torch.Tensor): The input tensor of shape (B, H, W, C) for 2D or (B, H, W, Z, C) for 3D.
        window_size (int): The size of the window.

    Returns:
        torch.Tensor: The partitioned windows.
    """
    if len(x.shape) == 4:  # 2D case
        # Rearrange the tensor using einops
        return rearrange(x, 'B (H p1) (W p2) C -> (B H W) (p1 p2) C', p1=window_size, p2=window_size)
    elif len(x.shape) == 5:  # 3D case
        # Rearrange the tensor using einops
        return rearrange(x, 'B (H p1) (W p2) (Z p3) C -> (B H W Z) (p1 p2 p3) C', p1=window_size, p2=window_size, p3=window_size)
    else:
        raise ValueError("Input tensor must be either 4D (2D case) or 5D (3D case).")

def window_reverse(windows, window_size, *spatial_dims):
    """
    Reverses the window partitioning, reconstructing the original tensor.
    Supports both 2D and 3D cases.

    Args:
        windows (torch.Tensor): The partitioned windows of shape (num_windows, window_size, ..., C).
        window_size (int): The size of the window.
        spatial_dims (int): The original spatial dimensions (H, W) for 2D or (H, W, Z) for 3D.

    Returns:
        torch.Tensor: The reconstructed tensor of shape (B, H, W, C) for 2D or (B, H, W, Z, C) for 3D.
    """
    B = windows.shape[0] // int(np.prod([dim // window_size for dim in spatial_dims]))
    num_dims = len(spatial_dims)  # Number of spatial dimensions (2 or 3)

    if num_dims == 2:  # 2D case
        H, W = spatial_dims
        x = rearrange(windows, '(B H W) (p1 p2) C -> B (H p1) (W p2) C', B=B, H=H // window_size, W=W // window_size, p1=window_size, p2=window_size)
    elif num_dims == 3:  # 3D case
        H, W, Z = spatial_dims
        x = rearrange(windows, '(B H W Z) (p1 p2 p3) C -> B (H p1) (W p2) (Z p3) C', B=B, H=H // window_size, W=W // window_size, Z=Z // window_size, p1=window_size, p2=window_size, p3=window_size)
    else:
        raise ValueError("Only 2D and 3D inputs are supported.")

    return x


class WindowAttention(nn.Module):
    r"""Window-based multi-head self-attention (W-MSA) module with relative position bias.
    Supports both 2D and 3D inputs.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The dimensions of the window (2D: [H, W], 3D: [H, W, Z]).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int], optional): The dimensions of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size #Here just give a number, was:  [Wh, Ww] for 2D or [Wh, Ww, Wz] for 3D
        self.num_heads = num_heads

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        #QVK layer with RMSNorm (need torch>2.4)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm_q = nn.RMSNorm([dim//num_heads])
        self.norm_k = nn.RMSNorm([dim//num_heads])

        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Precompute relative position indices and coordinates for 2D and 3D
        window_size_2d = (window_size, window_size)
        window_size_3d = (window_size, window_size, window_size)
        relative_position_index_2d = WindowAttention.get_relative_position_index(window_tuple=window_size_2d)
        relative_coords_table_2d = WindowAttention.get_relative_coords_table(window_tuple=window_size_2d)
        relative_position_index_3d = WindowAttention.get_relative_position_index(window_tuple=window_size_3d)
        relative_coords_table_3d = WindowAttention.get_relative_coords_table(window_tuple=window_size_3d)

        # Register buffers
        self.register_buffer("relative_position_index_2d", relative_position_index_2d, persistent=False)
        self.register_buffer("relative_coords_table_2d", relative_coords_table_2d, persistent=False)
        self.register_buffer("relative_position_index_3d", relative_position_index_3d, persistent=False)
        self.register_buffer("relative_coords_table_3d", relative_coords_table_3d, persistent=False)

    @staticmethod
    def get_relative_position_index(window_tuple: Tuple) -> Tensor:
        num_dims = len(window_tuple)
        coords = torch.stack(torch.meshgrid([torch.arange(window_tuple[i]) for i in range(num_dims)]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        for i in range(num_dims):
            relative_coords[:, :, i] += window_tuple[i] - 1
        if num_dims == 2:
            relative_coords[:, :, 0] *= 2 * window_tuple[1] - 1
        elif num_dims == 3:
            relative_coords[:, :, 0] *= (2 * window_tuple[1] - 1) * (2 * window_tuple[2] - 1)
            relative_coords[:, :, 1] *= 2 * window_tuple[2] - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    @staticmethod
    def get_relative_coords_table(window_tuple: Sequence[int]) -> torch.Tensor:
        num_dim = len(window_tuple)
        coords = [torch.arange(-(w - 1), w, dtype=torch.float32) for w in window_tuple[:num_dim]]
        meshgrids = torch.meshgrid(*coords)
        if num_dim == 2:
            relative_coords_table = torch.stack(meshgrids).permute(1, 2, 0).unsqueeze(0)
        elif num_dim == 3:
            relative_coords_table = torch.stack(meshgrids).permute(1, 2, 3, 0).unsqueeze(0)
        else:
            raise ValueError("num_dim must be either 2 or 3")

        for i in range(num_dim):
            relative_coords_table[..., i] /= window_tuple[i] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        return relative_coords_table
    
    def forward(self, x, num_dim, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        device = x.device
        B_, N, C = x.shape
        qkv = rearrange(self.qkv(x), 'B N (n H D) -> n B H N D', n=3, H=self.num_heads)
        q, k, v = qkv.unbind(0) #shape: [B_, num_heads, N, C]

        # Select the appropriate relative position buffers
        if num_dim == 2:
            relative_position_index = self.relative_position_index_2d  
            relative_coords_table = self.relative_coords_table_2d  
        elif num_dim == 3:
            relative_position_index = self.relative_position_index_3d  
            relative_coords_table = self.relative_coords_table_3d  
        else:
            raise ValueError("num_dim must be either 2 or 3")
        window_tuple = to_ntuple(num_dim)(self.window_size)
        #relative_position_index = self.get_relative_position_index(window_tuple=window_tuple) # Shape: [Wh*Ww*(Wz), Wh*Ww*(Wz)]
        #relative_coords_table = self.get_relative_coords_table(window_tuple=window_tuple) # Shape: [Wh*Ww*(Wz), Wh*Ww*(Wz), num_dims]
        q_norm = self.norm_q(q) #RMSNorm
        k_norm = self.norm_k(k) #RMSNorm

        if relative_coords_table.shape[-1] == 2:
            # Pad the last dimension with zeros to make it num_dims = 3
            relative_coords_table = F.pad(relative_coords_table, (0, 1), "constant", 0)  # Pads the last dimension by (0, 1)

        relative_position_bias_table = self.cpb_mlp(relative_coords_table).view(-1, self.num_heads) # Shape: [Wh*Ww*(Wz), num_heads]
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(np.prod(window_tuple), np.prod(window_tuple), -1)  # Wh*Ww*(Wz),Wh*Ww*(Wz),nH
        relative_position_bias = rearrange(relative_position_bias, 'N M H -> 1 H 1 N M')  # 1, nH, 1, Wh*Ww*(Wz), Wh*Ww*(Wz)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        #print('relative_position_bias shape:', relative_position_bias.shape)
        
        if mask is not None:
            nW = mask.shape[0] 
            m = rearrange(mask, 'nW N M -> 1 1 nW N M') + relative_position_bias
            #print('q norm shape', q_norm.shape)
            #print('k norm shape', k_norm.shape)
            #print('v shape', v.shape)
            #print('m shape', m.shape)
            y = F.scaled_dot_product_attention(rearrange(q_norm, '(B nW) H N C -> B H nW N C', nW=nW),
                                               rearrange(k_norm, '(B nW) H N C -> B H nW N C', nW=nW), 
                                               rearrange(v, '(B nW) H N C -> B H nW N C', nW=nW), 
                                               attn_mask=m, 
                                               dropout_p=self.attn_drop, 
                                               scale=1.0)
        elif mask is None:
            nW = relative_position_bias.shape[0]
            y = F.scaled_dot_product_attention(rearrange(q_norm, '(B nW) H N C -> B H nW N C', nW=nW), 
                                               rearrange(k_norm, '(B nW) H N C -> B H nW N C', nW=nW), 
                                               rearrange(v, '(B nW) H N C -> B H nW N C', nW=nW), 
                                               attn_mask=relative_position_bias, 
                                               dropout_p=self.attn_drop, 
                                               scale=1.0)

        x = rearrange(y, 'B H nW N C -> (B nW) N (H C)', nW=nW, H=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.GroupNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        hidden_dim,
        num_heads,
        #input_resolution=224,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.GroupNorm,
        pretrained_window_size=0,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.dim = hidden_dim
        #self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(num_heads, hidden_dim, affine=True)
        self.attn = WindowAttention(
            hidden_dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.GroupNorm(num_heads, hidden_dim, affine=True)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    @staticmethod
    def create_attn_mask(window_size, shift_size, bcs, device, H, W, Z = None):
        #print('BCS:', bcs)
        if Z is None:
            img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
            if shift_size > 0:
                h_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None),
                )
                w_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None),
                )
                cnt = 0
                if (bcs[0][0, 0] == BoundaryCondition["PERIODIC"].value) and (bcs[0][1, 0] == BoundaryCondition["PERIODIC"].value):
                    pass
                elif (bcs[0][0, 0] != BoundaryCondition["PERIODIC"].value) and (bcs[0][1, 0] == BoundaryCondition["PERIODIC"].value):
                    for h in h_slices:
                        img_mask[:, h, :, :] = cnt
                        cnt += 1
                elif (bcs[0][0, 0] == BoundaryCondition["PERIODIC"].value) and (bcs[0][1, 0] != BoundaryCondition["PERIODIC"].value):
                    for w in w_slices:
                        img_mask[:, :, w, :] = cnt
                        cnt += 1
                else:
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1
                mask_windows = window_partition(
                    img_mask, window_size
                )  # nW, window_size * window_size (* window_size), 1
                #mask_windows = mask_windows.reshape(-1, window_size * window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(
                    attn_mask != 0, float(-100.0)
                ).masked_fill(attn_mask == 0, float(0.0))

                #print('attn_mask shape:', attn_mask.shape)
                return attn_mask.squeeze(-1)
            else:
                return None
        else:
            # FIX THIS FOR 3D DATA
            img_mask = torch.zeros((1, H, W, Z, 1), device=device)  # 1 H W Z 1

            if shift_size > 0:
                h_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None),
                )
                w_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None),
                )
                z_slices = (
                    slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None),
                )
                cnt = 0

                # Have not included all possibilities yet
                if (bcs[0, 0] == 0) and (bcs[0, 1] == 0) and (bcs[0, 2] == 0):
                    for h in h_slices:
                        for w in w_slices:
                            for z in z_slices:
                                img_mask[:, h, w, z, :] = cnt
                                cnt += 1

                elif (bcs[0, 0] == 1) and (bcs[0, 1] == 0) and (bcs[0, 2] == 0):
                    for w in w_slices:
                        for z in z_slices:
                            img_mask[:, :, w, z, :] = cnt
                            cnt += 1

                elif (bcs[0, 0] == 1) and (bcs[0, 1] == 1) and (bcs[0, 2] == 0):
                    for z in z_slices:
                        img_mask[:, :, :, z, :] = cnt
                        cnt += 1
                else:
                    pass

                mask_windows = window_partition(img_mask, window_size)
                mask_windows = mask_windows.squeeze(-1)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(
                    attn_mask != 0, float(-100.0)
                ).masked_fill(attn_mask == 0, float(0.0))
                return attn_mask.squeeze(-1)
            else:
                return None
 
    def mini_window_partition(self,x, window_size):
        dims = x.shape[1:-1]
        new_dims = []
        for d in dims:
            new_dims.extend([d // window_size, window_size])
        return x.view(x.shape[0], *new_dims, x.shape[-1])

    def forward(self, x, bcs, axis_order, return_att=False):
        #print('shape before swin block:', x.shape)
        #x is B, C, H, W, (Z)
        num_dim = len(x.shape[2:])
        H, W, *rest = x.shape[2:]
        Z = rest[0] if rest else None
        resolution = (H, W, Z) if Z is not None else (H, W)

        if num_dim == 2:
            x = rearrange(x, 'b c h w -> b (h w) c')
        elif num_dim == 3:
            x = rearrange(x, 'b c h w z -> b (h w z) c')

        shortcut = x

        if num_dim not in [2, 3]:
            raise ValueError("Resolution must be a tuple of length 2 (H, W) or 3 (H, W, Z).")
        
        device = x.device
        attn_mask = SwinTransformerBlock.create_attn_mask(self.window_size, self.shift_size, bcs, device, H,W,Z)

        if num_dim == 2:
            x = rearrange(x, 'b (h w) c -> b h w c', h=H)
        elif num_dim == 3:
            x = rearrange(x, 'b (h w z) c-> b h w z c', h=H, w=W)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,)*num_dim, dims=tuple(range(1, num_dim+1)))
        else:
            shifted_x = x

        # partition windows takes B, H, W, C
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size * window_size (*window_size), C


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, num_dim=num_dim, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, *resolution)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,)*num_dim, dims=tuple(range(1, num_dim+1)))
        else:
            x = shifted_x

        if num_dim == 2:
            x = rearrange(x, 'b h w c -> b (h w) c', h=H)
        elif num_dim == 3:
            x = rearrange(x, 'b h w z c -> b (h w z) c', h=H, w=W)
        x = rearrange(x, 'b l c -> b c l')
        x = self.norm1(x)
        x = rearrange(x, 'b c l -> b l c')
        
        x = shortcut + self.drop_path(x)

        x = self.mlp(x)
        x = rearrange(x, 'b l c -> b c l')
        x = self.norm2(x)
        x = rearrange(x, 'b c l -> b l c')

        # FFN
        x = x + self.drop_path(x)

        if num_dim == 2:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        elif num_dim == 3:
            x = rearrange(x, 'b (h w z) c -> b c h w z', h=H, w=W)

        return x, []