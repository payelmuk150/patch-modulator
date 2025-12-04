from functools import reduce
from operator import mul
import torch.distributed as dist

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from controllable_patching_striding.models.shared_utils.flexi_utils import (
    choose_kernel_size_deterministic,
    choose_kernel_size_random,
    choose_kernel_size_alternating,
)
from controllable_patching_striding.models.shared_utils.mlps import (
    SubsampledLinear,  # Make this use library once lbirary is setup
)
from controllable_patching_striding.models.shared_utils.patch_jitterers import PatchJitterer


def dim_pad(x, max_d):
    """
    Assume T B C are first channels, then see how many spatial dims we need to append/
    """
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
    return x


class IsotropicModel(nn.Module):
    """
    Naive model that operates at a single dimension with a repeating block.

    Args:
        patch_size (tuple): Size of the input patch
        hidden_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
    """

    def __init__(
        self,
        encoder,
        decoder,
        processor,
        hidden_dim=768,
        processor_blocks=8,
        n_states=4,
        drop_path=0.2,
        groups=12,
        max_d=3,
        static_axes=False,
        jitter_patches=False, # Not used in flexible experiments. But used in main Walrus paper.
        weight_tied_axes=True,
        gradient_checkpointing=False,
        causal_in_time=False,
        oned_only=False,
        twod_only=True,  # Temporary
        threed_only=False,
        infer=(4, 4),
    ):
        super().__init__()
        self.drop_path = drop_path
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes
        # self.pos_emb = nn.Parameter(torch.randn(16, 1, hidden_dim, 128//16, 128//16, 1)*.02)
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.space_bag = SubsampledLinear(n_states, hidden_dim // 4)
        self.causal_in_time = causal_in_time
        self.static_axes = static_axes
        self.infer = infer

        if oned_only:
            self.embed = encoder(
                spatial_dims=1,
                in_chans=hidden_dim // 4,
                hidden_dim=hidden_dim,
                groups=groups,
            )
        if twod_only:
            self.embed = encoder(
                spatial_dims=2,
                in_chans=hidden_dim // 4,
                hidden_dim=hidden_dim,
                groups=groups,
            )
        if threed_only:
            self.embed = encoder(
                spatial_dims=3,
                in_chans=hidden_dim // 4,
                hidden_dim=hidden_dim,
                groups=groups,
            )
        
        self.patch_jitterer = PatchJitterer(
            stage_dim=hidden_dim // 4,
            patch_size=None,
            max_d=self.max_d,
            jitter_patches=jitter_patches,
        )
        if 'window_size' in processor.keywords['space_mixing'].keywords:
            self.window_size = processor.keywords['space_mixing'].keywords['window_size']
            self.shift_size = [
            0 if (i % 2 == 0) else self.window_size // 2 for i in range(processor_blocks)
            ]
            self.blocks = nn.ModuleList(
                [
                    processor(
                        hidden_dim=hidden_dim,
                        drop_path=self.dp[i],
                        causal_in_time=causal_in_time,
                        shift_size=self.shift_size[i],
                    )
                    for i in range(processor_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    processor(
                        hidden_dim=hidden_dim,
                        drop_path=self.dp[i],
                        causal_in_time=causal_in_time,
                    )
                    for i in range(processor_blocks)
                ]
            )
        if oned_only:
            self.debed = decoder(
                hidden_dim=hidden_dim, out_chans=n_states, spatial_dims=1, groups=groups
            )  
        if twod_only:
            self.debed = decoder(
                spatial_dims=2, out_chans=n_states, hidden_dim=hidden_dim, groups=groups
            )
        if threed_only:
            self.debed = decoder(
                spatial_dims=3, out_chans=n_states, hidden_dim=hidden_dim, groups=groups
            )

    def freeze_middle(self):
        # First just turn grad off for everything
        for param in self.parameters():
            param.requires_grad = False
        # Activate for embed/debed layers
        for param in self.readout_head.parameters():
            param.requires_grad = True
        for param in self.space_bag.parameters():
            param.requires_grad = True
        self.debed.out_kernel.requires_grad = True
        self.debed.out_bias.requires_grad = True

    def freeze_processor(self):
        # First just turn grad off for everything
        for param in self.parameters():
            param.requires_grad = False
        # Activate for embed/debed layers
        for param in self.readout_head.parameters():
            param.requires_grad = True
        for param in self.space_bag.parameters():
            param.requires_grad = True
        for param in self.debed.parameters():
            param.requires_grad = True
        for param in self.embed.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        seed=None,
    ):
        # x - T B C H [W D]
        # state_labels - 1, C
        # bcs - #dims, 2
        # proj axes - #dims - Permutes axes to discourage learning axes - dependent relationships
        n_spatial_dims = metadata.n_spatial_dims
        T, B, C = x.shape[:3]
        x_shape = x.shape[3:]

        dynamic_ks = []
        patch_size = []
        ks_random = [0, 0]
    
        # Choose the variable patches if applicable
        if (
            hasattr(self.embed, "variable_downsample")
            and (self.embed.variable_downsample)
            and self.embed.variable_deterministic_ds
        ):
            # support for variable but deterministic downsampling
            dynamic_ks = choose_kernel_size_deterministic(x_shape)
            patch_size = [reduce(mul, k) for k in dynamic_ks]
            # patch_size doesn't matter for the dimension that is higher than the number of spatial dims
            patch_size.extend([0] * (self.max_d - len(patch_size)))

        # support for variable and random downsampling.
        # this will probably not be used in Walrus but a needed feature for dedicated paper
        elif hasattr(self.embed, "variable_downsample") and (
            self.embed.variable_downsample
        ):
            if self.training:
                ks_random = choose_kernel_size_random(self.embed.kernel_scales_seq, seed=seed)
            else:
                # We don't need to randomize the kernel size for inference. Only during training. In inference, we use the fixed kernel size.
                # Except if one wants to use alternating kernel sizes during rollout.
                ks_random = self.infer

                # If you want to use alternating kernel sizes during rollout, use the following. 
                # TODO: Add this to the config.
                #ks_random = choose_kernel_size_alternating(self.embed.kernel_scales_seq, seed=seed)

                # Optionally, if you want to use random kernel sizes during rollout, use the following. 
                # TODO: Add this to the config.
                #ks_random = choose_kernel_size_random(self.embed.kernel_scales_seq, seed=seed)
            
            for _ in range(self.max_d):
                ks = ks_random
                patch_size.append(ks[0] * ks[1])
                #patch_size.append(ks)
                dynamic_ks.append(ks)
            dynamic_ks = tuple(dynamic_ks)
        # constant downsampling as with hmlp
        else:
            patch_size = [self.embed.patch_size] * self.max_d

        # Do not want to overfit to a specific anisotropic setting, so shuffle which axes are used
        if self.static_axes or self.weight_tied_axes:
            axis_order = torch.arange(self.max_d)  #
            if proj_axes is None:
                axis_order = axis_order[:n_spatial_dims]
            else:
                axis_order = axis_order[proj_axes]
        else:
            axis_order = torch.randperm(self.max_d)[:n_spatial_dims]

        if dynamic_ks:
            dynamic_ks = tuple([dynamic_ks[axis] for axis in axis_order])
        # Pad to max dims so we can just use 3D convs - same flops, but empirically would be faster
        # to dynamically adjust which conv is used, but more verbose for compiler-friendly version
        x = dim_pad(x, self.max_d)
        # Sparse proj
        x = rearrange(x, "t b c h w d -> t b h w d c")
        x = self.space_bag(x, state_labels)

        # Encode
        x = rearrange(x, "t b h w d c -> t b c h w d")
        # Note - not currently supporting different BCs per sample (on the same GPU)
        assert torch.allclose(
            bcs[0], bcs[-1]
        ), "Currently only supporting same BCs per GPU"

        # Jitter the patches
        if hasattr(self.embed, "learned_pad"):
            if hasattr(self, "window_size"):
                x, jitter_info = self.patch_jitterer(
                    x,
                    bcs[0],
                    metadata,
                    patch_size=patch_size,
                    learned_pad=self.embed.learned_pad,
                    random_kernel=dynamic_ks,
                    base_kernel=self.embed.base_kernel_size,
                    window_size=self.window_size,
                )
            else:
                x, jitter_info = self.patch_jitterer(
                    x,
                    bcs[0],
                    metadata,
                    patch_size=patch_size,
                    learned_pad=self.embed.learned_pad,
                    random_kernel=dynamic_ks,
                    base_kernel=self.embed.base_kernel_size,
                )
        else:
            if hasattr(self, "window_size"):
                x, jitter_info = self.patch_jitterer(
                    x,
                    bcs[0],
                    metadata,
                    patch_size=patch_size,
                    window_size=self.window_size,
                )
            else:
                x, jitter_info = self.patch_jitterer(
                    x, bcs[0], metadata, patch_size=patch_size
                )
        x, stage_info = self.embed(x, bcs[0], metadata, random_kernel=dynamic_ks)

        # Process
        all_att_maps = []

        for blk in self.blocks:
            x, att_maps = blk(x, bcs, axis_order, return_att=return_att)
            all_att_maps += att_maps

        # Decode
        # If not causal, no need to debed all time steps
        if not self.causal_in_time:
            x = x[-1:]

        # Debed in appropriate dimension
        x = self.debed(x, state_labels[0], stage_info, metadata)

        # Unjitter patches
        if hasattr(self.embed, "learned_pad"):
            x = self.patch_jitterer.unjitter(
                x, jitter_info, learned_pad=self.embed.learned_pad
            )

        # De-inflate the extra channels
        for _ in range(x.ndim - 3 - n_spatial_dims):
            x = x.squeeze(-1)
        # Return T, B, C, H, W, D
        return x  # TODO - Return attention maps for debugging
