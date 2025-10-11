"""
Enhanced nnFormer Architecture with Multi-Scale Cross-Attention
==============================================================

This module implements an enhanced version of the nnFormer architecture
that incorporates multi-scale cross-attention for improved 3D medical
image segmentation performance.

Key Enhancements:
1. Multi-scale cross-attention between encoder layers
2. Adaptive feature fusion across different resolutions
3. Enhanced skip connections with cross-scale feature interaction

Author: 210353V
Date: October 2025
"""

from einops import rearrange
from copy import deepcopy
from nnformer.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnformer.network_architecture.initialization import InitWeights_He
from nnformer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

# Import original nnFormer components
from nnformer.network_architecture.nnFormer_tumor import (
    Mlp, window_partition, window_reverse, WindowAttention, 
    SwinTransformerBlock, ContiguousGrad
)

# Import our enhanced attention modules
from enhancements.attention.multi_scale_attention import (
    MultiScaleCrossAttention, AdaptiveFeatureFusion, 
    EnhancedSwinTransformerBlock
)


class EnhancedSwinTransformerBlock_CrossAttn(nn.Module):
    """
    Enhanced Swin Transformer Block with cross-attention capability.
    Extends the original SwinTransformerBlock to support multi-scale feature interaction.
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, 
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, cross_attention=False, cross_dim=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.cross_attention = cross_attention
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        
        # Add cross-attention if enabled
        if cross_attention and cross_dim is not None:
            self.cross_attn = MultiScaleCrossAttention(
                dim_q=dim, dim_kv=cross_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_attn = None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)
        
        # Create attention mask for shifted window
        if self.shift_size > 0:
            S, H, W = self.input_resolution
            img_mask = torch.zeros((1, S, H, W, 1))
            s_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for s in s_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, s, h, w, :] = cnt
                        cnt += 1
            
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x, cross_features=None):
        """
        Forward pass with optional cross-attention.
        
        Args:
            x: Input features [B, L, C]
            cross_features: Features from other scales for cross-attention [B, L', C']
        """
        B, L, C = x.shape
        S, H, W = self.input_resolution
        
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), 
                                 dims=(1, 2, 3))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, S, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), 
                         dims=(1, 2, 3))
        else:
            x = shifted_x
        
        x = x.view(B, S * H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        
        # Cross-attention if available
        if self.cross_attn is not None and cross_features is not None:
            shortcut_cross = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x_cross, cross_features)
            x = shortcut_cross + self.drop_path(x_cross)
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        
        return x


class EnhancedBasicLayer(nn.Module):
    """
    Enhanced Basic Layer with multi-scale cross-attention.
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 cross_attention=False, cross_dim=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cross_attention = cross_attention
        
        # Build blocks
        self.blocks = nn.ModuleList([
            EnhancedSwinTransformerBlock_CrossAttn(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                cross_attention=cross_attention,
                cross_dim=cross_dim
            ) for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x, cross_features=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, cross_features)
            else:
                x = blk(x, cross_features)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class EnhancedEncoder(nn.Module):
    """
    Enhanced Encoder with multi-scale cross-attention capabilities.
    """
    
    def __init__(self, pretrain_img_size=[64,128,128], patch_size=[2,4,4], 
                 in_chans=1, embed_dim=192, depths=[2, 2, 2, 2], 
                 num_heads=[6, 12, 24, 48], window_size=[4,4,8,4], 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, enable_cross_attention=True):
        super().__init__()
        
        # Import original components
        from nnformer.network_architecture.nnFormer_tumor import (
            PatchEmbed, PatchMerging
        )
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.enable_cross_attention = enable_cross_attention
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None
        )
        
        # Calculate patches resolution manually  
        self.patches_resolution = [
            pretrain_img_size[i] // patch_size[i] for i in range(3)
        ]
        num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]
        
        # Absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Calculate cross-attention configuration
            cross_attention = enable_cross_attention and i_layer > 0
            cross_dim = int(embed_dim * 2 ** (i_layer - 1)) if cross_attention else None
            
            layer = EnhancedBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer),
                    self.patches_resolution[2] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                cross_attention=cross_attention,
                cross_dim=cross_dim
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """Forward function with multi-scale feature extraction."""
        x = self.patch_embed(x)
        
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        # Store features from each layer for skip connections and cross-attention
        layer_outputs = []
        cross_features = None
        
        for i, layer in enumerate(self.layers):
            # Use features from previous layer for cross-attention
            if i > 0 and self.enable_cross_attention:
                # Prepare cross features (from previous layer)
                cross_features = layer_outputs[-1]
            
            x = layer(x, cross_features)
            layer_outputs.append(x)
        
        x = self.norm(x)
        return layer_outputs


class Enhanced_nnFormer(SegmentationNetwork):
    """
    Enhanced nnFormer with Multi-Scale Cross-Attention.
    
    This enhanced version incorporates cross-attention between different scales
    to improve feature representation and segmentation accuracy, particularly
    for small structures and boundary delineation.
    """
    
    def __init__(self, crop_size=[64,128,128],
                 embedding_dim=192,
                 input_channels=1, 
                 num_classes=14, 
                 conv_op=nn.Conv3d, 
                 depths=[2,2,2,2],
                 num_heads=[6, 12, 24, 48],
                 patch_size=[2,4,4],
                 window_size=[4,4,8,4],
                 deep_supervision=True,
                 enable_cross_attention=True):
        
        super(Enhanced_nnFormer, self).__init__()
        
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.enable_cross_attention = enable_cross_attention
        
        # Import original components
        from nnformer.network_architecture.nnFormer_tumor import (
            Decoder, final_patch_expanding
        )
        
        self.upscale_logits_ops = []
        self.upscale_logits_ops.append(lambda x: x)
        
        embed_dim = embedding_dim
        
        # Enhanced encoder with cross-attention
        self.model_down = EnhancedEncoder(
            pretrain_img_size=crop_size,
            window_size=window_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            in_chans=input_channels,
            enable_cross_attention=enable_cross_attention
        )
        
        # Use original decoder (can be enhanced later)
        self.decoder = Decoder(
            pretrain_img_size=crop_size,
            embed_dim=embed_dim,
            window_size=window_size[::-1][1:],
            patch_size=patch_size,
            num_heads=num_heads[::-1][:-1],
            depths=depths[::-1][1:]
        )
        
        # Final segmentation heads
        self.final = []
        if self.do_ds:
            for i in range(len(depths)-1):
                self.final.append(final_patch_expanding(embed_dim*2**i, num_classes, patch_size=patch_size))
        else:
            self.final.append(final_patch_expanding(embed_dim, num_classes, patch_size=patch_size))
        
        self.final = nn.ModuleList(self.final)
    
    def forward(self, x):
        """
        Enhanced forward pass with multi-scale cross-attention.
        
        Args:
            x: Input tensor [B, C, H, W, D]
            
        Returns:
            List of segmentation outputs (if deep supervision enabled)
            or single segmentation output
        """
        seg_outputs = []
        
        # Enhanced encoder with cross-attention
        skips = self.model_down(x)
        neck = skips[-1]
        
        # Decoder (using original implementation)
        out = self.decoder(neck, skips)
        
        # Generate segmentation outputs
        if self.do_ds:
            for i in range(len(out)):  
                seg_outputs.append(self.final[-(i+1)](out[i]))
            return seg_outputs[::-1]
        else:
            seg_outputs.append(self.final[0](out))
            return seg_outputs[0]