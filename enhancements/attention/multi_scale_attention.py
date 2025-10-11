"""
Multi-Scale Cross-Attention Enhancement for nnFormer
====================================================

This module implements multi-scale cross-attention mechanisms to improve
feature interaction between different resolution levels in the nnFormer architecture.

Key Innovations:
1. Cross-scale attention for capturing long-range dependencies
2. Adaptive feature fusion across different scales
3. Enhanced boundary detection through multi-resolution feature interaction

Author: 210353V
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from timm.models.layers import DropPath, trunc_normal_


class MultiScaleCrossAttention(nn.Module):
    """
    Multi-Scale Cross-Attention mechanism that enables feature interaction
    between different encoder scales for improved segmentation accuracy.
    
    Args:
        dim_q (int): Query feature dimension
        dim_kv (int): Key-Value feature dimension  
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to add bias to qkv projection
        attn_drop (float): Attention dropout rate
        proj_drop (float): Projection dropout rate
        scale_factor (int): Scale factor between query and key-value features
    """
    
    def __init__(self, dim_q, dim_kv, num_heads=8, qkv_bias=True, 
                 attn_drop=0., proj_drop=0., scale_factor=2):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        
        # Ensure dimensions are divisible by num_heads
        assert dim_q % num_heads == 0, f"dim_q {dim_q} must be divisible by num_heads {num_heads}"
        
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)  # Project to query dimension
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=qkv_bias)  # Project to query dimension
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Layer normalization for stability
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, query_features, key_value_features):
        """
        Forward pass of multi-scale cross-attention.
        
        Args:
            query_features: High-resolution features [B, H*W*D, dim_q]
            key_value_features: Lower-resolution features [B, (H/scale)*(W/scale)*(D/scale), dim_kv]
            
        Returns:
            Enhanced query features [B, H*W*D, dim_q]
        """
        B_q, N_q, C_q = query_features.shape
        B_kv, N_kv, C_kv = key_value_features.shape
        
        # Apply normalization
        q_norm = self.norm_q(query_features)
        kv_norm = self.norm_kv(key_value_features)
        
        # Generate Q, K, V
        q = self.q_proj(q_norm).reshape(B_q, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_norm).reshape(B_kv, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_norm).reshape(B_kv, N_kv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Apply softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C_q)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Residual connection
        return query_features + x


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion module that combines multi-scale features
    with learned attention weights.
    """
    
    def __init__(self, dims, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.dims = dims
        
        # Attention weights for each scale
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(dim, dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv3d(dim // 4, 1, 1),
                nn.Sigmoid()
            ) for dim in dims
        ])
        
        # Feature projection layers
        self.projections = nn.ModuleList([
            nn.Conv3d(dim, dims[0], 1) for dim in dims
        ])
    
    def forward(self, features_list):
        """
        Args:
            features_list: List of features from different scales
                          [(B, C1, H1, W1, D1), (B, C2, H2, W2, D2), ...]
        Returns:
            Fused features at the highest resolution
        """
        if len(features_list) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} feature maps, got {len(features_list)}")
        
        target_shape = features_list[0].shape[2:]  # Target spatial dimensions
        weighted_features = []
        
        for i, features in enumerate(features_list):
            # Compute attention weights
            weights = self.attention_weights[i](features)
            
            # Apply weights and project to target dimension
            weighted_feat = features * weights
            projected_feat = self.projections[i](weighted_feat)
            
            # Upsample to target resolution if needed
            if projected_feat.shape[2:] != target_shape:
                projected_feat = F.interpolate(
                    projected_feat, size=target_shape, 
                    mode='trilinear', align_corners=False
                )
            
            weighted_features.append(projected_feat)
        
        # Sum all weighted features
        fused_features = sum(weighted_features)
        return fused_features


class EnhancedSwinTransformerBlock(nn.Module):
    """
    Enhanced Swin Transformer Block with multi-scale cross-attention capability.
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
        
        # Import original attention classes (assuming they're available)
        from nnformer.network_architecture.nnFormer_tumor import WindowAttention, Mlp
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size, window_size), 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # Cross-attention if enabled
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
    
    def forward(self, x, mask_matrix=None, cross_features=None):
        """
        Forward pass with optional cross-attention.
        
        Args:
            x: Input features
            mask_matrix: Attention mask
            cross_features: Features from other scales for cross-attention
        """
        shortcut = x
        x = self.norm1(x)
        
        # Self-attention
        x = self.attn(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        
        # Cross-attention if available
        if self.cross_attn is not None and cross_features is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x_cross, cross_features)
            x = shortcut + self.drop_path(x_cross)
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.drop_path(x)
        
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features at multiple scales for cross-attention computation.
    """
    
    def __init__(self, input_dim, scales=[1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(None if scale == 1 else (
                    lambda s: tuple(max(1, dim // s) for dim in [64, 128, 128])
                )(scale)),
                nn.Conv3d(input_dim, input_dim, 1),
                nn.BatchNorm3d(input_dim),
                nn.ReLU(inplace=True)
            ) for scale in scales
        ])
    
    def forward(self, x):
        """Extract features at multiple scales."""
        features = []
        for extractor in self.extractors:
            features.append(extractor(x))
        return features


def create_cross_attention_mask(query_shape, key_shape, device):
    """
    Create attention mask for cross-scale attention.
    
    Args:
        query_shape: Shape of query features (H, W, D)
        key_shape: Shape of key features (H', W', D')
        device: Device to create mask on
    
    Returns:
        Attention mask tensor
    """
    # Simple implementation - can be made more sophisticated
    return None  # No masking for now, but can be extended


# Utility functions for integration
def interpolate_features(features, target_shape, mode='trilinear'):
    """
    Interpolate features to target shape.
    """
    if features.shape[2:] == target_shape:
        return features
    return F.interpolate(features, size=target_shape, mode=mode, align_corners=False)


def compute_feature_similarity(feat1, feat2):
    """
    Compute similarity between features from different scales.
    Useful for adaptive fusion weights.
    """
    # Normalize features
    feat1_norm = F.normalize(feat1, dim=1)
    feat2_norm = F.normalize(feat2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(feat1_norm * feat2_norm, dim=1, keepdim=True)
    return similarity