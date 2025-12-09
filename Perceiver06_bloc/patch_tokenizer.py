# -*- coding: utf-8 -*-
"""Patch-based tokenization for physiological signals.

Converts time-series into patches/blocks that serve as tokens for the Perceiver.
Each patch is processed through a small frontend (conv stack or linear) and
combined with time, modality, and channel embeddings.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """
    Tokenize physiological signals using patch-based representation.
    
    Instead of per-sample tokens, we chunk waveforms into patches (e.g., 16-256 samples).
    Each patch becomes a token via:
    - Projection: Linear(patch_len) → d_model or Conv1d stack
    - Time embedding: Position/time of patch (using Fourier features)
    - Modality embedding: Which modality (ECG, PPG, EDA, etc.)
    - Channel embedding: Which channel/lead within modality
    
    Final token = proj(patch) + emb_time + emb_modality + emb_channel
    
    Args:
        patch_len: Number of samples per patch (e.g., 16, 32, 64, 128, 256)
        num_channels: Total number of channels across all modalities
        d_model: Output dimension for tokens
        num_modalities: Number of modality types (e.g., 5 for ECG/GSR/BVP/TEMP/ACC)
        max_channels_per_modality: Max channels per modality (for channel embedding)
        use_conv_frontend: If True, use Conv1d stack; if False, use simple Linear
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        patch_len: int,
        num_channels: int,
        d_model: int,
        num_modalities: int = 5,
        max_channels_per_modality: int = 1,
        use_conv_frontend: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.num_channels = num_channels
        self.d_model = d_model
        self.use_conv_frontend = use_conv_frontend
        
        # Patch projection frontend
        if use_conv_frontend:
            # Conv stack: handles local morphology, phase-invariant
            self.patch_proj = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),  # Pool to single value per channel
                nn.Flatten(),
                nn.Linear(64, d_model),
                nn.Dropout(dropout),
            )
        else:
            # Simple linear projection
            self.patch_proj = nn.Sequential(
                nn.Linear(patch_len, d_model),
                nn.Dropout(dropout),
            )
        
        # Learnable embeddings
        self.modality_embedding = nn.Embedding(num_modalities, d_model)
        self.channel_embedding = nn.Embedding(max_channels_per_modality, d_model)
        
        # Layer norm for stabilization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        patches: torch.Tensor,
        time_embeddings: torch.Tensor,
        modality_ids: torch.Tensor,
        channel_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tokenize patches with embeddings.
        
        Args:
            patches: (batch, num_patches, patch_len) patch data
            time_embeddings: (batch, num_patches, d_model) time position encodings
            modality_ids: (batch, num_patches) modality indices [0, num_modalities)
            channel_ids: (batch, num_patches) channel indices within modality
        
        Returns:
            tokens: (batch, num_patches, d_model) tokenized representations
        """
        batch, num_patches, patch_len = patches.shape
        
        if self.use_conv_frontend:
            # Reshape for Conv1d: (batch * num_patches, 1, patch_len)
            patches_flat = patches.reshape(-1, 1, patch_len)
            patch_features = self.patch_proj(patches_flat)  # (batch * num_patches, d_model)
            patch_features = patch_features.reshape(batch, num_patches, self.d_model)
        else:
            # Linear projection: (batch, num_patches, patch_len) -> (batch, num_patches, d_model)
            patch_features = self.patch_proj(patches)
        
        # Add embeddings
        modality_emb = self.modality_embedding(modality_ids)  # (batch, num_patches, d_model)
        channel_emb = self.channel_embedding(channel_ids)    # (batch, num_patches, d_model)
        
        # Combine: token = proj(patch) + time_emb + modality_emb + channel_emb
        tokens = patch_features + time_embeddings + modality_emb + channel_emb
        tokens = self.norm(tokens)
        
        return tokens


def create_patches(
    signals: torch.Tensor,
    patch_len: int,
    stride: Optional[int] = None,
) -> torch.Tensor:
    """
    Create overlapping or non-overlapping patches from signals.
    
    Args:
        signals: (batch, seq_len, num_channels) input signals
        patch_len: Length of each patch
        stride: Stride between patches (default: same as patch_len for non-overlapping)
    
    Returns:
        patches: (batch, num_patches, num_channels, patch_len)
    """
    if stride is None:
        stride = patch_len
    
    batch, seq_len, num_channels = signals.shape
    
    # Calculate number of patches
    num_patches = (seq_len - patch_len) // stride + 1
    
    # Extract patches using unfold
    patches = signals.transpose(1, 2).unfold(2, patch_len, stride)
    # patches: (batch, num_channels, num_patches, patch_len)
    
    # Rearrange to (batch, num_patches, num_channels, patch_len)
    patches = patches.permute(0, 2, 1, 3)
    
    return patches


def flatten_patches(
    patches: torch.Tensor,
    modality_assignments: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten multi-channel patches into a sequence of single-channel tokens.
    
    Args:
        patches: (batch, num_patches, num_channels, patch_len) patches
        modality_assignments: (num_channels,) modality ID for each channel (optional)
    
    Returns:
        flat_patches: (batch, num_patches * num_channels, patch_len)
        modality_ids: (batch, num_patches * num_channels)
        channel_ids: (batch, num_patches * num_channels)
    """
    batch, num_patches, num_channels, patch_len = patches.shape
    
    # Flatten: (batch, num_patches, num_channels, patch_len) -> (batch, num_patches*num_channels, patch_len)
    flat_patches = patches.reshape(batch, num_patches * num_channels, patch_len)
    
    # Create modality IDs
    if modality_assignments is None:
        # Assume each channel is its own modality
        modality_assignments = torch.arange(num_channels, device=patches.device)
    
    # Expand to (batch, num_patches * num_channels)
    modality_ids = modality_assignments.unsqueeze(0).unsqueeze(0).expand(batch, num_patches, -1)
    modality_ids = modality_ids.reshape(batch, -1)
    
    # Create channel IDs (within modality) - simplified: assume 1 channel per modality
    channel_ids = torch.zeros_like(modality_ids)
    
    return flat_patches, modality_ids, channel_ids
