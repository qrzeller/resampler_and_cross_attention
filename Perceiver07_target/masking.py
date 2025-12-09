# -*- coding: utf-8 -*-
"""Masking strategies for patch-based masked autoencoding."""

from __future__ import annotations

import torch


def random_patch_mask(
    batch_size: int,
    num_patches: int,
    num_channels: int,
    mask_ratio: float = 0.5,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Random independent patch masking.
    
    Args:
        batch_size: Batch size
        num_patches: Number of patches per channel
        num_channels: Number of channels
        mask_ratio: Fraction of patches to mask (0.0 to 1.0)
        device: Device to create mask on
    
    Returns:
        mask: (batch, num_patches, num_channels) binary mask (1=masked, 0=visible)
    """
    mask = torch.rand(batch_size, num_patches, num_channels, device=device) < mask_ratio
    return mask.float()


def span_mask(
    batch_size: int,
    num_patches: int,
    num_channels: int,
    mask_ratio: float = 0.5,
    min_span_len: int = 2,
    max_span_len: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Span masking: mask contiguous temporal spans per channel.
    
    This respects temporal structure better than random masking.
    Physiological signals have autocorrelation - masking spans forces
    the model to do longer-range inference.
    
    Args:
        batch_size: Batch size
        num_patches: Number of patches per channel
        num_channels: Number of channels
        mask_ratio: Target fraction of patches to mask
        min_span_len: Minimum span length in patches
        max_span_len: Maximum span length in patches
        device: Device to create mask on
    
    Returns:
        mask: (batch, num_patches, num_channels) binary mask (1=masked, 0=visible)
    """
    mask = torch.zeros(batch_size, num_patches, num_channels, device=device)
    
    for b in range(batch_size):
        for c in range(num_channels):
            # Calculate how many patches to mask for this channel
            target_masked = int(num_patches * mask_ratio)
            masked_count = 0
            
            while masked_count < target_masked:
                # Random span length
                span_len = torch.randint(min_span_len, max_span_len + 1, (1,)).item()
                # Random start position
                start = torch.randint(0, max(1, num_patches - span_len + 1), (1,)).item()
                
                # Mask the span (set to 1)
                end = min(start + span_len, num_patches)
                mask[b, start:end, c] = 1
                masked_count += (end - start)
                
                # Safety: avoid infinite loop
                if masked_count >= num_patches:
                    break
    
    return mask


def channel_drop_mask(
    batch_size: int,
    num_patches: int,
    num_channels: int,
    drop_prob: float = 0.3,
    device: torch.device = None,
    physio_channels: int = 6,
) -> torch.Tensor:
    """
    Channel dropout for arousal channels only (last 3 channels).
    
    Keeps physio channels (first 6) always visible, randomly drops arousal channels.
    This allows predicting arousal from physio signals.
    
    Args:
        batch_size: Batch size
        num_patches: Number of patches per channel
        num_channels: Total number of channels (e.g., 9 = 6 physio + 3 arousal)
        drop_prob: Probability of dropping each arousal channel
        device: Device to create mask on
        physio_channels: Number of physio channels to keep visible (default: 6)
    
    Returns:
        mask: (batch, num_patches, num_channels) binary mask (1=masked/dropped, 0=visible)
    """
    mask = torch.zeros(batch_size, num_patches, num_channels, device=device)
    
    # Only drop arousal channels (channels beyond physio_channels)
    if num_channels > physio_channels:
        arousal_start = physio_channels
        arousal_count = num_channels - physio_channels
        
        # Sample which arousal channels to drop: (batch, 1, arousal_count)
        arousal_drop = torch.rand(batch_size, 1, arousal_count, device=device) < drop_prob
        
        # Broadcast to all patches and place in arousal channel positions
        mask[:, :, arousal_start:] = arousal_drop.expand(-1, num_patches, -1).float()
    
    return mask


def mixed_mask(
    batch_size: int,
    num_patches: int,
    num_channels: int,
    mask_ratio: float = 0.5,
    channel_drop_prob: float = 0.2,
    span_prob: float = 0.7,
    min_span_len: int = 2,
    max_span_len: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Mixed masking strategy combining span masks and channel drops.
    
    With probability span_prob, use span masking on non-dropped channels.
    This gives the best of both: temporal structure + cross-modal learning.
    
    Args:
        batch_size: Batch size
        num_patches: Number of patches per channel
        num_channels: Number of channels
        mask_ratio: Target fraction for span masking
        channel_drop_prob: Probability of dropping entire channels
        span_prob: Probability of using span vs random masking
        min_span_len: Min span length
        max_span_len: Max span length
        device: Device to create mask on
    
    Returns:
        mask: (batch, num_patches, num_channels) binary mask (1=masked, 0=visible)
    """
    # Start with channel dropout (arousal only)
    physio_channels = kwargs.get('physio_channels', 6)
    mask = channel_drop_mask(batch_size, num_patches, num_channels, channel_drop_prob, device, physio_channels)
    
    # For each sample, decide span vs random
    for b in range(batch_size):
        if torch.rand(1).item() < span_prob:
            # Span masking on kept (visible) channels
            for c in range(num_channels):
                if mask[b, 0, c] == 0:  # Channel not dropped (visible)
                    # Apply span mask to this channel
                    span_m = span_mask(1, num_patches, 1, mask_ratio, min_span_len, max_span_len, device)
                    # Combine: channel is masked if either dropped OR span-masked
                    mask[b, :, c] = torch.maximum(mask[b, :, c], span_m[0, :, 0])
        else:
            # Random patch masking on kept channels
            random_m = random_patch_mask(1, num_patches, num_channels, mask_ratio, device)
            # Combine: channel is masked if either dropped OR random-masked
            mask[b] = torch.maximum(mask[b], random_m[0])
    
    return mask


def get_mask(
    batch_size: int,
    num_patches: int,
    num_channels: int,
    mask_type: str = 'mixed',
    mask_ratio: float = 0.5,
    device: torch.device = None,
    **kwargs,
) -> torch.Tensor:
    """
    Get mask using specified strategy.
    
    Args:
        batch_size: Batch size
        num_patches: Number of patches per channel
        num_channels: Number of channels
        mask_type: 'random', 'span', 'channel_drop', or 'mixed'
        mask_ratio: Masking ratio
        device: Device
        **kwargs: Additional args for specific strategies
    
    Returns:
        mask: (batch, num_patches, num_channels) binary mask (1=masked, 0=visible)
    """
    if mask_type == 'random':
        return random_patch_mask(batch_size, num_patches, num_channels, mask_ratio, device)
    elif mask_type == 'span':
        return span_mask(batch_size, num_patches, num_channels, mask_ratio, 
                        kwargs.get('span_len_min', 2), kwargs.get('span_len_max', 8), device)
    elif mask_type == 'channel_drop':
        return channel_drop_mask(batch_size, num_patches, num_channels, 
                               kwargs.get('channel_drop_prob', mask_ratio), device,
                               kwargs.get('physio_channels', 6))
    elif mask_type == 'mixed':
        return mixed_mask(batch_size, num_patches, num_channels, mask_ratio,
                         kwargs.get('channel_drop_prob', 0.2), kwargs.get('span_prob', 0.7),
                         kwargs.get('span_len_min', 2), kwargs.get('span_len_max', 8), device, **kwargs)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
