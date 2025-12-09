# -*- coding: utf-8 -*-
"""Perceiver model with patch-based tokenization for physiological signals."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from fourier_features import FourierFeatures
from patch_tokenizer import PatchTokenizer, create_patches, flatten_patches


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block (Perceiver IO pre-norm style).
    X = Attn(LN(XQ), LN(XKV))
    X = X + XQ (query residual)
    X = X + MLP(LN(X))
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim,
            dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(latent_dim)
        self.ff = FeedForward(latent_dim, dropout=dropout)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm cross-attention with query residual
        q_normed = self.norm_q(latents)
        kv_normed = self.norm_kv(tokens)
        attn_out, _ = self.attn(q_normed, kv_normed, kv_normed, key_padding_mask=mask)
        latents = latents + attn_out
        
        # Pre-norm MLP with residual
        latents = latents + self.ff(self.norm_mlp(latents))
        return latents


class DecoderCrossAttentionBlock(nn.Module):
    """
    Decoder cross-attention (Perceiver IO pre-norm style).
    Optional query residual for flexibility.
    """

    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_query_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_query_residual = use_query_residual
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(latent_dim)
        self.ff = FeedForward(latent_dim, dropout=dropout)

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm cross-attention with optional query residual
        q_normed = self.norm_q(queries)
        kv_normed = self.norm_kv(latents)
        attn_out, _ = self.attn(q_normed, kv_normed, kv_normed)
        
        if self.use_query_residual:
            queries = queries + attn_out
        else:
            queries = attn_out
        
        # Pre-norm MLP with residual
        queries = queries + self.ff(self.norm_mlp(queries))
        return queries


class PatchPerceiverAutoencoder(nn.Module):
    """
    Perceiver autoencoder using patch-based tokenization for physiological signals.
    
    Architecture:
    1. Patchify signals into blocks
    2. Tokenize patches with embeddings (time, modality, channel)
    3. Encoder cross-attention: latents attend to patch tokens
    4. L × self-attention on latents
    5. Decoder cross-attention: output patches attend to latents
    6. Unpatchify to reconstruct signals
    
    Masked Modeling:
    - MAE-style (default): Remove masked patches from encoder input entirely.
      Only visible patches attend to latents. Decoder must reconstruct all patches.
      This is compute-efficient: O(N_latents × M_visible) instead of O(N_latents × M_total).
    
    - BERT-style (optional): Keep masked patches but replace with [MASK] token.
      Less efficient but simpler. Set mask_strategy='bert' to use.
    
    Masking strategies:
    - Span masking: Mask contiguous time spans (respects temporal structure)
    - Channel drop: Mask entire modalities (simulates sensor dropout)
    - Random patch masking: Independent per-patch masking
    
    Args:
        signal_dim: Number of signal channels
        seq_len: Total sequence length in samples
        sample_rate_hz: Sampling rate in Hz
        patch_len: Samples per patch (16-256)
        latent_dim: Latent dimension
        num_latents: Number of latent vectors
        num_self_attn_layers: Number of self-attention layers on latents
        num_heads: Number of attention heads
        num_modalities: Number of modality types
        max_channels_per_modality: Max channels per modality
        use_conv_frontend: Use Conv1d stack for patch projection
        num_fourier_bands: Number of Fourier bands for time encoding
        dropout: Dropout probability
        use_query_residual: Use query residual in decoder
        mask_strategy: 'mae' (remove masked tokens) or 'bert' (keep with [MASK] token)
    """
    
    def __init__(
        self,
        signal_dim: int,
        seq_len: int,
        sample_rate_hz: float,
        patch_len: int = 64,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_self_attn_layers: int = 8,
        num_heads: int = 8,
        num_modalities: int = 5,
        max_channels_per_modality: int = 1,
        use_conv_frontend: bool = False,
        num_fourier_bands: int = 32,
        dropout: float = 0.05,
        use_query_residual: bool = True,
        mask_strategy: str = 'mae',
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.seq_len = seq_len
        self.sample_rate_hz = sample_rate_hz
        self.patch_len = patch_len
        self.latent_dim = latent_dim
        self.mask_strategy = mask_strategy
        
        # Calculate number of patches
        self.num_patches = seq_len // patch_len
        assert seq_len % patch_len == 0, f"seq_len ({seq_len}) must be divisible by patch_len ({patch_len})"
        
        # Patch tokenizer
        self.tokenizer = PatchTokenizer(
            patch_len=patch_len,
            num_channels=signal_dim,
            d_model=latent_dim,
            num_modalities=num_modalities,
            max_channels_per_modality=max_channels_per_modality,
            use_conv_frontend=use_conv_frontend,
            dropout=dropout,
        )
        
        # [MASK] token for BERT-style masking (learnable)
        if mask_strategy == 'bert':
            self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
        
        # Time encoding with Fourier features
        # Fourier features encode time t as: [t, sin(2π f₁ t), cos(2π f₁ t), ..., sin(2π fₖ t), cos(2π fₖ t)]
        # where frequencies f₁...fₖ are log-spaced between min_freq_hz and max_freq_hz
        # This provides position-aware encoding that works well for continuous time
        window_duration = (seq_len - 1) / sample_rate_hz
        self.time_encoder = FourierFeatures(
            num_fourier_bands,
            min_freq_hz=1.0 / window_duration,  # Captures slow trends over window
            max_freq_hz=sample_rate_hz / 2.0,    # Nyquist frequency
            include_positions=True,               # Include raw time t
        )
        
        # Project Fourier features to latent_dim
        # Output dim = 1 (raw time) + 2 * num_fourier_bands (sin/cos pairs)
        self.time_proj = nn.Linear(self.time_encoder.output_dim, latent_dim)
        
        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        
        # Encoder cross-attention
        self.encoder_cross = CrossAttentionBlock(latent_dim, latent_dim, num_heads, dropout)
        
        # Self-attention layers on latents
        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                batch_first=True,
                activation="gelu",
                dropout=dropout,
            )
            for _ in range(num_self_attn_layers)
        ])
        
        # Decoder cross-attention
        self.decoder_cross = DecoderCrossAttentionBlock(
            latent_dim, num_heads, dropout, use_query_residual=use_query_residual
        )
        
        # Output projection: latent_dim -> patch_len (per channel)
        self.output_proj = nn.Linear(latent_dim, patch_len)
        
        # Modality assignments (assume one modality per channel for simplicity)
        # This maps each channel to a modality ID: [0, 1, 2, 3, 4] for 5 channels
        # In practice, you might want: ECG=0, GSR=1, BVP=2, TEMP=3, ACC=4
        # Or for multi-lead ECG: all ECG leads get same modality ID
        self.register_buffer(
            "modality_assignments",
            torch.arange(signal_dim),
            persistent=False
        )
    
    def forward(
        self,
        signals: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional patch masking.
        
        Args:
            signals: (batch, seq_len, signal_dim) input signals
            patch_mask: (batch, num_patches, signal_dim) binary mask where:
                       - 1 = visible patch (keep in encoder)
                       - 0 = masked patch (reconstruct from latents)
                       If None, no masking (standard autoencoding)
            return_latents: If True, return (output, latents) tuple
        
        Returns:
            output: (batch, seq_len, signal_dim) reconstructed signals
            latents: (batch, num_latents, latent_dim) if return_latents=True
        """
        batch = signals.shape[0]
        device = signals.device
        
        # Create patches: (batch, num_patches, num_channels, patch_len)
        patches = create_patches(signals, self.patch_len)
        
        # Flatten to tokens: (batch, num_patches*num_channels, patch_len)
        flat_patches, modality_ids, channel_ids = flatten_patches(
            patches, self.modality_assignments
        )
        
        num_tokens = flat_patches.shape[1]  # num_patches * num_channels
        
        # Time embeddings for patch center times
        # Each patch covers [t_start, t_start + patch_len] samples
        # We use the center time: t_center = t_start + patch_len/2
        patch_times = torch.arange(self.num_patches, device=device, dtype=torch.float32)
        patch_times = (patch_times * self.patch_len + self.patch_len / 2) / self.sample_rate_hz
        patch_times = patch_times.unsqueeze(0).unsqueeze(-1)  # (1, num_patches, 1)
        
        # Expand for all channels: (batch, num_patches*num_channels, 1)
        patch_times = patch_times.repeat(1, self.signal_dim, 1).reshape(batch, -1, 1)
        
        # Encode time with Fourier features
        # This converts scalar time t into rich representation: [t, sin(2π f₁ t), cos(2π f₁ t), ...]
        time_features = self.time_encoder(patch_times)  # (batch, num_tokens, fourier_dim)
        time_embeddings = self.time_proj(time_features)  # (batch, num_tokens, latent_dim)
        
        # Tokenize patches: proj(patch) + time_emb + modality_emb + channel_emb
        # This creates rich token representations that know:
        # - WHAT the patch data is (from projection)
        # - WHEN it occurs (time_emb via Fourier features)
        # - WHICH modality it's from (modality_emb: ECG vs GSR vs BVP...)
        # - WHICH channel/lead (channel_emb: for multi-lead signals)
        tokens = self.tokenizer(
            flat_patches, time_embeddings, modality_ids, channel_ids
        )  # (batch, num_tokens, latent_dim)
        
        # === MAE-style masking: remove masked tokens from encoder ===
        if patch_mask is not None and self.mask_strategy == 'mae':
            # Flatten mask: (batch, num_patches, signal_dim) -> (batch, num_tokens)
            flat_mask = patch_mask.reshape(batch, -1)  # 1=visible, 0=masked
            
            # Keep only visible tokens for encoder (saves compute!)
            # This is the key MAE insight: O(N_latents × M_visible) << O(N_latents × M_total)
            visible_tokens = []
            visible_indices = []
            for b in range(batch):
                vis_idx = flat_mask[b].nonzero(as_tuple=True)[0]
                visible_tokens.append(tokens[b, vis_idx])
                visible_indices.append(vis_idx)
            
            # Pad to same length for batching (use max visible count)
            max_visible = max(len(vt) for vt in visible_tokens)
            encoder_tokens = torch.zeros(batch, max_visible, self.latent_dim, device=device)
            attention_mask = torch.ones(batch, max_visible, dtype=torch.bool, device=device)
            
            for b in range(batch):
                n_vis = len(visible_tokens[b])
                encoder_tokens[b, :n_vis] = visible_tokens[b]
                attention_mask[b, :n_vis] = False  # False = attend, True = ignore
        
        # === BERT-style masking: replace masked tokens with [MASK] ===
        elif patch_mask is not None and self.mask_strategy == 'bert':
            flat_mask = patch_mask.reshape(batch, -1).unsqueeze(-1)  # (batch, num_tokens, 1)
            # Replace masked tokens with learnable [MASK] token
            mask_tokens = self.mask_token.expand(batch, num_tokens, -1)
            encoder_tokens = tokens * flat_mask + mask_tokens * (1 - flat_mask)
            attention_mask = None  # All tokens attend
        
        # === No masking: standard autoencoding ===
        else:
            encoder_tokens = tokens
            attention_mask = None
        
        # Encoder: cross-attention + self-attention
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents = self.encoder_cross(latents, encoder_tokens, mask=attention_mask)
        for layer in self.self_layers:
            latents = layer(latents)
        
        # Decoder: decode ALL patches (including masked ones)
        # Use original full token set for queries (with time/modality/channel info)
        # But DON'T include patch data - decoder must reconstruct from latents only
        decoder_queries = time_embeddings + self.tokenizer.modality_embedding(modality_ids) + \
                         self.tokenizer.channel_embedding(channel_ids)
        decoder_queries = self.tokenizer.norm(decoder_queries)
        
        decoded_tokens = self.decoder_cross(decoder_queries, latents)  # (batch, num_tokens, latent_dim)
        
        # Project to patch values
        decoded_patches = self.output_proj(decoded_tokens)  # (batch, num_tokens, patch_len)
        
        # Reshape back to (batch, num_patches, num_channels, patch_len)
        decoded_patches = decoded_patches.reshape(batch, self.num_patches, self.signal_dim, self.patch_len)
        
        # Unpatchify: (batch, num_patches, num_channels, patch_len) -> (batch, seq_len, num_channels)
        output = decoded_patches.permute(0, 2, 1, 3).reshape(batch, self.signal_dim, -1)
        output = output.transpose(1, 2)  # (batch, seq_len, num_channels)
        
        if return_latents:
            return output, latents
        return output
