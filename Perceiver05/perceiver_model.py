# -*- coding: utf-8 -*-
"""Core Perceiver model architecture for time-series autoencoders."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from fourier_features import FourierFeatures


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
    """Cross-attention block: latents attend to input tokens."""

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim,
            dropout=dropout,
        )
        self.ff = FeedForward(latent_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: (batch, num_latents, latent_dim)
            tokens: (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Updated latents (batch, num_latents, latent_dim)
        """
        attn_out, _ = self.attn(latents, tokens, tokens, key_padding_mask=mask)
        latents = self.norm1(latents + attn_out)
        latents = self.norm2(latents + self.ff(latents))
        return latents


class DecoderCrossAttentionBlock(nn.Module):
    """Decoder cross-attention: queries attend to latent array."""

    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ff = FeedForward(latent_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch, seq_len, latent_dim)
            latents: (batch, num_latents, latent_dim)

        Returns:
            Updated queries (batch, seq_len, latent_dim)
        """
        attn_out, _ = self.attn(queries, latents, latents)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ff(queries))
        return queries


class PerceiverResampler(nn.Module):
    """
    Minimal Perceiver IO-style autoencoder for time-series.

    Architecture:
    1. Input signals -> project to latent_dim with Fourier features
    2. Cross-attention: latents attend to input tokens
    3. Self-attention: latents process among themselves
    4. Decoder cross-attention: reconstruction queries attend to latents
    5. Output projection to signal space (optionally residual)

    Args:
        signal_dim: Number of signal channels (e.g., 5 for ECG, GSR, etc.)
        seq_len: Sequence length (number of samples)
        sample_rate_hz: Sampling rate in Hz
        latent_dim: Dimension of latent space
        num_latents: Number of latent vectors
        num_self_attn_layers: Number of self-attention layers
        num_heads: Number of attention heads
        num_fourier_bands: Number of Fourier frequency bands
        max_freq_hz: Maximum frequency in Hz (default: Nyquist)
        min_freq_hz: Minimum frequency in Hz (default: ~1/window_duration)
        decoder_seq_len: Length of decoder output (default: same as seq_len)
        dropout: Dropout probability
        use_residual: Whether to use residual decoding (predict delta + input)
    """

    def __init__(
        self,
        signal_dim: int,
        seq_len: int,
        sample_rate_hz: float,
        latent_dim: int = 256,
        num_latents: int = 256,
        num_self_attn_layers: int = 8,
        num_heads: int = 8,
        num_fourier_bands: int = 32,
        max_freq_hz: Optional[float] = None,
        min_freq_hz: Optional[float] = None,
        decoder_seq_len: Optional[int] = None,
        dropout: float = 0.05,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.decoder_seq_len = int(decoder_seq_len or seq_len)
        self.sample_rate_hz = float(sample_rate_hz)
        self.use_residual = bool(use_residual)

        # Duration covered by indexed samples [0 .. seq_len-1] at sample_rate_hz
        self.window_duration_sec = (self.seq_len - 1) / max(self.sample_rate_hz, 1e-6)

        if max_freq_hz is None:
            max_freq_hz = self.sample_rate_hz / 2.0  # Nyquist
        if min_freq_hz is None:
            min_freq_hz = 1.0 / max(self.window_duration_sec, 1e-6)

        # Encoder and decoder Fourier feature encoders
        self.encoder_pos = FourierFeatures(
            num_fourier_bands, min_freq_hz, max_freq_hz, include_positions=True
        )
        self.decoder_pos = FourierFeatures(
            num_fourier_bands, min_freq_hz, max_freq_hz, include_positions=True
        )

        # Time positions in seconds
        t_enc = (
            torch.arange(self.seq_len, dtype=torch.float32)
            / max(self.sample_rate_hz, 1e-6)
        ).view(1, self.seq_len, 1)

        # If decoder len differs, spread queries across the same duration
        if self.decoder_seq_len == self.seq_len:
            t_dec = t_enc.clone()
        else:
            t_dec = torch.linspace(
                0.0,
                self.window_duration_sec,
                self.decoder_seq_len,
                dtype=torch.float32,
            ).view(1, self.decoder_seq_len, 1)

        self.register_buffer("encoder_positions", t_enc, persistent=False)
        self.register_buffer("decoder_positions", t_dec, persistent=False)

        # Encoder path
        token_dim = signal_dim + self.encoder_pos.output_dim
        self.input_proj = nn.Linear(token_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        
        # Learnable mask token for each signal channel (for modality dropout)
        # Shape: (1, 1, signal_dim) - broadcasts to (batch, seq_len, signal_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, signal_dim) * 0.02)

        self.encoder_cross = CrossAttentionBlock(latent_dim, latent_dim, num_heads, dropout)
        self.self_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=latent_dim * 4,
                    batch_first=True,
                    activation="gelu",
                    dropout=dropout,
                )
                for _ in range(num_self_attn_layers)
            ]
        )

        # Decoder path
        self.query_proj = nn.Linear(self.decoder_pos.output_dim, latent_dim)
        self.decoder_cross = DecoderCrossAttentionBlock(latent_dim, num_heads, dropout)
        self.output_proj = nn.Linear(latent_dim, signal_dim)

    def forward(
        self, 
        series: torch.Tensor, 
        return_latents: bool = False, 
        residual_base: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Perceiver.

        Args:
            series: (batch, seq_len, signal_dim) input signals
            return_latents: If True, return (outputs, latents) tuple
            residual_base: (batch, seq_len, signal_dim) optional base for residual connection.
                          If provided and use_residual=True, output will be delta + residual_base.
                          This allows using the original unmasked input as residual base during training
                          with modality dropout.
            channel_mask: (batch, 1, signal_dim) binary mask where 1=keep channel, 0=use mask token.
                         Used for modality dropout - dropped channels are replaced with learnable mask token.

        Returns:
            outputs: (batch, decoder_seq_len, signal_dim) reconstructions
            latents: (batch, num_latents, latent_dim) if return_latents=True
        """
        batch, seq_len, _ = series.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, received {seq_len}")

        # Apply mask tokens to dropped channels if channel_mask is provided
        if channel_mask is not None:
            # channel_mask: (batch, 1, signal_dim) where 1=keep, 0=replace with mask token
            # Broadcast mask token and replace dropped channels
            mask_tokens = self.mask_token.expand(batch, seq_len, -1)  # (batch, seq_len, signal_dim)
            series = series * channel_mask + mask_tokens * (1.0 - channel_mask)

        # Encoder: combine signals with Fourier positional features
        enc_pos = self.encoder_positions.to(series.device, dtype=series.dtype).expand(
            batch, -1, -1
        )
        enc_features = self.encoder_pos(enc_pos)
        tokens = torch.cat([series, enc_features], dim=-1)
        tokens = self.input_proj(tokens)

        # Encoder: cross-attention + self-attention on latents
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents = self.encoder_cross(latents, tokens)
        for layer in self.self_layers:
            latents = layer(latents)

        # Decoder: decode to output sequence
        dec_pos = self.decoder_positions.to(series.device, dtype=series.dtype).expand(
            batch, -1, -1
        )
        dec_features = self.decoder_pos(dec_pos)
        queries = self.query_proj(dec_features)
        decoded = self.decoder_cross(queries, latents)

        delta = self.output_proj(decoded)
        
        # Use residual connection if enabled
        if self.use_residual and self.decoder_seq_len == self.seq_len:
            # Use provided residual_base if available, otherwise use input series
            base = residual_base if residual_base is not None else series
            outputs = delta + base
        else:
            outputs = delta

        if return_latents:
            return outputs, latents
        return outputs
