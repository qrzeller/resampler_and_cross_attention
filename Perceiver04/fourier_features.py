# -*- coding: utf-8 -*-
"""Fourier features for continuous coordinates (time in seconds)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Proper Fourier features for continuous coordinates (time in seconds):
        [t, sin(2π f1 t), cos(2π f1 t), ..., sin(2π fk t), cos(2π fk t)]
    Frequencies f are in Hz.

    We use log-spaced frequencies between min_freq_hz and max_freq_hz.

    Args:
        num_bands: Number of frequency bands to use
        min_freq_hz: Minimum frequency in Hz
        max_freq_hz: Maximum frequency in Hz
        include_positions: Whether to include raw positions in output
        pos_dim: Dimensionality of position coordinates (typically 1 for time)
    """

    def __init__(
        self,
        num_bands: int,
        min_freq_hz: float,
        max_freq_hz: float,
        include_positions: bool = True,
        pos_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.min_freq_hz = float(min_freq_hz)
        self.max_freq_hz = float(max_freq_hz)
        self.include_positions = include_positions
        self.pos_dim = int(pos_dim)

    @property
    def output_dim(self) -> int:
        """Total output dimension of encoded features."""
        base = self.pos_dim if self.include_positions else 0
        return base + (self.pos_dim * self.num_bands * 2)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Encode positions with Fourier features.

        Args:
            positions: (..., pos_dim) where values are in seconds

        Returns:
            (..., output_dim) encoded features
        """
        if self.num_bands <= 0:
            return (
                positions
                if self.include_positions
                else positions.new_zeros(*positions.shape[:-1], 0)
            )

        # Keep frequency range sane and strictly positive for logspace
        f_lo = max(self.min_freq_hz, 1e-6)
        f_hi = max(self.max_freq_hz, f_lo)

        freqs = torch.logspace(
            math.log10(f_lo),
            math.log10(f_hi),
            self.num_bands,
            base=10.0,
            device=positions.device,
            dtype=positions.dtype,
        )
        shape = [1] * (positions.dim() - 1) + [self.num_bands]
        freqs = freqs.view(*shape)  # broadcast across batch/time

        angles = positions.unsqueeze(-1) * freqs * (2.0 * math.pi)  # 2π f t
        sin = angles.sin()
        cos = angles.cos()
        emb = torch.cat([sin, cos], dim=-1)
        emb = emb.reshape(*positions.shape[:-1], -1)

        if self.include_positions:
            emb = torch.cat([positions, emb], dim=-1)

        return emb
