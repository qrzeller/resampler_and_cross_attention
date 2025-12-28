# -*- coding: utf-8 -*-
"""Minimal dataset classes for physio time-series."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch.utils.data import Dataset


class PhysioTimeSeriesDataset(Dataset):
    """
    Base dataset for training on synthetic or loaded physio time-series.

    Subclasses should implement _load_signal() to load actual data.
    For demonstration, this supports dummy data generation.
    """

    def __init__(
        self,
        window_size_sec: float = 12.0,
        hop_size_sec: float = 6.0,
        target_fs: float = 50.0,
        num_signals: int = 5,
        num_windows: int = 100,
        seed: int = 0,
        use_dummy_data: bool = True,
        # Masking options (dynamic masking generated per __getitem__)
        mask_frac: float = 0.0,
        apply_mask: bool = False,
    ) -> None:
        """
        Args:
            window_size_sec: Window duration in seconds
            hop_size_sec: Hop size between windows in seconds
            target_fs: Target sampling rate in Hz
            num_signals: Number of signal channels
            num_windows: Number of windows to generate/load
            seed: Random seed
            use_dummy_data: If True, generate dummy data. Otherwise load via _load_signal.
        """
        super().__init__()
        self.window_size_sec = float(window_size_sec)
        self.hop_size_sec = float(hop_size_sec)
        self.target_fs = float(target_fs)
        self.num_signals = int(num_signals)
        self.window_samples = int(round(self.window_size_sec * self.target_fs))
        self.num_windows = int(num_windows)
        self.seed = int(seed)
        self.use_dummy_data = bool(use_dummy_data)
        # Masking configuration
        self.mask_frac = float(mask_frac)
        self.apply_mask = bool(apply_mask)

        # Generate or load windows
        if self.use_dummy_data:
            self._generate_dummy_data()
        else:
            self.windows = []  # Subclasses should populate this

    def _generate_dummy_data(self) -> None:
        """Generate synthetic physiological signals for testing."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.windows = []

        for _ in range(self.num_windows):
            # Generate multi-frequency sine waves with noise
            t = np.arange(self.window_samples) / self.target_fs
            window = np.zeros((self.window_samples, self.num_signals), dtype=np.float32)

            for ch in range(self.num_signals):
                # Different base frequencies for each channel
                base_freq = 0.5 + ch * 0.3
                window[:, ch] = (
                    np.sin(2 * np.pi * base_freq * t)
                    + 0.5 * np.sin(2 * np.pi * base_freq * 3 * t)
                    + 0.1 * np.random.randn(self.window_samples)
                )

            self.windows.append(torch.from_numpy(window).float())

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"physio": self.windows[idx]}
        if self.apply_mask and self.mask_frac > 0.0:
            mask = self._create_channel_chunk_mask(self.window_samples, self.num_signals, self.mask_frac)
            item["mask"] = mask
        return item

    def _create_channel_chunk_mask(self, seq_len: int, channels: int, mask_frac: float) -> torch.Tensor:
        """Create a boolean mask (seq_len, channels) with a contiguous chunk masked per channel.

        Each channel receives a contiguous masked chunk whose length is approximately
        mask_frac * seq_len. Returns a boolean torch.Tensor with True indicating masked positions.
        """
        mask = np.zeros((seq_len, channels), dtype=bool)
        for ch in range(channels):
            # per-channel chunk length (at least 1)
            chunk_len = max(1, int(round(seq_len * float(mask_frac))))
            if chunk_len >= seq_len:
                start = 0
            else:
                start = np.random.randint(0, seq_len - chunk_len + 1)
            mask[start : start + chunk_len, ch] = True

        return torch.from_numpy(mask)


class PhysioResampledDataset(PhysioTimeSeriesDataset):
    """
    Dataset with high-rate physio that gets resampled/pooled to a target rate.

    Useful when you have high-resolution physiological data (e.g., 512 Hz)
    that needs to be downsampled to match other modalities (e.g., 50 Hz audio features).
    """

    def __init__(
        self,
        window_size_sec: float = 12.0,
        hop_size_sec: float = 6.0,
        target_fs: float = 50.0,
        physio_fs: float = 512.0,
        num_signals: int = 5,
        num_windows: int = 100,
        smoothing_kernel: int = 0,
        downsample_strategy: str = "polyphase",
        seed: int = 0,
        use_dummy_data: bool = True,
        mask_frac: float = 0.0,
        apply_mask: bool = False,
    ) -> None:
        """
        Args:
            window_size_sec: Window duration in seconds
            hop_size_sec: Hop size in seconds
            target_fs: Target resampling rate (Hz)
            physio_fs: Native physio sampling rate (Hz)
            num_signals: Number of signal channels
            num_windows: Number of windows to generate/load
            smoothing_kernel: Kernel size for post-pooling smoothing (0=disabled)
            downsample_strategy: "polyphase" or "avg" pooling
            seed: Random seed
            use_dummy_data: If True, generate dummy high-rate data
        """
        # Set attributes needed by parent initialization
        self.window_size_sec = float(window_size_sec)
        self.physio_fs = float(physio_fs)
        self.smoothing_kernel = max(0, int(smoothing_kernel))
        self.downsample_strategy = downsample_strategy
        self.num_signals = int(num_signals)

        # Store polyphase resampling factors EARLY
        frac = Fraction(int(round(target_fs)), int(round(physio_fs)))
        self._poly_up = frac.numerator
        self._poly_down = frac.denominator

        # Generate high-res data first
        self._physio_windows_highres = []
        if use_dummy_data:
            self._generate_dummy_highres_data(num_windows, seed)

        # Initialize parent (which would generate low-res data)
        super().__init__(
            window_size_sec=window_size_sec,
            hop_size_sec=hop_size_sec,
            target_fs=target_fs,
            num_signals=num_signals,
            num_windows=num_windows,
            seed=seed,
            use_dummy_data=False,  # We'll pool manually
            mask_frac=mask_frac,
            apply_mask=apply_mask,
        )

        # Pool high-res data to target rate
        self._pool_windows()

    def _generate_dummy_highres_data(self, num_windows: int, seed: int) -> None:
        """Generate synthetic high-rate physio data."""
        np.random.seed(seed)
        highres_samples = int(round(self.window_size_sec * self.physio_fs))

        for _ in range(num_windows):
            t = np.arange(highres_samples) / self.physio_fs
            window = np.zeros((highres_samples, self.num_signals), dtype=np.float32)

            for ch in range(self.num_signals):
                base_freq = 0.5 + ch * 0.3
                window[:, ch] = (
                    np.sin(2 * np.pi * base_freq * t)
                    + 0.5 * np.sin(2 * np.pi * base_freq * 3 * t)
                    + 0.05 * np.random.randn(highres_samples)
                )

            self._physio_windows_highres.append(window)

    def _polyphase_downsample(self, physio: np.ndarray) -> np.ndarray:
        """Polyphase resampling via scipy.signal."""
        return signal.resample_poly(
            physio,
            self._poly_up,
            self._poly_down,
            axis=0,
            padtype="mean",
        )

    def _avg_downsample(self, physio: np.ndarray) -> np.ndarray:
        """Average pooling downsampling."""
        factor = max(1, int(round(self.physio_fs / self.target_fs)))
        x = torch.from_numpy(physio).transpose(0, 1).unsqueeze(0)
        pooled = F.avg_pool1d(
            x,
            kernel_size=factor,
            stride=factor,
            ceil_mode=False,
        )
        return pooled.squeeze(0).transpose(0, 1).numpy()

    def _smooth(self, physio: np.ndarray) -> np.ndarray:
        """Apply smoothing kernel if enabled."""
        if self.smoothing_kernel <= 1:
            return physio

        # Convert to torch, apply conv1d, convert back
        x = torch.from_numpy(physio).transpose(0, 1).unsqueeze(0)
        kernel = (
            torch.ones(physio.shape[1], 1, self.smoothing_kernel)
            / float(self.smoothing_kernel)
        )
        smoothed = F.conv1d(
            x,
            kernel,
            padding=self.smoothing_kernel // 2,
            groups=physio.shape[1],
        )
        return smoothed.squeeze(0).transpose(0, 1).numpy()

    def _pool_windows(self) -> None:
        """Pool high-res windows to target rate."""
        for highres_window in self._physio_windows_highres:
            # Downsample
            if self.downsample_strategy == "polyphase":
                pooled = self._polyphase_downsample(highres_window)
            else:
                pooled = self._avg_downsample(highres_window)

            # Smooth
            pooled = self._smooth(pooled)

            # Ensure correct length (interpolate if needed)
            if pooled.shape[0] != self.window_samples:
                pooled_torch = torch.from_numpy(pooled).transpose(0, 1).unsqueeze(0)
                pooled_torch = F.interpolate(
                    pooled_torch,
                    size=self.window_samples,
                    mode="linear",
                    align_corners=False,
                )
                pooled = pooled_torch.squeeze(0).transpose(0, 1).numpy()

            self.windows.append(torch.from_numpy(pooled).float())


class SyntheticPhysioDataset(PhysioTimeSeriesDataset):
    """Convenience class for synthetic multi-signal physio data."""

    SIGNAL_NAMES = ["ECG", "GSR", "Temp", "Plet", "Resp"]

    def __init__(
        self,
        window_size_sec: float = 12.0,
        target_fs: float = 50.0,
        num_windows: int = 100,
        seed: int = 0,
    ) -> None:
        """
        Args:
            window_size_sec: Window duration in seconds
            target_fs: Sampling rate in Hz
            num_windows: Number of synthetic windows
            seed: Random seed
        """
        super().__init__(
            window_size_sec=window_size_sec,
            hop_size_sec=window_size_sec,  # No overlap for simplicity
            target_fs=target_fs,
            num_signals=len(self.SIGNAL_NAMES),
            num_windows=num_windows,
            seed=seed,
            use_dummy_data=True,
        )

    def get_signal_names(self):
        """Return list of signal channel names."""
        return self.SIGNAL_NAMES
