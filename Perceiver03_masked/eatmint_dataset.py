# -*- coding: utf-8 -*-
"""EATMINT physio dataset loader for real physiological signals."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EATMINTConfig:
    """Configuration for EATMINT dataset paths and parameters."""

    # Base data path - UPDATE THIS to your data location
    data_root: str = "../data/EATMINT/researchdata"

    # Subpaths for each modality
    physio_path: str = "Physio"
    times_path: str = "StartEndTimes"

    # Sampling parameters
    physio_fs: int = 512  # Native physio sampling rate (Hz)

    # Dyads to exclude (D14 has no data)
    excluded_dyads: List[int] = field(default_factory=lambda: [14])

    @property
    def physio_dir(self) -> Path:
        return Path(self.data_root) / self.physio_path

    @property
    def times_dir(self) -> Path:
        return Path(self.data_root) / self.times_path


# =============================================================================
# Data Availability Checking
# =============================================================================


def check_physio_availability(config: EATMINTConfig) -> pd.DataFrame:
    """
    Check which dyads have physio data available.
    Returns a DataFrame with availability information.
    """
    all_dyads = [d for d in range(2, 32) if d not in config.excluded_dyads]

    availability = []

    for dyad in all_dyads:
        dyad_str = f"D{dyad:02d}"

        for participant in [1, 2]:
            part_str = f"P{participant:02d}"

            record = {
                "dyad": dyad,
                "participant": participant,
                "dyad_str": dyad_str,
                "part_str": part_str,
            }

            # Check physio (one file per dyad)
            physio_file = config.physio_dir / f"{dyad_str}.mat"
            record["physio"] = physio_file.exists()

            # Check times file (required for synchronization)
            times_file = config.times_dir / f"{dyad_str}_times.csv"
            record["times"] = times_file.exists()

            # Usable if both physio and times exist
            record["usable"] = record["physio"] and record["times"]

            availability.append(record)

    df = pd.DataFrame(availability)
    return df


def print_availability_summary(availability_df: pd.DataFrame) -> pd.DataFrame:
    """Print a summary of data availability and return usable rows."""
    print("=" * 50)
    print("EATMINT Physio Data Availability Summary")
    print("=" * 50)
    print(f"\nTotal participant slots: {len(availability_df)}")
    print(f"Physio available: {availability_df['physio'].sum()}")
    print(f"Times available: {availability_df['times'].sum()}")

    usable_df = availability_df[availability_df["usable"]]
    print(f"Usable (physio + times): {len(usable_df)}")

    return usable_df


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_times(config: EATMINTConfig, dyad: int) -> Dict[str, float]:
    """Load start/end times for a dyad. Times are in milliseconds."""
    times_file = config.times_dir / f"D{dyad:02d}_times.csv"
    df = pd.read_csv(times_file)
    return {
        "baseline_start": df["Baseline start"].iloc[0],
        "baseline_stop": df["Baseline stop"].iloc[0],
        "collab_start": df["Collaboration start"].iloc[0],
        "collab_stop": df["Collaboration stop"].iloc[0],
    }


def load_physio(
    config: EATMINTConfig, dyad: int, participant: int
) -> Dict[str, np.ndarray]:
    """
    Load physiological signals for a participant.
    Returns dict with keys: GSR, Temp, Plet, ECG, Resp and metadata.
    """
    physio_file = config.physio_dir / f"D{dyad:02d}.mat"
    mat = sio.loadmat(physio_file)

    fs = mat["fs"].flatten()[0]  # Sampling frequency (512 Hz)
    collab_data = mat["collabData"]
    labels = [l.strip() for l in mat["label"]]

    # Find channels for this participant
    prefix = f"{participant}-"
    signals = {}
    exg1_idx = None
    exg2_idx = None

    for i, label in enumerate(labels):
        if label.startswith(prefix):
            signal_name = label.split("-")[1]
            if signal_name in ["GSR1", "Temp", "Plet", "Resp"]:
                # Map GSR1 -> GSR
                key = "GSR" if signal_name == "GSR1" else signal_name
                signals[key] = collab_data[:, i]
            elif signal_name == "EXG1":
                exg1_idx = i
            elif signal_name == "EXG2":
                exg2_idx = i

    # Compute ECG from EXG1 - EXG2
    if exg1_idx is not None and exg2_idx is not None:
        signals["ECG"] = collab_data[:, exg1_idx] - collab_data[:, exg2_idx]

    signals["fs"] = fs
    signals["n_samples"] = collab_data.shape[0]

    return signals


# =============================================================================
# EATMINT Physio Dataset
# =============================================================================


class EATMINTPhysioDataset(Dataset):
    """
    PyTorch Dataset for EATMINT physiological data.

    Loads real physio signals (ECG, GSR, Resp, Temp, Plet) from .mat files,
    applies z-score normalization, and segments into windows.

    Args:
        config: EATMINTConfig with data paths
        availability_df: DataFrame with dyad/participant availability
        window_size_sec: Window duration in seconds
        hop_size_sec: Hop between windows in seconds
        target_fs: Target sampling frequency (Hz) for output
        physio_fs: Native physio sampling rate (Hz)
        smoothing_kernel: Kernel size for post-downsampling smoothing
        downsample_strategy: "polyphase" or "avg"
        preload: Whether to preload all data into memory
    """

    # Physio signals to use (in order)
    PHYSIO_SIGNALS = ["GSR", "ECG", "Resp", "Temp", "Plet"]

    def __init__(
        self,
        config: EATMINTConfig,
        availability_df: pd.DataFrame,
        window_size_sec: float = 12.0,
        hop_size_sec: float = 6.0,
        target_fs: float = 50.0,
        physio_fs: float = 512.0,
        smoothing_kernel: int = 0,
        downsample_strategy: str = "polyphase",
        ecg_band: Optional[tuple[float, float]] = (0.5, 40.0),
        detrend_non_ecg: bool = True,
        preload: bool = False,
        cache_windows: bool = False,
    ):
        super().__init__()
        self.config = config
        self.window_size_sec = float(window_size_sec)
        self.hop_size_sec = float(hop_size_sec)
        self.target_fs = float(target_fs)
        self.physio_fs = float(physio_fs)
        self.smoothing_kernel = max(0, int(smoothing_kernel))
        self.downsample_strategy = downsample_strategy
        self.ecg_band = ecg_band
        self.detrend_non_ecg = bool(detrend_non_ecg)

        # Calculate window samples at target rate
        self.window_samples = int(round(self.window_size_sec * self.target_fs))

        # Calculate high-rate window samples
        self.highres_window_samples = int(round(self.window_size_sec * self.physio_fs))

        # Polyphase resampling factors
        frac = Fraction(int(round(target_fs)), int(round(physio_fs)))
        self._poly_up = frac.numerator
        self._poly_down = frac.denominator

        # Filter to usable participants
        self.participants = availability_df[availability_df["usable"]].reset_index(
            drop=True
        )

        # Build index of all windows
        self.windows = self._build_window_index()

        # Cache for loaded participant-level data
        self.cache = {}
        if preload:
            self._preload_all()

        # Optionally precompute and cache all low-rate windows (in CPU RAM)
        # This speeds up epoch iterations but increases RAM usage.
        self.cache_windows = bool(cache_windows)
        self.windows_data: Optional[list] = None
        if self.cache_windows:
            self._precompute_all_windows()

    def _preload_all(self):
        """Preload all participant data into memory."""
        from tqdm.auto import tqdm

        print("Preloading participant data...")
        for _, row in tqdm(
            self.participants.iterrows(),
            total=len(self.participants),
            desc="Loading",
        ):
            try:
                self._load_participant_data(row["dyad"], row["participant"])
            except Exception as e:
                warnings.warn(f"Error loading D{row['dyad']}P{row['participant']}: {e}")
        print(f"Preloaded {len(self.cache)} participants")

    def _build_window_index(self) -> List[Dict]:
        """Build index of all training windows."""
        windows = []

        for _, row in self.participants.iterrows():
            try:
                times = load_times(self.config, row["dyad"])
                duration_sec = (times["collab_stop"] - times["collab_start"]) / 1000.0

                # Generate window start times
                n_windows = (
                    int((duration_sec - self.window_size_sec) / self.hop_size_sec) + 1
                )

                for w_idx in range(max(1, n_windows)):
                    windows.append(
                        {
                            "dyad": row["dyad"],
                            "participant": row["participant"],
                            "window_idx": w_idx,
                            "start_sec": w_idx * self.hop_size_sec,
                        }
                    )
            except Exception as e:
                warnings.warn(
                    f"Error building windows for D{row['dyad']}P{row['participant']}: {e}"
                )

        print(
            f"Built {len(windows)} training windows from {len(self.participants)} participants"
        )
        return windows

    def _load_participant_data(
        self, dyad: int, participant: int
    ) -> Dict[str, np.ndarray]:
        """Load and normalize physio data for a participant."""
        cache_key = f"D{dyad:02d}P{participant:02d}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        times = load_times(self.config, dyad)
        collab_duration = (times["collab_stop"] - times["collab_start"]) / 1000.0

        # Load physio signals
        physio = load_physio(self.config, dyad, participant)
        physio_signals = []

        for sig in self.PHYSIO_SIGNALS:
            if sig in physio:
                physio_signals.append(physio[sig])
            else:
                # Fill with zeros if signal missing
                physio_signals.append(np.zeros(physio["n_samples"]))

        if not physio_signals:
            raise ValueError(f"No physio signals found for D{dyad}P{participant}")

        physio_data = np.stack(physio_signals, axis=1).astype(np.float32)

        # Optional preprocessing: ECG band-pass + detrend other channels
        fs = float(physio["fs"])
        # ECG is channel index 1 in PHYSIO_SIGNALS
        ecg_idx = self.PHYSIO_SIGNALS.index("ECG") if "ECG" in self.PHYSIO_SIGNALS else None

        if ecg_idx is not None and self.ecg_band is not None:
            low, high = self.ecg_band
            # Clamp to valid range and design a Butterworth band-pass
            nyq = 0.5 * fs
            low = max(0.001, float(low) / nyq)
            high = min(0.999, float(high) / nyq)
            if low < high:
                b, a = signal.butter(4, [low, high], btype="bandpass")
                physio_data[:, ecg_idx] = signal.filtfilt(b, a, physio_data[:, ecg_idx])

        if self.detrend_non_ecg:
            for ch_idx, sig_name in enumerate(self.PHYSIO_SIGNALS):
                if sig_name == "ECG":
                    continue
                physio_data[:, ch_idx] = signal.detrend(physio_data[:, ch_idx], type="constant")

        # Z-score normalize each channel (after filtering/detrend)
        physio_mean = np.mean(physio_data, axis=0, keepdims=True)
        physio_std = np.std(physio_data, axis=0, keepdims=True)
        physio_std = np.where(physio_std < 1e-8, 1.0, physio_std)
        physio_data = (physio_data - physio_mean) / physio_std

        # Create timestamps (physio is synced to collab start/stop)
        physio_ts = np.linspace(0, collab_duration, len(physio_data))

        data = {
            "physio": physio_data,
            "timestamps": physio_ts,
            "fs": physio["fs"],
        }

        self.cache[cache_key] = data
        return data

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

        x = torch.from_numpy(physio).float().transpose(0, 1).unsqueeze(0)
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

    def _extract_window(
        self, data: Dict[str, np.ndarray], start_sec: float
    ) -> np.ndarray:
        """
        Extract and downsample a window of physio data.

        Args:
            data: Participant data dict with 'physio', 'timestamps', 'fs'
            start_sec: Window start time in seconds

        Returns:
            (window_samples, num_signals) numpy array
        """
        physio = data["physio"]
        timestamps = data["timestamps"]
        end_sec = start_sec + self.window_size_sec

        # Find samples within window
        mask = (timestamps >= start_sec) & (timestamps < end_sec)

        if mask.sum() < 2:
            return np.zeros((self.window_samples, len(self.PHYSIO_SIGNALS)))

        window_data = physio[mask]

        # Ensure correct high-res length before downsampling
        if len(window_data) != self.highres_window_samples:
            # Interpolate to exact length
            target_ts = np.linspace(0, 1, self.highres_window_samples)
            source_ts = np.linspace(0, 1, len(window_data))
            resampled = np.zeros(
                (self.highres_window_samples, window_data.shape[1]), dtype=np.float32
            )
            for ch in range(window_data.shape[1]):
                resampled[:, ch] = np.interp(target_ts, source_ts, window_data[:, ch])
            window_data = resampled

        # Downsample to target rate
        if self.downsample_strategy == "polyphase":
            pooled = self._polyphase_downsample(window_data)
        else:
            pooled = self._avg_downsample(window_data)

        # Apply smoothing
        pooled = self._smooth(pooled)

        # Ensure correct final length
        if len(pooled) != self.window_samples:
            x = torch.from_numpy(pooled).float().transpose(0, 1).unsqueeze(0)
            x = F.interpolate(x, size=self.window_samples, mode="linear", align_corners=False)
            pooled = x.squeeze(0).transpose(0, 1).numpy()

        return pooled.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_windows and self.windows_data is not None:
            physio = self.windows_data[idx]
            return {"physio": physio, "mask": torch.tensor(True, dtype=torch.bool)}

        window_info = self.windows[idx]
        data = self._load_participant_data(
            window_info["dyad"], window_info["participant"]
        )

        physio = self._extract_window(data, window_info["start_sec"])

        return {
            "physio": torch.from_numpy(physio).float(),
            "mask": torch.tensor(True, dtype=torch.bool),
        }

    def _precompute_all_windows(self) -> None:
        """Precompute all windows and store them as torch tensors in memory."""
        from tqdm.auto import tqdm

        print("Precomputing all windows into memory (this may use a lot of RAM)...")
        self.windows_data = [None] * len(self.windows)
        for i, win in enumerate(tqdm(self.windows, desc="Precompute")):
            try:
                data = self._load_participant_data(win["dyad"], win["participant"])
                physio_np = self._extract_window(data, win["start_sec"])
                self.windows_data[i] = torch.from_numpy(physio_np).float()
            except Exception as e:
                warnings.warn(f"Error precomputing window {i}: {e}")

        print(f"Precomputed {len(self.windows_data)} windows")

    def transfer_cache_to_device(self, device: torch.device) -> None:
        """Move precomputed windows to a device (e.g., CUDA). Use with care — may exhaust VRAM."""
        if not self.cache_windows or self.windows_data is None:
            raise RuntimeError("No precomputed windows available to transfer; set cache_windows=True first.")

        print(f"Transferring {len(self.windows_data)} windows to device: {device}")
        for i in range(len(self.windows_data)):
            self.windows_data[i] = self.windows_data[i].to(device)
        print("Transfer complete")

    def get_signal_names(self) -> List[str]:
        """Return list of signal channel names."""
        return self.PHYSIO_SIGNALS


# =============================================================================
# Convenience function
# =============================================================================


def create_eatmint_dataset(
    data_root: str = "../data/EATMINT/researchdata",
    window_size_sec: float = 12.0,
    hop_size_sec: float = 6.0,
    target_fs: float = 50.0,
    physio_fs: float = 512.0,
    smoothing_kernel: int = 0,
    downsample_strategy: str = "polyphase",
    ecg_band: Optional[tuple[float, float]] = (0.5, 40.0),
    detrend_non_ecg: bool = True,
    preload: bool = True,
    cache_windows: bool = True,
) -> EATMINTPhysioDataset:
    """
    Convenience function to create an EATMINT physio dataset.

    Args:
        data_root: Path to EATMINT data root (containing Physio/, StartEndTimes/)
        window_size_sec: Window duration in seconds
        hop_size_sec: Hop between windows in seconds
        target_fs: Target sampling frequency (Hz)
        physio_fs: Native physio sampling rate (Hz)
        smoothing_kernel: Kernel size for smoothing (0=disabled)
        downsample_strategy: "polyphase" or "avg"
        ecg_band: Optional (low, high) band-pass for ECG in Hz; set to None to disable
        detrend_non_ecg: If True, remove DC from non-ECG channels
        preload: Whether to preload all data

    Returns:
        EATMINTPhysioDataset instance
    """
    config = EATMINTConfig(data_root=data_root)

    # Check availability
    availability = check_physio_availability(config)
    usable = print_availability_summary(availability)

    if len(usable) == 0:
        raise ValueError(
            f"No usable data found in {data_root}. "
            "Please ensure Physio/ and StartEndTimes/ directories exist."
        )

    return EATMINTPhysioDataset(
        config=config,
        availability_df=usable,
        window_size_sec=window_size_sec,
        hop_size_sec=hop_size_sec,
        target_fs=target_fs,
        physio_fs=physio_fs,
        smoothing_kernel=smoothing_kernel,
        downsample_strategy=downsample_strategy,
        ecg_band=ecg_band,
        detrend_non_ecg=detrend_non_ecg,
        preload=preload,
        cache_windows=cache_windows,
    )
