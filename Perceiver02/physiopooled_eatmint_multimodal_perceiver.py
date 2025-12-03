"""Physio-pooled variant of the EATMINT Multimodal Perceiver.

This script keeps physiological data at high resolution (≈500 Hz) in the
EATMINTDataset and leverages a dedicated convolutional encoder/decoder pair to
pool the long sequences (T=4000) down to the 50 Hz grid (T=400) used by the
other modalities. The higher-resolution physio samples are passed directly to
the model, where PhysioConvEncoder performs temporal pooling before
concatenation with the remaining modalities. A mirrored PhysioConvDecoder
upsamples the reconstruction back to the original 500 Hz resolution so the
training objective remains unchanged.

Usage:
    python physiopooled_eatmint_multimodal_perceiver.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eatmint_multimodal_perceiver import (
    AudioFeatureExtractor,
    EATMINTConfig,
    EATMINTDataset,
    EATMINTPerceiver,
    check_modality_availability,
    plot_training_history,
    print_availability_summary,
    train,
    visualize_reconstruction,
)


# =============================================================================
# Dataset
# =============================================================================


class PhysioPooledEATMINTDataset(EATMINTDataset):
    """Dataset that preserves high-rate physio samples for downstream pooling."""

    def __init__(
        self,
        config: EATMINTConfig,
        availability_df,
        audio_extractor: Optional[AudioFeatureExtractor] = None,
        window_size_sec: float = 8.0,
        hop_size_sec: float = 4.0,
        target_fs: float = 50.0,
        physio_target_fs: float = 500.0,
        min_modalities: int = 2,
        preload: bool = False,
        skip_audio: bool = False,
        use_precomputed_audio: bool = True,
    ) -> None:
        super().__init__(
            config=config,
            availability_df=availability_df,
            audio_extractor=audio_extractor,
            window_size_sec=window_size_sec,
            hop_size_sec=hop_size_sec,
            target_fs=target_fs,
            min_modalities=min_modalities,
            preload=preload,
            skip_audio=skip_audio,
            use_precomputed_audio=use_precomputed_audio,
        )

        self.physio_target_fs = physio_target_fs
        self.physio_window_samples = int(round(self.window_size_sec * self.physio_target_fs))
        self.physio_downsample_factor = max(1, int(round(self.physio_target_fs / self.target_fs)))
        if self.physio_downsample_factor <= 0:
            raise ValueError("Physio downsample factor must be positive")

        pooled_len = self.physio_window_samples // self.physio_downsample_factor
        if pooled_len != self.window_samples:
            warnings.warn(
                "Physio/audio window mismatch: pooled physio length "
                f"{pooled_len} != audio length {self.window_samples}. "
                "Adjust window_size_sec or sampling rates so that physio_target_fs / target_fs "
                "is an integer and window_size_sec * target_fs == desired pooled length.",
                stacklevel=2,
            )

    def _resample_physio_window(self, modality, start_sec: float, end_sec: float):
        """High-resolution resampling helper for physio windows."""
        if not modality.available or len(modality.data) == 0:
            return None

        mask = (modality.timestamps >= start_sec) & (modality.timestamps < end_sec)
        if mask.sum() < 2:
            return None

        window_data = modality.data[mask]
        window_ts = modality.timestamps[mask]

        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)

        target_ts = np.linspace(start_sec, end_sec, self.physio_window_samples, endpoint=False)
        resampled = np.zeros((self.physio_window_samples, window_data.shape[1]), dtype=np.float32)
        for feat_idx in range(window_data.shape[1]):
            resampled[:, feat_idx] = np.interp(target_ts, window_ts, window_data[:, feat_idx])
        return resampled

    def __getitem__(self, idx: int):
        window_info = self.windows[idx]
        data = self._load_participant_data(window_info["dyad"], window_info["participant"])

        start_sec = window_info["start_sec"]
        end_sec = start_sec + self.window_size_sec

        sample = {}
        modality_mask = []

        audio_features = self._extract_audio_window(data, start_sec, end_sec)
        if audio_features is not None:
            sample["audio"] = torch.from_numpy(audio_features).float()
            modality_mask.append(True)
        else:
            sample["audio"] = torch.zeros(self.window_samples, 768)
            modality_mask.append(False)

        physio = self._resample_physio_window(data["physio"], start_sec, end_sec)
        if physio is not None:
            sample["physio"] = torch.from_numpy(physio).float()
            modality_mask.append(True)
        else:
            sample["physio"] = torch.zeros(self.physio_window_samples, len(self.PHYSIO_SIGNALS))
            modality_mask.append(False)

        openface = self._resample_to_window(data["openface"], start_sec, end_sec)
        if openface is not None:
            sample["openface"] = torch.from_numpy(openface).float()
            modality_mask.append(True)
        else:
            sample["openface"] = torch.zeros(self.window_samples, len(self.OPENFACE_COLS))
            modality_mask.append(False)

        eyetracker = self._resample_to_window(data["eyetracker"], start_sec, end_sec)
        if eyetracker is not None:
            sample["eyetracker"] = torch.from_numpy(eyetracker).float()
            modality_mask.append(True)
        else:
            sample["eyetracker"] = torch.zeros(self.window_samples, len(self.EYETRACKER_COLS))
            modality_mask.append(False)

        sample["modality_mask"] = torch.tensor(modality_mask)
        return sample


# =============================================================================
# Physio pooling modules
# =============================================================================


class PhysioConvEncoder(nn.Module):
    """Temporal conv encoder that downsamples physio sequences."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        downsample_factor: int,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.project = nn.Conv1d(
            hidden_dim,
            output_dim,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        x = x.transpose(1, 2)
        x = self.temporal(x)
        x = self.project(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class PhysioConvDecoder(nn.Module):
    """Inverse temporal module that upsamples physio reconstructions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        upsample_factor: int,
        target_len: int,
    ) -> None:
        super().__init__()
        self.target_len = target_len
        self.upsample = nn.ConvTranspose1d(
            input_dim,
            hidden_dim,
            kernel_size=upsample_factor,
            stride=upsample_factor,
        )
        self.reconstruct = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return x
        x = x.transpose(1, 2)
        x = self.upsample(x)
        x = F.gelu(x)
        x = self.reconstruct(x)
        x = x.transpose(1, 2)
        seq_len = x.size(1)
        if seq_len > self.target_len:
            x = x[:, : self.target_len, :]
        elif seq_len < self.target_len:
            pad = self.target_len - seq_len
            x = F.pad(x, (0, 0, 0, pad))
        return x


# =============================================================================
# Model
# =============================================================================


class PhysioPooledEATMINTPerceiver(EATMINTPerceiver):
    """Perceiver variant with a dedicated physio pooling pipeline."""

    def __init__(
        self,
        *,
        physio_input_dim: int,
        physio_conv_dim: int,
        physio_seq_len: int,
        physio_downsample_factor: int,
        **kwargs,
    ) -> None:
        super().__init__(physio_dim=physio_conv_dim, **kwargs)
        self.physio_seq_len = physio_seq_len
        self.physio_conv_dim = physio_conv_dim
        self.physio_temporal_encoder = PhysioConvEncoder(
            input_dim=physio_input_dim,
            hidden_dim=max(physio_input_dim * 2, 8),
            output_dim=physio_conv_dim,
            downsample_factor=physio_downsample_factor,
        )
        self.physio_temporal_decoder = PhysioConvDecoder(
            input_dim=physio_conv_dim,
            hidden_dim=max(physio_conv_dim * 2, 8),
            output_dim=physio_input_dim,
            upsample_factor=physio_downsample_factor,
            target_len=physio_seq_len,
        )

    def _pool_physio(self, physio: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.physio_temporal_encoder(physio)
        if pooled.size(1) != self.seq_len:
            pooled = F.interpolate(
                pooled.transpose(1, 2),
                size=self.seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        mask = modality_mask[:, 1].float().view(-1, 1, 1)
        return pooled * mask

    def encode(
        self,
        audio: torch.Tensor,
        physio: torch.Tensor,
        openface: torch.Tensor,
        eyetracker: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled_physio = self._pool_physio(physio, modality_mask)
        return super().encode(audio, pooled_physio, openface, eyetracker, modality_mask)

    def forward(
        self,
        audio: torch.Tensor,
        physio: torch.Tensor,
        openface: torch.Tensor,
        eyetracker: torch.Tensor,
        modality_mask: torch.Tensor,
    ):
        latents = self.encode(audio, physio, openface, eyetracker, modality_mask)
        outputs = {
            "audio": self.decode(latents, "audio"),
            "openface": self.decode(latents, "openface"),
            "eyetracker": self.decode(latents, "eyetracker"),
            "latents": latents,
        }
        physio_lowres = self.decode(latents, "physio")
        outputs["physio_lowres"] = physio_lowres
        outputs["physio"] = self.physio_temporal_decoder(physio_lowres)
        return outputs


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    script_dir = Path(__file__).parent.parent
    data_root = script_dir / "data" / "researchdata"

    config = EATMINTConfig(
        data_root=str(data_root),
        window_size_sec=8.0,
        hop_size_sec=4.0,
    )

    print("=" * 60)
    print("Physio-Pooled EATMINT Multimodal Perceiver")
    print("=" * 60)
    print(f"\nData root: {config.data_root}")

    availability_df = check_modality_availability(config)
    usable_df = print_availability_summary(availability_df)

    precomputed_dir = config.sound_features_dir
    precomputed_files = list(precomputed_dir.glob("*.npy")) if precomputed_dir.exists() else []
    use_precomputed = len(precomputed_files) > 0
    print("\nPrecomputed audio features: ", end="")
    if use_precomputed:
        print(f"✓ Found {len(precomputed_files)} files in {precomputed_dir}")
    else:
        print("✗ Not found. Run 'python tools/precompute.py' for faster training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    audio_extractor = None
    if not use_precomputed:
        print("\nLoading Wav2Vec2 audio encoder (on-the-fly extraction)...")
        audio_extractor = AudioFeatureExtractor(device=device)

    dataset = PhysioPooledEATMINTDataset(
        config=config,
        availability_df=usable_df,
        audio_extractor=audio_extractor,
        window_size_sec=config.window_size_sec,
        hop_size_sec=config.hop_size_sec,
        target_fs=50.0,
        physio_target_fs=500.0,
        min_modalities=2,
        use_precomputed_audio=use_precomputed,
        preload=True,
    )
    print(f"Dataset windows: {len(dataset)}")

    sample = dataset[0]
    print("Sample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")

    model = PhysioPooledEATMINTPerceiver(
        hidden_dim=256,
        latent_dim=256,
        num_latents=64,
        num_self_attention_layers=6,
        num_cross_attention_layers=1,
        num_heads=8,
        audio_dim=768,
        physio_input_dim=len(EATMINTDataset.PHYSIO_SIGNALS),
        physio_conv_dim=16,
        physio_seq_len=dataset.physio_window_samples,
        physio_downsample_factor=dataset.physio_downsample_factor,
        openface_dim=len(EATMINTDataset.OPENFACE_COLS),
        eyetracker_dim=len(EATMINTDataset.EYETRACKER_COLS),
        seq_len=dataset.window_samples,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = train(
        model=model,
        dataset=dataset,
        n_epochs=10,
        batch_size=8,
        lr=1e-4,
        device=device,
        save_path="physiopooled_eatmint_perceiver_checkpoint.pt",
    )

    plot_training_history(history, save_path="physiopooled_training_history.png")
    visualize_reconstruction(
        model,
        dataset,
        sample_idx=0,
        device=device,
        save_path="physiopooled_reconstruction.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
