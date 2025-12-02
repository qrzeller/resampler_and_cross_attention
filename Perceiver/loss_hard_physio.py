"""Physio-pooled Perceiver variant with advanced physio-only masking.

This script keeps the high-resolution physio stream (~500 Hz) all the way into
the model (via :mod:`physiopooled_eatmint_multimodal_perceiver`) and applies a
set of structured masking augmentations exclusively to the physio modality.
The masking policy mixes channel dropout, temporal span dropout, and mild
Gaussian noise, encouraging the model to compress physio signals while relying
on the other modalities for reconstruction.

Usage:
    python physiopooled_masked_physio.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from eatmint_multimodal_perceiver import (
    AudioFeatureExtractor,
    EATMINTConfig,
    EATMINTDataset,
    check_modality_availability,
    compute_masked_loss,
    plot_training_history,
    print_availability_summary,
    visualize_reconstruction,
)
from physiopooled_eatmint_multimodal_perceiver import (
    PhysioPooledEATMINTDataset,
    PhysioPooledEATMINTPerceiver,
)


# =============================================================================
# Physio masking policy
# =============================================================================


@dataclass
class PhysioMaskingConfig:
    """Configuration for the physio-only masking policy."""

    channel_drop_prob: float = 0.2
    min_channels_to_drop: int = 0
    max_channels_to_drop: int = 2
    temporal_mask_prob: float = 0.85
    min_temporal_span_ratio: float = 0.05
    max_temporal_span_ratio: float = 0.25
    max_temporal_masks: int = 3
    temporal_mask_band_prob: float = 0.4
    noise_std: float = 0.01
    min_severity: float = 0.4
    max_severity: float = 1.0
    schedule: str = "cosine"  # cosine progression across epochs


class PhysioMasker:
    """Applies structured masks to physio tokens before they enter the model."""

    def __init__(self, config: PhysioMaskingConfig) -> None:
        self.config = config
        self._severity = config.max_severity

    def update_progress(self, progress: float) -> None:
        progress = float(min(max(progress, 0.0), 1.0))
        if self.config.schedule == "cosine":
            scaled = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif self.config.schedule == "linear":
            scaled = progress
        else:
            scaled = 1.0
        span = max(self.config.max_severity - self.config.min_severity, 1e-3)
        self._severity = self.config.min_severity + span * scaled

    def __call__(
        self,
        physio: torch.Tensor,
        physio_available: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if physio is None:
            return physio, 0.0

        masked = physio.clone()
        total_masked = 0
        total_tokens = 0
        available_indices = torch.nonzero(physio_available.bool(), as_tuple=False).flatten()

        for idx in available_indices:
            seq = masked[idx]
            seq_masked, masked_count = self._mask_single(seq)
            masked[idx] = seq_masked
            total_masked += masked_count
            total_tokens += seq.numel()

        ratio = float(total_masked) / float(total_tokens) if total_tokens > 0 else 0.0
        return masked, ratio

    def _mask_single(self, seq: torch.Tensor) -> tuple[torch.Tensor, int]:
        seq_len, n_channels = seq.shape
        mask = torch.zeros(seq_len, n_channels, dtype=torch.bool, device=seq.device)
        result = seq.clone()

        # Channel dropout
        drop_prob = self.config.channel_drop_prob * self._severity
        if n_channels > 0 and torch.rand(1, device=seq.device).item() < drop_prob:
            max_drop = min(n_channels, self.config.max_channels_to_drop)
            min_drop = min(max_drop, self.config.min_channels_to_drop)
            if max_drop > 0 and min_drop > 0:
                drop_count = torch.randint(min_drop, max_drop + 1, (1,), device=seq.device).item()
                drop_idx = torch.randperm(n_channels, device=seq.device)[:drop_count]
                mask[:, drop_idx] = True

        # Temporal span dropout
        temp_prob = self.config.temporal_mask_prob * self._severity
        if seq_len > 1 and torch.rand(1, device=seq.device).item() < temp_prob:
            max_masks = max(1, self.config.max_temporal_masks)
            n_masks = torch.randint(1, max_masks + 1, (1,), device=seq.device).item()
            for _ in range(n_masks):
                span_ratio = torch.empty(1, device=seq.device).uniform_(
                    self.config.min_temporal_span_ratio,
                    self.config.max_temporal_span_ratio,
                ).item()
                span_len = max(1, int(round(span_ratio * seq_len)))
                span_len = min(span_len, seq_len)
                max_start = max(1, seq_len - span_len + 1)
                start = torch.randint(0, max_start, (1,), device=seq.device).item()
                end = min(seq_len, start + span_len)

                if torch.rand(1, device=seq.device).item() < self.config.temporal_mask_band_prob:
                    mask[start:end, :] = True
                else:
                    subset = max(1, int(round(self._severity * n_channels * 0.5)))
                    channel_sel = torch.randperm(n_channels, device=seq.device)[:subset]
                    mask[start:end, channel_sel] = True

        # Mild noise on unmasked tokens to encourage denoising
        if self.config.noise_std > 0:
            noise = torch.randn_like(result) * (self.config.noise_std * self._severity)
            result = torch.where(mask, result, result + noise)

        result = result.masked_fill(mask, 0.0)
        return result, int(mask.sum().item())


# =============================================================================
# Loss weighting policy
# =============================================================================


@dataclass
class PhysioLossWeightConfig:
    """Configuration for prioritizing physio reconstruction losses."""

    focus_fraction: float = 0.65  # Portion of training dedicated to physio-first
    max_physio_weight: float = 4.0
    min_other_weight: float = 0.25
    mid_physio_weight: float = 1.25
    mid_other_weight: float = 0.9


class PhysioLossWeightScheduler:
    """Smoothly anneals modality weights to focus on physio early in training."""

    def __init__(self, config: PhysioLossWeightConfig) -> None:
        self.config = config
        self._modality_names = ["audio", "physio", "openface", "eyetracker"]

    def __call__(self, progress: float) -> Dict[str, float]:
        progress = float(min(max(progress, 0.0), 1.0))
        focus = float(min(max(self.config.focus_fraction, 1e-3), 0.99))

        if progress < focus:
            phase = progress / focus
            physio_weight = self._cosine_blend(
                self.config.max_physio_weight,
                self.config.mid_physio_weight,
                phase,
            )
            other_weight = self._cosine_blend(
                self.config.min_other_weight,
                self.config.mid_other_weight,
                phase,
            )
        else:
            tail = (progress - focus) / max(1.0 - focus, 1e-3)
            physio_weight = self._cosine_blend(
                self.config.mid_physio_weight,
                1.0,
                tail,
            )
            other_weight = self._cosine_blend(
                self.config.mid_other_weight,
                1.0,
                tail,
            )

        physio_weight = max(1e-3, physio_weight)
        other_weight = max(1e-3, other_weight)

        weights = {mod: other_weight for mod in self._modality_names}
        weights["physio"] = physio_weight
        return weights

    @staticmethod
    def _cosine_blend(start: float, end: float, t: float) -> float:
        t = float(min(max(t, 0.0), 1.0))
        # Cosine easing keeps the schedule smooth and monotonic.
        blend = 0.5 - 0.5 * math.cos(math.pi * t)
        return start + (end - start) * blend


# =============================================================================
# Training helpers
# =============================================================================


def train_epoch_masked(
    model: PhysioPooledEATMINTPerceiver,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    physio_masker: Optional[PhysioMasker] = None,
    epoch_progress: float = 1.0,
    loss_weight_schedule: Optional[Callable[[float], Dict[str, float]]] = None,
) -> Dict[str, float]:
    model.train()
    if physio_masker is not None:
        physio_masker.update_progress(epoch_progress)

    epoch_weights = None
    if loss_weight_schedule is not None:
        epoch_weights = loss_weight_schedule(epoch_progress)

    total_loss = 0.0
    modality_losses = {"audio": 0.0, "physio": 0.0, "openface": 0.0, "eyetracker": 0.0}
    mask_ratios: List[float] = []
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training (physio masked)")
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        physio_input = batch["physio"]
        if physio_masker is not None:
            physio_masked, mask_ratio = physio_masker(physio_input, batch["modality_mask"][:, 1])
            mask_ratios.append(mask_ratio)
        else:
            physio_masked = physio_input

        outputs = model(
            audio=batch["audio"],
            physio=physio_masked,
            openface=batch["openface"],
            eyetracker=batch["eyetracker"],
            modality_mask=batch["modality_mask"],
        )

        loss, losses = compute_masked_loss(
            outputs,
            {k: batch[k] for k in ["audio", "physio", "openface", "eyetracker"]},
            batch["modality_mask"],
            modality_weights=epoch_weights,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for mod_name, mod_loss in losses.items():
            modality_losses[mod_name] += mod_loss
        n_batches += 1
        pbar.set_postfix({"loss": loss.item()})

    metrics = {"total_loss": total_loss / max(1, n_batches)}
    for mod_name in modality_losses:
        metrics[f"{mod_name}_loss"] = modality_losses[mod_name] / max(1, n_batches)
    if mask_ratios:
        metrics["physio_mask_ratio"] = float(sum(mask_ratios) / len(mask_ratios))
    if epoch_weights:
        for mod_name, weight in epoch_weights.items():
            metrics[f"weight_{mod_name}"] = float(weight)
    return metrics


def train_masked(
    model: PhysioPooledEATMINTPerceiver,
    dataset: PhysioPooledEATMINTDataset,
    *,
    physio_masker: Optional[PhysioMasker] = None,
    loss_weight_schedule: Optional[Callable[[float], Dict[str, float]]] = None,
    n_epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        progress = epoch / max(1, n_epochs - 1)
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        metrics = train_epoch_masked(
            model,
            dataloader,
            optimizer,
            device,
            physio_masker=physio_masker,
            epoch_progress=progress,
            loss_weight_schedule=loss_weight_schedule,
        )
        scheduler.step()
        history.append(metrics)
        mask_ratio = metrics.get("physio_mask_ratio", 0.0)
        print(
            "  Total loss: {total:.4f} | Audio: {audio:.4f} | Physio: {physio:.4f} | "
            "OpenFace: {openface:.4f} | Eye: {eye:.4f} | Mask ratio: {mask:.2%}".format(
                total=metrics["total_loss"],
                audio=metrics["audio_loss"],
                physio=metrics["physio_loss"],
                openface=metrics["openface_loss"],
                eye=metrics["eyetracker_loss"],
                mask=mask_ratio,
            )
        )

        if save_path:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                save_path,
            )

    return history


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
    print("Physio-Pooled Perceiver with Physio Masking")
    print("=" * 60)
    print(f"\nData root: {config.data_root}")

    availability_df = check_modality_availability(config)
    usable_df = print_availability_summary(availability_df)

    precomputed_dir = config.sound_features_dir
    precomputed_files = list(precomputed_dir.glob("*.npy")) if precomputed_dir.exists() else []
    use_precomputed = len(precomputed_files) > 0
    print("\nPrecomputed audio features: ", end="")
    if use_precomputed:
        print(f"Found {len(precomputed_files)} files in {precomputed_dir}")
    else:
        print("Not found. Run 'python tools/precompute.py' for faster training.")

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
        num_latents=256,
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

    mask_config = PhysioMaskingConfig()
    physio_masker = PhysioMasker(mask_config)
    loss_weight_scheduler = PhysioLossWeightScheduler(PhysioLossWeightConfig())

    history = train_masked(
        model=model,
        dataset=dataset,
        physio_masker=physio_masker,
        loss_weight_schedule=loss_weight_scheduler,
        n_epochs=10,
        batch_size=4,
        lr=1e-4,
        device=device,
        save_path="checkpoints/physio_first_physiopooled_masked_physio_checkpoint.pt",
    )

    plot_training_history(history, save_path="checkpoints/physio_first_physiopooled_masked_history.png")
    visualize_reconstruction(
        model,
        dataset,
        sample_idx=0,
        device=device,
        save_path="checkpoints/physio_first_physiopooled_masked_reconstruction.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
