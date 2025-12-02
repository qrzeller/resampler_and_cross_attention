"""Physio-centered training variant that downsamples physio streams to 50 Hz.

This script mirrors ``loss_hard_physio.py`` but removes the dedicated
high-resolution convolutional encoder/decoder. Instead, it performs a simple
average-pooling downsampling from 500 Hz to 50 Hz (with optional smoothing) so
all modalities are aligned on the same temporal grid. The goal is to simplify
physio reconstruction and allow the Perceiver to focus on inter-physio
relationships without the added complexity of the temporal conv stack.

Usage:
    python loss_hard_physio_50hz.py
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scipy import signal

from eatmint_multimodal_perceiver import (
    AudioFeatureExtractor,
    EATMINTConfig,
    EATMINTDataset,
    EATMINTPerceiver,
    check_modality_availability,
    compute_masked_loss,
    plot_training_history,
    print_availability_summary,
    visualize_reconstruction,
)
from physiopooled_eatmint_multimodal_perceiver import PhysioPooledEATMINTDataset

from loss_hard_physio import (
    PhysioMasker,
    PhysioMaskingConfig,
    PhysioLossWeightConfig,
    PhysioLossWeightScheduler,
)


# =============================================================================
# Dataset helper
# =============================================================================


class Physio50HzDataset(PhysioPooledEATMINTDataset):
    """Downsample high-rate physio tokens to 50 Hz with low-pass polyphase filtering."""

    def __init__(
        self,
        *args,
        smoothing_kernel: int = 0,
        downsample_strategy: str = "polyphase",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.smoothing_kernel = max(0, smoothing_kernel)
        self.downsample_strategy = downsample_strategy

        frac = Fraction(int(round(self.target_fs)), int(round(self.physio_target_fs)))
        self._poly_up = frac.numerator
        self._poly_down = frac.denominator

    def _polyphase_downsample(self, physio: torch.Tensor) -> torch.Tensor:
        arr = physio.detach().cpu().numpy().astype(np.float32)
        resampled = signal.resample_poly(arr, self._poly_up, self._poly_down, axis=0, padtype="mean")
        return torch.from_numpy(resampled).to(physio.device)

    def _avg_downsample(self, physio: torch.Tensor) -> torch.Tensor:
        x = physio.transpose(0, 1).unsqueeze(0)
        pooled = F.avg_pool1d(
            x,
            kernel_size=self.physio_downsample_factor,
            stride=self.physio_downsample_factor,
            ceil_mode=False,
        )
        pooled = pooled.squeeze(0).transpose(0, 1)
        return pooled

    def _smooth(self, physio: torch.Tensor) -> torch.Tensor:
        if self.smoothing_kernel <= 1:
            return physio
        kernel = torch.ones(
            physio.size(1),
            1,
            self.smoothing_kernel,
            device=physio.device,
            dtype=physio.dtype,
        )
        kernel = kernel / float(self.smoothing_kernel)
        padded = self.smoothing_kernel // 2
        return (
            F.conv1d(
                physio.transpose(0, 1).unsqueeze(0),
                kernel,
                padding=padded,
                groups=physio.size(1),
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    def _pool_physio(self, physio: torch.Tensor) -> torch.Tensor:
        if physio.dim() != 2:
            return physio
        if physio.size(0) == self.window_samples:
            return physio

        if self.downsample_strategy == "polyphase":
            pooled = self._polyphase_downsample(physio)
        else:
            pooled = self._avg_downsample(physio)

        pooled = self._smooth(pooled)

        if pooled.size(0) != self.window_samples:
            pooled = (
                F.interpolate(
                    pooled.transpose(0, 1).unsqueeze(0),
                    size=self.window_samples,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        return pooled

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        if bool(sample["modality_mask"][1].item()):
            sample["physio"] = self._pool_physio(sample["physio"])
        return sample


# =============================================================================
# Training helpers
# =============================================================================


def train_epoch(
    model: EATMINTPerceiver,
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

    pbar = tqdm(dataloader, desc="Training (physio 50Hz)")
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


def train(
    model: EATMINTPerceiver,
    dataset: Physio50HzDataset,
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
        metrics = train_epoch(
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
    print("Physio-Pooled Perceiver (50 Hz physio)")
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

    dataset = Physio50HzDataset(
        config=config,
        availability_df=usable_df,
        audio_extractor=audio_extractor,
        window_size_sec=config.window_size_sec,
        hop_size_sec=config.hop_size_sec,
        target_fs=50.0,
        physio_target_fs=512.0,
        min_modalities=2,
        use_precomputed_audio=use_precomputed,
        preload=True,
        smoothing_kernel=0,
    )
    print(f"Dataset windows: {len(dataset)}")

    sample = dataset[0]
    print("Sample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")

    model = EATMINTPerceiver(
        hidden_dim=256,
        latent_dim=256,
        num_latents=256,
        num_self_attention_layers=6,
        num_cross_attention_layers=1,
        num_heads=8,
        audio_dim=768,
        physio_dim=len(EATMINTDataset.PHYSIO_SIGNALS),
        openface_dim=len(EATMINTDataset.OPENFACE_COLS),
        eyetracker_dim=len(EATMINTDataset.EYETRACKER_COLS),
        seq_len=dataset.window_samples,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    mask_config = PhysioMaskingConfig()
    physio_masker = PhysioMasker(mask_config)
    loss_weight_scheduler = PhysioLossWeightScheduler(PhysioLossWeightConfig())

    history = train(
        model=model,
        dataset=dataset,
        physio_masker=physio_masker,
        loss_weight_schedule=loss_weight_scheduler,
        n_epochs=10,
        batch_size=4,
        lr=1e-4,
        device=device,
        save_path="checkpoints/physio_50hz_masked_checkpoint.pt",
    )

    plot_training_history(history, save_path="checkpoints/physio_50hz_masked_history.png")
    visualize_reconstruction(
        model,
        dataset,
        sample_idx=0,
        device=device,
        save_path="checkpoints/physio_50hz_masked_reconstruction.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
