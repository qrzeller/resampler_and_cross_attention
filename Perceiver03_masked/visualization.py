# -*- coding: utf-8 -*-
"""Visualization utilities for physio reconstruction analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def select_indices(
    dataset_size: int,
    requested: Optional[Sequence[int]] = None,
    max_samples: int = 4,
) -> Sequence[int]:
    """
    Select which dataset indices to visualize.

    Args:
        dataset_size: Total size of dataset
        requested: Specific indices to use (optional)
        max_samples: Maximum number of samples to show

    Returns:
        List of indices to visualize
    """
    if requested:
        return [idx for idx in requested if 0 <= idx < dataset_size][:max_samples]
    return list(range(min(max_samples, dataset_size)))


def resolve_feature_names(
    num_channels: int,
    provided: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    """
    Resolve feature names for plotting.

    Args:
        num_channels: Number of channels to name
        provided: Provided names (optional)

    Returns:
        List of feature names
    """
    if provided:
        names = list(provided)
    else:
        names = []

    if len(names) < num_channels:
        names = names + [f"Channel {idx}" for idx in range(len(names), num_channels)]
    return names[:num_channels]


def plot_physio_reconstructions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    sample_indices: Optional[Iterable[int]] = None,
    max_samples: int = 4,
    save_path: str = "physio_recon.png",
    feature_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Render ground-truth vs. reconstructed physio signals.

    Useful for quick visual inspection of reconstruction quality.

    Args:
        model: The model (should be in eval mode)
        dataset: Dataset to sample from
        device: Device to run inference on
        sample_indices: Specific indices to plot (optional)
        max_samples: Maximum number of samples to show
        save_path: Where to save the figure
        feature_names: Names for each channel (optional)

    Returns:
        Path to saved figure
    """
    model.eval()
    indices = select_indices(len(dataset), sample_indices, max_samples)

    if not indices:
        raise ValueError("No samples available for plotting")

    first_sample = dataset[indices[0]]
    num_channels = first_sample["physio"].shape[1]
    names = resolve_feature_names(num_channels, feature_names)

    rows = len(indices)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            sample = dataset[idx]
            physio = sample["physio"].unsqueeze(0).to(device)
            mask = sample.get("mask")
            if mask is not None:
                mask = mask.unsqueeze(0).to(device)

            recon = model(physio, mask=mask).cpu().squeeze(0)
            target = physio.cpu().squeeze(0)

            # Only show model output on masked positions; elsewhere show ground truth
            if mask is not None:
                mask_bool = mask.squeeze(0).cpu().bool()
                recon_display = target.clone()
                recon_display[mask_bool] = recon[mask_bool]
            else:
                mask_bool = None
                recon_display = recon

            time = np.arange(target.shape[0])

            for feat in range(num_channels):
                gt_label = f"{names[feat]} (gt)"
                recon_label = f"{names[feat]} (recon masked)" if mask is not None else f"{names[feat]} (recon)"
                ax.plot(time, target[:, feat], label=gt_label, linewidth=1.5)
                ax.plot(
                    time,
                    recon_display[:, feat],
                    linestyle="--",
                    label=recon_label,
                    linewidth=1.2,
                    alpha=0.9,
                )

            # Shade masked regions (any channel masked)
            if mask_bool is not None:
                masked_any = mask_bool.any(dim=1).numpy()
                # find contiguous masked spans
                in_span = False
                start = 0
                for i, m in enumerate(masked_any.tolist() + [False]):
                    if m and not in_span:
                        start = i
                        in_span = True
                    elif in_span and not m:
                        ax.axvspan(start, i, color="gray", alpha=0.12)
                        in_span = False

            ax.set_xlabel("Sample index")
            ax.set_ylabel("Signal value")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Sample {idx}: Ground-truth vs Reconstruction")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    return save_path


def plot_training_history(
    history: list[dict],
    save_path: str = "training_history.png",
) -> Path:
    """
    Plot training and validation loss curves.

    Args:
        history: List of dicts with keys: epoch, train_loss, val_loss
        save_path: Where to save the figure

    Returns:
        Path to saved figure
    """
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, marker="s", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    return save_path
