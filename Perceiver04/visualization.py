# -*- coding: utf-8 -*-
"""Visualization utilities for physio reconstruction analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
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
            recon = model(physio).cpu().squeeze(0)
            target = physio.cpu().squeeze(0)

            time = range(target.shape[0])

            for feat in range(num_channels):
                gt_label = f"{names[feat]} (gt)"
                recon_label = f"{names[feat]} (recon)"
                ax.plot(time, target[:, feat], label=gt_label, linewidth=1.5)
                ax.plot(
                    time,
                    recon[:, feat],
                    linestyle="--",
                    label=recon_label,
                    linewidth=1.2,
                    alpha=0.8,
                )

            ax.set_xlabel("Sample index")
            ax.set_ylabel("Signal value (normalized)")
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


def plot_modality_dropout_reconstructions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    sample_indices: Optional[Iterable[int]] = None,
    max_samples: int = 4,
    save_path: str = "modality_dropout_recon.png",
    feature_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Render reconstructions with modality dropout for each modality.

    For each sample, shows:
    - Full input (all modalities)
    - Reconstruction with each modality dropped one at a time

    Useful for analyzing model robustness to missing modalities.

    Args:
        model: The model (should be in eval mode)
        dataset: Dataset to sample from
        device: Device to run inference on
        sample_indices: Specific indices to plot (optional)
        max_samples: Maximum number of samples to show
        save_path: Where to save the figure
        feature_names: Names for each channel/modality (optional)

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

    # Number of scenarios: 1 (full) + num_channels (each dropped)
    num_scenarios = num_channels + 1
    rows = len(indices)
    cols = num_scenarios

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=False)
    if rows == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            physio = sample["physio"].unsqueeze(0).to(device)  # (1, seq_len, channels)
            time = range(physio.shape[1])

            # Full reconstruction (no dropout)
            ax = axes[row, 0]
            full_recon = model(physio).cpu().squeeze(0)
            target = physio.cpu().squeeze(0)

            for feat in range(num_channels):
                ax.plot(
                    time,
                    target[:, feat],
                    label=f"{names[feat]} (gt)",
                    linewidth=1.5,
                )
                ax.plot(
                    time,
                    full_recon[:, feat],
                    linestyle="--",
                    label=f"{names[feat]} (recon)",
                    linewidth=1.2,
                    alpha=0.8,
                )

            ax.set_title(f"Sample {idx}: Full (all modalities)")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Signal value")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)

            # Reconstructions with each modality dropped
            for dropped_feat in range(num_channels):
                ax = axes[row, dropped_feat + 1]

                # Create input with one modality dropped
                physio_dropped = physio.clone()
                physio_dropped[:, :, dropped_feat] = 0.0

                recon_dropped = model(physio_dropped).cpu().squeeze(0)

                # Plot all channels
                for feat in range(num_channels):
                    if feat == dropped_feat:
                        # Show the input as zero (dropped modality)
                        ax.plot(
                            time,
                            target[:, feat],
                            label=f"{names[feat]} (dropped input)",
                            linewidth=1.5,
                            alpha=0.3,
                            linestyle=":",
                        )
                        ax.plot(
                            time,
                            recon_dropped[:, feat],
                            linestyle="--",
                            label=f"{names[feat]} (recon w/o input)",
                            linewidth=1.2,
                            alpha=0.8,
                            color="red",
                        )
                    else:
                        ax.plot(
                            time,
                            target[:, feat],
                            label=f"{names[feat]} (gt)",
                            linewidth=1.5,
                        )
                        ax.plot(
                            time,
                            recon_dropped[:, feat],
                            linestyle="--",
                            label=f"{names[feat]} (recon)",
                            linewidth=1.2,
                            alpha=0.8,
                        )

                ax.set_title(f"Sample {idx}: {names[dropped_feat]} dropped")
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Signal value")
                ax.legend(loc="best", fontsize=7)
                ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    return save_path
