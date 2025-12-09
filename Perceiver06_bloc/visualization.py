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
    mask_ratio: float = 0.75,
    save_path: str = "physio_recon.png",
    feature_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Render ground-truth vs. reconstructed physio signals WITH patch masking.
    
    This shows the model's ability to reconstruct masked patches from visible context,
    as it does during training. Masked regions are highlighted.

    Args:
        model: The model (should be in eval mode)
        dataset: Dataset to sample from
        device: Device to run inference on
        sample_indices: Specific indices to plot (optional)
        max_samples: Maximum number of samples to show
        mask_ratio: Ratio of patches to mask (default: 0.75)
        save_path: Where to save the figure
        feature_names: Names for each channel (optional)

    Returns:
        Path to saved figure
    """
    from masking import random_patch_mask
    
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
            physio = sample["physio"].unsqueeze(0).to(device)  # (1, seq_len, channels)
            
            # Calculate number of patches
            seq_len = physio.shape[1]
            patch_len = model.patch_len
            num_patches = seq_len // patch_len
            
            # Generate patch mask (1=masked, 0=visible)
            patch_mask = random_patch_mask(
                batch_size=1,
                num_patches=num_patches,
                num_channels=num_channels,
                mask_ratio=mask_ratio,
                device=device
            )
            
            # Forward pass WITH masking
            recon = model(physio, patch_mask=patch_mask).cpu().squeeze(0)
            target = physio.cpu().squeeze(0)
            patch_mask_cpu = patch_mask.cpu().squeeze(0)  # (num_patches, num_channels)

            time = range(target.shape[0])

            # First, highlight masked/visible regions for all channels
            for patch_idx in range(num_patches):
                start = patch_idx * patch_len
                end = start + patch_len
                # Check if this patch is masked for ANY channel
                is_masked = patch_mask_cpu[patch_idx, :].any().item()
                if is_masked:
                    ax.axvspan(start, end, alpha=0.12, color='red', zorder=0)
                else:
                    ax.axvspan(start, end, alpha=0.08, color='green', zorder=0)
            
            # Then plot signals on top
            for feat in range(num_channels):
                gt_label = f"{names[feat]} (gt)"
                recon_label = f"{names[feat]} (recon)"
                ax.plot(time, target[:, feat], label=gt_label, linewidth=1.5, alpha=0.7)
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
            mask_percent = int(mask_ratio * 100)
            ax.set_title(f"Sample {idx}: {mask_percent}% Patch Masked Reconstruction (red=masked, green=visible)")

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

    For each sample, shows only the dropped modality and its ground truth.
    Each column represents a modality dropout scenario.

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

    # One column per modality dropout
    rows = len(indices)
    cols = num_channels

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), sharex=False)
    if rows == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            physio = sample["physio"].unsqueeze(0).to(device)  # (1, seq_len, channels)
            target = physio.cpu().squeeze(0)
            time = range(target.shape[0])

            # Reconstructions with each modality dropped
            for dropped_feat in range(num_channels):
                ax = axes[row, dropped_feat]

                # Create input with one modality dropped
                physio_dropped = physio.clone()
                physio_dropped[:, :, dropped_feat] = 0.0

                recon_dropped = model(physio_dropped).cpu().squeeze(0)

                # Plot only the dropped modality
                ax.plot(
                    time,
                    target[:, dropped_feat],
                    label=f"{names[dropped_feat]} (ground truth)",
                    linewidth=2.5,
                    color="steelblue",
                )
                ax.plot(
                    time,
                    recon_dropped[:, dropped_feat],
                    linestyle="--",
                    label=f"{names[dropped_feat]} (reconstruction)",
                    linewidth=2.0,
                    alpha=0.9,
                    color="red",
                )

                ax.set_title(f"Sample {idx}: {names[dropped_feat]} dropped")
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Signal value")
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def plot_unmasked_reconstructions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    sample_indices: Optional[Iterable[int]] = None,
    max_samples: int = 4,
    save_path: str = "unmasked_recon.png",
    feature_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Render full unmasked reconstructions (standard autoencoding, no masking).
    
    Shows all modalities/channels side-by-side with ground truth overlaid.
    Unlike modality dropout plots, this passes complete input without any masking.
    
    Useful for evaluating overall reconstruction quality without masked modeling.

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

    # One row per sample, one column per channel
    rows = len(indices)
    cols = num_channels

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharex=False)
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            physio = sample["physio"].unsqueeze(0).to(device)  # (1, seq_len, channels)
            
            # Forward pass WITHOUT masking (patch_mask=None)
            recon = model(physio, patch_mask=None).cpu().squeeze(0)
            target = physio.cpu().squeeze(0)
            time = range(target.shape[0])

            for col in range(num_channels):
                ax = axes[row][col]
                
                # Plot ground truth
                ax.plot(
                    time,
                    target[:, col],
                    label="Ground truth",
                    linewidth=2.0,
                    color="steelblue",
                    alpha=0.8,
                )
                
                # Plot reconstruction
                ax.plot(
                    time,
                    recon[:, col],
                    linestyle="--",
                    label="Reconstruction",
                    linewidth=1.8,
                    color="orange",
                    alpha=0.9,
                )

                ax.set_title(f"Sample {idx}: {names[col]}")
                ax.set_xlabel("Sample index")
                ax.set_ylabel("Signal value")
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)

    fig.suptitle("Unmasked Reconstruction (Full Autoencoding)", fontsize=14, y=1.00)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path
