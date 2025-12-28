# -*- coding: utf-8 -*-
"""Training and evaluation utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    diff_weight: float = 0.2,
) -> Tuple[torch.Tensor, int]:
    """
    Compute masked reconstruction loss with optional first-difference term.

    Uses L1 loss (mean absolute error) instead of MSE for more robust training.
    The first-difference loss helps combat over-smoothing by encouraging
    the model to preserve temporal dynamics.

    Args:
        predictions: (batch, seq_len, channels) predicted signals
        targets: (batch, seq_len, channels) target signals
        mask: (batch,) boolean mask indicating valid samples
        diff_weight: Weight for first-difference loss term (0.0 to disable)

    Returns:
        loss: Scalar loss value
        num_valid: Number of valid samples in batch
    """
    # Element-wise Huber loss: shape (batch, seq_len, channels)
    elem_loss = F.huber_loss(predictions, targets, reduction="none")

    batch = predictions.size(0)

    if mask is not None:
        # Normalize mask to (batch, seq_len, channels)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.dim() == 2:
            # (batch, seq_len) -> expand across channels
            mask = mask.unsqueeze(-1).expand(-1, -1, predictions.size(-1))
        elif mask.dim() == 2 and mask.size(0) == predictions.size(1):
            mask = mask.unsqueeze(0).expand(batch, -1, -1)

        valid_counts = mask.float().sum(dim=(1, 2))
        # If no valid masked positions in batch, return zero loss and zero valid
        if valid_counts.sum() <= 0:
            return predictions.sum() * 0.0, 0

        rec_per_sample = (elem_loss * mask.float()).sum(dim=(1, 2)) / (valid_counts + 1e-8)

        # First-difference loss computed where both adjacent positions are masked
        if (
            diff_weight > 0.0
            and predictions.size(1) > 1
            and predictions.size(1) == targets.size(1)
        ):
            dp = predictions[:, 1:] - predictions[:, :-1]
            dt = targets[:, 1:] - targets[:, :-1]
            dloss = F.huber_loss(dp, dt, reduction="none")
            mask_diff = mask[:, 1:, :] & mask[:, :-1, :]
            valid_diff = mask_diff.float().sum(dim=(1, 2))
            dl1_per_sample = (dloss * mask_diff.float()).sum(dim=(1, 2)) / (valid_diff + 1e-8)
            # If no diff-valid positions, dl1_per_sample will be zeros due to division by eps
            per_sample = rec_per_sample + diff_weight * dl1_per_sample
        else:
            per_sample = rec_per_sample

        # Average over samples that had any valid masked positions
        valid_sample_mask = (valid_counts > 0)
        if valid_sample_mask.sum() == 0:
            return predictions.sum() * 0.0, 0
        loss = per_sample[valid_sample_mask].mean()
        return loss, int(valid_sample_mask.sum().item())
    else:
        # No mask provided: compute full-sequence loss averaged per-sample
        per_sample = elem_loss.mean(dim=(1, 2))
        return per_sample.mean(), len(per_sample)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    diff_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on (cpu/cuda)
        diff_weight: Weight for first-difference loss term

    Returns:
        Dictionary with "loss" key
    """
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        physio = batch["physio"].to(device, non_blocking=True)
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon = model(physio, mask=mask)
        loss, valid = masked_reconstruction_loss(
            recon, physio, mask=mask, diff_weight=diff_weight
        )

        if valid == 0:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return {"loss": total_loss / max(1, steps)}


def evaluate(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    diff_weight: float = 0.2,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: Validation data loader (or None to skip)
        device: Device to evaluate on
        diff_weight: Weight for first-difference loss term

    Returns:
        Dictionary with "loss" key
    """
    if dataloader is None:
        return {"loss": float("nan")}

    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            physio = batch["physio"].to(device, non_blocking=True)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)

            recon = model(physio)
            loss, valid = masked_reconstruction_loss(
                recon, physio, mask=mask, diff_weight=diff_weight
            )

            if valid == 0:
                continue

            total_loss += loss.item()
            steps += 1

    return {"loss": total_loss / max(1, steps)}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    diff_weight: float = 0.2,
) -> List[Dict[str, float]]:
    """
    Run full training loop.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: Optimizer instance
        device: Device to train on
        num_epochs: Number of training epochs
        diff_weight: Weight for first-difference loss

    Returns:
        List of history dictionaries with keys: epoch, train_loss, val_loss
    """
    history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, diff_weight=diff_weight
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, diff_weight=diff_weight)

        # Record
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )

        # Print
        val_str = (
            f"{val_metrics['loss']:.4f}"
            if not math.isnan(val_metrics["loss"])
            else "n/a"
        )
        print(f"  train_loss={train_metrics['loss']:.4f} | val_loss={val_str}")

    return history
