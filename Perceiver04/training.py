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
    modality_mask: Optional[torch.Tensor] = None,
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
        modality_mask: (batch, 1, channels) mask for modality dropout (1=dropped, 0=kept)

    Returns:
        loss: Scalar loss value
        num_valid: Number of valid samples in batch
    """
    # Reconstruction loss: was L1 (mean absolute error)
    # l1 = F.l1_loss(predictions, targets, reduction="none").mean(dim=(1, 2))
    # Switched to Huber loss for smoother robustness
    recon_loss = F.huber_loss(predictions, targets, reduction="none")  # (batch, seq_len, channels)
    
    # Apply modality mask: only compute loss on masked (dropped) channels
    if modality_mask is not None:
        recon_loss = recon_loss * modality_mask  # Zero out loss for unmasked channels
        # Normalize by number of masked channels
        num_masked = modality_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (batch, seq_len, 1)
        l1 = recon_loss.sum(dim=-1) / num_masked.squeeze(-1)  # (batch, seq_len)
        l1 = l1.mean(dim=1)  # (batch,)
    else:
        l1 = recon_loss.mean(dim=(1, 2))

    # First-difference loss (finite differences) - also L1
    if (
        diff_weight > 0.0
        and predictions.size(1) > 1
        and predictions.size(1) == targets.size(1)
    ):
        dp = predictions[:, 1:] - predictions[:, :-1]
        dt = targets[:, 1:] - targets[:, :-1]
        # First-difference loss: was L1, now Huber for consistency
        # dl1 = F.l1_loss(dp, dt, reduction="none").mean(dim=(1, 2))
        diff_loss = F.huber_loss(dp, dt, reduction="none")  # (batch, seq_len-1, channels)
        
        # Apply modality mask to diff loss as well
        if modality_mask is not None:
            # modality_mask is (batch, 1, channels), broadcast to (batch, seq_len-1, channels)
            diff_loss = diff_loss * modality_mask  # Broadcast mask across time dimension
            num_masked = modality_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (batch, 1, 1)
            dl1 = diff_loss.sum(dim=-1) / num_masked.squeeze(-1)  # (batch, seq_len-1)
            dl1 = dl1.mean(dim=1)  # (batch,)
        else:
            dl1 = diff_loss.mean(dim=(1, 2))
        
        per_sample = l1 + diff_weight * dl1
    else:
        per_sample = l1

    # Apply mask if provided
    if mask is not None:
        weights = mask.float()
        valid = weights.sum()
        if valid <= 0:
            return predictions.sum() * 0.0, 0
        return (per_sample * weights).sum() / valid, int(valid.item())
    else:
        return per_sample.mean(), len(per_sample)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    diff_weight: float = 0.2,
    modality_dropout_p: float = 0.2,
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
        physio = batch["physio"].to(device)

        # Modality dropout: randomly drop entire channels in the input
        if modality_dropout_p > 0.0:
            keep_prob = max(0.0, min(1.0, 1.0 - modality_dropout_p))
            b, _, c = physio.shape
            mod_mask = torch.bernoulli(
                torch.full((b, 1, c), keep_prob, device=physio.device)
            )
            physio_in = physio * mod_mask
            # Create loss mask: 1 for dropped channels (where mod_mask=0), 0 for kept channels
            loss_mask = 1.0 - mod_mask
            
            # Skip if no channels were dropped in this batch
            if loss_mask.sum() == 0:
                continue
        else:
            physio_in = physio
            loss_mask = None

        optimizer.zero_grad(set_to_none=True)
        # Use masked input as residual_base - model must predict full signal for dropped channels
        recon = model(physio_in, residual_base=physio_in)
        loss, valid = masked_reconstruction_loss(
            recon, physio, mask=None, diff_weight=diff_weight, modality_mask=loss_mask
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
    modality_dropout_p: float = 0.0,
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
            physio = batch["physio"].to(device)

            # Optionally apply modality dropout at evaluation (default disabled)
            if modality_dropout_p > 0.0:
                keep_prob = max(0.0, min(1.0, 1.0 - modality_dropout_p))
                b, _, c = physio.shape
                mod_mask = torch.bernoulli(
                    torch.full((b, 1, c), keep_prob, device=physio.device)
                )
                physio_in = physio * mod_mask
                # Create loss mask: 1 for dropped channels, 0 for kept channels
                loss_mask = 1.0 - mod_mask
                
                # Skip if no channels were dropped in this batch
                if loss_mask.sum() == 0:
                    continue
            else:
                physio_in = physio
                loss_mask = None

            # Pass original physio as residual_base so residual connections use unmasked input
            recon = model(physio_in, residual_base=physio_in)
            loss, valid = masked_reconstruction_loss(
                recon, physio, mask=None, diff_weight=diff_weight, modality_mask=loss_mask
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
    modality_dropout_p: float = 0.2,
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
            model,
            train_loader,
            optimizer,
            device,
            diff_weight=diff_weight,
            modality_dropout_p=modality_dropout_p,
        )

        # Validate
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            diff_weight=diff_weight,
            modality_dropout_p=0.0,
        )

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
