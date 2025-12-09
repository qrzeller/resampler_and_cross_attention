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

from masking import get_mask


def masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    diff_weight: float = 0.2,
    modality_mask: Optional[torch.Tensor] = None,
    patch_len: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Compute masked reconstruction loss with optional first-difference term.

    Uses Huber loss for robust training with smoother gradients.
    The first-difference loss helps combat over-smoothing by encouraging
    the model to preserve temporal dynamics.

    Args:
        predictions: (batch, seq_len, channels) predicted signals
        targets: (batch, seq_len, channels) target signals
        mask: (batch,) boolean mask indicating valid samples
        diff_weight: Weight for first-difference loss term (0.0 to disable)
        modality_mask: (batch, num_patches, channels) mask for patches (1=masked/compute loss, 0=visible/skip)
        patch_len: Length of each patch in samples (required if modality_mask is provided)

    Returns:
        loss: Scalar loss value
        num_valid: Number of valid samples in batch
    """
    # Reconstruction loss: Huber loss for smoother robustness
    recon_loss = F.huber_loss(predictions, targets, reduction="none")  # (batch, seq_len, channels)
    
    # Apply modality mask: only compute loss on masked patches
    if modality_mask is not None:
        if patch_len is None:
            raise ValueError("patch_len must be provided when modality_mask is used")
        
        # Expand patch-level mask to sequence-level
        # modality_mask: (batch, num_patches, channels)
        # Repeat each mask value patch_len times
        b, num_patches, c = modality_mask.shape
        seq_mask = modality_mask.unsqueeze(2).repeat(1, 1, patch_len, 1)  # (batch, num_patches, patch_len, channels)
        seq_mask = seq_mask.reshape(b, num_patches * patch_len, c)  # (batch, seq_len, channels)
        
        recon_loss = recon_loss * seq_mask  # Zero out loss for unmasked patches
        # Normalize by number of masked elements
        num_masked = seq_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)  # (batch, 1, 1)
        l1 = recon_loss.sum(dim=(1, 2), keepdim=True) / num_masked  # (batch, 1, 1)
        l1 = l1.squeeze()  # (batch,)
    else:
        l1 = recon_loss.mean(dim=(1, 2))

    # First-difference loss (finite differences)
    if (
        diff_weight > 0.0
        and predictions.size(1) > 1
        and predictions.size(1) == targets.size(1)
    ):
        dp = predictions[:, 1:] - predictions[:, :-1]
        dt = targets[:, 1:] - targets[:, :-1]
        diff_loss = F.huber_loss(dp, dt, reduction="none")  # (batch, seq_len-1, channels)
        
        # Apply modality mask to diff loss as well
        if modality_mask is not None:
            # Use seq_mask but trim first timestep to match diff shape
            seq_mask_diff = seq_mask[:, 1:, :]  # (batch, seq_len-1, channels)
            diff_loss = diff_loss * seq_mask_diff
            num_masked = seq_mask_diff.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            dl1 = diff_loss.sum(dim=(1, 2), keepdim=True) / num_masked
            dl1 = dl1.squeeze()  # (batch,)
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
    mask_config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Run one training epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device to train on (cpu/cuda)
        diff_weight: Weight for first-difference loss term
        modality_dropout_p: Not used (kept for compatibility)
        mask_config: Dictionary with mask parameters (mask_type, mask_ratio, etc.)

    Returns:
        Dictionary with "loss" key
    """
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        physio = batch["physio"].to(device)  # (batch, seq_len, channels)
        b, seq_len, num_channels = physio.shape

        # Generate patch-level mask using configured strategy
        if mask_config is not None:
            # Calculate number of patches
            patch_len = model.tokenizer.patch_len
            num_patches = seq_len // patch_len
            
            # Generate mask: (batch, num_patches, num_channels)
            # 1 = masked (compute loss), 0 = visible (skip loss)
            patch_mask = get_mask(
                batch_size=b,
                num_patches=num_patches,
                num_channels=num_channels,
                device=device,
                **mask_config
            )
            
            # Skip if no patches were masked in this batch
            if patch_mask.sum() == 0:
                continue
        else:
            patch_mask = None

        optimizer.zero_grad(set_to_none=True)
        # Model forward pass with patch_mask
        recon = model(physio, patch_mask=patch_mask)
        
        # Pass patch_len for proper loss masking
        patch_len = model.tokenizer.patch_len if mask_config is not None else None
        loss, valid = masked_reconstruction_loss(
            recon, physio, mask=None, diff_weight=diff_weight, 
            modality_mask=patch_mask, patch_len=patch_len
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
    mask_config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: Validation data loader (or None to skip)
        device: Device to evaluate on
        diff_weight: Weight for first-difference loss term
        modality_dropout_p: Not used (kept for compatibility)
        mask_config: Dictionary with mask parameters (mask_type, mask_ratio, etc.)

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
            physio = batch["physio"].to(device)  # (batch, seq_len, channels)
            b, seq_len, num_channels = physio.shape

            # Generate patch-level mask using configured strategy
            if mask_config is not None:
                # Calculate number of patches
                patch_len = model.tokenizer.patch_len
                num_patches = seq_len // patch_len
                
                # Generate mask: (batch, num_patches, num_channels)
                # 1 = masked (compute loss), 0 = visible (skip loss)
                patch_mask = get_mask(
                    batch_size=b,
                    num_patches=num_patches,
                    num_channels=num_channels,
                    device=device,
                    **mask_config
                )
                
                # Skip if no patches were masked in this batch
                if patch_mask.sum() == 0:
                    continue
            else:
                patch_mask = None

            # Model forward pass with patch_mask
            recon = model(physio, patch_mask=patch_mask)
            
            # Pass patch_len for proper loss masking
            patch_len = model.tokenizer.patch_len if mask_config is not None else None
            loss, valid = masked_reconstruction_loss(
                recon, physio, mask=None, diff_weight=diff_weight,
                modality_mask=patch_mask, patch_len=patch_len
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
    mask_config: Optional[Dict] = None,
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
        modality_dropout_p: Not used (kept for compatibility)
        mask_config: Dictionary with mask parameters (mask_type, mask_ratio, etc.)

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
            mask_config=mask_config,
        )

        # Validate
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            diff_weight=diff_weight,
            modality_dropout_p=0.0,
            mask_config=mask_config,
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
