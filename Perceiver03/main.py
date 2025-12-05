# -*- coding: utf-8 -*-
"""
Perceiver IO-style physio autoencoder (standalone, modular version).

Main entry point for training with proper Fourier features and clean architecture.

Features:
- Time positions in seconds (not [-1, 1]) with proper Fourier encoding
- Frequencies in Hz (Nyquist-based defaults)
- Residual decoding to fight over-smoothing
- Optional first-difference loss term
- Modular design: perceiver_model, dataset, training, visualization

Usage:
    python main.py --epochs 10 --batch-size 8
    python main.py --epochs 5 --num-windows 50 --target-fs 50 --physio-fs 512
    python main.py --help
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, Subset

from dataset import PhysioResampledDataset, SyntheticPhysioDataset
from perceiver_model import PerceiverResampler
from training import train_model
from visualization import plot_physio_reconstructions, plot_training_history


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perceiver IO autoencoder for physio time-series with proper Fourier features"
    )

    # Dataset parameters
    parser.add_argument(
        "--window-size",
        type=float,
        default=12.0,
        help="Window size in seconds (default: 12.0)",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=50.0,
        help="Target sampling rate in Hz (default: 50.0)",
    )
    parser.add_argument(
        "--physio-fs",
        type=float,
        default=512.0,
        help="Native physio sampling rate in Hz (default: 512.0)",
    )
    parser.add_argument(
        "--smoothing-kernel",
        type=int,
        default=0,
        help="Smoothing kernel size (0=disabled)",
    )
    parser.add_argument(
        "--downsample-strategy",
        choices=["polyphase", "avg"],
        default="polyphase",
        help="Downsampling strategy",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=200,
        help="Number of synthetic windows to generate (default: 200)",
    )
    parser.add_argument(
        "--num-signals",
        type=int,
        default=5,
        help="Number of signal channels (default: 5)",
    )

    # Model parameters
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent space dimension (default: 256)",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=128,
        help="Number of latent vectors (default: 128)",
    )
    parser.add_argument(
        "--self-layers",
        type=int,
        default=4,
        help="Number of self-attention layers (default: 4)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--fourier-bands",
        type=int,
        default=32,
        help="Number of Fourier frequency bands (default: 32)",
    )
    parser.add_argument(
        "--max-freq-hz",
        type=float,
        default=None,
        help="Maximum frequency in Hz (default: Nyquist)",
    )
    parser.add_argument(
        "--min-freq-hz",
        type=float,
        default=None,
        help="Minimum frequency in Hz (default: ~1/window_duration)",
    )
    parser.add_argument(
        "--decoder-len",
        type=int,
        default=None,
        help="Decoder output length (default: same as encoder)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="Dropout probability (default: 0.05)",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable residual decoding",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--diff-loss-weight",
        type=float,
        default=0.2,
        help="Weight for first-difference loss (default: 0.2, set to 0 to disable)",
    )

    # I/O parameters
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Save checkpoint to this path (optional)",
    )
    parser.add_argument(
        "--recon-fig",
        type=str,
        default="Perceiver03/physio_recon.png",
        help="Save reconstruction figure to this path",
    )
    parser.add_argument(
        "--history-fig",
        type=str,
        default="Perceiver03/training_history.png",
        help="Save training history figure to this path",
    )

    # Other
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'cpu', or None (auto-detect)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training loop."""
    args = parse_args()

    # Setup device and seeds
    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 78)
    print("Perceiver IO Physio Autoencoder (Standalone, Modular)")
    print("=" * 78)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print()

    # =========================================================================
    # Dataset setup
    # =========================================================================
    print("Creating dataset...")
    dataset = PhysioResampledDataset(
        window_size_sec=args.window_size,
        hop_size_sec=args.window_size,  # No overlap for synthetic
        target_fs=args.target_fs,
        physio_fs=args.physio_fs,
        num_signals=args.num_signals,
        num_windows=args.num_windows,
        smoothing_kernel=args.smoothing_kernel,
        downsample_strategy=args.downsample_strategy,
        seed=args.seed,
        use_dummy_data=True,
    )

    seq_len = dataset.window_samples
    print(f"  Total windows: {len(dataset)}")
    print(f"  Window size: {seq_len} samples @ {args.target_fs} Hz")
    print(f"  Signal channels: {args.num_signals}")
    print()

    # Calculate effective Fourier parameters
    window_duration_sec = (seq_len - 1) / max(args.target_fs, 1e-6)
    max_freq_hz = (args.target_fs / 2.0) if args.max_freq_hz is None else args.max_freq_hz
    min_freq_hz = (
        (1.0 / max(window_duration_sec, 1e-6))
        if args.min_freq_hz is None
        else args.min_freq_hz
    )
    print(f"Fourier encoding:")
    print(f"  min_freq: {min_freq_hz:.4f} Hz")
    print(f"  max_freq: {max_freq_hz:.2f} Hz (Nyquist: {args.target_fs/2:.2f} Hz)")
    print(f"  num_bands: {args.fourier_bands}")
    print()

    # Split into train/val
    if args.val_fraction > 0 and len(dataset) > 1:
        val_size = max(1, int(len(dataset) * args.val_fraction))
        train_size = len(dataset) - val_size
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if val_ds is not None
        else None
    )

    print(f"Train set: {len(train_ds)} windows")
    if val_ds is not None:
        print(f"Val set: {len(val_ds)} windows")
    print()

    # =========================================================================
    # Model setup
    # =========================================================================
    print("Creating model...")
    model = PerceiverResampler(
        signal_dim=args.num_signals,
        seq_len=seq_len,
        sample_rate_hz=args.target_fs,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
        num_self_attn_layers=args.self_layers,
        num_heads=args.num_heads,
        num_fourier_bands=args.fourier_bands,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        decoder_seq_len=args.decoder_len or seq_len,
        dropout=args.dropout,
        use_residual=(not args.no_residual),
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Num latents: {args.num_latents}")
    print(f"  Self-attn layers: {args.self_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Residual: {'enabled' if not args.no_residual else 'disabled'}")
    print()

    # =========================================================================
    # Training setup
    # =========================================================================
    print("Training...")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    print(f"  Optimizer: AdamW (lr={args.lr}, weight_decay=1e-2)")
    print(f"  Epochs: {args.epochs}")
    print(f"  Diff loss weight: {args.diff_loss_weight}")
    print()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        diff_weight=args.diff_loss_weight,
    )

    # =========================================================================
    # Post-training visualization
    # =========================================================================
    print("\n" + "=" * 78)
    print("Post-training visualization and analysis")
    print("=" * 78)

    # Plot training history
    if history:
        print(f"\nSaving training history plot to {args.history_fig}...")
        plot_training_history(history, save_path=args.history_fig)

    # Plot reconstructions
    plot_dataset = val_ds if val_ds is not None else train_ds
    if plot_dataset is not None and len(plot_dataset) > 0:
        try:
            print(f"Saving reconstruction plot to {args.recon_fig}...")
            plot_physio_reconstructions(
                model,
                plot_dataset,
                device,
                max_samples=4,
                save_path=args.recon_fig,
            )
        except Exception as exc:
            print(f"Warning: failed to plot reconstructions ({exc})")

    # =========================================================================
    # Checkpoint saving (optional)
    # =========================================================================
    if args.save_checkpoint:
        print(f"\nSaving checkpoint to {args.save_checkpoint}...")
        ckpt_path = Path(args.save_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": vars(args),
            },
            ckpt_path,
        )

    print("\n" + "=" * 78)
    print("Training complete!")
    print("=" * 78)


if __name__ == "__main__":
    main()
