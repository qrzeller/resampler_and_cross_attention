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
- Supports both synthetic data and real EATMINT physio data

Usage:
    # Synthetic data (for testing)
    python main.py --epochs 10 --batch-size 8

    # Real EATMINT data
    python main.py --use-eatmint --data-root ../data/researchdata --epochs 20

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
from visualization import (
    plot_physio_reconstructions,
    plot_training_history,
    plot_modality_dropout_reconstructions,
)
from run_logger import create_run_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perceiver IO autoencoder for physio time-series with proper Fourier features"
    )

    # Data source selection
    parser.add_argument(
        "--use-eatmint",
        action="store_true",
        help="Use real EATMINT physio data instead of synthetic data",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/EATMINT/researchdata",
        help="Path to EATMINT data root (default: ../data/EATMINT/researchdata)",
    )

    # Dataset parameters
    parser.add_argument(
        "--window-size",
        type=float,
        default=12.0,
        help="Window size in seconds (default: 12.0)",
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        default=6.0,
        help="Hop size in seconds for EATMINT data (default: 6.0)",
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
        help="Number of synthetic windows to generate (default: 200, ignored with --use-eatmint)",
    )
    parser.add_argument(
        "--num-signals",
        type=int,
        default=5,
        help="Number of signal channels (default: 5)",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all data into memory (recommended for EATMINT)",
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
        default=1e-4,
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
    parser.add_argument(
        "--modality-dropout-p",
        type=float,
        default=0.2,
        help="Probability of dropping each modality/channel during training (default: 0.2)",
    )

    # I/O parameters
    parser.add_argument(
        "--run-output",
        type=str,
        default="runs",
        help="Base directory for run outputs (default: runs)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for this run (appended to timestamp)",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="Comment or goal description for this run (saved in report)",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="Save checkpoint to this path (optional, or use --run-output)",
    )
    parser.add_argument(
        "--recon-fig",
        type=str,
        default=None,
        help="Save reconstruction figure to this path (default: auto in run folder)",
    )
    parser.add_argument(
        "--history-fig",
        type=str,
        default=None,
        help="Save training history figure to this path (default: auto in run folder)",
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

    # Setup run logger for automatic documentation
    run_logger = create_run_logger(
        output_base=args.run_output,
        run_name=args.run_name,
        comment=args.comment,
    )
    paths = run_logger.get_paths()
    
    # Use run logger paths unless explicitly overridden
    recon_fig_path = args.recon_fig if args.recon_fig else paths["recon_fig"]
    history_fig_path = args.history_fig if args.history_fig else paths["history_fig"]
    checkpoint_path = args.save_checkpoint if args.save_checkpoint else paths["checkpoint"]

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
    print(f"Run output: {paths['run_dir']}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    if args.comment:
        print(f"Comment: {args.comment}")
    print()

    # =========================================================================
    # Dataset setup
    # =========================================================================
    if args.use_eatmint:
        # Use real EATMINT physio data
        print("Loading EATMINT physio dataset...")
        print(f"  Data root: {args.data_root}")
        from eatmint_dataset import create_eatmint_dataset

        dataset = create_eatmint_dataset(
            data_root=args.data_root,
            window_size_sec=args.window_size,
            hop_size_sec=args.hop_size,
            target_fs=args.target_fs,
            physio_fs=args.physio_fs,
            smoothing_kernel=args.smoothing_kernel,
            downsample_strategy=args.downsample_strategy,
            preload=args.preload,
        )
        feature_names = dataset.get_signal_names()
        num_signals = len(feature_names)
        print(f"  Signals: {feature_names}")
    else:
        # Use synthetic data
        print("Creating synthetic dataset...")
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
        feature_names = [f"Signal_{i}" for i in range(args.num_signals)]
        num_signals = args.num_signals

    seq_len = dataset.window_samples
    print(f"  Total windows: {len(dataset)}")
    print(f"  Window size: {seq_len} samples @ {args.target_fs} Hz")
    print(f"  Signal channels: {num_signals}")
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
        signal_dim=num_signals,
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
    print(f"  Modality dropout p: {args.modality_dropout_p}")
    print()

    history = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.epochs,
        diff_weight=args.diff_loss_weight,
        modality_dropout_p=args.modality_dropout_p,
    )

    # Store config and history in run logger
    run_logger.set_config(vars(args))
    run_logger.set_history(history)
    run_logger.add_metric("total_parameters", num_params)

    # =========================================================================
    # Post-training visualization
    # =========================================================================
    print("\n" + "=" * 78)
    print("Post-training visualization and analysis")
    print("=" * 78)

    # Plot training history
    if history:
        print(f"\nSaving training history plot to {history_fig_path}...")
        plot_training_history(history, save_path=history_fig_path)

    # Plot reconstructions
    plot_dataset = val_ds if val_ds is not None else train_ds
    if plot_dataset is not None and len(plot_dataset) > 0:
        try:
            print(f"Saving reconstruction plot to {recon_fig_path}...")
            plot_physio_reconstructions(
                model,
                plot_dataset,
                device,
                max_samples=4,
                save_path=recon_fig_path,
                feature_names=feature_names,
            )
        except Exception as exc:
            print(f"Warning: failed to plot reconstructions ({exc})")

        # Plot modality dropout reconstructions
        try:
            modality_dropout_path = recon_fig_path.replace(
                "physio_recon.png", "modality_dropout_recon.png"
            )
            print(f"Saving modality dropout plot to {modality_dropout_path}...")
            plot_modality_dropout_reconstructions(
                model,
                plot_dataset,
                device,
                max_samples=4,
                save_path=modality_dropout_path,
                feature_names=feature_names,
            )
        except Exception as exc:
            print(f"Warning: failed to plot modality dropout reconstructions ({exc})")

    # =========================================================================
    # Checkpoint saving
    # =========================================================================
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": vars(args),
            "feature_names": feature_names,
        },
        ckpt_path,
    )

    # =========================================================================
    # Generate run report
    # =========================================================================
    print(f"\nGenerating run report...")
    report_path = run_logger.generate_report()
    print(f"Report saved to: {report_path}")

    print("\n" + "=" * 78)
    print("Training complete!")
    print(f"All outputs saved to: {paths['run_dir']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
