# -*- coding: utf-8 -*-
"""Utility script to load a checkpoint and render reconstruction plots.

Examples
--------
python run_recon_plot.py \
    --checkpoint runs/20251227_223920_compression_long_50hz_32x32/checkpoint.pt \
    --save-path runs/20251227_223920_compression_long_50hz_32x32/recon.png \
    --max-samples 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from perceiver_model import PerceiverResampler
from training import masked_reconstruction_loss  # noqa: F401 (imported for completeness)
from visualization import plot_physio_reconstructions
from dataset import PhysioResampledDataset


def build_model(cfg: dict, device: torch.device) -> PerceiverResampler:
    seq_len = int(round(cfg["window_size"] * cfg["target_fs"]))
    decoder_len = cfg.get("decoder_len")
    decoder_len = int(round(cfg["window_size"] * cfg["target_fs"])) if decoder_len is None else decoder_len

    model = PerceiverResampler(
        signal_dim=cfg["num_signals"],
        seq_len=seq_len,
        sample_rate_hz=cfg["target_fs"],
        latent_dim=cfg["latent_dim"],
        num_latents=cfg["num_latents"],
        num_self_attn_layers=cfg["self_layers"],
        num_heads=cfg["num_heads"],
        num_fourier_bands=cfg["fourier_bands"],
        min_freq_hz=cfg.get("min_freq_hz"),
        max_freq_hz=cfg.get("max_freq_hz"),
        decoder_seq_len=decoder_len,
        dropout=cfg["dropout"],
        use_residual=not cfg.get("no_residual", False),
    ).to(device)
    return model


def make_dataset(cfg: dict):
    if cfg.get("use_eatmint", False):
        from eatmint_dataset import create_eatmint_dataset

        base = create_eatmint_dataset(
            data_root=cfg.get("data_root", "../data/EATMINT/researchdata"),
            window_size_sec=cfg["window_size"],
            hop_size_sec=cfg["hop_size"],
            target_fs=cfg["target_fs"],
            physio_fs=cfg["physio_fs"],
            smoothing_kernel=cfg["smoothing_kernel"],
            downsample_strategy=cfg["downsample_strategy"],
            preload=cfg.get("preload", True),
            cache_windows=cfg.get("preload", True),
        )

        # Inline mask wrapper (contiguous chunk per channel, matches training behavior)
        if cfg.get("masking", False):
            import numpy as np
            from torch.utils.data import Dataset as _DS

            class MaskWrapper(_DS):
                def __init__(self, base_ds, mask_frac: float):
                    self.base = base_ds
                    self.mask_frac = float(mask_frac)

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    item = self.base[idx]
                    phys = item.get("physio") if isinstance(item, dict) else item
                    seq_len, channels = phys.shape
                    mask = np.zeros((seq_len, channels), dtype=bool)
                    for ch in range(channels):
                        clen = max(1, int(round(seq_len * self.mask_frac)))
                        start = 0 if clen >= seq_len else np.random.randint(0, seq_len - clen + 1)
                        mask[start : start + clen, ch] = True
                    out = {"physio": phys, "mask": torch.from_numpy(mask)}
                    if isinstance(item, dict):
                        # keep extra fields if any
                        for k, v in item.items():
                            if k not in out:
                                out[k] = v
                    return out

                @property
                def window_samples(self):
                    return getattr(self.base, "window_samples", None)

            return MaskWrapper(base, cfg["mask_frac"])

        return base

    # Synthetic / resampled dataset
    return PhysioResampledDataset(
        window_size_sec=cfg["window_size"],
        hop_size_sec=cfg["hop_size"],
        target_fs=cfg["target_fs"],
        physio_fs=cfg["physio_fs"],
        num_signals=cfg["num_signals"],
        num_windows=cfg.get("num_windows", 50),
        smoothing_kernel=cfg["smoothing_kernel"],
        downsample_strategy=cfg["downsample_strategy"],
        seed=cfg.get("seed", 0),
        use_dummy_data=True,
        mask_frac=cfg.get("mask_frac", 0.0),
        apply_mask=cfg.get("masking", False),
    )


def main():
    parser = argparse.ArgumentParser(description="Render reconstructions from a saved checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/20251227_223920_compression_long_50hz_32x32/checkpoint.pt",
        help="Path to checkpoint.pt",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save the reconstruction figure (default: alongside checkpoint)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Maximum samples to plot",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional sample indices to plot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu/cuda). Default: auto",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    feature_names = ckpt.get("feature_names")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = make_dataset(cfg)

    save_path = args.save_path
    if save_path is None:
        save_path = ckpt_path.parent / "recon.png"

    out_path = plot_physio_reconstructions(
        model,
        dataset,
        device,
        sample_indices=args.indices,
        max_samples=args.max_samples,
        save_path=str(save_path),
        feature_names=feature_names,
    )
    print(f"Saved reconstruction plot to: {out_path}")


if __name__ == "__main__":
    main()
