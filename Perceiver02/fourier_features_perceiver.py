"""
Perceiver IO-style physio autoencoder (physio-only) — with *proper* Fourier features.

Fixes included:
- Uses time positions in **seconds** (not [-1, 1]) and Fourier features as:
      [sin(2π f t), cos(2π f t)]
  where f is in **Hz**.
- max_freq defaults to Nyquist: target_fs / 2
- min_freq defaults to ~one cycle per window: 1 / window_duration
  (important for slow physio trends)
- Default window size is 12s (hop 6s)
- Residual decoding (predict delta + input)
- Optional first-difference loss term to fight over-smoothing

Usage:
    python physio_perceiver_resampler.py --epochs 5 --batch-size 8
"""

from __future__ import annotations

import argparse
import math
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.auto import tqdm

from scipy import signal

from eatmint_multimodal_perceiver import (
    EATMINTConfig,
    EATMINTDataset,
    check_modality_availability,
    print_availability_summary,
)
from physiopooled_eatmint_multimodal_perceiver import PhysioPooledEATMINTDataset
from visualization.reconstruction_plotter import plot_physio_reconstructions


# =============================================================================
# Dataset
# =============================================================================


class PhysioResampledDataset(PhysioPooledEATMINTDataset):
    """Downsample high-rate physio streams onto a low-rate grid."""

    def __init__(
        self,
        *args,
        smoothing_kernel: int = 0,
        downsample_strategy: str = "polyphase",
        drop_missing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, skip_audio=True, min_modalities=1, **kwargs)
        self.smoothing_kernel = max(0, smoothing_kernel)
        self.downsample_strategy = downsample_strategy
        self.drop_missing = drop_missing

        frac = Fraction(int(round(self.target_fs)), int(round(self.physio_target_fs)))
        self._poly_up = frac.numerator
        self._poly_down = frac.denominator

    def _polyphase_downsample(self, physio: torch.Tensor) -> torch.Tensor:
        arr = physio.detach().cpu().numpy()
        resampled = signal.resample_poly(
            arr,
            self._poly_up,
            self._poly_down,
            axis=0,
            padtype="mean",
        )
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
        ) / float(self.smoothing_kernel)
        padding = self.smoothing_kernel // 2
        return (
            F.conv1d(
                physio.transpose(0, 1).unsqueeze(0),
                kernel,
                padding=padding,
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        has_physio = bool(sample["modality_mask"][1].item())

        if not has_physio:
            if self.drop_missing:
                raise ValueError("Encountered missing physio despite drop_missing=True")
            physio = torch.zeros(
                self.window_samples,
                len(EATMINTDataset.PHYSIO_SIGNALS),
                dtype=torch.float32,
            )
        else:
            physio = self._pool_physio(sample["physio"])

        return {
            "physio": physio,
            "mask": torch.tensor(has_physio, dtype=torch.bool),
        }


# =============================================================================
# Perceiver resampler
# =============================================================================


class FourierFeatures(nn.Module):
    """
    Proper Fourier features for continuous coordinates (time in seconds):
        [t, sin(2π f1 t), cos(2π f1 t), ..., sin(2π fk t), cos(2π fk t)]
    Frequencies f are in Hz.

    We use log-spaced frequencies between min_freq_hz and max_freq_hz.
    """

    def __init__(
        self,
        num_bands: int,
        min_freq_hz: float,
        max_freq_hz: float,
        include_positions: bool = True,
        pos_dim: int = 1,
    ) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.min_freq_hz = float(min_freq_hz)
        self.max_freq_hz = float(max_freq_hz)
        self.include_positions = include_positions
        self.pos_dim = int(pos_dim)

    @property
    def output_dim(self) -> int:
        base = self.pos_dim if self.include_positions else 0
        return base + (self.pos_dim * self.num_bands * 2)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (..., pos_dim) where pos_dim=1, values in seconds
        if self.num_bands <= 0:
            return positions if self.include_positions else positions.new_zeros(*positions.shape[:-1], 0)

        # Keep frequency range sane and strictly positive for logspace
        f_lo = max(self.min_freq_hz, 1e-6)
        f_hi = max(self.max_freq_hz, f_lo)

        freqs = torch.logspace(
            math.log10(f_lo),
            math.log10(f_hi),
            self.num_bands,
            base=10.0,
            device=positions.device,
            dtype=positions.dtype,
        )
        shape = [1] * (positions.dim() - 1) + [self.num_bands]
        freqs = freqs.view(*shape)  # broadcast across batch/time

        angles = positions.unsqueeze(-1) * freqs * (2.0 * math.pi)  # 2π f t
        sin = angles.sin()
        cos = angles.cos()
        emb = torch.cat([sin, cos], dim=-1)
        emb = emb.reshape(*positions.shape[:-1], -1)
        if self.include_positions:
            emb = torch.cat([positions, emb], dim=-1)
        return emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """Latents attend to input tokens."""

    def __init__(self, latent_dim: int, input_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim,
            dropout=dropout,
        )
        self.ff = FeedForward(latent_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(latents, tokens, tokens, key_padding_mask=mask)
        latents = self.norm1(latents + attn_out)
        latents = self.norm2(latents + self.ff(latents))
        return latents


class DecoderCrossAttentionBlock(nn.Module):
    """Decoder queries attend to latent array to produce outputs."""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.ff = FeedForward(latent_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(queries, latents, latents)
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ff(queries))
        return queries


class PerceiverResampler(nn.Module):
    """Minimal Perceiver IO-style autoencoder for time-series."""

    def __init__(
        self,
        signal_dim: int,
        seq_len: int,
        sample_rate_hz: float,
        latent_dim: int = 256,
        num_latents: int = 256,
        num_self_attn_layers: int = 8,
        num_heads: int = 8,
        num_fourier_bands: int = 32,
        max_freq_hz: Optional[float] = None,  # default -> Nyquist
        min_freq_hz: Optional[float] = None,  # default -> 1/window_duration
        decoder_seq_len: Optional[int] = None,
        dropout: float = 0.05,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.decoder_seq_len = int(decoder_seq_len or seq_len)
        self.sample_rate_hz = float(sample_rate_hz)
        self.use_residual = bool(use_residual)

        # Duration covered by indexed samples [0 .. seq_len-1] at sample_rate_hz
        self.window_duration_sec = (self.seq_len - 1) / max(self.sample_rate_hz, 1e-6)

        if max_freq_hz is None:
            max_freq_hz = self.sample_rate_hz / 2.0  # Nyquist
        if min_freq_hz is None:
            min_freq_hz = 1.0 / max(self.window_duration_sec, 1e-6)  # ~1 cycle per window

        self.encoder_pos = FourierFeatures(num_fourier_bands, min_freq_hz, max_freq_hz, include_positions=True)
        self.decoder_pos = FourierFeatures(num_fourier_bands, min_freq_hz, max_freq_hz, include_positions=True)

        # Time positions in seconds
        t_enc = (torch.arange(self.seq_len, dtype=torch.float32) / max(self.sample_rate_hz, 1e-6)).view(1, self.seq_len, 1)
        # If decoder len differs, spread queries across the same duration
        if self.decoder_seq_len == self.seq_len:
            t_dec = t_enc.clone()
        else:
            t_dec = torch.linspace(0.0, self.window_duration_sec, self.decoder_seq_len, dtype=torch.float32).view(1, self.decoder_seq_len, 1)

        self.register_buffer("encoder_positions", t_enc, persistent=False)
        self.register_buffer("decoder_positions", t_dec, persistent=False)

        token_dim = signal_dim + self.encoder_pos.output_dim
        self.input_proj = nn.Linear(token_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        self.encoder_cross = CrossAttentionBlock(latent_dim, latent_dim, num_heads, dropout)
        self.self_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=latent_dim * 4,
                    batch_first=True,
                    activation="gelu",
                    dropout=dropout,
                )
                for _ in range(num_self_attn_layers)
            ]
        )

        self.query_proj = nn.Linear(self.decoder_pos.output_dim, latent_dim)
        self.decoder_cross = DecoderCrossAttentionBlock(latent_dim, num_heads, dropout)
        self.output_proj = nn.Linear(latent_dim, signal_dim)

    def forward(self, series: torch.Tensor, return_latents: bool = False):
        batch, seq_len, _ = series.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, received {seq_len}")

        enc_pos = self.encoder_positions.to(series.device, dtype=series.dtype).expand(batch, -1, -1)
        enc_features = self.encoder_pos(enc_pos)
        tokens = torch.cat([series, enc_features], dim=-1)
        tokens = self.input_proj(tokens)

        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        latents = self.encoder_cross(latents, tokens)
        for layer in self.self_layers:
            latents = layer(latents)

        dec_pos = self.decoder_positions.to(series.device, dtype=series.dtype).expand(batch, -1, -1)
        dec_features = self.decoder_pos(dec_pos)
        queries = self.query_proj(dec_features)
        decoded = self.decoder_cross(queries, latents)

        delta = self.output_proj(decoded)
        outputs = delta + series if (self.use_residual and self.decoder_seq_len == self.seq_len) else delta

        if return_latents:
            return outputs, latents
        return outputs


# =============================================================================
# Training helpers
# =============================================================================


def masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    diff_weight: float = 0.2,
) -> Tuple[torch.Tensor, float]:
    mse = F.mse_loss(predictions, targets, reduction="none").mean(dim=(1, 2))

    if diff_weight > 0.0 and predictions.size(1) > 1 and predictions.size(1) == targets.size(1):
        dp = predictions[:, 1:] - predictions[:, :-1]
        dt = targets[:, 1:] - targets[:, :-1]
        dmse = F.mse_loss(dp, dt, reduction="none").mean(dim=(1, 2))
        per_sample = mse + diff_weight * dmse
    else:
        per_sample = mse

    weights = mask.float()
    valid = weights.sum()
    if valid <= 0:
        return predictions.sum() * 0.0, 0.0
    return (per_sample * weights).sum() / valid, float(valid.item())


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    diff_weight: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        physio = batch["physio"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        recon = model(physio)
        loss, valid = masked_reconstruction_loss(recon, physio, mask, diff_weight=diff_weight)
        if valid == 0.0:
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
    diff_weight: float,
) -> Dict[str, float]:
    if dataloader is None:
        return {"loss": float("nan")}

    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            physio = batch["physio"].to(device)
            mask = batch["mask"].to(device)
            recon = model(physio)
            loss, valid = masked_reconstruction_loss(recon, physio, mask, diff_weight=diff_weight)
            if valid == 0.0:
                continue
            total_loss += loss.item()
            steps += 1
    return {"loss": total_loss / max(1, steps)}


# =============================================================================
# Utilities
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Physio-only Perceiver resampler (proper Fourier features)")
    parser.add_argument("--window-size", type=float, default=12.0)
    parser.add_argument("--hop-size", type=float, default=6.0)
    parser.add_argument("--target-fs", type=float, default=50.0)
    parser.add_argument("--physio-fs", type=float, default=512.0)
    parser.add_argument("--smoothing-kernel", type=int, default=0)
    parser.add_argument("--downsample-strategy", choices=["polyphase", "avg"], default="polyphase")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-latents", type=int, default=128)
    parser.add_argument("--self-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--fourier-bands", type=int, default=32)

    # Hz (not cycles/window anymore)
    parser.add_argument("--max-freq-hz", type=float, default=None)  # default -> Nyquist
    parser.add_argument("--min-freq-hz", type=float, default=None)  # default -> ~1/window_duration

    parser.add_argument("--decoder-len", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--diff-loss-weight", type=float, default=0.2)
    parser.add_argument("--no-residual", action="store_true")

    parser.add_argument("--save-path", type=str, default="Perceiver02/physio512_perceiver_resampler.pt")
    parser.add_argument("--recon-fig", type=str, default="Perceiver02/physio512_resampler_recon.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def split_dataset(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Optional[Dataset]]:
    if val_fraction <= 0 or len(dataset) < 2:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).parent.parent
    data_root = script_dir / "data" / "researchdata"
    config = EATMINTConfig(
        data_root=str(data_root),
        window_size_sec=args.window_size,
        hop_size_sec=args.hop_size,
    )

    print("=" * 78)
    print("Physio-only Perceiver Resampler (12s default, proper Fourier in Hz)")
    print("=" * 78)
    print(f"Device: {device}")
    print(f"window_size={args.window_size}s | hop_size={args.hop_size}s | target_fs={args.target_fs}Hz")

    availability = check_modality_availability(config)
    usable = print_availability_summary(availability)

    dataset = PhysioResampledDataset(
        config=config,
        availability_df=usable,
        window_size_sec=args.window_size,
        hop_size_sec=args.hop_size,
        target_fs=args.target_fs,
        physio_target_fs=args.physio_fs,
        smoothing_kernel=args.smoothing_kernel,
        downsample_strategy=args.downsample_strategy,
        preload=args.preload,
    )
    seq_len = dataset.window_samples
    print(f"Total windows: {len(dataset)} | window samples: {seq_len}")

    # Print the effective defaults for min/max freq
    window_duration_sec = (seq_len - 1) / max(args.target_fs, 1e-6)
    max_freq_hz = (args.target_fs / 2.0) if args.max_freq_hz is None else args.max_freq_hz
    min_freq_hz = (1.0 / max(window_duration_sec, 1e-6)) if args.min_freq_hz is None else args.min_freq_hz
    print(f"Fourier freqs: min={min_freq_hz:.4f} Hz | max={max_freq_hz:.2f} Hz (Nyquist={args.target_fs/2:.2f})")
    print(f"diff_loss_weight={args.diff_loss_weight} | residual={'off' if args.no_residual else 'on'}")

    if args.max_windows is not None:
        limit = min(args.max_windows, len(dataset))
        dataset = Subset(dataset, list(range(limit)))
        print(f"Clipped dataset to first {limit} windows")

    train_ds, val_ds = split_dataset(dataset, args.val_fraction, args.seed)

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

    model = PerceiverResampler(
        signal_dim=len(EATMINTDataset.PHYSIO_SIGNALS),
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
        use_residual=(not args.no_residual),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    history: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_metrics = train_epoch(model, train_loader, optimizer, device, diff_weight=args.diff_loss_weight)
        val_metrics = evaluate(model, val_loader, device, diff_weight=args.diff_loss_weight)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
            }
        )
        val_str = f"{val_metrics['loss']:.4f}" if not math.isnan(val_metrics["loss"]) else "n/a"
        print(f"  train_loss={train_metrics['loss']:.4f} | val_loss={val_str}")

    plot_source = val_ds if val_ds is not None else train_ds
    if plot_source is not None and len(plot_source) > 0:
        try:
            print("\nRendering reconstruction figure...")
            plot_physio_reconstructions(
                model,
                plot_source,
                device,
                save_path=args.recon_fig,
                feature_names=EATMINTDataset.PHYSIO_SIGNALS,
            )
            print(f"Saved reconstructions to {args.recon_fig}")
        except Exception as exc:
            print(f"Warning: failed to plot reconstructions ({exc})")

    # If you want saving back, uncomment:
    # ckpt_path = Path(args.save_path)
    # ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "history": history,
    #         "config": vars(args),
    #     },
    #     ckpt_path,
    # )
    # print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
