"""Plot helpers for physio reconstruction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import torch


def _select_indices(
    dataset_size: int,
    requested: Optional[Sequence[int]] = None,
    max_samples: int = 4,
) -> Sequence[int]:
    if requested:
        return [idx for idx in requested if 0 <= idx < dataset_size][:max_samples]
    return list(range(min(max_samples, dataset_size)))


def _walk_attr(dataset, attr: str):
    if hasattr(dataset, attr):
        value = getattr(dataset, attr)
        if value is not None:
            return value
    inner = getattr(dataset, "dataset", None)
    if inner is not None:
        return _walk_attr(inner, attr)
    return None


def _resolve_feature_names(
    dataset,
    provided: Optional[Sequence[str]],
    num_channels: int,
) -> Sequence[str]:
    if provided:
        names = list(provided)
    else:
        names = (
            _walk_attr(dataset, "feature_names")
            or _walk_attr(dataset, "physio_feature_names")
            or _walk_attr(dataset, "PHYSIO_SIGNALS")
            or []
        )
        if not names and hasattr(dataset, "PHYSIO_SIGNALS"):
            names = getattr(dataset, "PHYSIO_SIGNALS")
        names = list(names)

    if len(names) < num_channels:
        names = names + [f"feat_{idx}" for idx in range(len(names), num_channels)]
    return names[:num_channels]


def plot_physio_reconstructions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    sample_indices: Optional[Iterable[int]] = None,
    max_samples: int = 4,
    save_path: str = "checkpoints/physio_resampler_recon.png",
    feature_names: Optional[Sequence[str]] = None,
) -> Path:
    """Render ground-truth vs. reconstructed physio signals for quick inspection."""

    model.eval()
    indices = _select_indices(len(dataset), sample_indices, max_samples)
    if not indices:
        raise ValueError("No samples available for plotting")

    first_sample = dataset[indices[0]]
    num_channels = first_sample["physio"].shape[1]
    names = _resolve_feature_names(dataset, feature_names, num_channels)

    rows = len(indices)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=False)
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
                )

            ax.set_title(f"Sample {idx}")
            ax.set_ylabel("z-norm signal")
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time steps")
    fig.tight_layout()

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
