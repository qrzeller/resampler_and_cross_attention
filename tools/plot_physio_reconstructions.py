#!/usr/bin/env python3
"""Plot per-channel physio reconstructions using a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PERCIEVER_DIR = REPO_ROOT / "Perceiver"
if str(PERCIEVER_DIR) not in sys.path:
    sys.path.insert(0, str(PERCIEVER_DIR))

from eatmint_multimodal_perceiver import (  # type: ignore  # noqa: E402
    AudioFeatureExtractor,
    EATMINTConfig,
    EATMINTDataset,
    check_modality_availability,
    print_availability_summary,
)
from physiopooled_eatmint_multimodal_perceiver import (  # type: ignore  # noqa: E402
    PhysioPooledEATMINTDataset,
    PhysioPooledEATMINTPerceiver,
)


def parse_args() -> argparse.Namespace:
    device_default = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Plot the reconstruction of each physio signal when that channel is masked from the input."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "physio_50hz_noaudio_checkpoint.pt"),
        help="Path to the trained checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "tools" / "plots" / "physio_recon"),
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--sample-idx",
        dest="sample_indices",
        type=int,
        action="append",
        help="Specific dataset indices to visualize. Can be provided multiple times.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="If no sample indices are passed, visualize the first N samples.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=8.0,
        help="Window size in seconds (must match training).",
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        default=4.0,
        help="Hop size in seconds (must match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=device_default,
        help="Computation device (cuda or cpu).",
    )
    parser.add_argument(
        "--no-precomputed-audio",
        action="store_true",
        help="Force extraction of audio features on-the-fly (slow).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all participant data into memory before iterating.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


@dataclass
class DatasetBundle:
    dataset: PhysioPooledEATMINTDataset
    use_precomputed_audio: bool


def resolve_precomputed_audio_usage(config: EATMINTConfig, no_precomputed_flag: bool) -> bool:
    if no_precomputed_flag:
        return False
    precomputed_dir = config.sound_features_dir
    if not precomputed_dir.exists():
        return False
    return any(precomputed_dir.glob("*.npy"))


def prepare_dataset(args: argparse.Namespace) -> DatasetBundle:
    data_root = REPO_ROOT / "data" / "researchdata"
    config = EATMINTConfig(
        data_root=str(data_root),
        window_size_sec=args.window_size,
        hop_size_sec=args.hop_size,
    )

    availability_df = check_modality_availability(config)
    usable_df = print_availability_summary(availability_df)

    use_precomputed = resolve_precomputed_audio_usage(config, args.no_precomputed_audio)
    audio_extractor = None
    if not use_precomputed:
        print("Precomputed audio features not found. Using on-the-fly extraction via Wav2Vec2.")
        audio_extractor = AudioFeatureExtractor(device=args.device)

    dataset = PhysioPooledEATMINTDataset(
        config=config,
        availability_df=usable_df,
        audio_extractor=audio_extractor,
        window_size_sec=config.window_size_sec,
        hop_size_sec=config.hop_size_sec,
        target_fs=50.0,
        physio_target_fs=500.0,
        min_modalities=2,
        preload=args.preload,
        use_precomputed_audio=use_precomputed,
    )
    print(f"Dataset windows: {len(dataset)}")
    return DatasetBundle(dataset=dataset, use_precomputed_audio=use_precomputed)


def build_model(bundle: DatasetBundle, device: str, checkpoint_path: str) -> PhysioPooledEATMINTPerceiver:
    dataset = bundle.dataset
    model = PhysioPooledEATMINTPerceiver(
        hidden_dim=256,
        latent_dim=256,
        num_latents=256,
        num_self_attention_layers=6,
        num_cross_attention_layers=1,
        num_heads=8,
        audio_dim=768,
        physio_input_dim=len(EATMINTDataset.PHYSIO_SIGNALS),
        physio_conv_dim=16,
        physio_seq_len=dataset.physio_window_samples,
        physio_downsample_factor=dataset.physio_downsample_factor,
        openface_dim=len(EATMINTDataset.OPENFACE_COLS),
        eyetracker_dim=len(EATMINTDataset.EYETRACKER_COLS),
        seq_len=dataset.window_samples,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def iter_sample_indices(dataset_len: int, indices: Sequence[int] | None, num_samples: int) -> Iterable[int]:
    if indices:
        for idx in indices:
            if 0 <= idx < dataset_len:
                yield idx
            else:
                print(f"Skipping index {idx} (out of range).")
        return
    limit = min(num_samples, dataset_len)
    for idx in range(limit):
        yield idx


def reconstruct_and_plot(
    model: PhysioPooledEATMINTPerceiver,
    dataset: PhysioPooledEATMINTDataset,
    sample_idx: int,
    device: str,
    output_dir: Path,
    show: bool,
) -> Path:
    sample = dataset[sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

    channel_names = EATMINTDataset.PHYSIO_SIGNALS
    time_axis = (
        torch.arange(dataset.physio_window_samples, dtype=torch.float32)
        / float(dataset.physio_target_fs)
    ).numpy()
    n_channels = len(channel_names)
    n_cols = 2
    n_rows = (n_channels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    ground_truth = sample["physio"].numpy()

    model_inputs = batch.copy()
    for ch_idx, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        masked_physio = model_inputs["physio"].clone()
        masked_physio[:, :, ch_idx] = 0.0
        with torch.no_grad():
            outputs = model(
                audio=model_inputs["audio"],
                physio=masked_physio,
                openface=model_inputs["openface"],
                eyetracker=model_inputs["eyetracker"],
                modality_mask=model_inputs["modality_mask"],
            )
        reconstructed = outputs["physio"].detach().cpu().numpy()[0, :, ch_idx]
        target = ground_truth[:, ch_idx]
        ax.plot(time_axis, target, label="target", linewidth=1.5)
        ax.plot(time_axis, reconstructed, label="recon", linewidth=1.2, linestyle="--")
        ax.set_title(f"{ch_name} (masked input)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z-score")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    for leftover_ax in axes[n_channels:]:
        leftover_ax.axis("off")

    fig.suptitle(f"Sample {sample_idx} | Physio channel reconstructions", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"50hz_{sample_idx:05d}_physio_recon.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()
    bundle = prepare_dataset(args)
    model = build_model(bundle, args.device, args.checkpoint)
    dataset = bundle.dataset

    output_dir = Path(args.output_dir)
    plotted_paths: List[Path] = []
    for sample_idx in iter_sample_indices(len(dataset), args.sample_indices, args.num_samples):
        plotted_paths.append(
            reconstruct_and_plot(
                model=model,
                dataset=dataset,
                sample_idx=sample_idx,
                device=args.device,
                output_dir=output_dir,
                show=args.show,
            )
        )

    if not plotted_paths:
        print("No plots were generated.")


if __name__ == "__main__":
    main()
