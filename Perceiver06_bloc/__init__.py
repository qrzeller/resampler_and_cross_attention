# -*- coding: utf-8 -*-
"""Perceiver IO module for physio time-series autoencoders."""

from fourier_features import FourierFeatures
from perceiver_model import PerceiverResampler, CrossAttentionBlock, DecoderCrossAttentionBlock, FeedForward
from dataset import PhysioTimeSeriesDataset, PhysioResampledDataset, SyntheticPhysioDataset
from training import train_epoch, evaluate, train_model, masked_reconstruction_loss
from visualization import plot_physio_reconstructions, plot_training_history

# EATMINT real data loading (optional import)
try:
    from eatmint_dataset import (
        EATMINTConfig,
        EATMINTPhysioDataset,
        create_eatmint_dataset,
        check_physio_availability,
    )
    _EATMINT_AVAILABLE = True
except ImportError:
    _EATMINT_AVAILABLE = False

__all__ = [
    "FourierFeatures",
    "PerceiverResampler",
    "CrossAttentionBlock",
    "DecoderCrossAttentionBlock",
    "FeedForward",
    "PhysioTimeSeriesDataset",
    "PhysioResampledDataset",
    "SyntheticPhysioDataset",
    "train_epoch",
    "evaluate",
    "train_model",
    "masked_reconstruction_loss",
    "plot_physio_reconstructions",
    "plot_training_history",
    # EATMINT
    "EATMINTConfig",
    "EATMINTPhysioDataset",
    "create_eatmint_dataset",
    "check_physio_availability",
]

__version__ = "1.0.0"
