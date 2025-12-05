# -*- coding: utf-8 -*-
"""Perceiver IO module for physio time-series autoencoders."""

from fourier_features import FourierFeatures
from perceiver_model import PerceiverResampler, CrossAttentionBlock, DecoderCrossAttentionBlock, FeedForward
from dataset import PhysioTimeSeriesDataset, PhysioResampledDataset, SyntheticPhysioDataset
from training import train_epoch, evaluate, train_model, masked_reconstruction_loss
from visualization import plot_physio_reconstructions, plot_training_history

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
]

__version__ = "1.0.0"
