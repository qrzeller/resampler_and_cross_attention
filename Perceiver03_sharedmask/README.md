# Perceiver03: Clean, Modular Perceiver IO for Physio Time-Series

A refactored, production-ready implementation of Perceiver IO for physiological signal autoencoding. This is a cleaner, modular version of the code in `Perceiver02`.

## Key Improvements

### Architecture
- **Modular design**: Separated into logical components (model, dataset, training, visualization)
- **Proper Fourier features**: Time in seconds, frequencies in Hz (Nyquist-based defaults)
- **Clean codebase**: No unnecessary dependencies from the original EATMINT multimodal implementation
- **Standalone datasets**: Synthetic datasets for easy testing without external data
- **Comprehensive CLI**: Full argparse interface with sensible defaults

### Features
- ✅ Residual decoding to fight over-smoothing
- ✅ First-difference loss term for temporal dynamics
- ✅ Flexible attention architecture (configurable layers, heads, dimensions)
- ✅ Polyphase/average pooling downsampling strategies
- ✅ Smooth post-downsampling
- ✅ Visualization utilities for reconstruction inspection
- ✅ Training history plotting

## File Structure

```
Perceiver03/
├── __init__.py                 # Package initialization
├── main.py                     # Entry point script with full CLI
├── perceiver_model.py          # Core Perceiver architecture
├── fourier_features.py         # Fourier feature encoding
├── dataset.py                  # Dataset classes (synthetic + resampling)
├── training.py                 # Training and evaluation loops
├── visualization.py            # Plotting utilities
└── README.md                   # This file
```

## Usage

### Quick Start: Train on Synthetic Data

```bash
cd Perceiver03

# Train with defaults (200 synthetic windows, 10 epochs)
python main.py

# Train with custom parameters
python main.py --epochs 20 --batch-size 16 --latent-dim 512 --num-windows 500

# See all options
python main.py --help
```

### Example: Multi-Rate Physio Dataset

Train a model that downsamples high-rate physio (512 Hz) to 50 Hz:

```bash
python main.py \
  --window-size 12.0 \
  --target-fs 50.0 \
  --physio-fs 512.0 \
  --smoothing-kernel 5 \
  --downsample-strategy polyphase \
  --num-windows 200 \
  --epochs 20 \
  --batch-size 8 \
  --latent-dim 256 \
  --num-latents 128 \
  --self-layers 6 \
  --diff-loss-weight 0.2
```

### Example: Save Checkpoint

```bash
python main.py \
  --epochs 10 \
  --save-checkpoint checkpoints/perceiver_physio.pt \
  --recon-fig results/recon.png \
  --history-fig results/history.png
```

## Module Documentation

### `perceiver_model.py`

Core Perceiver architecture with cross-attention and self-attention blocks.

**Main class:** `PerceiverResampler`
- Takes signals → encodes with Fourier features → processes via latent arrays → decodes outputs
- Supports optional residual connections and variable decoder lengths

**Helper classes:**
- `FourierFeatures`: Log-spaced frequency encoding in Hz
- `CrossAttentionBlock`: Latents attend to input tokens
- `DecoderCrossAttentionBlock`: Reconstruction queries attend to latents
- `FeedForward`: Position-wise MLP

### `fourier_features.py`

Fourier feature encoding for time coordinates.

```python
from fourier_features import FourierFeatures

fourier = FourierFeatures(
    num_bands=32,
    min_freq_hz=0.1,
    max_freq_hz=25.0,
    include_positions=True
)

# Encode time positions (batch, seq_len, 1)
positions = torch.randn(8, 500, 1)  # 500 samples, in seconds
encoded = fourier(positions)  # (8, 500, 1 + 32*2)
```

### `dataset.py`

Dataset classes for physio time-series.

**Classes:**
- `PhysioTimeSeriesDataset`: Base class with optional synthetic data generation
- `PhysioResampledDataset`: Downsamples high-rate data to target rate (with polyphase or avg pooling)
- `SyntheticPhysioDataset`: Convenience class for standard 5-channel physio (ECG, GSR, Temp, Plet, Resp)

```python
from dataset import SyntheticPhysioDataset

# Create 200 synthetic 12-second windows at 50 Hz
dataset = SyntheticPhysioDataset(
    window_size_sec=12.0,
    target_fs=50.0,
    num_windows=200,
    seed=42
)

sample = dataset[0]
print(sample["physio"].shape)  # (600, 5)
```

### `training.py`

Training utilities.

**Functions:**
- `masked_reconstruction_loss()`: MSE + optional first-difference loss
- `train_epoch()`: Single training loop
- `evaluate()`: Validation loop
- `train_model()`: Full training pipeline

```python
from training import train_model

history = train_model(
    model, train_loader, val_loader, 
    optimizer, device, 
    num_epochs=10,
    diff_weight=0.2
)
```

### `visualization.py`

Plotting utilities.

**Functions:**
- `plot_physio_reconstructions()`: Side-by-side GT vs reconstructed signals
- `plot_training_history()`: Loss curves over epochs

```python
from visualization import plot_physio_reconstructions, plot_training_history

# Plot reconstructions for 4 samples
plot_physio_reconstructions(
    model, dataset, device,
    max_samples=4,
    save_path="recon.png",
    feature_names=["ECG", "GSR", "Temp", "Plet", "Resp"]
)

# Plot training history
plot_training_history(history, save_path="history.png")
```

## Configuration Defaults

- **Window size**: 12s
- **Target FS**: 50 Hz
- **Physio FS**: 512 Hz
- **Latent dim**: 256
- **Num latents**: 128
- **Self-attn layers**: 4
- **Num heads**: 8
- **Fourier bands**: 32
- **Dropout**: 0.05
- **Diff loss weight**: 0.2
- **Learning rate**: 1e-3
- **Epochs**: 10
- **Batch size**: 8
- **Val fraction**: 10%

## Key Concepts

### Fourier Features

Instead of learnable position embeddings, we use proper Fourier features:

$$\text{features} = [t, \sin(2\pi f_1 t), \cos(2\pi f_1 t), \ldots, \sin(2\pi f_k t), \cos(2\pi f_k t)]$$

where:
- $t$ is time in **seconds**
- $f_i$ are log-spaced frequencies in **Hz**
- Default $f_{\max} = \text{Nyquist} = \text{target_fs}/2$
- Default $f_{\min} \approx 1/\text{window_duration}$ (captures slow trends)

### Residual Decoding

Instead of predicting signals directly, we predict residuals/deltas:

$$\hat{x} = \Delta + x_{\text{input}}$$

This helps the model preserve input structure and improves convergence.

### First-Difference Loss

Optional auxiliary loss on temporal derivatives:

$$\mathcal{L}_{\text{diff}} = \|\Delta\hat{x} - \Delta x\|_2^2$$

where $\Delta = x_{t+1} - x_t$. Helps fight over-smoothing.

## Extending the Code

### Custom Datasets

Subclass `PhysioTimeSeriesDataset`:

```python
from dataset import PhysioTimeSeriesDataset

class MyDataset(PhysioTimeSeriesDataset):
    def _generate_dummy_data(self):
        # Load your data here
        self.windows = [...]
```

### Custom Models

Subclass `PerceiverResampler`:

```python
from perceiver_model import PerceiverResampler

class MyModel(PerceiverResampler):
    def __init__(self, ...):
        super().__init__(...)
        # Add custom layers
```

### Custom Training Loops

Use `train_epoch()` and `evaluate()` as building blocks:

```python
from training import train_epoch, evaluate

for epoch in range(num_epochs):
    train_metrics = train_epoch(model, train_loader, opt, device)
    val_metrics = evaluate(model, val_loader, device)
    # Custom logging/scheduling here
```

## Design Philosophy

This codebase prioritizes:

1. **Clarity**: Each file has a single responsibility
2. **Modularity**: Easy to import and extend individual components
3. **Completeness**: No hidden dependencies or assumptions
4. **Flexibility**: Reasonable defaults with full CLI control
5. **Reproducibility**: Seed control and deterministic operations

## Performance Tips

- Use `--smoothing-kernel 3` to 5 for polyphase downsampling to reduce artifacts
- Increase `--self-layers` (6-8) for longer windows or complex dynamics
- Use `--diff-loss-weight 0.1` to 0.3 to balance smoothing vs. fidelity
- Residual decoding usually helps; avoid `--no-residual` unless testing
- Multi-GPU training: wrap model with `nn.DataParallel()` before passing to `train_model()`

## Troubleshooting

**Q: Loss stays high/doesn't decrease**
- Try reducing learning rate: `--lr 5e-4`
- Increase latent dim: `--latent-dim 512`
- Increase self-attention layers: `--self-layers 6`

**Q: Reconstructions are too smooth**
- Increase diff loss weight: `--diff-loss-weight 0.5`
- Increase num Fourier bands: `--fourier-bands 64`
- Add smoothing kernel: `--smoothing-kernel 3`

**Q: Out of memory**
- Reduce batch size: `--batch-size 4`
- Reduce latent dim: `--latent-dim 128`
- Reduce num latents: `--num-latents 64`

**Q: How do I use my own data?**
- Subclass `PhysioTimeSeriesDataset` and override `_generate_dummy_data()` with your data loading
- Or use `PhysioResampledDataset` with pre-computed high-rate data

## References

- Perceiver IO: https://arxiv.org/abs/2107.14795
- Original EATMINT: Available in parent `Perceiver02/` directory

## License

Same as parent repository.
