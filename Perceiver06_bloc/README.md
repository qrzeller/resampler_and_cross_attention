# Perceiver05: Masked Autoencoder with Learnable Mask Tokens

A masked autoencoding implementation of Perceiver IO for multi-modal physiological signal reconstruction. This version introduces **learnable mask tokens** and **modality dropout** for robust cross-modal learning.

## Key Features

### New in Perceiver05
- **🎭 Learnable mask tokens**: Explicit embeddings replace dropped channels (vs zeros)
- **🎲 Modality dropout**: Randomly drop entire physiological channels during training
- **🔍 Masked loss computation**: Loss computed ONLY on dropped channels
- **🔄 Smart residual handling**: Residual base uses masked input for proper reconstruction
- **🧠 Cross-modal learning**: Model learns to infer missing modalities from available ones

### Architecture Improvements
- **Channel-aware masking**: Model knows which channels are masked vs naturally zero
- **Fixed loss computation**: Prevents cheating by computing loss only on masked modalities
- **Proper broadcasting**: Correct handling of channel-wise masks across time
- **Skip empty batches**: Avoids training on batches where no channels are dropped

## How It Works

### Masked Autoencoding Process

1. **Input**: 5 physiological channels `[ECG, GSR, BVP, TEMP, ACC]`
2. **Modality dropout**: Randomly drop channels (e.g., `[ECG, 0, 0, TEMP, ACC]`)
3. **Mask tokens**: Replace dropped channels with learnable embeddings
4. **Encoder**: Build latent representation from available channels + mask tokens
5. **Decoder**: Reconstruct ALL channels from latent space
6. **Loss**: Compute reconstruction error ONLY on dropped channels (GSR, BVP)

### Why Mask Tokens?

Without mask tokens, the model cannot distinguish:
- "Zero because this channel is masked" ❌
- "Zero because the signal value is actually zero" ✓

Learnable mask tokens provide an explicit signal: **"reconstruct this channel from context"**

## Usage

### Quick Start

```bash
cd Perceiver05

# Train with modality dropout (20% per channel)
python main.py --use-eatmint --epochs 50 --batch-size 8 \
  --modality-dropout-p 0.2 --latent-dim 128 --num-latents 32

# Train at 50Hz with residuals
python main.py --use-eatmint --epochs 50 --target-fs 50 \
  --diff-loss-weight 0.3 --run-name "masked_50hz"
```

### Example: High Masking Rate

```bash
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

Fourier feature encoding for time coordinates.python main.py --use-eatmint --modality-dropout-p 0.4 \
  --epochs 50 --batch-size 8 --latent-dim 128
```

## File Structure

```
Perceiver05/
├── __init__.py                 # Package initialization
├── main.py                     # Entry point script with full CLI
├── perceiver_model.py          # Perceiver with learnable mask tokens
├── fourier_features.py         # Fourier feature encoding
├── dataset.py                  # Dataset classes
├── eatmint_dataset.py          # EATMINT physiological data loader
├── training.py                 # Masked training loops with modality dropout
├── visualization.py            # Plotting utilities
└── README.md                   # This file
```

## Key Parameters

### Masking & Training
- `--modality-dropout-p`: Probability of dropping each channel (default: 0.2)
- `--no-residual`: Disable residual connections (not recommended with masking)
- `--diff-loss-weight`: Weight for first-difference loss term (default: 0.3)

### Architecture  
- `--latent-dim`: Dimension of latent space (default: 128)
- `--num-latents`: Number of latent vectors (default: 32)
- `--self-layers`: Number of self-attention layers (default: 4)
- `--num-heads`: Number of attention heads (default: 8)

### Data
- `--target-fs`: Target sampling frequency in Hz (default: 50.0)
- `--window-size`: Window duration in seconds (default: 12.0)
- `--use-eatmint`: Use real EATMINT dataset instead of synthetic

## Technical Details

### Loss Computation with Masking

```python
# For each batch:
# 1. Random channel mask: [1, 1, 0, 0, 1] (keep ch 0,1,4; drop ch 2,3)
# 2. Input with mask tokens: [ECG, GSR, [MASK], [MASK], ACC]
# 3. Model reconstructs: [ECG', GSR', BVP', TEMP', ACC']
# 4. Loss computed ONLY on ch 2,3 (dropped channels)
loss = reconstruction_loss[dropped_channels].mean()
```

### Residual Connections with Masking

When residuals are enabled:
- **Kept channels**: `output = delta + original_signal` (refinement)
- **Dropped channels**: `output = delta + 0 = delta` (full reconstruction)

The model learns:
- Small deltas for kept channels (residual helps)
- Full signal values for dropped channels (from latent inference)

## Improvements Over Perceiver04

### Bug Fixes
1. ✅ Loss now computed ONLY on masked channels (was computing on all)
2. ✅ Residual base uses masked input, not original (prevents cheating)
3. ✅ Skip batches where no channels are dropped (avoids zero loss)
4. ✅ Proper broadcasting for channel masks across time dimension

### New Features
1. 🎭 Learnable mask tokens distinguish "masked" from "zero value"
2. 🎯 Explicit channel-wise masking in model architecture
3. 📊 Better validation of masked autoencoding setup

## Module Documentation

### `perceiver_model.py`

Core Perceiver architecture with mask token support.

**Key additions:**
- `self.mask_token`: Learnable (1, 1, signal_dim) parameter
- `channel_mask` argument in `forward()`: Binary mask for dropped channels
- Automatic replacement of dropped channels with mask tokens

```python
from perceiver_model import PerceiverResampler

model = PerceiverResampler(
    signal_dim=5,
    seq_len=600,
    sample_rate_hz=50.0,
    latent_dim=128,
    num_latents=32,
    use_residual=True
)

# Forward with channel masking
# channel_mask: (batch, 1, 5) where 1=keep, 0=use mask token
output = model(input_signals, channel_mask=channel_mask)
```

### `training.py`

Masked training utilities.

**Key changes:**
- `modality_mask` parameter in `masked_reconstruction_loss()`
- Channel dropout logic creates both `channel_mask` and `loss_mask`
- Loss normalization by number of dropped channels
- Batch skipping when no masking occurs

from training import train_model

history = train_model(
    model, train_loader, val_loader, 
    optimizer, device, 
    num_epochs=50,
    diff_weight=0.3,
    modality_dropout_p=0.2  # 20% dropout per channel
)
```

## Comparison with Other Versions

| Version | Key Feature | Use Case |
|---------|-------------|----------|
| **Perceiver03** | Clean baseline | Standard autoencoding |
| **Perceiver04** | Modality dropout | Cross-modal learning (buggy) |
| **Perceiver05** | Mask tokens + fixes | Robust masked autoencoding ✅ |

## Configuration Defaults

- **Window size**: 12s
- **Target FS**: 50 Hz  
- **Physio FS**: 512 Hz
- **Latent dim**: 128
- **Num latents**: 32
- **Self-attn layers**: 4
- **Num heads**: 8
- **Fourier bands**: 32
- **Dropout**: 0.05
- **Diff loss weight**: 0.3
- **Modality dropout**: 0.2
- **Learning rate**: 1e-4
- **Epochs**: 50
- **Batch size**: 8
- **Val fraction**: 10%

## Expected Behavior

### Training Losses
- **Without masking**: Loss ~0.02-0.08 (standard reconstruction)
- **With masking (20% dropout)**: Loss ~0.06-0.15 (harder task)
- **Val loss similar to train loss**: Model generalizes well

### What to Watch For
- ❌ **Zero loss after epoch 1**: Bug in loss computation
- ❌ **Val loss << train loss**: Model is cheating (computing loss on kept channels)
- ✅ **Val loss > train loss**: Normal for masked autoencoding
- ✅ **Losses decrease steadily**: Model is learning cross-modal inference

## Performance Tips

- Use `--modality-dropout-p 0.2` to 0.4 for best cross-modal learning
- Increase `--num-latents` (64-128) for better capacity with masking
- Use `--diff-loss-weight 0.3` for temporal coherence
- **Keep residuals enabled** - they help refine kept channels
- Higher `--self-layers` (6-8) improves inference quality
- Validation loss will be higher than Perceiver03 (expected - harder task!)

## Troubleshooting

### Loss goes to zero
- Check that `modality_dropout_p > 0`
- Verify batches have dropped channels (check logs)
- Ensure loss computed only on dropped channels

### Poor reconstruction quality  
- Increase `--num-latents` and `--latent-dim`
- Add more `--self-layers` for better reasoning
- Lower `--modality-dropout-p` initially, then increase

### Model not learning cross-modal patterns
- Increase dropout rate (`--modality-dropout-p 0.3` to 0.5)
- Verify mask tokens are being used (check model parameters)
- Ensure sufficient model capacity

## Citation

Based on:
- Jaegle et al., "Perceiver: General Perception with Iterative Attention" (2021)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022)

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
