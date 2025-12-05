# Quick Start Guide - Perceiver03

## Setup

```bash
# Install dependencies
cd Perceiver03
pip install -r requirements.txt
```

## Running the Code

### 1. Basic Training (with defaults)
```bash
python main.py
```

### 2. Training with Custom Parameters
```bash
python main.py \
  --epochs 20 \
  --batch-size 8 \
  --num-windows 500 \
  --latent-dim 256 \
  --num-latents 128 \
  --self-layers 6 \
  --fourier-bands 32
```

### 3. Multi-Rate Physio (512 Hz → 50 Hz)
```bash
python main.py \
  --window-size 12.0 \
  --target-fs 50.0 \
  --physio-fs 512.0 \
  --smoothing-kernel 5 \
  --downsample-strategy polyphase \
  --epochs 20
```

### 4. Save Checkpoint
```bash
python main.py \
  --epochs 10 \
  --save-checkpoint checkpoints/my_model.pt \
  --recon-fig results/recon.png \
  --history-fig results/history.png
```

### 5. See All Options
```bash
python main.py --help
```

## Module Structure

- **`fourier_features.py`** - Fourier positional encoding (time in seconds, frequencies in Hz)
- **`perceiver_model.py`** - Core Perceiver architecture (cross-attention, self-attention, decoder)
- **`dataset.py`** - Dataset classes (PhysioTimeSeriesDataset, PhysioResampledDataset, SyntheticPhysioDataset)
- **`training.py`** - Training loops (train_epoch, evaluate, train_model)
- **`visualization.py`** - Plotting (reconstructions, training history)
- **`main.py`** - Entry point with full CLI
- **`__init__.py`** - Package exports

## Using as a Library

```python
from perceiver_model import PerceiverResampler
from dataset import SyntheticPhysioDataset
from training import train_model
from visualization import plot_physio_reconstructions

# Create dataset
dataset = SyntheticPhysioDataset(
    window_size_sec=12.0,
    target_fs=50.0,
    num_windows=200
)

# Create model
model = PerceiverResampler(
    signal_dim=5,
    seq_len=600,
    sample_rate_hz=50.0,
    latent_dim=256,
    num_latents=128
)

# Train (see training.py for full API)
from torch.utils.data import DataLoader
from torch.optim import AdamW

loader = DataLoader(dataset, batch_size=8)
optimizer = AdamW(model.parameters(), lr=1e-3)

# ... training loop
```

## Key Features Tested ✓

- ✓ All modules import successfully
- ✓ PyTorch 2.9.1 with CUDA support
- ✓ Model forward pass works (811K parameters)
- ✓ Synthetic dataset generation
- ✓ Polyphase downsampling
- ✓ Training loop converges (loss decreases)
- ✓ Reconstruction visualization
- ✓ Training history plotting
- ✓ Full CLI with sensible defaults

## Outputs

When running training, you'll get:
- `training_history.png` - Loss curves
- `physio_recon.png` - Ground-truth vs reconstructed signals
- Optional checkpoint: `checkpoints/model.pt`

## Documentation

See `README.md` for:
- Detailed module documentation
- Mathematical background (Fourier features, residual decoding)
- Configuration reference
- Troubleshooting guide
- How to extend the code

## Differences from Perceiver02

| Aspect | Perceiver02 | Perceiver03 |
|--------|------------|-----------|
| **Structure** | Monolithic file | Modular 6-file design |
| **Dependencies** | Full EATMINT codebase | Minimal, standalone |
| **Dataset** | Complex EATMINT classes | Clean synthetic + resampling |
| **Main entry** | No clear CLI | Full argparse interface |
| **Reusability** | Hard to extract pieces | Easy imports as library |
| **Documentation** | Limited | Comprehensive README |
| **Testing** | Not provided | Full validation tests |

## Next Steps

1. **Custom data**: Subclass `PhysioTimeSeriesDataset` to load your own data
2. **Model variants**: Modify `PerceiverResampler` for your architecture
3. **Integration**: Import components into your own training pipeline
4. **Production**: Save checkpoints and load them for inference
