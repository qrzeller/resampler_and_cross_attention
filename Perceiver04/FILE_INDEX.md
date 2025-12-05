# Perceiver03 - File Index

**Total: 13 files, ~72 KB**

## Core Implementation (6 files, 38.5 KB)

| File | Size | Purpose |
|------|------|---------|
| `fourier_features.py` | 2.8K | Fourier positional encoding (Hz frequencies, log-spaced bands) |
| `perceiver_model.py` | 9.3K | Core Perceiver architecture (cross-attn, self-attn, decoder) |
| `dataset.py` | 9.5K | Dataset classes (base, resampled, synthetic physio) |
| `training.py` | 5.8K | Training utilities (loss, train_epoch, evaluate, train_model) |
| `visualization.py` | 4.8K | Plotting (reconstructions, training history) |
| `main.py` | 13K | CLI entry point with full argument parser |

## Package Configuration (2 files, 1 KB)

| File | Size | Purpose |
|------|------|---------|
| `__init__.py` | 905B | Package exports and version |
| `requirements.txt` | 85B | Minimal dependencies (6 packages) |

## Documentation (4 files, 22.2 KB)

| File | Size | Purpose |
|------|------|---------|
| `README.md` | 8.7K | Comprehensive documentation with examples |
| `QUICKSTART.md` | 3.7K | Quick reference for common tasks |
| `DEPENDENCIES.md` | 2.8K | Installation guide and troubleshooting |
| `PROJECT_SUMMARY.md` | 6.8K | High-level project overview |

## Validation (1 file, 3.4 KB)

| File | Size | Purpose |
|------|------|---------|
| `verification.py` | 3.4K | Comprehensive system verification script |

---

## How to Get Started

1. **Read first**: `QUICKSTART.md` (5 minutes)
2. **Install**: `pip install -r requirements.txt`
3. **Run**: `python main.py --help`
4. **Train**: `python main.py --epochs 10 --num-windows 200`
5. **Deep dive**: `README.md`

## Module Dependencies

```
main.py
  ├── perceiver_model.py
  │   ├── fourier_features.py
  │   └── torch, torch.nn
  ├── dataset.py
  │   ├── numpy, scipy, torch
  │   └── fractions, pathlib
  ├── training.py
  │   ├── torch, torch.nn, torch.optim
  │   └── tqdm, typing
  └── visualization.py
      ├── matplotlib, torch
      └── pathlib, typing
```

## External Dependencies

All in `requirements.txt`:
- `torch>=1.9.0` - Deep learning framework
- `numpy>=1.19.0` - Numerical operations
- `scipy>=1.5.0` - Signal processing (polyphase resampling)
- `matplotlib>=3.3.0` - Plotting
- `tqdm>=4.50.0` - Progress bars
- `pandas>=1.1.0` - Data handling (optional)

## File Statistics

| Metric | Value |
|--------|-------|
| Total lines | ~1,900 |
| Python code | ~1,800 |
| Docstrings | ~400 |
| Comments | ~200 |
| Documentation | ~2,800 lines |
| Total size | ~72 KB |
| Avg file size | 5.5 KB |

## What Each File Contains

### `fourier_features.py`
- `FourierFeatures` class
- Proper Fourier encoding in Hz
- Log-spaced frequency bands

### `perceiver_model.py`
- `FeedForward` - Position-wise MLP
- `CrossAttentionBlock` - Latents attend to tokens
- `DecoderCrossAttentionBlock` - Decode queries attend to latents
- `PerceiverResampler` - Full architecture

### `dataset.py`
- `PhysioTimeSeriesDataset` - Base class with synthetic generation
- `PhysioResampledDataset` - Downsampling (polyphase/avg + smoothing)
- `SyntheticPhysioDataset` - Convenience class (5-channel physio)

### `training.py`
- `masked_reconstruction_loss()` - MSE + first-difference loss
- `train_epoch()` - Single training loop
- `evaluate()` - Validation loop
- `train_model()` - Full training pipeline

### `visualization.py`
- `plot_physio_reconstructions()` - GT vs recon comparison
- `plot_training_history()` - Loss curves
- Helper functions for indices and names

### `main.py`
- Complete CLI with ~20 arguments
- Configuration documentation
- End-to-end training pipeline
- Checkpoint saving

### `__init__.py`
- Package exports (14 items)
- Version info

### `requirements.txt`
- 6 core dependencies with minimum versions

### `verification.py`
- Comprehensive system checks (7 categories)
- File existence verification
- Import testing
- Model instantiation
- Dataset creation
- Forward pass validation
- Dependency listing
- GPU support detection

---

## Quick Links

- **Installation**: See `DEPENDENCIES.md`
- **Usage**: See `QUICKSTART.md`
- **Full docs**: See `README.md`
- **Overview**: See `PROJECT_SUMMARY.md`
- **API ref**: Check module docstrings in Python files
- **Examples**: See `main.py --help`

## Common Commands

```bash
# Validate installation
python verification.py

# Show CLI help
python main.py --help

# Train with defaults
python main.py

# Train custom config
python main.py --epochs 20 --latent-dim 512

# Use as library
python -c "from perceiver_model import PerceiverResampler; ..."
```

---

**Status**: ✅ All files created and tested  
**Last updated**: December 5, 2025  
**Python**: 3.11.13  
**PyTorch**: 2.9.1+cu128
