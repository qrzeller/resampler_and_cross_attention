# Perceiver06_bloc Usage Guide

## Quick Start

Basic training with default patch settings:
```bash
python Perceiver06_bloc/main.py --use-eatmint --epochs 20 --batch-size 8
```

## Default Configuration

- **Patch length**: 64 samples (1.28s at 50Hz)
- **Masking strategy**: MAE (remove masked patches from encoder)
- **Mask type**: mixed (combines span + channel drop)
- **Mask ratio**: 0.5 (50% of patches masked)
- **Channel drop probability**: 0.2 (20% chance to drop entire modality)
- **Span length range**: 2-8 patches for temporal masking

## Key Arguments

### Patch Configuration
- `--patch-len SAMPLES`: Samples per patch (default: 64)
  - Smaller = more tokens, finer granularity
  - Larger = fewer tokens, faster training
  - Must divide evenly into window size

- `--use-conv-frontend`: Use 1D convolution for patch tokenization
  - Default: linear projection
  - Conv can capture local patterns better

### Masking Strategy
- `--mask-strategy {mae,bert}`: How to handle masked patches
  - `mae`: Remove masked patches from encoder (compute efficient, default)
  - `bert`: Replace with learnable [MASK] token

- `--mask-type {random,span,channel_drop,mixed}`: Masking pattern
  - `random`: Random independent patches
  - `span`: Contiguous temporal blocks (good for temporal reasoning)
  - `channel_drop`: Drop entire modalities (good for cross-modal learning)
  - `mixed`: Combines span + channel_drop (default)

- `--mask-ratio RATIO`: Fraction of patches to mask (default: 0.5)
  - Higher = harder task, more training signal
  - Lower = easier, may underfit

### Mixed Masking Parameters
- `--channel-drop-prob PROB`: Probability of dropping a full modality (default: 0.2)
- `--span-len-min PATCHES`: Minimum span length (default: 2)
- `--span-len-max PATCHES`: Maximum span length (default: 8)

### Model Architecture
- `--latent-dim DIM`: Dimension of latent representations (default: 256)
- `--num-latents N`: Number of latent tokens (default: 128)
- `--self-layers N`: Number of self-attention layers (default: 3)
- `--num-heads N`: Number of attention heads (default: 8)
- `--dropout RATE`: Dropout rate (default: 0.1)

### Fourier Features
- `--fourier-bands N`: Number of frequency bands (default: 64)
- `--min-freq-hz HZ`: Minimum frequency (default: 0.01 Hz)
- `--max-freq-hz HZ`: Maximum frequency (default: 25.0 Hz, Nyquist/2)

## Example Configurations

### High masking ratio for aggressive self-supervised learning
```bash
python Perceiver06_bloc/main.py --use-eatmint --epochs 50 \
    --mask-ratio 0.75 --mask-type mixed --mask-strategy mae
```

### Temporal span masking (like TS-TCC)
```bash
python Perceiver06_bloc/main.py --use-eatmint --epochs 50 \
    --mask-type span --span-len-min 4 --span-len-max 16 \
    --mask-ratio 0.5 --mask-strategy mae
```

### Channel dropout focused (for cross-modal learning)
```bash
python Perceiver06_bloc/main.py --use-eatmint --epochs 50 \
    --mask-type channel_drop --channel-drop-prob 0.4 \
    --mask-strategy mae
```

### BERT-style masking with smaller patches
```bash
python Perceiver06_bloc/main.py --use-eatmint --epochs 50 \
    --patch-len 32 --mask-strategy bert --mask-ratio 0.15
```

### Convolutional frontend with larger patches
```bash
python Perceiver06_bloc/main.py --use-eatmint --epochs 50 \
    --use-conv-frontend --patch-len 128
```

## Output Structure

Training creates a run directory under `Perceiver06_bloc/runs/`:
```
runs/YYYYMMDD_HHMMSS_<description>/
├── config.yaml          # All hyperparameters
├── metrics.yaml         # Training metrics
├── training_history.png # Loss curves
├── physio_recon.png     # Reconstruction visualizations
└── modality_dropout_recon.png  # Modality dropout analysis
```

## Architecture Details

The model uses:
1. **Patch tokenizer**: Converts raw signals → patch embeddings
   - Signal patches (linear/conv projection)
   - Fourier time embeddings
   - Modality embeddings (ECG, GSR, BVP, TEMP, ACC)
   - Channel embeddings

2. **Encoder**: Cross-attention + self-attention layers
   - Query: learnable latent tokens
   - Keys/Values: visible patch tokens (MAE) or all tokens (BERT)

3. **Decoder**: Cross-attention to reconstruct full sequence
   - Query: all patch positions (including masked)
   - Keys/Values: encoder latent states

4. **Loss**: Huber loss on masked patches only
   - Reconstruction loss
   - First-difference loss (weighted 0.3) for temporal dynamics

## Tips

- Start with defaults to ensure everything works
- MAE strategy is usually more compute-efficient than BERT
- Mixed masking works well for multimodal physiological signals
- Larger patch_len = faster but may miss fine details
- Validation loss should be close to training loss (no overfitting on masked modeling)
