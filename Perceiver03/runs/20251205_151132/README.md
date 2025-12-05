# Training Run Report

**Date:** 2025-12-05 15:52:45

## 🚀 Command

```bash
main.py --use-eatmint --epochs 10 --batch-size 8 --target-fs 512 --latent-dim 128 --num-latents 128 --diff-loss-weight 0.3 --no-residual
```

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| `batch_size` | `8` |
| `comment` | `None` |
| `data_root` | `../data/EATMINT/researchdata` |
| `decoder_len` | `None` |
| `device` | `None` |
| `diff_loss_weight` | `0.3` |
| `downsample_strategy` | `polyphase` |
| `dropout` | `0.05` |
| `epochs` | `10` |
| `fourier_bands` | `32` |
| `hop_size` | `6.0` |
| `latent_dim` | `128` |
| `lr` | `0.0001` |
| `max_freq_hz` | `None` |
| `min_freq_hz` | `None` |
| `no_residual` | `True` |
| `num_heads` | `8` |
| `num_latents` | `128` |
| `num_signals` | `5` |
| `num_windows` | `200` |
| `num_workers` | `0` |
| `physio_fs` | `512.0` |
| `preload` | `False` |
| `run_name` | `None` |
| `seed` | `0` |
| `self_layers` | `4` |
| `smoothing_kernel` | `0` |
| `target_fs` | `512.0` |
| `use_eatmint` | `True` |
| `val_fraction` | `0.1` |
| `window_size` | `12.0` |

## 📊 Training Results

- **Final Train Loss:** 0.158723
- **Final Val Loss:** 0.162161
- **Best Train Loss:** 0.158723
- **Best Val Loss:** 0.158439
- **Total Epochs:** 10

### Epoch-by-Epoch History

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 1 | 0.267130 | 0.257034 |
| 2 | 0.251025 | 0.247401 |
| 3 | 0.248094 | 0.250734 |
| 4 | 0.245845 | 0.248352 |
| 5 | 0.242446 | 0.216769 |
| 6 | 0.208942 | 0.197600 |
| 7 | 0.187259 | 0.179831 |
| 8 | 0.173083 | 0.161765 |
| 9 | 0.164404 | 0.158439 |
| 10 | 0.158723 | 0.162161 |

## 📈 Additional Metrics

- **total_parameters:** 1224197

## 📷 Visualizations

### Training History

![Training History](training_history.png)

### Reconstruction Examples

![Reconstruction](reconstruction.png)

## 💾 Checkpoint

Saved to: `checkpoint.pt`

## 📁 Files

- `checkpoint.pt` (14441.2 KB)
- `reconstruction.png` (364.7 KB)
- `training_history.png` (44.5 KB)
