# Dependencies

## Core Requirements

The `requirements.txt` file specifies minimal versions:

```
torch>=1.9.0          # PyTorch (tested with 2.9.1+cu128)
numpy>=1.19.0         # Numerical operations
scipy>=1.5.0          # Signal processing (polyphase resampling)
matplotlib>=3.3.0     # Plotting
tqdm>=4.50.0          # Progress bars
pandas>=1.1.0         # Data handling
```

## Installation

```bash
# Using requirements.txt
pip install -r requirements.txt

# Or manually
pip install torch numpy scipy matplotlib tqdm pandas

# If you have issues, try upgrading pip
pip install --upgrade pip
```

## Version Notes

- **PyTorch**: Tested with 2.9.1+cu128. Older versions (1.9+) will work but may have performance differences
- **NumPy**: 1.19+ for compatibility. 1.21+ recommended for better performance
- **SciPy**: 1.5+ for `signal.resample_poly`. Older versions may lack this function
- **Matplotlib**: 3.3+ for robust plotting
- **pandas**: 1.1+ for optional data handling utilities

## Optional: GPU Support

If you want to train on GPU, ensure your PyTorch installation includes CUDA:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install PyTorch with CUDA support
# See https://pytorch.org/get-started/locally/
```

For NVIDIA GPUs, CUDA 11.8+ is typically required for PyTorch 2.x.

## Development Dependencies (Optional)

For development, testing, or contributing:

```bash
# Code linting
pip install black flake8 isort

# Type checking
pip install mypy

# Testing
pip install pytest pytest-cov

# Jupyter (for notebook experimentation)
pip install jupyter
```

## Compatibility

The code is designed to work with:
- **Python 3.7+** (tested with 3.11.13)
- **Linux, macOS, Windows**
- **CPU and GPU** (CUDA/cuDNN)

## Troubleshooting

### Import Error: "No module named torch"
- Verify installation: `pip show torch`
- If not installed: `pip install torch`
- Check you're in the correct virtual environment

### ImportError: "cannot import name 'resample_poly'" (from scipy)
- Your scipy is too old. Update: `pip install --upgrade scipy`

### CUDA out of memory (CUDA OOM)
- Reduce batch size: `--batch-size 4`
- Reduce model size: `--latent-dim 128 --num-latents 64`
- Use CPU: `--device cpu`

### Slow performance on GPU
- Verify CUDA is detected: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU usage: `nvidia-smi`

## Performance

Typical training performance on modern hardware:
- **CPU (Intel i7)**: ~2-3 sec/epoch (batch_size=8, 600 samples)
- **GPU (NVIDIA A100)**: ~0.5 sec/epoch
- **GPU (NVIDIA RTX 3060)**: ~1 sec/epoch

Memory requirements:
- **Model**: ~50 MB (with default latent_dim=256)
- **Per sample**: ~2 MB (at 600 samples × 5 channels)
- **Batch of 8**: ~20 MB total
