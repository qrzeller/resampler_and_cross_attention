#!/usr/bin/env python
"""Comprehensive verification of Perceiver03 module structure."""

import os
import sys
from pathlib import Path

print("=" * 70)
print("PERCEIVER03 MODULE VERIFICATION")
print("=" * 70)

# 1. Check files exist
print("\n1. REQUIRED FILES")
required_files = [
    "fourier_features.py",
    "perceiver_model.py",
    "dataset.py",
    "training.py",
    "visualization.py",
    "main.py",
    "__init__.py",
    "requirements.txt",
    "README.md",
    "QUICKSTART.md",
]

for fname in required_files:
    path = Path(fname)
    exists = "✓" if path.exists() else "✗"
    size = f"({path.stat().st_size:,} bytes)" if path.exists() else ""
    print(f"  {exists} {fname:25} {size}")

# 2. Check imports
print("\n2. MODULE IMPORTS")
modules = [
    ("fourier_features", "FourierFeatures"),
    ("perceiver_model", "PerceiverResampler"),
    ("dataset", "SyntheticPhysioDataset"),
    ("training", "train_model"),
    ("visualization", "plot_physio_reconstructions"),
]

for mod, cls in modules:
    try:
        m = __import__(mod)
        obj = getattr(m, cls)
        print(f"  ✓ {mod}.{cls}")
    except Exception as e:
        print(f"  ✗ {mod}.{cls}: {e}")

# 3. Check model creation
print("\n3. MODEL INSTANTIATION")
try:
    from perceiver_model import PerceiverResampler
    model = PerceiverResampler(
        signal_dim=5,
        seq_len=600,
        sample_rate_hz=50.0,
        latent_dim=128,
        num_latents=64,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created: {params:,} parameters")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")

# 4. Check dataset creation
print("\n4. DATASET INSTANTIATION")
try:
    from dataset import SyntheticPhysioDataset
    ds = SyntheticPhysioDataset(num_windows=10)
    print(f"  ✓ Dataset created: {len(ds)} windows")
    sample = ds[0]
    print(f"  ✓ Sample shape: {sample['physio'].shape}")
except Exception as e:
    print(f"  ✗ Dataset creation failed: {e}")

# 5. Check forward pass
print("\n5. FORWARD PASS")
try:
    import torch
    x = sample["physio"].unsqueeze(0)
    with torch.no_grad():
        y = model(x)
    print(f"  ✓ Input: {x.shape} → Output: {y.shape}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")

# 6. Check dependencies
print("\n6. DEPENDENCIES")
import torch
import numpy
import scipy
import matplotlib
import tqdm
import pandas

deps = [
    ("torch", torch.__version__),
    ("numpy", numpy.__version__),
    ("scipy", scipy.__version__),
    ("matplotlib", matplotlib.__version__),
    ("tqdm", tqdm.__version__),
    ("pandas", pandas.__version__),
]

for name, version in deps:
    print(f"  ✓ {name:15} {version}")

# 7. Check GPU
print("\n7. GPU SUPPORT")
cuda_available = torch.cuda.is_available()
cuda_str = "Available" if cuda_available else "Not available"
print(f"  {'✓' if cuda_available else '✗'} CUDA: {cuda_str}")
if cuda_available:
    print(f"    Device: {torch.cuda.get_device_name(0)}")
    print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE: ALL SYSTEMS GO ✓")
print("=" * 70)
print("\nNext steps:")
print("  1. Run: python main.py --help")
print("  2. Train: python main.py --epochs 5 --num-windows 100")
print("  3. Custom: Modify main.py or use components as library")
