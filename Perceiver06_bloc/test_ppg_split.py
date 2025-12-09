#!/usr/bin/env python3
"""Test PPG baseline/pulsatile split functionality."""

import numpy as np
from scipy import signal as sp_signal
import matplotlib.pyplot as plt

# Import the function we added
import sys
sys.path.insert(0, '.')
from eatmint_dataset import extract_ppg_baseline


def test_ppg_split():
    """Test PPG baseline/pulsatile extraction."""
    print("Testing PPG baseline/pulsatile split...")
    
    # Generate synthetic PPG signal
    fs = 512.0  # Sampling rate
    duration = 30.0  # 30 seconds
    t = np.arange(0, duration, 1/fs)
    
    # Components:
    # 1. Baseline drift (very slow - 0.02 Hz)
    baseline_component = 2.0 * np.sin(2 * np.pi * 0.02 * t)
    
    # 2. Vasomotor oscillations (0.04 Hz - Mayer waves)
    vasomotor = 0.5 * np.sin(2 * np.pi * 0.04 * t)
    
    # 3. Heart rate component (~1.2 Hz = 72 bpm)
    hr_component = 1.0 * np.sin(2 * np.pi * 1.2 * t)
    
    # 4. Respiratory modulation (~0.25 Hz = 15 breaths/min)
    resp_mod = 0.3 * np.sin(2 * np.pi * 0.25 * t)
    
    # Combine all components
    ppg = baseline_component + vasomotor + hr_component + resp_mod
    
    # Add some noise
    ppg += 0.1 * np.random.randn(len(ppg))
    
    # Extract baseline and pulsatile
    baseline, pulsatile = extract_ppg_baseline(ppg, fs, cutoff_hz=0.05, order=4)
    
    # Verify reconstruction
    reconstruction_error = np.mean((ppg - (baseline + pulsatile))**2)
    print(f"  Reconstruction error: {reconstruction_error:.6e}")
    
    # Check that baseline captures low-frequency content
    baseline_freq_content = np.abs(np.fft.rfft(baseline))
    pulsatile_freq_content = np.abs(np.fft.rfft(pulsatile))
    freqs = np.fft.rfftfreq(len(ppg), 1/fs)
    
    # Find energy in <0.05 Hz band
    low_freq_mask = freqs < 0.05
    baseline_low_energy = np.sum(baseline_freq_content[low_freq_mask]**2)
    pulsatile_low_energy = np.sum(pulsatile_freq_content[low_freq_mask]**2)
    
    print(f"  Baseline low-freq energy: {baseline_low_energy:.2e}")
    print(f"  Pulsatile low-freq energy: {pulsatile_low_energy:.2e}")
    print(f"  Ratio (baseline/pulsatile): {baseline_low_energy / (pulsatile_low_energy + 1e-10):.2f}x")
    
    # Find energy in >0.8 Hz band (heart rate)
    high_freq_mask = freqs > 0.8
    baseline_high_energy = np.sum(baseline_freq_content[high_freq_mask]**2)
    pulsatile_high_energy = np.sum(pulsatile_freq_content[high_freq_mask]**2)
    
    print(f"  Baseline high-freq energy: {baseline_high_energy:.2e}")
    print(f"  Pulsatile high-freq energy: {pulsatile_high_energy:.2e}")
    print(f"  Ratio (pulsatile/baseline): {pulsatile_high_energy / (baseline_high_energy + 1e-10):.2f}x")
    
    # Verification
    assert reconstruction_error < 1e-10, "Reconstruction error too high!"
    assert baseline_low_energy > pulsatile_low_energy * 10, "Baseline should have more low-freq energy!"
    assert pulsatile_high_energy > baseline_high_energy * 10, "Pulsatile should have more high-freq energy!"
    
    print("✓ All tests passed!")
    
    # Optional: Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot original signal
    axes[0].plot(t[:2048], ppg[:2048], 'b-', linewidth=0.5)
    axes[0].set_title('Original PPG Signal (first 4 seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot baseline
    axes[1].plot(t[:2048], baseline[:2048], 'g-', linewidth=1)
    axes[1].set_title('BVP_baseline: Vasomotor tone & perfusion (<0.05 Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Plot pulsatile
    axes[2].plot(t[:2048], pulsatile[:2048], 'r-', linewidth=0.5)
    axes[2].set_title('BVP_pulsatile: Heart rate morphology (>0.05 Hz)')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    
    # Plot frequency spectrum
    axes[3].semilogy(freqs[:500], baseline_freq_content[:500], 'g-', label='Baseline', linewidth=1)
    axes[3].semilogy(freqs[:500], pulsatile_freq_content[:500], 'r-', label='Pulsatile', linewidth=1)
    axes[3].axvline(0.05, color='k', linestyle='--', alpha=0.5, label='Cutoff (0.05 Hz)')
    axes[3].set_title('Frequency Spectrum')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Magnitude')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppg_baseline_split_test.png', dpi=150)
    print(f"  Saved visualization to ppg_baseline_split_test.png")
    plt.close()


if __name__ == "__main__":
    test_ppg_split()
