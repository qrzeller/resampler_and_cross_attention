# Critical Fixes Applied to Perceiver-MAE Implementation

## Summary of Changes (Dec 9, 2025)

### 1. Fixed `patch_times` Construction ✅
**Problem**: Incorrect batch handling and channel ordering
**Fix**: 
```python
t = (torch.arange(num_patches, device=device, dtype=torch.float32) * self.patch_len
     + self.patch_len / 2.0) / self.sample_rate_hz
t = t[None, :, None]                       # (1, num_patches, 1)
t = t.expand(batch, -1, -1)                # (batch, num_patches, 1)
t = t[:, :, None, :]                       # (batch, num_patches, 1, 1)
t = t.expand(-1, -1, self.signal_dim, -1)  # (batch, num_patches, C, 1)
patch_times = t.reshape(batch, num_patches * self.signal_dim, 1)
```
Now correctly aligns with `flat_patches` ordering: `[patch0_ch0, patch0_ch1, ..., patch1_ch0, ...]`

### 2. Capped Fourier Max Frequency ✅
**Problem**: Using raw sample rate Nyquist (too high for patch centers)
**Fix**:
```python
token_rate = sample_rate_hz / patch_len  # Effective patch token rate
max_freq_hz = 0.5 * token_rate            # Nyquist for TOKENS, not samples
```
**Rationale**: Patches are spaced by Δt = patch_len / sample_rate, so max representable frequency is 1/(2Δt)

### 3. Pre-norm Self-Attention ✅
**Problem**: Using post-norm (PyTorch default) instead of Perceiver IO pre-norm
**Fix**:
```python
nn.TransformerEncoderLayer(..., norm_first=True)
```
**Impact**: More stable training, matches Perceiver IO paper

### 4. Runtime `num_patches` Calculation ✅
**Problem**: Hardcoded `self.num_patches` breaks with overlapping patches
**Fix**: Derive from actual data: `num_patches = patches.shape[1]`
**Rationale**: Safer, more flexible for future overlap support

### 5. Modality ID Validation ✅
**Problem**: No check that `signal_dim <= num_modalities`
**Fix**: Added assertion in `__init__`
**Prevents**: `nn.Embedding` index out of range errors

### 6. MAE Edge Case Protection ✅
**Problem**: All-masked samples crash MultiheadAttention with 0 tokens
**Fix**:
```python
if len(vis_idx) == 0:
    vis_idx = torch.tensor([0], device=device)  # Keep first token as fallback
```

### 7. Fixed Docstrings ✅
**Problem**: `PatchTokenizer` docstring said `(batch, num_patches, patch_len)` but actually receives `(batch, num_tokens, patch_len)`
**Fix**: Updated all docstrings to reflect actual shapes

### 8. Removed Stride Support ✅
**Problem**: `create_patches` supported overlap but unpatchify assumed non-overlapping
**Fix**: Removed `stride` parameter, enforce non-overlapping only
**Alternative**: Could implement proper overlap-add reconstruction in future

### 9. Fixed Channel Embeddings ✅
**Problem**: `channel_ids` was all zeros (collapsed to single value)
**Fix**:
```python
channel_idx = torch.arange(num_channels, device=patches.device)
channel_ids = channel_idx.unsqueeze(0).unsqueeze(0).expand(batch, num_patches, -1)
channel_ids = channel_ids.reshape(batch, -1)
```
Now properly represents channel index within modality (0..C-1)

### 10. Added Residual Dropout ✅
**Problem**: Only attention weights had dropout, not residual paths
**Fix**: Added `nn.Dropout(dropout)` to both `CrossAttentionBlock` and `DecoderCrossAttentionBlock`
**Impact**: Additional stability knob for training

### 11. Mask Convention Consistency ✅
**Problem**: Inconsistent mask meaning across codebase
**Fix**: Standardized to **1=masked (compute loss), 0=visible (no loss)**
**Updated**:
- All masking functions (`random_patch_mask`, `span_mask`, `channel_drop_mask`, `mixed_mask`)
- Model forward pass (MAE and BERT paths)
- Training loss computation
- Documentation

### 12. Loss Computation on Masked Only ✅
**Status**: Already correct in `training.py`
**Verification**: `modality_mask` with convention 1=masked properly masks loss to only masked patches
**Rationale**: MAE pretraining should focus on hard task (reconstruct masked), not easy task (identity on visible)

## Testing Checklist

Before training:
- [ ] Verify model initializes: `python Perceiver06_bloc/main.py --use-eatmint --epochs 1 --batch-size 2`
- [ ] Check mask generation: `get_mask()` returns correct shapes and convention
- [ ] Validate time embeddings align with patches
- [ ] Confirm loss is computed only on masked patches

## Architecture Verification

✅ Proper Perceiver IO style:
1. Input → Patch tokens (with time/modality/channel embeddings)
2. Encoder cross-attention: latents ← tokens
3. L × self-attention on latents (pre-norm)
4. Decoder cross-attention: output queries ← latents
5. Project to patches → unpatchify

✅ MAE masking:
- Remove masked tokens from encoder (compute efficient)
- Decoder reconstructs ALL patches (including masked)
- Loss computed ONLY on masked patches

✅ BERT masking (optional):
- Replace masked tokens with [MASK]
- All tokens in encoder
- Loss on masked patches

## Performance Expectations

With these fixes:
- Training should be stable (no NaN/explosion)
- Validation loss should track training loss reasonably
- Reconstructions should improve over epochs
- MAE strategy should be faster than BERT (fewer encoder tokens)

## Next Steps

1. Run full training: `python Perceiver06_bloc/main.py --use-eatmint --epochs 50`
2. Monitor for:
   - Stable loss curves
   - Reasonable val/train gap
   - Good reconstructions on masked patches
3. Experiment with:
   - Different mask ratios (0.5, 0.75)
   - Different patch lengths (25, 50, 75, 100)
   - MAE vs BERT masking strategies
