# Perceiver06_bloc: Patch-Based Masked Autoencoding

## Key Concepts

### 1. Patch Tokenization

Instead of treating each sample as a token, we chunk signals into **patches**:
- **Patch length**: 16-256 samples (e.g., 64 samples @ 512Hz = 125ms chunks)
- **Per-channel patching**: Each channel chunked independently
- **Token = Patch**: One patch becomes one token

**Why patches?**
- ✅ Captures local morphology (QRS complex, pulse shapes)
- ✅ Reduces token count: 6144 samples → 96 patches (64× fewer tokens!)
- ✅ Phase-invariant with conv frontend
- ✅ More efficient attention: O(N_latents × M_patches) << O(N_latents × M_samples)

### 2. Token Composition

Each token is the sum of 4 components:

```python
token = proj(patch) + time_emb + modality_emb + channel_emb
```

**a) Patch Projection** `proj(patch)`
- **Simple**: `Linear(patch_len) → d_model`
- **Conv frontend** (better): Conv1d stack → pooling → Linear
  - Handles local morphology
  - Phase-invariant
  - Extracts salient features from patch

**b) Time Embedding** `time_emb`
Uses **Fourier Features** for continuous time encoding:
```
time_features = [t, sin(2π f₁ t), cos(2π f₁ t), ..., sin(2π fₖ t), cos(2π fₖ t)]
```
Where:
- `t` = patch center time in seconds
- `f₁...fₖ` = log-spaced frequencies from `min_freq_hz` to `max_freq_hz`
- `min_freq_hz` = 1 / window_duration (captures slow trends)
- `max_freq_hz` = Nyquist = sample_rate / 2 (captures fast dynamics)

**Why Fourier features?**
- ✅ Encodes continuous time (not discrete positions)
- ✅ Multi-scale: low frequencies for trends, high frequencies for details
- ✅ Generalizes to irregular sampling rates
- ✅ More expressive than learned positional embeddings

**c) Modality Embedding** `modality_emb`
Learnable embedding indicating signal type:
- ECG → 0
- GSR (EDA) → 1  
- BVP (PPG) → 2
- TEMP → 3
- ACC → 4

**Why modality embeddings?**
- ✅ Model knows "this is ECG" vs "this is temperature"
- ✅ Learns modality-specific processing
- ✅ Enables cross-modal fusion

**d) Channel Embedding** `channel_emb`
For multi-lead signals (e.g., ECG Lead I, II, III):
- Lead I → 0
- Lead II → 1
- Lead III → 2

### 3. Masked Autoencoding

Two strategies implemented:

#### A) MAE-style (Recommended) ✅

```python
mask_strategy = 'mae'
```

**How it works:**
1. Create patch mask: 1=visible, 0=masked
2. **Remove masked patches from encoder input** (key insight!)
3. Encoder only processes visible patches
4. Decoder must reconstruct ALL patches from latents

**Advantages:**
- ⚡ **Compute efficient**: O(N_latents × M_visible) vs O(N_latents × M_total)
- 📈 Better learning: model can't peek at masked regions
- 🎯 Forces latent space to capture global structure

**Example:**
```
Input: 10 patches per channel, 50% masked
Encoder sees: 5 patches per channel (50% fewer tokens!)
Decoder outputs: 10 patches per channel (must reconstruct all)
```

#### B) BERT-style (Simpler but less efficient)

```python
mask_strategy = 'bert'
```

**How it works:**
1. Create patch mask: 1=visible, 0=masked
2. **Replace masked patches with learnable [MASK] token**
3. Encoder processes ALL patches (visible + [MASK])
4. Decoder reconstructs from latent

**Disadvantages:**
- 🐌 Wastes compute on mask tokens
- 🔍 Model knows "where" masks are (may leak too much info)

### 4. Masking Strategies

Multiple masking patterns in `masking.py`:

#### Random Patch Masking
```python
strategy = 'random'
```
- Independent per-patch masking
- Simple baseline

#### Span Masking (Recommended for Physio) ✅
```python
strategy = 'span'
min_span_len = 2  # patches
max_span_len = 8  # patches
```
- Masks contiguous temporal spans
- Respects autocorrelation structure
- Forces longer-range inference

**Why spans?**
Physiology has temporal structure:
- ECG: PQRST waves correlated over 0.5-1s
- Respiration: cycles over 2-5s
- Random masking is "too easy" - model can interpolate from neighbors

#### Channel Drop (Cross-Modal Learning) ✅
```python
strategy = 'channel_drop'
drop_prob = 0.3
```
- Drops **entire modalities**
- Simulates sensor dropout
- Forces cross-modal inference

**Example:**
```
Input:  ECG | GSR | BVP | TEMP | ACC
Masked: ECG | --- | --- | TEMP | ACC
Task:   Reconstruct GSR and BVP from ECG, TEMP, ACC
```

#### Mixed Strategy (Best of Both) ⭐
```python
strategy = 'mixed'
mask_ratio = 0.5
channel_drop_prob = 0.2
span_prob = 0.7
```
- Combines channel drops + span/random masking
- Most challenging and realistic

### 5. Architecture Flow

```
Input: (batch, seq_len, channels)
  ↓
Patchify: → (batch, num_patches, channels, patch_len)
  ↓
Flatten: → (batch, num_patches×channels, patch_len)
  ↓
Tokenize: → (batch, num_tokens, d_model)
  [proj(patch) + time_emb + modality_emb + channel_emb]
  ↓
[MAE: Remove masked tokens] ← Compute savings!
  ↓
Encoder Cross-Attn: latents ← visible tokens
  ↓
Latent Self-Attn × L layers
  ↓
Decoder Cross-Attn: ALL patch queries ← latents
  ↓
Output Projection: → (batch, num_tokens, patch_len)
  ↓
Reshape: → (batch, num_patches, channels, patch_len)
  ↓
Unpatchify: → (batch, seq_len, channels)
```

### 6. Loss Computation

**Only compute loss on MASKED patches:**

```python
# patch_mask: (batch, num_patches, channels) where 0=masked
recon_loss = F.mse_loss(prediction, target, reduction='none')
recon_loss = recon_loss * (1 - patch_mask)  # Zero out visible patches
loss = recon_loss.sum() / (1 - patch_mask).sum()  # Normalize by #masked
```

This prevents "cheating" - model can't get credit for reconstructing patches it saw!

### 7. Comparison: Perceiver05 vs Perceiver06

| Feature | Perceiver05 | Perceiver06 |
|---------|-------------|-------------|
| **Tokenization** | Per-sample | Patch-based |
| **Token count** | seq_len × channels | (seq_len/patch_len) × channels |
| **Example** | 6144 × 5 = 30,720 tokens | 96 × 5 = 480 tokens (64× fewer!) |
| **Masking** | Channel-wise | Patch-wise + span + channel drop |
| **Encoder input** | Zeros for masked | MAE: removes masked patches |
| **Position encoding** | Per-sample Fourier | Per-patch Fourier (same principle) |
| **Modality info** | Implicit | Explicit modality embeddings |

## Usage Example

```python
from perceiver_patch_model import PatchPerceiverAutoencoder
from masking import get_mask

# Create model
model = PatchPerceiverAutoencoder(
    signal_dim=5,
    seq_len=6144,  # Must be divisible by patch_len
    sample_rate_hz=512.0,
    patch_len=64,  # 6144/64 = 96 patches
    latent_dim=256,
    num_latents=128,
    num_self_attn_layers=8,
    mask_strategy='mae',  # Use MAE-style (remove masked tokens)
)

# Training step
signals = ...  # (batch, 6144, 5)
num_patches = 6144 // 64  # 96

# Create span mask with channel drops
patch_mask = get_mask(
    batch_size=batch,
    num_patches=num_patches,
    num_channels=5,
    strategy='mixed',
    mask_ratio=0.5,
    channel_drop_prob=0.2,
)

# Forward pass
reconstruction = model(signals, patch_mask=patch_mask)

# Compute loss only on masked patches
loss_mask = (1 - patch_mask).reshape(batch, -1, 1)  # Flatten to match unpatchified
reconstruction_flat = reconstruction.reshape(batch, num_patches, 5, 64)
target_flat = signals.reshape(batch, num_patches, 5, 64)

loss = F.mse_loss(reconstruction_flat, target_flat, reduction='none')
loss = (loss * loss_mask.reshape(batch, num_patches, 5, 1)).sum() / loss_mask.sum()
```

## Key Takeaways

1. **Patches > Samples**: Tokens represent meaningful chunks, not individual samples
2. **Fourier features**: Rich time encoding that generalizes across sampling rates
3. **Modality embeddings**: Explicit signal type information
4. **MAE-style masking**: Remove masked patches from encoder (compute efficient!)
5. **Span masking**: Respects temporal structure of physiology
6. **Channel drops**: Forces cross-modal learning
7. **Loss on masked only**: Prevents cheating, focuses learning on hard task
