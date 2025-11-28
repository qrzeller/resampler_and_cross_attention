"""
Precompute Wav2Vec2 features for EATMINT dataset.
Run this ONCE before training.
"""

import os
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from tqdm import tqdm

# Configuration (Matches your main script)
DATA_ROOT = "./data/researchdata"
SOUND_PATH = "Sound/clean"
OUTPUT_DIR = "Sound/wav2vec2_features"

def precompute():
    # 1. Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for feature extraction")
    
    root_path = Path(DATA_ROOT)
    in_dir = root_path / SOUND_PATH
    out_dir = root_path / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Foundation Model
    print("Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    model.eval()
    
    # 3. Process Files
    audio_files = list(in_dir.glob("*_Collab.wav"))
    print(f"Found {len(audio_files)} audio files.")

    with torch.no_grad():
        for file_path in tqdm(audio_files, desc="Processing Audio"):
            # Check if already exists
            out_name = out_dir / file_path.with_suffix('.npy').name
            if out_name.exists():
                continue
                
            try:
                # Load Audio
                sr, audio = wavfile.read(file_path)
                
                # Normalize to float [-1, 1]
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0
                elif audio.dtype == np.int32:
                    audio = audio.astype(np.float32) / 2147483648.0
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float()
                
                # Wav2Vec2 requires 16kHz
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    # Resample expects (channel, time)
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    audio_tensor = resampler(audio_tensor).squeeze()
                
                # Process in chunks (60 seconds) to avoid OOM on GPU
                # Wav2Vec2 is heavy on memory.
                chunk_sec = 60
                chunk_samples = 16000 * chunk_sec
                total_samples = len(audio_tensor)
                
                all_features = []
                
                for i in range(0, total_samples, chunk_samples):
                    chunk = audio_tensor[i : i + chunk_samples]
                    
                    # Pad short last chunk if necessary (min length requirement)
                    if len(chunk) < 1600: # 0.1s
                        continue

                    inputs = processor(
                        chunk.numpy(), 
                        sampling_rate=16000, 
                        return_tensors="pt", 
                        padding=True
                    ).input_values.to(device)

                    outputs = model(inputs)
                    # Shape: (1, seq_len, 768)
                    features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                    all_features.append(features)
                
                if all_features:
                    final_features = np.concatenate(all_features, axis=0)
                    np.save(out_name, final_features)
                    
            except Exception as e:
                print(f"\nError processing {file_path.name}: {e}")

    print(f"\nDone! Features saved to {out_dir}")

if __name__ == "__main__":
    precompute()