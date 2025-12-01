"""
EATMINT Multimodal Perceiver

A Perceiver-based model for multimodal learning on the EATMINT dataset.
Handles: Sound (via Wav2Vec2), Physio, OpenFace, Eye-tracker modalities.
Gracefully handles missing modalities with masking.

Usage:
    python eatmint_multimodal_perceiver.py

Author: Adapted for EATMINT dataset
"""

import os
import glob
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.io.wavfile as wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EATMINTConfig:
    """Configuration for EATMINT dataset paths and parameters."""
    
    # Base data path - UPDATE THIS to your data location
    data_root: str = "../data/researchdata"
    
    # Subpaths for each modality
    sound_path: str = "Sound/clean"  # Use denoised audio
    sound_features_path: str = "Sound/wav2vec2_features"  # Precomputed Wav2Vec2 features
    physio_path: str = "Physio"
    openface_path: str = "openface"
    eyetracker_path: str = "Eyetracker"
    times_path: str = "StartEndTimes"
    
    # Sampling parameters
    physio_fs: int = 512  # Physio sampling rate (Hz)
    target_audio_sr: int = 16000  # Target audio sample rate for foundation model
    window_size_sec: float = 2.0  # Window size in seconds for training
    hop_size_sec: float = 1.0  # Hop size in seconds
    
    # Dyads to exclude (D14 has no data, D11/D19 have limited data)
    excluded_dyads: List[int] = field(default_factory=lambda: [14])
    
    @property
    def sound_dir(self) -> Path:
        return Path(self.data_root) / self.sound_path
    
    @property
    def sound_features_dir(self) -> Path:
        return Path(self.data_root) / self.sound_features_path
    
    @property
    def physio_dir(self) -> Path:
        return Path(self.data_root) / self.physio_path
    
    @property
    def openface_dir(self) -> Path:
        return Path(self.data_root) / self.openface_path
    
    @property
    def eyetracker_dir(self) -> Path:
        return Path(self.data_root) / self.eyetracker_path
    
    @property
    def times_dir(self) -> Path:
        return Path(self.data_root) / self.times_path


# =============================================================================
# Data Availability Checking
# =============================================================================

def check_modality_availability(config: EATMINTConfig) -> pd.DataFrame:
    """
    Check which modalities are available for each dyad and participant.
    Returns a DataFrame with availability information.
    """
    # List all possible dyads (2-31, excluding 14)
    all_dyads = [d for d in range(2, 32) if d not in config.excluded_dyads]
    
    availability = []
    
    for dyad in all_dyads:
        dyad_str = f"D{dyad:02d}"
        
        for participant in [1, 2]:
            part_str = f"P{participant:02d}"
            
            record = {
                'dyad': dyad,
                'participant': participant,
                'dyad_str': dyad_str,
                'part_str': part_str,
            }
            
            # Check physio (one file per dyad)
            physio_file = config.physio_dir / f"{dyad_str}.mat"
            record['physio'] = physio_file.exists()
            
            # Check sound
            sound_file = config.sound_dir / f"{dyad_str}_{part_str}_Collab.wav"
            record['sound'] = sound_file.exists()
            
            # Check OpenFace
            openface_file = config.openface_dir / f"{dyad_str}{part_str}_User_Collab.mp4-features.csv"
            openface_times = config.openface_dir / f"{dyad_str}{part_str}_User_times.mat"
            record['openface'] = openface_file.exists() and openface_times.exists()
            
            # Check Eye-tracker
            eyetracker_file = config.eyetracker_dir / f"{dyad_str}{part_str}_Tobii.mat"
            record['eyetracker'] = eyetracker_file.exists()
            
            # Check times file (required for synchronization)
            times_file = config.times_dir / f"{dyad_str}_times.csv"
            record['times'] = times_file.exists()
            
            # Count available modalities
            record['n_modalities'] = sum([
                record['physio'], record['sound'], 
                record['openface'], record['eyetracker']
            ])
            
            availability.append(record)
    
    df = pd.DataFrame(availability)
    return df


def print_availability_summary(availability_df: pd.DataFrame):
    """Print a summary of data availability."""
    print("=" * 50)
    print("EATMINT Data Availability Summary")
    print("=" * 50)
    print(f"\nTotal participants: {len(availability_df)}")
    print(f"\nModality availability:")
    for mod in ['physio', 'sound', 'openface', 'eyetracker', 'times']:
        count = availability_df[mod].sum()
        pct = 100 * count / len(availability_df)
        print(f"  {mod:12s}: {count:2d}/{len(availability_df)} ({pct:.1f}%)")
    
    print(f"\n# Modalities per participant:")
    print(availability_df['n_modalities'].value_counts().sort_index())
    
    # Filter to participants with at least 2 modalities + times
    usable_df = availability_df[
        (availability_df['n_modalities'] >= 2) & (availability_df['times'])
    ]
    print(f"\nUsable participants (≥2 modalities + times): {len(usable_df)}")
    return usable_df


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_times(config: EATMINTConfig, dyad: int) -> Dict[str, float]:
    """Load start/end times for a dyad. Times are in milliseconds (UNIX time * 1000)."""
    times_file = config.times_dir / f"D{dyad:02d}_times.csv"
    df = pd.read_csv(times_file)
    return {
        'baseline_start': df['Baseline start'].iloc[0],
        'baseline_stop': df['Baseline stop'].iloc[0],
        'collab_start': df['Collaboration start'].iloc[0],
        'collab_stop': df['Collaboration stop'].iloc[0],
    }


def load_physio(config: EATMINTConfig, dyad: int, participant: int) -> Dict[str, np.ndarray]:
    """
    Load physiological signals for a participant.
    Returns dict with keys: GSR, Temp, Plet, ECG, Resp and metadata.
    """
    physio_file = config.physio_dir / f"D{dyad:02d}.mat"
    mat = sio.loadmat(physio_file)
    
    fs = mat['fs'].flatten()[0]  # Sampling frequency (512 Hz)
    collab_data = mat['collabData']
    labels = [l.strip() for l in mat['label']]
    
    # Find channels for this participant
    prefix = f"{participant}-"
    signals = {}
    exg1_idx = None
    exg2_idx = None
    
    for i, label in enumerate(labels):
        if label.startswith(prefix):
            signal_name = label.split('-')[1]
            if signal_name in ['GSR1', 'Temp', 'Plet', 'Resp']:
                # Map GSR1 -> GSR
                key = 'GSR' if signal_name == 'GSR1' else signal_name
                signals[key] = collab_data[:, i]
            elif signal_name == 'EXG1':
                exg1_idx = i
            elif signal_name == 'EXG2':
                exg2_idx = i
    
    # Compute ECG from EXG1 - EXG2
    if exg1_idx is not None and exg2_idx is not None:
        signals['ECG'] = collab_data[:, exg1_idx] - collab_data[:, exg2_idx]
    
    signals['fs'] = fs
    signals['n_samples'] = collab_data.shape[0]
    
    return signals


def load_sound(config: EATMINTConfig, dyad: int, participant: int) -> Tuple[np.ndarray, int]:
    """Load audio for a participant. Returns (audio, sample_rate)."""
    sound_file = config.sound_dir / f"D{dyad:02d}_P{participant:02d}_Collab.wav"
    sr, audio = wavfile.read(sound_file)
    
    # Normalize to [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    return audio, sr


def load_precomputed_audio_features(config: EATMINTConfig, dyad: int, participant: int) -> Optional[np.ndarray]:
    """
    Load precomputed Wav2Vec2 features for a participant.
    
    Returns:
        features: Array of shape (n_frames, 768) or None if not available
        
    Note: Features are at ~50Hz (one frame per 20ms of audio).
    Run tools/precompute.py first to generate these files.
    """
    features_file = config.sound_features_dir / f"D{dyad:02d}_P{participant:02d}_Collab.npy"
    
    if not features_file.exists():
        return None
    
    try:
        features = np.load(features_file)
        return features.astype(np.float32)
    except Exception as e:
        warnings.warn(f"Error loading precomputed features {features_file}: {e}")
        return None


def load_openface(config: EATMINTConfig, dyad: int, participant: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load OpenFace features and timestamps.
    Returns (features_df, timestamps_microseconds).
    """
    features_file = config.openface_dir / f"D{dyad:02d}P{participant:02d}_User_Collab.mp4-features.csv"
    times_file = config.openface_dir / f"D{dyad:02d}P{participant:02d}_User_times.mat"
    
    # Load features - don't strip whitespace to preserve column names as-is
    df = pd.read_csv(features_file)
    
    # Load timestamps (in microseconds, UNIX time)
    mat = sio.loadmat(times_file)
    timestamps = mat['collabTime'].flatten()  # microseconds
    
    # Ensure timestamps align with frames
    if len(timestamps) != len(df):
        # Truncate to shorter length
        min_len = min(len(timestamps), len(df))
        timestamps = timestamps[:min_len]
        df = df.iloc[:min_len]
    
    return df, timestamps


def load_eyetracker(config: EATMINTConfig, dyad: int, participant: int) -> Dict[str, Any]:
    """
    Load eye-tracker data for a participant.
    Returns dict with gaze data and timestamps.
    """
    tobii_file = config.eyetracker_dir / f"D{dyad:02d}P{participant:02d}_Tobii.mat"
    mat = sio.loadmat(tobii_file)
    
    data = mat['data']
    # Extract the structured data - collab_data has shape (1, n_columns)
    collab_data = data['collabData'][0, 0]
    labels_raw = data['label'][0, 0]
    labels = [str(l[0]) for l in labels_raw.flatten()]
    
    result = {'labels': labels}
    
    # Find columns by label - data is accessed via collab_data[0, col_idx]
    for i, label in enumerate(labels):
        col_data = collab_data[0, i].flatten()
        # Skip non-numeric columns
        if col_data.dtype.kind in ['i', 'f', 'u']:
            if 'ExpeServerTimeMicroSecond' in label:
                result['timestamps'] = col_data.astype(np.float64)
            else:
                result[label] = col_data.astype(np.float64)
    
    result['screen_size'] = mat['screenSize'].flatten()
    
    return result


# =============================================================================
# Audio Feature Extractor (Wav2Vec2)
# =============================================================================

class AudioFeatureExtractor:
    """
    Extract audio features using Wav2Vec2 foundation model.
    This reduces raw audio to compact embeddings suitable for the Perceiver.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Wav2Vec2 expects 16kHz audio
        self.target_sr = 16000
        
    def extract_features(self, audio: np.ndarray, sr: int, 
                        return_attention: bool = False) -> torch.Tensor:
        """
        Extract features from audio.
        
        Args:
            audio: Audio array (mono)
            sr: Sample rate of input audio
            return_attention: Whether to return attention weights
            
        Returns:
            features: Tensor of shape (seq_len, hidden_size=768)
                     ~50 features per second of audio (320 samples per feature)
        """
        # Resample if needed
        if sr != self.target_sr:
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            audio_tensor = resampler(audio_tensor).squeeze(0).numpy()
        else:
            audio_tensor = audio
        
        # Process with Wav2Vec2
        inputs = self.processor(
            audio_tensor, 
            sampling_rate=self.target_sr, 
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_values.to(self.device),
                output_attentions=return_attention
            )
        
        # Return last hidden state: (batch, seq_len, hidden_dim=768)
        features = outputs.last_hidden_state.squeeze(0).cpu()
        
        return features
    
    def get_feature_timestamps(self, audio_len: int, sr: int) -> np.ndarray:
        """
        Get timestamps for each extracted feature frame.
        Wav2Vec2 produces ~50 frames per second (stride of 320 samples at 16kHz).
        """
        # After resampling
        resampled_len = int(audio_len * self.target_sr / sr)
        # Wav2Vec2 stride is 320 samples
        n_frames = resampled_len // 320
        # Each frame corresponds to 20ms = 0.02s
        timestamps = np.arange(n_frames) * 0.02
        return timestamps


# =============================================================================
# Multimodal Dataset
# =============================================================================

class ModalityData(NamedTuple):
    """Container for a single modality's data with timestamps."""
    data: np.ndarray  # Shape: (n_samples, n_features) or (n_samples,)
    timestamps: np.ndarray  # UNIX timestamps in seconds
    available: bool


class EATMINTDataset(Dataset):
    """
    PyTorch Dataset for EATMINT multimodal data.
    
    Handles:
    - Variable modality availability
    - Temporal synchronization across modalities
    - Windowed sampling
    - Audio feature extraction with foundation model
    """
    
    # OpenFace feature columns to use (Action Units + head pose)
    # Note: CSV columns have leading space after comma separator
    OPENFACE_COLS = [
        # Action Units (intensity) - with space prefix as in CSV
        ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r',
        ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r',
        ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r',
        # Head pose
        ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz',
        # Gaze
        ' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y',
    ]
    
    # Eye-tracker columns to use
    EYETRACKER_COLS = [
        'GazePointX', 'GazePointY', 'PupilLeft', 'PupilRight',
        'ValidityLeft', 'ValidityRight'
    ]
    
    # Physio signals to use
    PHYSIO_SIGNALS = ['GSR', 'ECG', 'Resp', 'Temp', 'Plet']
    
    def __init__(
        self,
        config: EATMINTConfig,
        availability_df: pd.DataFrame,
        audio_extractor: Optional[AudioFeatureExtractor] = None,
        window_size_sec: float = 2.0,
        hop_size_sec: float = 1.0,
        target_fs: float = 50.0,  # Target frequency for resampling all modalities
        min_modalities: int = 2,
        preload: bool = False,
        skip_audio: bool = False,  # Skip audio extraction for faster training
        use_precomputed_audio: bool = True,  # Use precomputed Wav2Vec2 features if available
    ):
        """
        Args:
            config: Dataset configuration
            availability_df: DataFrame with modality availability
            audio_extractor: Wav2Vec2 feature extractor (optional, created if None)
            window_size_sec: Window duration in seconds
            hop_size_sec: Hop between windows in seconds
            target_fs: Target sampling frequency for alignment
            min_modalities: Minimum modalities required per sample
            preload: Whether to preload all data into memory
            skip_audio: Skip audio feature extraction for faster training
            use_precomputed_audio: Use precomputed Wav2Vec2 features (from tools/precompute.py)
        """
        self.config = config
        self.audio_extractor = audio_extractor
        self.skip_audio = skip_audio
        self.use_precomputed_audio = use_precomputed_audio
        self.window_size_sec = window_size_sec
        self.hop_size_sec = hop_size_sec
        self.target_fs = target_fs
        self.window_samples = int(window_size_sec * target_fs)
        
        # Filter to usable participants
        self.participants = availability_df
        self.participants = self.participants[
            (self.participants['n_modalities'] >= min_modalities) &
            (self.participants['times'])
        ].reset_index(drop=True)
        
        # Build index of all windows
        self.windows = self._build_window_index()
        
        # Cache for loaded data - always use cache for efficiency
        self.cache = {}
        if preload:
            self._preload_all()
    
    def _preload_all(self):
        """Preload all participant data into memory for faster training."""
        print("Preloading participant data into memory...")
        for _, row in tqdm(self.participants.iterrows(), total=len(self.participants), desc="Loading"):
            try:
                self._load_participant_data(row['dyad'], row['participant'])
            except Exception as e:
                warnings.warn(f"Error loading D{row['dyad']}P{row['participant']}: {e}")
        print(f"Preloaded {len(self.cache)} participants")
    
    def _build_window_index(self) -> List[Dict]:
        """Build index of all training windows."""
        windows = []
        
        for _, row in self.participants.iterrows():
            try:
                times = load_times(self.config, row['dyad'])
                duration_sec = (times['collab_stop'] - times['collab_start']) / 1000.0
                
                # Generate window start times
                n_windows = int((duration_sec - self.window_size_sec) / self.hop_size_sec) + 1
                
                for w_idx in range(max(1, n_windows)):
                    windows.append({
                        'dyad': row['dyad'],
                        'participant': row['participant'],
                        'window_idx': w_idx,
                        'start_sec': w_idx * self.hop_size_sec,
                        'has_sound': row['sound'],
                        'has_physio': row['physio'],
                        'has_openface': row['openface'],
                        'has_eyetracker': row['eyetracker'],
                    })
            except Exception as e:
                warnings.warn(f"Error building windows for D{row['dyad']}P{row['participant']}: {e}")
        
        print(f"Built {len(windows)} training windows from {len(self.participants)} participants")
        return windows
    
    def _load_participant_data(self, dyad: int, participant: int) -> Dict[str, ModalityData]:
        """Load all available modalities for a participant."""
        cache_key = f"D{dyad:02d}P{participant:02d}"
        
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]
        
        times = load_times(self.config, dyad)
        collab_start_sec = times['collab_start'] / 1000.0
        collab_duration = (times['collab_stop'] - times['collab_start']) / 1000.0
        
        data = {}
        
        # Load Physio
        try:
            physio = load_physio(self.config, dyad, participant)
            physio_signals = []
            for sig in self.PHYSIO_SIGNALS:
                if sig in physio:
                    physio_signals.append(physio[sig])
            
            if physio_signals:
                physio_data = np.stack(physio_signals, axis=1)
                # Z-score normalize each channel (physio has very different scales)
                physio_mean = np.mean(physio_data, axis=0, keepdims=True)
                physio_std = np.std(physio_data, axis=0, keepdims=True)
                physio_std = np.where(physio_std < 1e-8, 1.0, physio_std)  # Avoid div by zero
                physio_data = (physio_data - physio_mean) / physio_std
                
                # Create timestamps (physio is synced to collab start/stop)
                physio_ts = np.linspace(0, collab_duration, len(physio_data))
                data['physio'] = ModalityData(physio_data, physio_ts, True)
            else:
                raise ValueError("No physio signals found")
        except Exception as e:
            data['physio'] = ModalityData(np.array([]), np.array([]), False)
        
        # Load Sound - try precomputed features first, then fall back to raw audio
        try:
            precomputed_features = None
            if self.use_precomputed_audio:
                precomputed_features = load_precomputed_audio_features(self.config, dyad, participant)
            
            if precomputed_features is not None:
                # Use precomputed Wav2Vec2 features (~50Hz, 768-dim)
                # Create timestamps: Wav2Vec2 produces ~50 frames/sec (20ms per frame)
                n_frames = len(precomputed_features)
                audio_ts = np.linspace(0, collab_duration, n_frames)
                data['audio'] = ModalityData(precomputed_features, audio_ts, True)
                data['audio_sr'] = None  # Not needed for precomputed
                data['audio_precomputed'] = True
            else:
                # Fall back to raw audio (will be processed by Wav2Vec2 on-the-fly)
                audio, sr = load_sound(self.config, dyad, participant)
                audio_ts = np.linspace(0, collab_duration, len(audio))
                data['audio'] = ModalityData(
                    np.column_stack([audio, np.full_like(audio, sr)]),
                    audio_ts, 
                    True
                )
                data['audio_sr'] = sr
                data['audio_precomputed'] = False
        except Exception as e:
            data['audio'] = ModalityData(np.array([]), np.array([]), False)
            data['audio_sr'] = None
            data['audio_precomputed'] = False
        
        # Load OpenFace
        try:
            of_df, of_times = load_openface(self.config, dyad, participant)
            
            # Select columns - try both with and without space prefix
            available_cols = []
            for c in self.OPENFACE_COLS:
                if c in of_df.columns:
                    available_cols.append(c)
                elif c.strip() in of_df.columns:
                    available_cols.append(c.strip())
            
            if len(available_cols) == 0:
                raise ValueError(f"No OpenFace columns found. Available: {list(of_df.columns)[:10]}...")
            
            of_data = of_df[available_cols].values.astype(np.float32)
            
            # Handle NaN values
            of_data = np.nan_to_num(of_data, nan=0.0)
            
            # Z-score normalize each channel
            of_mean = np.mean(of_data, axis=0, keepdims=True)
            of_std = np.std(of_data, axis=0, keepdims=True)
            of_std = np.where(of_std < 1e-8, 1.0, of_std)
            of_data = (of_data - of_mean) / of_std
            
            # Convert timestamps to seconds relative to collab start
            of_ts_sec = (of_times / 1e6) - collab_start_sec
            
            data['openface'] = ModalityData(of_data, of_ts_sec, True)
        except Exception as e:
            data['openface'] = ModalityData(np.array([]), np.array([]), False)
        
        # Load Eye-tracker
        try:
            eye_data = load_eyetracker(self.config, dyad, participant)
            
            # Extract relevant columns
            eye_signals = []
            for col in self.EYETRACKER_COLS:
                if col in eye_data:
                    eye_signals.append(eye_data[col])
            
            if eye_signals and 'timestamps' in eye_data:
                eye_arr = np.stack(eye_signals, axis=1)
                
                # Z-score normalize each channel
                eye_mean = np.mean(eye_arr, axis=0, keepdims=True)
                eye_std = np.std(eye_arr, axis=0, keepdims=True)
                eye_std = np.where(eye_std < 1e-8, 1.0, eye_std)
                eye_arr = (eye_arr - eye_mean) / eye_std
                
                eye_ts_sec = (eye_data['timestamps'] / 1e6) - collab_start_sec
                data['eyetracker'] = ModalityData(eye_arr, eye_ts_sec, True)
            else:
                raise ValueError("Missing eye-tracker data")
        except Exception as e:
            data['eyetracker'] = ModalityData(np.array([]), np.array([]), False)
        
        if self.cache is not None:
            self.cache[cache_key] = data
        
        return data
    
    def _resample_to_window(
        self, 
        modality: ModalityData, 
        start_sec: float, 
        end_sec: float
    ) -> Optional[np.ndarray]:
        """
        Resample modality data to a fixed window at target_fs.
        Returns array of shape (window_samples, n_features) or None if unavailable.
        """
        if not modality.available or len(modality.data) == 0:
            return None
        
        # Find samples within window
        mask = (modality.timestamps >= start_sec) & (modality.timestamps < end_sec)
        
        if mask.sum() < 2:
            return None
        
        window_data = modality.data[mask]
        window_ts = modality.timestamps[mask]
        
        # Resample to target frequency
        target_ts = np.linspace(start_sec, end_sec, self.window_samples, endpoint=False)
        
        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)
        
        # Linear interpolation
        resampled = np.zeros((self.window_samples, window_data.shape[1]))
        for feat_idx in range(window_data.shape[1]):
            resampled[:, feat_idx] = np.interp(target_ts, window_ts, window_data[:, feat_idx])
        
        return resampled
    
    def _extract_audio_window(
        self,
        data: Dict[str, Any],
        start_sec: float,
        end_sec: float,
    ) -> Optional[np.ndarray]:
        """
        Extract audio features for a specific time window.
        
        If precomputed features are available, simply slice and resample them.
        Otherwise, use Wav2Vec2 to extract features on-the-fly.
        
        Args:
            data: Participant data dict containing 'audio' ModalityData
            start_sec: Start time of window in seconds
            end_sec: End time of window in seconds
            
        Returns:
            Array of shape (window_samples, 768) with Wav2Vec2 features, or None
        """
        if self.skip_audio or not data['audio'].available:
            return None
        
        audio_data = data['audio']
        timestamps = audio_data.timestamps
        
        # Check if using precomputed features
        if data.get('audio_precomputed', False):
            # Precomputed features: just slice and resample
            return self._resample_to_window(audio_data, start_sec, end_sec)
        
        # On-the-fly extraction with Wav2Vec2
        if self.audio_extractor is None:
            return None
        
        sr = data.get('audio_sr')
        if sr is None:
            return None
        
        # Audio data is stored as (n_samples, 2) where col 0 is audio, col 1 is sr
        raw_audio = audio_data.data[:, 0]
        
        # Find audio samples within the window
        mask = (timestamps >= start_sec) & (timestamps < end_sec)
        
        if mask.sum() < sr // 2:  # Need at least 0.5s of audio
            return None
        
        audio_window = raw_audio[mask]
        
        # Extract Wav2Vec2 features for this window
        try:
            features = self.audio_extractor.extract_features(audio_window, sr).numpy()
            
            if len(features) == 0:
                return None
            
            # Interpolate to target length
            if len(features) != self.window_samples:
                src_ts = np.linspace(0, 1, len(features))
                tgt_ts = np.linspace(0, 1, self.window_samples)
                resampled = np.zeros((self.window_samples, features.shape[1]))
                for i in range(features.shape[1]):
                    resampled[:, i] = np.interp(tgt_ts, src_ts, features[:, i])
                features = resampled
            
            return features.astype(np.float32)
            
        except Exception as e:
            return None
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns dict with:
            - 'physio': (window_samples, n_physio_features) or zeros + mask
            - 'audio': (window_samples, 768) audio embeddings or zeros + mask
            - 'openface': (window_samples, n_openface_features) or zeros + mask
            - 'eyetracker': (window_samples, n_eye_features) or zeros + mask
            - 'modality_mask': (4,) boolean mask indicating available modalities
        """
        window_info = self.windows[idx]
        
        # Load participant data
        data = self._load_participant_data(window_info['dyad'], window_info['participant'])
        
        start_sec = window_info['start_sec']
        end_sec = start_sec + self.window_size_sec
        
        # Resample each modality to window
        sample = {}
        modality_mask = []
        
        # Audio - extract features on-the-fly for this window
        audio_features = self._extract_audio_window(data, start_sec, end_sec)
        if audio_features is not None:
            sample['audio'] = torch.from_numpy(audio_features).float()
            modality_mask.append(True)
        else:
            sample['audio'] = torch.zeros(self.window_samples, 768)  # Wav2Vec2 dim
            modality_mask.append(False)
        
        # Physio
        physio = self._resample_to_window(data['physio'], start_sec, end_sec)
        if physio is not None:
            sample['physio'] = torch.from_numpy(physio).float()
            modality_mask.append(True)
        else:
            sample['physio'] = torch.zeros(self.window_samples, len(self.PHYSIO_SIGNALS))
            modality_mask.append(False)
        
        # OpenFace
        openface = self._resample_to_window(data['openface'], start_sec, end_sec)
        if openface is not None:
            sample['openface'] = torch.from_numpy(openface).float()
            modality_mask.append(True)
        else:
            sample['openface'] = torch.zeros(self.window_samples, len(self.OPENFACE_COLS))
            modality_mask.append(False)
        
        # Eye-tracker
        eyetracker = self._resample_to_window(data['eyetracker'], start_sec, end_sec)
        if eyetracker is not None:
            sample['eyetracker'] = torch.from_numpy(eyetracker).float()
            modality_mask.append(True)
        else:
            sample['eyetracker'] = torch.zeros(self.window_samples, len(self.EYETRACKER_COLS))
            modality_mask.append(False)
        
        sample['modality_mask'] = torch.tensor(modality_mask)
        
        return sample


# =============================================================================
# Model Components
# =============================================================================

class ModalityEncoder(nn.Module):
    """Encode a single modality to a common embedding space."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        return self.projection(x)


class EATMINTPerceiverEncoder(nn.Module):
    """
    Custom encoder that handles multiple EATMINT modalities.
    Projects each modality to a common space and concatenates with modality tokens.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        audio_dim: int = 768,
        physio_dim: int = 5,
        openface_dim: int = 28,  # 17 AUs + 6 pose + 5 gaze
        eyetracker_dim: int = 6,
        max_seq_len: int = 100,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Modality-specific encoders
        self.audio_encoder = ModalityEncoder(audio_dim, hidden_dim)
        self.physio_encoder = ModalityEncoder(physio_dim, hidden_dim)
        self.openface_encoder = ModalityEncoder(openface_dim, hidden_dim)
        self.eyetracker_encoder = ModalityEncoder(eyetracker_dim, hidden_dim)
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(4, hidden_dim)  # 4 modalities
        
        # Positional encoding (Fourier)
        self.register_buffer(
            'position_encoding',
            self._create_sinusoidal_positions(max_seq_len, hidden_dim)
        )
    
    def _create_sinusoidal_positions(self, length: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(
        self,
        audio: torch.Tensor,  # (batch, seq, 768)
        physio: torch.Tensor,  # (batch, seq, 5)
        openface: torch.Tensor,  # (batch, seq, 29)
        eyetracker: torch.Tensor,  # (batch, seq, 6)
        modality_mask: torch.Tensor,  # (batch, 4) - which modalities are available
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all modalities into a single sequence.
        
        Returns:
            embeddings: (batch, total_seq_len, hidden_dim)
            attention_mask: (batch, total_seq_len)
        """
        batch_size, seq_len = audio.shape[:2]
        device = audio.device
        
        # Encode each modality
        encoded = []
        masks = []
        
        # Audio (modality 0)
        audio_enc = self.audio_encoder(audio)  # (batch, seq, hidden)
        audio_enc = audio_enc + self.modality_embeddings(torch.zeros(1, dtype=torch.long, device=device))
        audio_enc = audio_enc + self.position_encoding[:seq_len].unsqueeze(0)
        encoded.append(audio_enc)
        masks.append(modality_mask[:, 0:1].expand(-1, seq_len))
        
        # Physio (modality 1)
        physio_enc = self.physio_encoder(physio)
        physio_enc = physio_enc + self.modality_embeddings(torch.ones(1, dtype=torch.long, device=device))
        physio_enc = physio_enc + self.position_encoding[:seq_len].unsqueeze(0)
        encoded.append(physio_enc)
        masks.append(modality_mask[:, 1:2].expand(-1, seq_len))
        
        # OpenFace (modality 2)
        openface_enc = self.openface_encoder(openface)
        openface_enc = openface_enc + self.modality_embeddings(torch.full((1,), 2, dtype=torch.long, device=device))
        openface_enc = openface_enc + self.position_encoding[:seq_len].unsqueeze(0)
        encoded.append(openface_enc)
        masks.append(modality_mask[:, 2:3].expand(-1, seq_len))
        
        # Eye-tracker (modality 3)
        eyetracker_enc = self.eyetracker_encoder(eyetracker)
        eyetracker_enc = eyetracker_enc + self.modality_embeddings(torch.full((1,), 3, dtype=torch.long, device=device))
        eyetracker_enc = eyetracker_enc + self.position_encoding[:seq_len].unsqueeze(0)
        encoded.append(eyetracker_enc)
        masks.append(modality_mask[:, 3:4].expand(-1, seq_len))
        
        # Concatenate all modalities: (batch, 4*seq_len, hidden_dim)
        embeddings = torch.cat(encoded, dim=1)
        attention_mask = torch.cat(masks, dim=1).float()
        
        return embeddings, attention_mask


class EATMINTPerceiver(nn.Module):
    """
    Perceiver model for EATMINT multimodal data.
    
    Uses cross-attention to compress variable-length multimodal inputs
    into a fixed set of latent vectors, then decodes for each modality.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_self_attention_layers: int = 6,
        num_cross_attention_layers: int = 1,
        num_heads: int = 8,
        audio_dim: int = 768,
        physio_dim: int = 5,
        openface_dim: int = 28,  # 17 AUs + 6 pose + 5 gaze
        eyetracker_dim: int = 6,
        seq_len: int = 100,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.seq_len = seq_len
        
        # Multimodal encoder
        self.encoder = EATMINTPerceiverEncoder(
            hidden_dim=hidden_dim,
            audio_dim=audio_dim,
            physio_dim=physio_dim,
            openface_dim=openface_dim,
            eyetracker_dim=eyetracker_dim,
            max_seq_len=seq_len,
        )
        
        # Latent array (learnable)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        
        # Cross-attention: latents attend to inputs
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
            for _ in range(num_cross_attention_layers)
        ])
        self.cross_norm = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_cross_attention_layers)
        ])
        
        # Self-attention on latents
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                batch_first=True,
                activation='gelu',
            )
            for _ in range(num_self_attention_layers)
        ])
        
        # Decoder queries for each modality
        self.audio_queries = nn.Parameter(torch.randn(seq_len, latent_dim))
        self.physio_queries = nn.Parameter(torch.randn(seq_len, latent_dim))
        self.openface_queries = nn.Parameter(torch.randn(seq_len, latent_dim))
        self.eyetracker_queries = nn.Parameter(torch.randn(seq_len, latent_dim))
        
        # Decoder cross-attention
        self.decoder_attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.decoder_norm = nn.LayerNorm(latent_dim)
        
        # Output heads
        self.audio_head = nn.Linear(latent_dim, audio_dim)
        self.physio_head = nn.Linear(latent_dim, physio_dim)
        self.openface_head = nn.Linear(latent_dim, openface_dim)
        self.eyetracker_head = nn.Linear(latent_dim, eyetracker_dim)
    
    def encode(
        self,
        audio: torch.Tensor,
        physio: torch.Tensor,
        openface: torch.Tensor,
        eyetracker: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode inputs to latent space."""
        batch_size = audio.shape[0]
        
        # Get multimodal embeddings
        embeddings, attention_mask = self.encoder(
            audio, physio, openface, eyetracker, modality_mask
        )
        
        # Prepare latents
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: latents query the inputs
        # Convert mask: 1 = attend, 0 = ignore -> need to invert for key_padding_mask
        key_padding_mask = (attention_mask == 0)
        
        for cross_attn, norm in zip(self.cross_attention, self.cross_norm):
            attn_out, _ = cross_attn(
                latents, embeddings, embeddings,
                key_padding_mask=key_padding_mask
            )
            latents = norm(latents + attn_out)
        
        # Self-attention on latents
        for layer in self.self_attention_layers:
            latents = layer(latents)
        
        return latents
    
    def decode(
        self,
        latents: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Decode latents to a specific modality."""
        batch_size = latents.shape[0]
        
        # Get queries for this modality
        queries = getattr(self, f"{modality}_queries")
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: queries attend to latents
        decoded, _ = self.decoder_attention(queries, latents, latents)
        decoded = self.decoder_norm(decoded)
        
        # Project to output space
        head = getattr(self, f"{modality}_head")
        output = head(decoded)
        
        return output
    
    def forward(
        self,
        audio: torch.Tensor,
        physio: torch.Tensor,
        openface: torch.Tensor,
        eyetracker: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode all modalities, decode to reconstruct each.
        
        Returns dict with reconstructions for each modality.
        """
        # Encode
        latents = self.encode(audio, physio, openface, eyetracker, modality_mask)
        
        # Decode each modality
        outputs = {
            'audio': self.decode(latents, 'audio'),
            'physio': self.decode(latents, 'physio'),
            'openface': self.decode(latents, 'openface'),
            'eyetracker': self.decode(latents, 'eyetracker'),
            'latents': latents,
        }
        
        return outputs


# =============================================================================
# Training Functions
# =============================================================================

def compute_masked_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute reconstruction loss only on available modalities.
    
    Args:
        outputs: Model outputs for each modality
        targets: Ground truth for each modality
        modality_mask: (batch, 4) - boolean mask for [audio, physio, openface, eyetracker]
    
    Returns:
        total_loss: Scalar loss
        loss_dict: Per-modality losses for logging
    """
    modality_names = ['audio', 'physio', 'openface', 'eyetracker']
    losses = {}
    total_loss = torch.tensor(0.0, device=modality_mask.device)
    n_active = 0
    
    for i, mod_name in enumerate(modality_names):
        mask = modality_mask[:, i]  # (batch,)
        
        if mask.sum() > 0:
            # Select samples where this modality is available
            pred = outputs[mod_name][mask]
            target = targets[mod_name][mask]
            
            # MSE loss
            mod_loss = F.mse_loss(pred, target)
            losses[mod_name] = mod_loss.item()
            total_loss = total_loss + mod_loss
            n_active += 1
    
    if n_active > 0:
        total_loss = total_loss / n_active  # Average across active modalities
    
    return total_loss, losses


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mask_physio_submodality: bool = True,
) -> Dict[str, float]:
    """
    Train for one epoch.

    If mask_physio_submodality is True, one physio sub-modality (channel)
    is randomly zeroed in the input for the whole batch, so the model
    must reconstruct it from the remaining physio channels and the other
    modalities.
    """
    model.train()
    
    total_loss = 0.0
    modality_losses = {'audio': 0.0, 'physio': 0.0, 'openface': 0.0, 'eyetracker': 0.0}
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Optionally mask one physio sub-modality in the input
        physio_input = batch['physio']
        if mask_physio_submodality and physio_input is not None:
            # Choose a random physio channel to drop for this batch
            n_channels = physio_input.shape[-1]
            drop_idx = torch.randint(0, n_channels, (1,), device=device).item()
            physio_masked = physio_input.clone()
            physio_masked[:, :, drop_idx] = 0.0
        else:
            physio_masked = physio_input
        
        # Forward pass with masked physio input
        outputs = model(
            audio=batch['audio'],
            physio=physio_masked,
            openface=batch['openface'],
            eyetracker=batch['eyetracker'],
            modality_mask=batch['modality_mask'],
        )
        
        # Compute loss against full targets (including the dropped physio channel)
        loss, losses = compute_masked_loss(
            outputs,
            {k: batch[k] for k in ['audio', 'physio', 'openface', 'eyetracker']},
            batch['modality_mask'],
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        for mod_name, mod_loss in losses.items():
            modality_losses[mod_name] += mod_loss
        n_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Average
    metrics = {'total_loss': total_loss / n_batches}
    for mod_name in modality_losses:
        metrics[f'{mod_name}_loss'] = modality_losses[mod_name] / n_batches
    
    return metrics


def train(
    model: nn.Module,
    dataset: Dataset,
    n_epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = 'gpu',
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Full training loop."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set >0 if not preloading
        pin_memory=True if device == 'cuda' else False,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    history = []
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Train with physio sub-modality masking enabled
        metrics = train_epoch(model, dataloader, optimizer, device, mask_physio_submodality=True)
        scheduler.step()
        
        history.append(metrics)
        
        print(f"  Total loss: {metrics['total_loss']:.4f}")
        print(f"  Audio: {metrics['audio_loss']:.4f}, Physio: {metrics['physio_loss']:.4f}")
        print(f"  OpenFace: {metrics['openface_loss']:.4f}, EyeTracker: {metrics['eyetracker_loss']:.4f}")
        
        # Save checkpoint
        if save_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_path)
    
    return history


# =============================================================================
# Visualization
# =============================================================================

def visualize_reconstruction(
    model: nn.Module,
    dataset: Dataset,
    sample_idx: int = 0,
    device: str = 'cpu',
    save_path: Optional[str] = None,
):
    """Visualize original vs reconstructed signals for all high-level modalities."""
    model.eval()
    
    sample = dataset[sample_idx]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    
    with torch.no_grad():
        outputs = model(
            audio=batch['audio'],
            physio=batch['physio'],
            openface=batch['openface'],
            eyetracker=batch['eyetracker'],
            modality_mask=batch['modality_mask'],
        )
    
    modality_mask = sample['modality_mask']
    modality_names = ['audio', 'physio', 'openface', 'eyetracker']
    
    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    for i, mod_name in enumerate(modality_names):
        ax_orig, ax_recon = axes[i]
        
        original = sample[mod_name].numpy()
        reconstructed = outputs[mod_name][0].cpu().numpy()
        
        available = modality_mask[i].item()
        
        # Plot first few features
        n_features = min(3, original.shape[1])
        
        for j in range(n_features):
            ax_orig.plot(original[:, j], label=f'Feat {j}', alpha=0.7)
            ax_recon.plot(reconstructed[:, j], label=f'Feat {j}', alpha=0.7)
        
        status = "✓" if available else "✗ (masked)"
        ax_orig.set_title(f'{mod_name.upper()} - Original {status}')
        ax_recon.set_title(f'{mod_name.upper()} - Reconstructed')
        ax_orig.legend(loc='upper right', fontsize=8)
        ax_recon.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_physio_cross_reconstruction(
    model: nn.Module,
    dataset: Dataset,
    sample_idx: int = 0,
    device: str = 'cpu',
    save_path: Optional[str] = None,
):
    """
    Visualize cross-reconstruction of individual physio sub-modalities.

    For a given sample, we remove one physio channel at a time from the input
    (set to zero) and let the model reconstruct it from the remaining
    physio channels and the other modalities. We then plot original vs
    reconstructed for all physio sub-modalities.
    """
    model.eval()

    # Try to ensure we pick a sample that actually has physio
    if sample_idx < 0 or sample_idx >= len(dataset):
        sample_idx = 0

    sample = dataset[sample_idx]
    if sample['modality_mask'][1].item() == 0:
        found = False
        for i in range(len(dataset)):
            s = dataset[i]
            if s['modality_mask'][1].item() == 1:
                sample = s
                sample_idx = i
                found = True
                break
        if not found:
            print("No sample with physio modality available for visualization.")
            return

    print(f"Using sample index {sample_idx} for physio cross-reconstruction visualization.")
    
    # Prepare batch (1, seq, ...)
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    
    n_physio = batch['physio'].shape[-1]
    physio_names = EATMINTDataset.PHYSIO_SIGNALS
    
    fig, axes = plt.subplots(n_physio, 1, figsize=(14, 2.5 * n_physio), sharex=True)
    if n_physio == 1:
        axes = [axes]
    
    with torch.no_grad():
        for idx in range(n_physio):
            # Mask one physio sub-modality
            physio_masked = batch['physio'].clone()
            physio_masked[:, :, idx] = 0.0
            
            outputs = model(
                audio=batch['audio'],
                physio=physio_masked,
                openface=batch['openface'],
                eyetracker=batch['eyetracker'],
                modality_mask=batch['modality_mask'],
            )
            
            recon = outputs['physio'][0, :, idx].detach().cpu().numpy()
            original = batch['physio'][0, :, idx].detach().cpu().numpy()
            
            ax = axes[idx]
            ax.plot(original, label='Original', alpha=0.7)
            ax.plot(recon, label='Reconstructed', alpha=0.7, linestyle='--')
            name = physio_names[idx] if idx < len(physio_names) else f'Channel {idx}'
            ax.set_title(f'Physio sub-modality: {name}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_training_history(history: List[Dict[str, float]], save_path: Optional[str] = None):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history) + 1)
    
    # Total loss
    axes[0].plot(epochs, [h['total_loss'] for h in history], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Per-modality loss
    for mod in ['audio', 'physio', 'openface', 'eyetracker']:
        axes[1].plot(epochs, [h[f'{mod}_loss'] for h in history], label=mod, linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Per-Modality Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    
    # Configuration - use absolute path or path relative to script
    script_dir = Path(__file__).parent.parent  # Go up from Perceiver/ to workspace root
    data_root = script_dir / "data" / "researchdata"
    
    config = EATMINTConfig(
        data_root=str(data_root),
        window_size_sec=2.0,
        hop_size_sec=1.0,
    )
    
    print("=" * 60)
    print("EATMINT Multimodal Perceiver")
    print("=" * 60)
    print(f"\nData root: {config.data_root}")
    
    # Check data availability
    print("\nChecking data availability...")
    availability_df = check_modality_availability(config)
    usable_df = print_availability_summary(availability_df)
    
    # Check for precomputed audio features
    precomputed_dir = config.sound_features_dir
    precomputed_files = list(precomputed_dir.glob("*.npy")) if precomputed_dir.exists() else []
    use_precomputed = len(precomputed_files) > 0
    
    print(f"\nPrecomputed audio features: ", end="")
    if use_precomputed:
        print(f"✓ Found {len(precomputed_files)} files in {precomputed_dir}")
    else:
        print(f"✗ Not found. Run 'python tools/precompute.py' first for faster training.")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Initialize audio feature extractor (only if no precomputed features)
    audio_extractor = None    # Always pass availability_df (even if we use usable_df for debug above)
    if not use_precomputed:
        print("\nLoading Wav2Vec2 audio encoder (for on-the-fly extraction)...")
        audio_extractor = AudioFeatureExtractor(
            model_name="facebook/wav2vec2-base",
            device=device
        )
        print("Wav2Vec2 loaded (768-dim features)")
    else:
        print("\nUsing precomputed Wav2Vec2 features (no model loading needed)")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = EATMINTDataset(
        config=config,
        availability_df=availability_df,
        audio_extractor=audio_extractor,
        window_size_sec=2.0,
        hop_size_sec=1.0,
        target_fs=50.0,
        min_modalities=2,
        use_precomputed_audio=use_precomputed,
        preload=True,  # Preload for faster training
    )
    print(f"Dataset: {len(dataset)} windows")
    
    # Test loading a sample
    print("\nTesting sample loading...")
    sample = dataset[0]
    print("Sample shapes:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    # Create model
    print("\nCreating model...")
    model = EATMINTPerceiver(
        hidden_dim=256,
        latent_dim=256,
        num_latents=64,
        num_self_attention_layers=6,
        num_cross_attention_layers=1,
        num_heads=8,
        audio_dim=768,
        physio_dim=len(EATMINTDataset.PHYSIO_SIGNALS),
        openface_dim=len(EATMINTDataset.OPENFACE_COLS),
        eyetracker_dim=len(EATMINTDataset.EYETRACKER_COLS),
        seq_len=dataset.window_samples,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    history = train(
        model=model,
        dataset=dataset,
        n_epochs=10,
        batch_size=8,
        lr=1e-4,
        device=device,
        save_path="eatmint_perceiver_checkpoint.pt",
    )
    
    # Plot results
    plot_training_history(history, save_path="training_history.png")
    
    # Visualize reconstruction of all high-level modalities
    visualize_reconstruction(
        model, dataset, sample_idx=0, device=device,
        save_path="reconstruction.png"
    )
    
    # Visualize physio sub-modality cross-reconstruction (5 physio signals)
    visualize_physio_cross_reconstruction(
        model, dataset, sample_idx=0, device=device,
        save_path="physio_cross_reconstruction.png"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
