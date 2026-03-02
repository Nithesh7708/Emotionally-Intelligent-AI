"""PyTorch Dataset for emotion recognition from WAV files.

Usage
-----
    from app.services.emotion_detection_service.cnn_dataset import build_splits

    train_ds, val_ds, test_ds = build_splits("backend/dataset/raw")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

Directory layout expected::

    raw/
      angry/   *.wav
      fear/    *.wav
      happy/   *.wav
      neutral/ *.wav
      sad/     *.wav
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from app.services.emotion_detection_service.cnn_model import EMOTIONS
from app.services.emotion_detection_service.features import extract_mel_spectrogram

try:
    import librosa
except Exception:
    librosa = None


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _add_noise(y: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    """Add white Gaussian noise at a target SNR (dB)."""
    signal_power = np.mean(y ** 2) + 1e-9
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(y)).astype(np.float32) * np.sqrt(noise_power)
    return y + noise


def _time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Stretch audio by *rate* (>1 = faster, <1 = slower)."""
    if librosa is None:
        return y
    return librosa.effects.time_stretch(y=y, rate=rate).astype(np.float32)


def _pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by *n_steps* semitones."""
    if librosa is None:
        return y
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps).astype(np.float32)


def _augment(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply stochastic augmentation (each transform independently at 50%)."""
    if random.random() < 0.5:
        rate = random.uniform(0.8, 1.2)
        y = _time_stretch(y, rate)

    if random.random() < 0.5:
        n_steps = random.uniform(-2.0, 2.0)
        y = _pitch_shift(y, sr, n_steps)

    if random.random() < 0.5:
        snr = random.uniform(20.0, 30.0)
        y = _add_noise(y, snr_db=snr)

    return y


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class EmotionDataset(Dataset):
    """Reads WAV files and returns mel-spectrogram tensors + integer labels.

    Args:
        samples:   List of ``(wav_path, label_index)`` tuples.
        augment:   If ``True``, apply on-the-fly audio augmentation.
    """

    LABEL_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
    IDX_TO_LABEL: dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}

    def __init__(self, samples: Sequence[tuple[Path, int]], augment: bool = False) -> None:
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        wav_path, label = self.samples[idx]

        try:
            if librosa is None:
                raise RuntimeError("librosa not available")
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
            y = y.astype(np.float32)

            if self.augment:
                y = _augment(y, sr)

            mel = extract_mel_spectrogram(y, sr)  # (1, 128, 128)
        except Exception:
            mel = np.zeros((1, 128, 128), dtype=np.float32)

        return torch.from_numpy(mel), label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_splits(
    raw_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[EmotionDataset, EmotionDataset, EmotionDataset]:
    """Stratified 70/15/15 split of WAV files found under *raw_dir*.

    Each sub-directory name must match one of the 5 emotion labels in
    ``EMOTIONS``.  Files are split per-class so class ratios are preserved.

    Args:
        raw_dir:     Root directory containing ``<emotion>/`` sub-folders.
        train_ratio: Fraction used for training.
        val_ratio:   Fraction used for validation (remainder → test).
        seed:        Random seed for reproducibility.

    Returns:
        ``(train_ds, val_ds, test_ds)`` — three :class:`EmotionDataset`
        instances, with augmentation enabled only on ``train_ds``.
    """
    raw_dir = Path(raw_dir)
    rng = random.Random(seed)

    label_to_idx = EmotionDataset.LABEL_TO_IDX

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []

    for emotion in EMOTIONS:
        emotion_dir = raw_dir / emotion
        if not emotion_dir.is_dir():
            continue

        files = sorted(emotion_dir.glob("*.wav"))
        if not files:
            continue

        rng.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        label = label_to_idx[emotion]
        train_samples.extend((f, label) for f in files[:n_train])
        val_samples.extend((f, label) for f in files[n_train: n_train + n_val])
        test_samples.extend((f, label) for f in files[n_train + n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    train_ds = EmotionDataset(train_samples, augment=True)
    val_ds = EmotionDataset(val_samples, augment=False)
    test_ds = EmotionDataset(test_samples, augment=False)

    print(
        f"[Dataset] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
        f"total={len(train_ds)+len(val_ds)+len(test_ds)}"
    )
    return train_ds, val_ds, test_ds
