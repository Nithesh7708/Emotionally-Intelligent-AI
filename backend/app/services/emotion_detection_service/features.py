from __future__ import annotations

import numpy as np

try:
    import librosa
except Exception:  # noqa: BLE001
    librosa = None


FEATURE_VERSION = 1
EMOTIONS = ["happy", "sad", "angry", "fear", "neutral"]


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def extract_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is required for feature extraction.")

    y = np.asarray(waveform, dtype=np.float32)
    if y.size == 0:
        return np.zeros((feature_dim(),), dtype=np.float32)

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = librosa.to_mono(y) if y.ndim > 1 else y

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if y_trim.size >= int(0.5 * sample_rate):
        y = y_trim

    max_len = int(8.0 * sample_rate)
    if y.size > max_len:
        y = y[:max_len]

    target_sr = 16000
    if sample_rate != target_sr and sample_rate > 0:
        y = librosa.resample(y=y, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    d_mean = np.mean(mfcc_delta, axis=1)
    d_std = np.std(mfcc_delta, axis=1)

    rms = librosa.feature.rms(y=y)[0]
    rms_mean, rms_std = _safe_mean_std(rms)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean, zcr_std = _safe_mean_std(zcr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    cent_mean, cent_std = _safe_mean_std(centroid)

    try:
        f0 = librosa.yin(y=y, fmin=50, fmax=400, sr=sample_rate)
        voiced = np.isfinite(f0)
        voiced_rate = float(np.mean(voiced)) if f0.size else 0.0
        f0_voiced = f0[voiced]
        f0_mean, f0_std = _safe_mean_std(f0_voiced)
    except Exception:  # noqa: BLE001
        f0_mean, f0_std, voiced_rate = 0.0, 0.0, 0.0

    feats = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            d_mean,
            d_std,
            np.array(
                [rms_mean, rms_std, zcr_mean, zcr_std, cent_mean, cent_std, f0_mean, f0_std, voiced_rate],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )

    if feats.shape[0] != feature_dim():
        out = np.zeros((feature_dim(),), dtype=np.float32)
        n = min(out.shape[0], feats.shape[0])
        out[:n] = feats[:n]
        return out

    return feats.astype(np.float32, copy=False)


def feature_dim() -> int:
    n_mfcc = 13
    mfcc_blocks = 4 * n_mfcc  # mean/std + delta mean/std
    extras = 9  # rms(2) + zcr(2) + centroid(2) + f0(2) + voiced_rate(1)
    return mfcc_blocks + extras


# ---------------------------------------------------------------------------
# Mel-spectrogram extraction (for CNN pipeline)
# ---------------------------------------------------------------------------

_MEL_SR = 16_000        # target sample rate
_MEL_DURATION = 3.0    # seconds — pad / crop to this length
_MEL_N_MELS = 128
_MEL_N_FFT = 2048
_MEL_HOP = 512
_MEL_SIZE = 128         # final image size (height=n_mels, width=time frames → resized)


def _preprocess_audio(waveform: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Return a clean, normalised, 3-second mono waveform at 16 kHz.

    Steps:
        1. Ensure float32 and clean NaN / Inf.
        2. Convert to mono (average channels).
        3. Normalise to peak amplitude 1.0.
        4. Trim leading/trailing silence (top_db=25).
        5. Resample to 16 kHz.
        6. Pad (zero) or crop to exactly 3 s.
    """
    if librosa is None:
        raise RuntimeError("librosa is required for mel-spectrogram extraction.")

    y = np.asarray(waveform, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # mono
    if y.ndim > 1:
        y = np.mean(y, axis=0) if y.shape[0] < y.shape[-1] else np.mean(y, axis=-1)

    # normalise
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # trim silence
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if y_trim.size >= int(0.5 * sr):
        y = y_trim

    # resample
    if sr != _MEL_SR and sr > 0:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=_MEL_SR)
        sr = _MEL_SR

    # pad / crop to _MEL_DURATION seconds
    target_len = int(_MEL_DURATION * sr)
    if y.size < target_len:
        y = np.pad(y, (0, target_len - y.size))
    else:
        y = y[:target_len]

    return y, sr


def extract_mel_spectrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Compute a log-mel spectrogram and return shape ``(1, 128, 128)`` in [0, 1].

    Args:
        waveform: Raw audio samples (any sample rate, any channel count).
        sr:       Sample rate of *waveform*.

    Returns:
        ``np.ndarray`` of shape ``(1, 128, 128)`` with dtype ``float32``,
        values in ``[0.0, 1.0]``.
    """
    if librosa is None:
        raise RuntimeError("librosa is required for mel-spectrogram extraction.")

    y, sr = _preprocess_audio(waveform, sr)

    # log-mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=_MEL_N_MELS,
        n_fft=_MEL_N_FFT,
        hop_length=_MEL_HOP,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (n_mels, time_frames)

    # resize to (128, 128) using pure numpy (no cv2 required)
    target_h, target_w = _MEL_SIZE, _MEL_SIZE
    src_h, src_w = mel_db.shape

    if src_h != target_h or src_w != target_w:
        # bilinear-like resize via repeated numpy indexing
        row_idx = (np.arange(target_h) * src_h / target_h).astype(np.int32).clip(0, src_h - 1)
        col_idx = (np.arange(target_w) * src_w / target_w).astype(np.int32).clip(0, src_w - 1)
        mel_db = mel_db[np.ix_(row_idx, col_idx)]

    # normalise to [0, 1]
    d_min, d_max = mel_db.min(), mel_db.max()
    if d_max > d_min:
        mel_db = (mel_db - d_min) / (d_max - d_min)
    else:
        mel_db = np.zeros_like(mel_db)

    return mel_db[np.newaxis, :, :].astype(np.float32)  # (1, 128, 128)

