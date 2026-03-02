"""Emotion predictor — CNN primary, SVM fallback, neutral safe default.

Priority chain
--------------
1. **CNN** (primary)  — loads ``emotion_cnn.pt`` if torch is available and the
   file exists.  Returns one of: happy / sad / angry / fear / neutral.
2. **SVM** (fallback) — loads ``model.joblib`` if CNN is not available.
3. **"neutral"** (safe default) — returned if no model file is found at all.

The FastAPI router calls ``predictor.predict_emotion(waveform, sr)`` and does
not need to know which backend is active.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.emotion_detection_service.features import EMOTIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model directory (shared by both CNN and SVM)
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "emotion_model"
_CNN_PATH = _MODEL_DIR / "emotion_cnn.pt"
_SVM_PATH = _MODEL_DIR / "model.joblib"


# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

@dataclass
class EmotionPredictor:
    """Unified predictor that selects CNN or SVM automatically at first call."""

    # Internal state — populated lazily on first predict call
    _backend: str = field(default="none", init=False, repr=False)
    _cnn_model: Any = field(default=None, init=False, repr=False)
    _cnn_device: Any = field(default=None, init=False, repr=False)
    _svm_model: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init(self) -> None:
        """Select and load the best available model."""
        if self._backend != "none":
            return  # already initialised

        # --- Try CNN first ---
        if _torch_available() and _CNN_PATH.exists():
            try:
                self._load_cnn()
                self._backend = "cnn"
                logger.info("EmotionPredictor: using CNN backend (%s)", _CNN_PATH.name)
                return
            except Exception as exc:
                logger.warning("EmotionPredictor: CNN load failed (%s), trying SVM …", exc)

        # --- Try SVM fallback ---
        if _SVM_PATH.exists():
            try:
                self._load_svm()
                self._backend = "svm"
                logger.info("EmotionPredictor: using SVM backend (%s)", _SVM_PATH.name)
                return
            except Exception as exc:
                logger.warning("EmotionPredictor: SVM load failed (%s), using neutral default.", exc)

        # --- No model available ---
        self._backend = "default"
        logger.warning(
            "EmotionPredictor: no model found — always returning 'neutral'.\n"
            "  CNN path checked : %s\n"
            "  SVM path checked : %s\n"
            "  To train the CNN: cd backend && python train_cnn.py",
            _CNN_PATH,
            _SVM_PATH,
        )

    def _load_cnn(self) -> None:
        import torch
        from app.services.emotion_detection_service.cnn_model import load_cnn_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cnn_model = load_cnn_model(_CNN_PATH, device=device)
        self._cnn_device = device

    def _load_svm(self) -> None:
        import joblib
        payload = joblib.load(_SVM_PATH)
        self._svm_model = payload.get("pipeline") if isinstance(payload, dict) else payload

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_emotion(self, waveform: np.ndarray, sample_rate: int) -> str:
        """Return the predicted emotion label for a raw audio waveform.

        Args:
            waveform:    1-D (or 2-D) float32 array of audio samples.
            sample_rate: Sample rate of *waveform* in Hz.

        Returns:
            One of ``"happy"``, ``"sad"``, ``"angry"``, ``"fear"``,
            ``"neutral"``.  Falls back to ``"neutral"`` on any error.
        """
        self._init()

        if self._backend == "cnn":
            return self._predict_cnn(waveform, sample_rate)
        if self._backend == "svm":
            return self._predict_svm(waveform, sample_rate)
        return "neutral"

    def _predict_cnn(self, waveform: np.ndarray, sample_rate: int) -> str:
        try:
            import torch
            from app.services.emotion_detection_service.features import extract_mel_spectrogram
            from app.services.emotion_detection_service.cnn_model import EMOTIONS as CNN_EMOTIONS

            mel = extract_mel_spectrogram(waveform, sample_rate)  # (1, 128, 128)
            tensor = torch.from_numpy(mel).unsqueeze(0).to(self._cnn_device)  # (1, 1, 128, 128)

            with torch.no_grad():
                logits = self._cnn_model(tensor)
                pred_idx = int(logits.argmax(dim=1).item())

            emotion = CNN_EMOTIONS[pred_idx]
            return emotion if emotion in EMOTIONS else "neutral"

        except Exception as exc:
            logger.error("CNN prediction failed: %s", exc)
            return "neutral"

    def _predict_svm(self, waveform: np.ndarray, sample_rate: int) -> str:
        try:
            from app.services.emotion_detection_service.features import extract_features
            feats = extract_features(waveform=waveform, sample_rate=sample_rate).reshape(1, -1)
            pred = self._svm_model.predict(feats)[0]
            pred_str = str(pred)
            return pred_str if pred_str in EMOTIONS else "neutral"
        except Exception as exc:
            logger.error("SVM prediction failed: %s", exc)
            return "neutral"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def active_backend(self) -> str:
        """Return which backend is active: ``'cnn'``, ``'svm'``, or ``'default'``."""
        self._init()
        return self._backend
