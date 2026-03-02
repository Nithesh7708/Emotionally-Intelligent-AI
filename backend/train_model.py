from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.services.emotion_detection_service.features import EMOTIONS, FEATURE_VERSION, extract_features, feature_dim


def iter_wav_files(dataset_raw: Path):
    for emotion in EMOTIONS:
        folder = dataset_raw / emotion
        if not folder.exists():
            continue
        for path in folder.rglob("*.wav"):
            yield emotion, path


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/raw", help="Path to dataset/raw")
    parser.add_argument("--out", default="app/models/emotion_model/model.joblib", help="Output model path")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dataset_raw = (here / args.dataset).resolve()
    out_path = (here / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X: list[np.ndarray] = []
    y: list[str] = []

    for emotion, wav_path in iter_wav_files(dataset_raw):
        waveform, sr = load_wav(wav_path)
        feats = extract_features(waveform=waveform, sample_rate=sr)
        if feats.shape[0] != feature_dim():
            continue
        X.append(feats)
        y.append(emotion)

    if not X:
        raise SystemExit(f"No .wav files found under: {dataset_raw}")

    X_mat = np.vstack([v.reshape(1, -1) for v in X])
    y_arr = np.asarray(y)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale")),
        ]
    )
    pipeline.fit(X_mat, y_arr)

    payload = {
        "pipeline": pipeline,
        "feature_version": FEATURE_VERSION,
        "classes": EMOTIONS,
        "feature_dim": int(feature_dim()),
        "num_samples": int(len(y_arr)),
    }
    joblib.dump(payload, out_path)
    print(f"Saved model -> {out_path} (samples={len(y_arr)})")


if __name__ == "__main__":
    main()

