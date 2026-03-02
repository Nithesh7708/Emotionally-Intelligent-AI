"""Organise downloaded datasets into dataset/raw/<emotion>/*.wav.

Usage
-----
    cd backend
    python prepare_dataset.py

Source directories (produced by download_datasets.py)
------------------------------------------------------
    dataset/downloads/ravdess/     RAVDESS actor folders
    dataset/downloads/cremad/AudioWAV/  CREMA-D WAV files
    dataset/downloads/tess/        TESS sub-folders

Emotion mapping
---------------
    RAVDESS filename: 03-01-<EMOTION>-…
        01,02 → neutral
        03    → happy
        04    → sad
        05    → angry
        06    → fear
        (07,08 — disgust/surprised — skipped)

    CREMA-D filename: <ID>_<WORD>_<EMOTION>_<LEVEL>.wav
        NEU → neutral | HAP → happy | SAD → sad | ANG → angry | FEA → fear
        (DIS — skipped)

    TESS filename: OAF_<word>_<emotion>.wav  or  YAF_<word>_<emotion>.wav
        angry, fear, happy, neutral, sad  (direct)
        (disgust, ps/pleasant_surprise — skipped)

Output
------
    dataset/raw/
        angry/    ravdess_<name>.wav  cremad_<name>.wav  tess_<name>.wav  …
        fear/     …
        happy/    …
        neutral/  …
        sad/      …
"""
from __future__ import annotations

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BACKEND_DIR / "dataset" / "downloads"
RAW_DIR = BACKEND_DIR / "dataset" / "raw"

EMOTIONS = ["angry", "fear", "happy", "neutral", "sad"]

# Ensure output directories exist
for _e in EMOTIONS:
    (RAW_DIR / _e).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy(src: Path, dest_dir: Path, prefix: str) -> None:
    """Copy *src* to *dest_dir* with a unique *prefix* prepended to its name."""
    dest = dest_dir / f"{prefix}_{src.name}"
    if dest.exists():
        return
    shutil.copy2(src, dest)


def _count_raw() -> dict[str, int]:
    return {e: len(list((RAW_DIR / e).glob("*.wav"))) for e in EMOTIONS}


# ---------------------------------------------------------------------------
# RAVDESS
# ---------------------------------------------------------------------------
# File naming: 03-01-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav
# Modality 03 = audio-only speech; emotion field (3rd) is relevant.

RAVDESS_MAP = {
    "01": "neutral",
    "02": "neutral",  # calm → neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    # 07 = disgust, 08 = surprised — skipped
}


def process_ravdess() -> int:
    ravdess_root = DOWNLOADS_DIR / "ravdess"
    if not ravdess_root.is_dir():
        print("  [skip] RAVDESS not found. Run download_datasets.py first.")
        return 0

    count = 0
    for wav in ravdess_root.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]
        emotion = RAVDESS_MAP.get(emotion_code)
        if emotion is None:
            continue
        _copy(wav, RAW_DIR / emotion, "ravdess")
        count += 1

    return count


# ---------------------------------------------------------------------------
# CREMA-D
# ---------------------------------------------------------------------------
# File naming: <ActorID>_<Sentence>_<Emotion>_<Level>.wav

CREMAD_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    # DIS = disgust — skipped
}


def process_cremad() -> int:
    cremad_root = DOWNLOADS_DIR / "cremad" / "AudioWAV"
    if not cremad_root.is_dir():
        print("  [skip] CREMA-D AudioWAV not found. Run download_datasets.py first.")
        return 0

    count = 0
    for wav in cremad_root.rglob("*.wav"):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        emotion_code = parts[2].upper()
        emotion = CREMAD_MAP.get(emotion_code)
        if emotion is None:
            continue
        _copy(wav, RAW_DIR / emotion, "cremad")
        count += 1

    return count


# ---------------------------------------------------------------------------
# TESS
# ---------------------------------------------------------------------------
# Folder names: OAF_<emotion>  or  YAF_<emotion>
# File names:   OAF_<word>_<emotion>.wav  or  YAF_<word>_<emotion>.wav

TESS_MAP = {
    "angry": "angry",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    # disgust / ps → skipped
}


def process_tess() -> int:
    tess_root = DOWNLOADS_DIR / "tess"
    if not tess_root.is_dir():
        print("  [skip] TESS not found. Run download_datasets.py first.")
        return 0

    count = 0
    for wav in tess_root.rglob("*.wav"):
        # Try to derive emotion from filename last segment after split
        # e.g. OAF_back_angry.wav → angry
        emotion_str = wav.stem.split("_")[-1].lower()
        emotion = TESS_MAP.get(emotion_str)
        if emotion is None:
            continue
        _copy(wav, RAW_DIR / emotion, "tess")
        count += 1

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Dataset Preparation — organising into dataset/raw/<emotion>/")
    print("=" * 60)

    print("\n>>> [1/3] Processing RAVDESS …")
    n_rav = process_ravdess()
    print(f"  Copied {n_rav} RAVDESS files.")

    print("\n>>> [2/3] Processing CREMA-D …")
    n_cre = process_cremad()
    print(f"  Copied {n_cre} CREMA-D files.")

    print("\n>>> [3/3] Processing TESS …")
    n_tes = process_tess()
    print(f"  Copied {n_tes} TESS files.")

    totals = _count_raw()
    grand_total = sum(totals.values())

    print("\n" + "=" * 60)
    print(f"Total files per class:")
    for emotion, cnt in totals.items():
        bar = "#" * (cnt // 50)
        print(f"  {emotion:<10} {cnt:>5}  {bar}")
    print(f"\n  Grand total: {grand_total} WAV files")
    print("=" * 60)

    if grand_total < 100:
        print("\nWARNING: Very few files found.")
        print("Make sure you ran download_datasets.py and all downloads succeeded.")
    else:
        print("\nNext step:  python train_cnn.py --epochs 100 --batch-size 32 --num-workers 0")


if __name__ == "__main__":
    main()
