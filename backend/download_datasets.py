"""Download RAVDESS, CREMA-D, and TESS datasets to backend/dataset/downloads/.

Usage
-----
    cd backend
    python download_datasets.py

Downloads
---------
    RAVDESS  (~750 MB)  — Zenodo  (public, no login)
    CREMA-D  (~2.8 GB) — GitHub  (public, no login)
    TESS     (~450 MB)  — Kaggle  (free account + API key required)

After downloading, run::

    python prepare_dataset.py

to organise files into ``dataset/raw/<emotion>/``.
"""
from __future__ import annotations

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import Callable

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = BACKEND_DIR / "dataset" / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(filename: str) -> Callable:
    """Return an urllib reporthook that prints a tqdm progress bar."""
    if not HAS_TQDM:
        def _simple(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                print(f"\r  {filename}: {pct}%", end="", flush=True)
        return _simple

    pbar: list = []  # mutable closure container

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if not pbar:
            pbar.append(tqdm(total=total_size, unit="B", unit_scale=True, desc=f"  {filename}"))
        pbar[0].update(block_size)

    return _hook


def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download *url* to *dest* (skip if already present)."""
    if dest.exists():
        print(f"  [skip] {dest.name} already downloaded.")
        return dest
    print(f"\nDownloading {desc or dest.name} …")
    hook = _progress_hook(dest.name)
    urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    print()  # newline after progress
    return dest


def _extract_zip(zip_path: Path, extract_to: Path, desc: str = "") -> None:
    """Extract *zip_path* into *extract_to* (skip if marker exists)."""
    marker = extract_to / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"  [skip] {desc or zip_path.name} already extracted.")
        return
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    marker.touch()
    print(f"  Done → {extract_to}")


# ---------------------------------------------------------------------------
# RAVDESS — Zenodo (public, direct download)
# ---------------------------------------------------------------------------
# The full audio-only actor archive per actor, 24 actors total.
# We download the single all-actors zip from the Zenodo record.

RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/"
    "Audio_Speech_Actors_01-24.zip?download=1"
)


def download_ravdess() -> None:
    dest_zip = DOWNLOADS_DIR / "ravdess_speech.zip"
    _download_file(RAVDESS_URL, dest_zip, "RAVDESS Speech (Zenodo)")
    _extract_zip(dest_zip, DOWNLOADS_DIR / "ravdess", "RAVDESS")


# ---------------------------------------------------------------------------
# CREMA-D — GitHub releases (public)
# ---------------------------------------------------------------------------
# AudioWAV directory is in a separate release asset (~2.8 GB split into parts).
# The simplest approach: clone sparse checkout of just the AudioWAV folder.

def download_cremad() -> None:
    cremad_dir = DOWNLOADS_DIR / "cremad" / "AudioWAV"
    if cremad_dir.is_dir() and any(cremad_dir.glob("*.wav")):
        print("  [skip] CREMA-D AudioWAV already present.")
        return

    print("\nDownloading CREMA-D via git sparse-checkout …")
    print("  (This clones ~2.8 GB — may take 10–20 min on slow connections.)")

    repo_dir = DOWNLOADS_DIR / "cremad_repo"
    if not repo_dir.is_dir():
        os.system(
            f'git clone --filter=blob:none --no-checkout '
            f'"https://github.com/CheyneyComputerScience/CREMA-D.git" '
            f'"{repo_dir}"'
        )
    os.system(f'git -C "{repo_dir}" sparse-checkout set AudioWAV')
    os.system(f'git -C "{repo_dir}" checkout')

    # Copy AudioWAV to canonical path
    src = repo_dir / "AudioWAV"
    if src.is_dir():
        import shutil
        dest = DOWNLOADS_DIR / "cremad"
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "AudioWAV").exists():
            shutil.copytree(str(src), str(dest / "AudioWAV"))
        print(f"  CREMA-D → {dest / 'AudioWAV'}")
    else:
        print("  ERROR: git sparse-checkout of CREMA-D failed — see messages above.")
        print("  Alternative: download manually from https://github.com/CheyneyComputerScience/CREMA-D")


# ---------------------------------------------------------------------------
# TESS — Kaggle (free, requires Kaggle API key)
# ---------------------------------------------------------------------------
# Dataset: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
# Requires ~/.kaggle/kaggle.json with your API credentials.

TESS_KAGGLE_DATASET = "ejlok1/toronto-emotional-speech-set-tess"


def download_tess() -> None:
    tess_dir = DOWNLOADS_DIR / "tess"
    if tess_dir.is_dir() and any(tess_dir.rglob("*.wav")):
        print("  [skip] TESS already present.")
        return

    print("\nDownloading TESS via Kaggle API …")
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "  ERROR: ~/.kaggle/kaggle.json not found.\n"
            "  Steps to fix:\n"
            "    1. Create a free Kaggle account at https://www.kaggle.com\n"
            "    2. Go to Account → API → Create New Token\n"
            "    3. Place the downloaded kaggle.json in ~/.kaggle/\n"
            "    4. Re-run this script.\n"
            "  Alternatively, download TESS manually:\n"
            f"    https://www.kaggle.com/datasets/{TESS_KAGGLE_DATASET}\n"
            f"  and unzip into: {tess_dir}\n"
        )
        return

    tess_dir.mkdir(parents=True, exist_ok=True)
    ret = os.system(
        f'kaggle datasets download -d "{TESS_KAGGLE_DATASET}" '
        f'--path "{tess_dir}" --unzip'
    )
    if ret != 0:
        print("  ERROR: kaggle command failed. Install with: pip install kaggle")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Emotion Dataset Downloader")
    print("=" * 60)
    print(f"Download destination: {DOWNLOADS_DIR}\n")

    print(">>> [1/3] RAVDESS")
    try:
        download_ravdess()
    except Exception as exc:
        print(f"  ERROR during RAVDESS download: {exc}")

    print("\n>>> [2/3] CREMA-D")
    try:
        download_cremad()
    except Exception as exc:
        print(f"  ERROR during CREMA-D download: {exc}")

    print("\n>>> [3/3] TESS")
    try:
        download_tess()
    except Exception as exc:
        print(f"  ERROR during TESS download: {exc}")

    print("\n" + "=" * 60)
    print("Download complete.")
    print("Next step:  python prepare_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
