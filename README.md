# Emotion Voice Chat

A full-stack voice assistant that listens to you, detects your emotion in real-time, and replies with an empathetic response — all from your browser.

**Stack:** React + Vite (frontend) · FastAPI + PyTorch/scikit-learn (backend)

---

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [1. Backend](#1-backend)
  - [2. Frontend](#2-frontend)
- [Train the Emotion Model](#train-the-emotion-model)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## How It Works

```
[Browser mic] → WAV audio → [FastAPI backend]
                                   │
                          Audio feature extraction
                          (MFCC, chroma, mel-spec)
                                   │
                          Emotion classifier (CNN / SVM)
                                   │
                          Detected emotion label
                          (angry · disgust · fear ·
                           happy · neutral · sad)
                                   │
                          Rule-based response text
                                   │
                    ← JSON response ← [Browser TTS speaks reply]
```

---

## Requirements

| Tool | Minimum version |
|------|----------------|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

> **GPU optional.** The CNN trainer uses PyTorch and will use CUDA if available, otherwise CPU.

---

## Quick Start

### 1. Backend

```bash
# from the repo root
cd backend

# create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# start the API server
uvicorn app.main:app --reload
```

The backend starts at **http://localhost:8000**

Quick health check:
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Interactive API docs: **http://localhost:8000/docs**

---

### 2. Frontend

Open a **second terminal**:

```bash
# from the repo root
cd frontend

# install dependencies (first time only)
npm install

# start the dev server
npm run dev
```

Open **http://localhost:5173** in your browser.

> Allow microphone access when the browser asks. Click the mic button, speak, and the system will detect your emotion and reply.

---

## Train the Emotion Model

The backend ships with a fallback to `neutral` when no model file is present. To train a real model:

### Option A — SVM (fast, CPU-only)

1. Place your WAV files under `backend/dataset/raw/<emotion>/`

   ```
   backend/dataset/raw/
   ├── angry/       *.wav
   ├── disgust/     *.wav
   ├── fear/        *.wav
   ├── happy/       *.wav
   ├── neutral/     *.wav
   └── sad/         *.wav
   ```

2. Run the trainer:

   ```bash
   cd backend
   python train_model.py
   # optional flags:
   #   --dataset path/to/raw    (default: dataset/raw)
   #   --out     path/to/model  (default: app/models/emotion_model/model.joblib)
   ```

### Option B — CNN (higher accuracy, uses PyTorch)

```bash
cd backend
python train_cnn.py
```

The trained model is saved to `backend/app/models/emotion_model/`.
Restart the backend after training so it loads the new model.

### Download the CREMA-D dataset (optional)

```bash
cd backend
python download_datasets.py   # clones CREMA-D audio into dataset/downloads/
python prepare_dataset.py     # organises WAVs into dataset/raw/<emotion>/
```

> Audio files are excluded from this repository (`.gitignore`). You must download or provide your own dataset.

---

## Project Structure

```
emotion-voice-chat/
├── backend/
│   ├── app/
│   │   ├── main.py                          # FastAPI entry point
│   │   ├── api/emotion_routes/router.py     # POST /api/voice-chat
│   │   ├── schemas/voice_chat.py            # Request / response models
│   │   ├── services/
│   │   │   ├── audio_processing_service/    # Load & decode audio
│   │   │   ├── emotion_detection_service/   # Feature extraction + predictor
│   │   │   ├── response_generation_service/ # Rule-based reply text
│   │   │   └── text_to_speech_service/      # (future TTS hook)
│   │   └── models/emotion_model/            # Saved model files (gitignored)
│   ├── dataset/                             # Raw WAVs for training (gitignored)
│   ├── train_model.py                       # SVM trainer
│   ├── train_cnn.py                         # CNN trainer
│   ├── download_datasets.py                 # CREMA-D downloader
│   ├── prepare_dataset.py                   # Dataset organiser
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                          # Main UI component
│   │   └── main.jsx                         # React entry point
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
├── docs/
│   ├── API_SPEC.md                          # Detailed API documentation
│   ├── DATASET_GUIDE.md                     # Dataset preparation guide
│   └── DEPLOYMENT.md                        # Production deployment notes
└── README.md
```

---

## API Reference

### `POST /api/voice-chat`

Accepts an audio recording and returns the detected emotion with an empathetic reply.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | yes | WAV file (browser records WAV by default) |
| `transcript` | string | no | Optional speech-to-text transcript from the browser |

**Response 200**

```json
{
  "emotion": "sad",
  "response_text": "I can hear that you're feeling down. I'm here — want to talk about it?"
}
```

**Error codes**

| Code | Reason |
|------|--------|
| 400 | Empty, invalid, or unsupported audio format |
| 500 | Unexpected server error |

### `GET /health`

```json
{ "status": "ok" }
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` on backend start | Activate your virtual environment and run `pip install -r requirements.txt` |
| Emotion always returns `neutral` | No trained model found — train one with `python train_model.py` |
| Mic button does nothing | Browser needs microphone permission; check the address bar padlock |
| CORS error in browser console | Make sure the backend is running on port 8000 |
| Frontend can't reach backend | Confirm `npm run dev` is on port 5173 and `uvicorn` is on port 8000 |
| `libsndfile` not found (Linux) | `sudo apt install libsndfile1` |

---

## Docs

See [`docs/`](docs/) for:
- [`API_SPEC.md`](docs/API_SPEC.md) — full API details
- [`DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) — dataset preparation
- [`DEPLOYMENT.md`](docs/DEPLOYMENT.md) — production deployment
