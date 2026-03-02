# Deployment (Dev / MVP)

## Local dev

### Backend

```bash
cd emotion_voice_system/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd emotion_voice_system/frontend
npm install
npm run dev
```

## Notes

- The Vite dev server proxies `/api` to `http://localhost:8000`.
- Browser TTS is used for “voice replies”, so there is no backend audio generation in this MVP.

