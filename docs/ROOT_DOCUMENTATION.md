# Root Documentation

## Architecture (logical)

Frontend (Voice UI)
→ `POST /api/voice-chat` (audio upload)
→ Audio preprocessing + feature extraction
→ Emotion prediction (ML model if trained; else neutral)
→ Response generation (rule-based templates)
→ Frontend displays emotion + text, then speaks via SpeechSynthesis

## Emotion behavior

| Emotion  | AI behavior style |
|---------|-------------------|
| happy   | Energetic, positive |
| sad     | Soft, comforting |
| angry   | Calm, grounding |
| fear    | Reassuring |
| neutral | Balanced |

