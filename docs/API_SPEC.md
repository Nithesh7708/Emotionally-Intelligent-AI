# API Spec

## `POST /api/voice-chat`

Accepts an audio upload and returns the detected emotion + an empathetic text reply.

### Request

- Content-Type: `multipart/form-data`
- Fields:
  - `audio` (file): **WAV** recommended (the frontend records WAV)
  - `transcript` (string, optional): if provided by browser speech recognition

### Response (200)

```json
{
  "emotion": "neutral",
  "response_text": "I’m with you. What would you like to talk about or figure out today?"
}
```

### Errors

- `400`: empty/invalid audio or unsupported format
- `500`: unexpected server errors

## `GET /health`

```json
{ "status": "ok" }
```

