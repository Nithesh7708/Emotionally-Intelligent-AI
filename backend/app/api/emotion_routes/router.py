from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.voice_chat import VoiceChatResponse
from app.services.audio_processing_service.loader import load_audio_from_upload
from app.services.emotion_detection_service.predictor import EmotionPredictor
from app.services.response_generation_service.generate import generate_response_text

router = APIRouter(tags=["voice-chat"])

_predictor = EmotionPredictor()


@router.post("/voice-chat", response_model=VoiceChatResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    transcript: str | None = Form(default=None),
):
    waveform, sample_rate = await load_audio_from_upload(audio)
    emotion = _predictor.predict_emotion(waveform=waveform, sample_rate=sample_rate)
    response_text = generate_response_text(emotion=emotion, transcript=transcript)
    return VoiceChatResponse(emotion=emotion, response_text=response_text)

