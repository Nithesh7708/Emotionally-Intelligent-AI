from pydantic import BaseModel, Field


class VoiceChatResponse(BaseModel):
    emotion: str = Field(..., examples=["neutral"])
    response_text: str

