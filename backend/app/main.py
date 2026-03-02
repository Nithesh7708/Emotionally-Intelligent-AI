from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.emotion_routes.router import router as emotion_router


app = FastAPI(title="Emotion Aware Voice Chat System", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(emotion_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}

