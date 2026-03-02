import io

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile


async def load_audio_from_upload(audio: UploadFile) -> tuple[np.ndarray, int]:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing audio file name.")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. Please upload WAV audio "
                "(the frontend records WAV by default)."
            ),
        ) from exc

    if data is None or (hasattr(data, "size") and data.size == 0):
        raise HTTPException(status_code=400, detail="Could not decode audio.")

    if data.ndim == 2:
        data = np.mean(data, axis=1)

    sr_int = int(sr)
    return data.astype(np.float32, copy=False), sr_int

