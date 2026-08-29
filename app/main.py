"""FastAPI server for the voice-enabled chatbot.

Endpoints
  GET  /              -> the chat UI
  POST /api/chat      -> {"text": "..."}          text-only turn
  POST /api/voice     -> multipart audio upload   full voice turn
  POST /api/transcribe-> multipart audio upload   speech recognition only
  GET  /api/info      -> model + intent metadata
  GET  /health        -> liveness probe
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-chatbot")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_AUDIO_BYTES = 25 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models before the container starts accepting traffic, so the
    # first visitor does not eat the ~10s cold load.
    log.info("loading models ...")
    pipeline.warmup()
    log.info("models ready")
    yield


app = FastAPI(title="Voice-Enabled Chatbot", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/info")
def info() -> dict:
    return pipeline.model_info()


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    result = pipeline.chat(text)
    result["transcript"] = text
    result["source"] = "text"
    log.info("text  %-40r -> %s (%.2f)", text[:40], result["intent"], result["confidence"])
    return result


async def _read_audio(file: UploadFile) -> bytes:
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio upload")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio too large (max 25 MB)")
    return audio


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...)) -> dict:
    data = await _read_audio(audio)
    try:
        result = pipeline.voice_chat(data)
    except Exception as exc:  # decoding failures surface here
        log.exception("voice turn failed")
        raise HTTPException(status_code=422, detail=f"could not process audio: {exc}") from exc
    result["source"] = "voice"
    log.info(
        "voice %-40r -> %s (%.2f) stt=%dms",
        result["transcript"][:40], result["intent"], result["confidence"], result["stt_ms"],
    )
    return result


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    data = await _read_audio(audio)
    try:
        return pipeline.transcribe(data)
    except Exception as exc:
        log.exception("transcription failed")
        raise HTTPException(status_code=422, detail=f"could not decode audio: {exc}") from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
