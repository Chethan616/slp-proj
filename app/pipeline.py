"""The two-stage voice pipeline: Whisper for speech recognition, DistilBERT for
intent classification, template lookup for the reply.

Both models are loaded once at import time and reused across requests -- on a
2-vCPU container, reloading per request would dominate the latency budget.
"""

import io
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

# Below this softmax probability we treat the utterance as out-of-scope even if
# the argmax says otherwise. Tuned on the validation set -- see
# results/threshold_sweep.png and train/evaluate.py.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.44"))

# base.en is English-only, which is both faster and more accurate on English
# than the multilingual base model of the same size. int8 quantisation roughly
# halves memory and speeds up CPU inference with negligible WER cost.
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "base.en")

_whisper: WhisperModel | None = None
_tokenizer = None
_classifier = None
_labels: list[str] = []
_responses: dict[str, list[str]] = {}


def _load() -> None:
    """Idempotent lazy load of both models plus the response templates."""
    global _whisper, _tokenizer, _classifier, _labels, _responses
    if _classifier is not None:
        return

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

    _whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _classifier.eval()

    _labels = json.loads((MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
    _responses = json.loads((APP_DIR / "responses.json").read_text(encoding="utf-8"))


def warmup() -> None:
    """Load models and run one tiny forward pass so the first real request is
    not paying initialisation cost."""
    _load()
    classify("hello")


def transcribe(audio_bytes: bytes) -> dict:
    """Speech -> text. Accepts any container PyAV can decode (webm/opus from
    MediaRecorder, wav, mp3, m4a)."""
    _load()
    t0 = time.perf_counter()
    segments, info = _whisper.transcribe(
        io.BytesIO(audio_bytes),
        beam_size=1,          # greedy: ~2x faster, negligible WER change here
        language="en",
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {
        "transcript": text,
        "stt_ms": round((time.perf_counter() - t0) * 1000),
        "audio_seconds": round(info.duration, 2),
    }


def classify(text: str) -> dict:
    """Text -> intent. Returns the argmax plus the top 3 for the UI."""
    _load()
    t0 = time.perf_counter()
    enc = _tokenizer(
        text, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _classifier(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]

    top_conf, top_idx = probs.topk(min(3, len(_labels)))
    top = [
        {"intent": _labels[i], "confidence": round(float(c), 4)}
        for c, i in zip(top_conf, top_idx)
    ]
    intent, confidence = top[0]["intent"], top[0]["confidence"]

    # Two routes to "I don't know": the model predicts the oos class outright,
    # or it is not confident enough in whatever it did predict.
    below_threshold = confidence < CONF_THRESHOLD
    if below_threshold:
        intent = "oos"

    return {
        "intent": intent,
        "predicted_intent": top[0]["intent"],
        "confidence": confidence,
        "below_threshold": below_threshold,
        "top": top,
        "infer_ms": round((time.perf_counter() - t0) * 1000),
    }


def respond(intent: str) -> str:
    """Pick a reply template and fill any live placeholders."""
    _load()
    template = random.choice(_responses.get(intent, _responses["oos"]))
    now = datetime.now()
    return (
        template.replace("{time}", now.strftime("%I:%M %p").lstrip("0"))
        .replace("{date}", now.strftime("%A, %d %B %Y"))
        .replace("{coin}", random.choice(["heads", "tails"]))
    )


def chat(text: str) -> dict:
    """Full text-side turn: classify, then answer."""
    result = classify(text)
    result["reply"] = respond(result["intent"])
    return result


def voice_chat(audio_bytes: bytes) -> dict:
    """Full voice turn: transcribe, classify, then answer."""
    stt = transcribe(audio_bytes)
    if not stt["transcript"]:
        return {
            **stt,
            "intent": "oos",
            "confidence": 0.0,
            "top": [],
            "infer_ms": 0,
            "reply": "I didn't catch any speech there -- try again a bit closer to the mic.",
            "empty_audio": True,
        }
    return {**stt, **chat(stt["transcript"])}


def model_info() -> dict:
    _load()
    return {
        "stt_model": f"faster-whisper {WHISPER_SIZE} (int8)",
        "intent_model": "DistilBERT (fine-tuned)",
        "num_intents": len(_labels),
        "conf_threshold": CONF_THRESHOLD,
        "intents": _labels,
    }
