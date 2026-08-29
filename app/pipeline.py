"""The two-stage voice pipeline: Whisper for speech recognition, DistilBERT for
intent classification, template lookup for the reply.

Neither model needs PyTorch at inference time. Whisper runs on CTranslate2 via
faster-whisper, and the fine-tuned classifier runs as an int8-quantised ONNX
graph under ONNX Runtime. That keeps the deployed image small enough for a free
hosting tier -- the default Linux torch wheel alone pulls in roughly 2.5 GB of
CUDA libraries that would never be used on a CPU host.

Both models are loaded once and reused across requests; reloading per request
would dominate the latency budget.
"""

import io
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from faster_whisper import WhisperModel
from transformers import AutoTokenizer

APP_DIR = Path(__file__).resolve().parent
LOCAL_ONNX_DIR = APP_DIR / "model_onnx"

# Prefer weights committed next to the app; fall back to the copy on the Hub when
# they are absent, which is how the Streamlit Cloud deployment gets them (that
# platform does not fetch Git LFS objects).
HUB_MODEL_ID = os.environ.get("MODEL_ID", "Chethan616/voice-chatbot-intent-distilbert")

# Below this softmax probability we treat the utterance as out-of-scope even if
# the argmax says otherwise. Selected on the validation split -- see
# results/threshold_sweep.png and train/evaluate.py.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.44"))

# base.en is English-only, which is both faster and more accurate on English than
# the multilingual model of the same size. int8 quantisation roughly quarters
# memory and speeds up CPU inference at negligible accuracy cost.
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "base.en")

MAX_LEN = 32

_whisper: WhisperModel | None = None
_tokenizer = None
_session: ort.InferenceSession | None = None
_labels: list[str] = []
_responses: dict[str, list[str]] = {}
_source = "?"


def _resolve_model() -> tuple[str, str, list[str]]:
    """Return (onnx_path, tokenizer_source, labels), preferring local files."""
    local_graph = LOCAL_ONNX_DIR / "model.onnx"
    if local_graph.exists():
        labels = json.loads((LOCAL_ONNX_DIR / "labels.json").read_text(encoding="utf-8"))
        return str(local_graph), str(LOCAL_ONNX_DIR), labels

    from huggingface_hub import hf_hub_download

    graph = hf_hub_download(HUB_MODEL_ID, "onnx/model.onnx")
    labels_path = hf_hub_download(HUB_MODEL_ID, "labels.json")
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    return graph, HUB_MODEL_ID, labels


def _load() -> None:
    """Idempotent lazy load of both models plus the response templates."""
    global _whisper, _tokenizer, _session, _labels, _responses, _source
    if _session is not None:
        return

    threads = int(os.environ.get("ORT_THREADS", "2"))

    graph_path, tok_source, _labels_local = _resolve_model()
    _source = "local files" if graph_path.startswith(str(LOCAL_ONNX_DIR)) else "Hugging Face Hub"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(graph_path, opts, providers=["CPUExecutionProvider"])

    _tokenizer = AutoTokenizer.from_pretrained(tok_source)
    _labels = _labels_local
    _responses = json.loads((APP_DIR / "responses.json").read_text(encoding="utf-8"))

    _whisper = WhisperModel(
        WHISPER_SIZE, device="cpu", compute_type="int8", cpu_threads=threads
    )


def warmup() -> None:
    """Load models and run one tiny forward pass so the first real request is not
    paying initialisation cost."""
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


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def classify(text: str) -> dict:
    """Text -> intent. Returns the argmax plus the top 3 for the UI."""
    _load()
    t0 = time.perf_counter()
    enc = _tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="np"
    )
    logits = _session.run(None, {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    })[0]
    probs = _softmax(logits[0])

    order = np.argsort(-probs)[:3]
    top = [{"intent": _labels[i], "confidence": round(float(probs[i]), 4)} for i in order]
    intent, confidence = top[0]["intent"], top[0]["confidence"]

    # Two routes to "I don't know": the model predicts the oos class outright, or
    # it is not confident enough in whatever it did predict.
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
        "intent_model": "DistilBERT, ONNX int8",
        "num_intents": len(_labels),
        "conf_threshold": CONF_THRESHOLD,
        "weights_from": _source,
        "intents": _labels,
    }
