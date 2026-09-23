# Container image for the voice-enabled chatbot API.
#
# Deployed to Google Cloud Run, which injects $PORT and expects the server to
# bind to it on 0.0.0.0. The React front end is deployed separately to Vercel
# and calls this service over CORS.
FROM python:3.11-slim

# CTranslate2 (the engine under faster-whisper) links against OpenMP, which the
# slim base image does not ship.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    ORT_THREADS=2 \
    OMP_NUM_THREADS=2 \
    PORT=8080
WORKDIR $HOME/app

COPY --chown=user requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Bake both sets of weights into the image so a cold start does not also pay for
# a ~210 MB download. Cloud Run scales to zero, so cold starts are routine.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"
RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('Chethan616/voice-chatbot-intent-distilbert', 'onnx/model.onnx'); \
hf_hub_download('Chethan616/voice-chatbot-intent-distilbert', 'labels.json'); \
hf_hub_download('Chethan616/voice-chatbot-intent-distilbert', 'tokenizer.json'); \
hf_hub_download('Chethan616/voice-chatbot-intent-distilbert', 'tokenizer_config.json')"

COPY --chown=user app/ ./

# Single worker: each one would load its own ~700 MB copy of the models.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
