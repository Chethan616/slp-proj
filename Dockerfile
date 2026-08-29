# Container image for the voice-enabled chatbot.
#
# This mirrors the deployed Streamlit app. The public deployment runs on
# Streamlit Community Cloud, which builds from requirements.txt directly; this
# Dockerfile exists so the same app can be reproduced or self-hosted anywhere.
FROM python:3.11-slim

# CTranslate2 (the engine under faster-whisper) links against OpenMP, which the
# slim base image does not ship.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Run as a non-root user, matching how most container platforms execute images.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    ORT_THREADS=2 \
    OMP_NUM_THREADS=2
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Bake the Whisper weights into the image so the first visitor does not wait for
# a ~145 MB download. The intent classifier is fetched from the Hub at startup.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

COPY --chown=user app/ ./app/
COPY --chown=user streamlit_app.py .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
