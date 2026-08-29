# Hugging Face Docker Space image for the voice-enabled chatbot.
FROM python:3.11-slim

# Spaces run the container as uid 1000; everything below is owned by that user
# so the app can read its own model files at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TORCH_THREADS=2 \
    OMP_NUM_THREADS=2
WORKDIR $HOME/app

COPY --chown=user requirements.txt .

# Install the CPU-only torch build explicitly. The default PyPI wheel drags in
# ~2.5 GB of CUDA libraries that are useless on a CPU Space.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt

# Bake the Whisper weights into the image so the first visitor does not wait for
# a ~145 MB download.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"

COPY --chown=user app/ ./

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
