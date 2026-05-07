FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACEFUSION_BASE_COMMAND="/opt/venvs/facefusion/bin/python facefusion.py" \
    FACEFUSION_CWD="/app/facefusion" \
    SEED_VC_BASE_COMMAND="/opt/venvs/seed-vc/bin/python inference.py" \
    SEED_VC_CWD="/app/seed-vc" \
    HF_HUB_CACHE="/app/seed-vc/checkpoints/hf_cache"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY app.py README.md ./
COPY templates ./templates
COPY static ./static
COPY facefusion ./facefusion
COPY seed-vc ./seed-vc

RUN python -m venv /opt/venvs/app \
    && /opt/venvs/app/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/app/bin/pip install -r /app/requirements.txt

RUN python -m venv /opt/venvs/facefusion \
    && /opt/venvs/facefusion/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/facefusion/bin/pip install -r /app/facefusion/requirements.txt \
    && /opt/venvs/facefusion/bin/pip install onnxruntime==1.24.1

RUN python -m venv /opt/venvs/seed-vc \
    && /opt/venvs/seed-vc/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/seed-vc/bin/pip install -r /app/requirements.txt

EXPOSE 5050

CMD ["/opt/venvs/app/bin/python", "app.py"]
