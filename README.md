# FaceFusion + Seed-VC Upload App

Local Flask app for:

- face swapping with FaceFusion
- optional voice conversion with Seed-VC
- MP4 download of the final processed video

The app accepts:

- a required source face image
- a required target video
- an optional reference voice clip

If a voice clip is uploaded, the app extracts the target video's speech, runs Seed-VC against the reference clip, and muxes the converted audio back into the FaceFusion result.

## Requirements

- Python 3.11
- FFmpeg and FFprobe
- internet access on first run so FaceFusion and Seed-VC can download model files

## Docker

Docker is the easiest way to run the full stack because the container already separates the app, FaceFusion, and Seed-VC into dedicated Python 3.11 virtualenvs.

### Start

```bash
docker compose up --build
```

Open:

```text
http://localhost:5050
```

### Notes

- FaceFusion models are cached in `./facefusion/.assets`
- Seed-VC checkpoints are cached in `./seed-vc/checkpoints`
- The compose file currently defaults `SEED_VC_FP16=false` for compatibility. If you are running on supported GPU hardware and want to tune performance, override the environment values in `docker-compose.yml`.

## Local Setup

The most reliable local setup mirrors Docker: use three Python 3.11 virtualenvs so FaceFusion and Seed-VC can keep their own dependency sets.

### 1. Create the app environment

```bash
python3.11 -m venv .venv-app
source .venv-app/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
deactivate
```

### 2. Create the FaceFusion environment

```bash
python3.11 -m venv .venv-facefusion
source .venv-facefusion/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r facefusion/requirements.txt
pip install onnxruntime==1.24.1
deactivate
```

### 3. Create the Seed-VC environment

```bash
python3.11 -m venv .venv-seed-vc
source .venv-seed-vc/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
deactivate
```

### 4. Export the runtime commands

```bash
export FACEFUSION_BASE_COMMAND="$PWD/.venv-facefusion/bin/python facefusion.py"
export FACEFUSION_CWD="$PWD/facefusion"
export SEED_VC_BASE_COMMAND="$PWD/.venv-seed-vc/bin/python inference.py"
export SEED_VC_CWD="$PWD/seed-vc"
export SEED_VC_FP16=false
```

Optional tuning:

```bash
export SEED_VC_DIFFUSION_STEPS=20
export SEED_VC_INFERENCE_CFG_RATE=0.7
export SEED_VC_F0_CONDITION=false
```

### 5. Start the web app

```bash
.venv-app/bin/python app.py
```

Open:

```text
http://localhost:5050
```

## Workflow

1. Upload the source face image.
2. Upload the target video.
3. Optionally upload a voice reference clip.
4. Wait for the job to finish.
5. Download the final MP4.

## Supported Formats

- Source image: `jpg`, `jpeg`, `png`, `webp`
- Target video: `mp4`, `mov`, `mkv`, `avi`, `webm`
- Voice reference: `mp3`, `wav`, `m4a`, `aac`, `flac`, `ogg`, `opus`

## Runtime Notes

- Non-MP4 target videos are converted to MP4 before FaceFusion runs.
- If no voice reference is uploaded, the app skips Seed-VC and keeps the original video audio.
- If a voice reference is uploaded, the target video must contain an audio track.
- Processing runs in the background and the UI polls `/status/<job_id>`.
- Temporary job files are removed on failure and deleted after download.
