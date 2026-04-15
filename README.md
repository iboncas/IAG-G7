# FaceFusion Upload App

Small local Flask app for one job: upload a source face image and a target video, convert the target to `mp4` when needed, run FaceFusion in headless mode, and download the generated `output.mp4`. The interface shows the current processing step while the job runs.

## Requirements

- Git
- Conda
- FFmpeg
- Local FaceFusion installation

## Full Installation

### 1. Install Git and Conda

Install these tools first:

- Git
- Miniconda or Anaconda

Use the official installers or your operating system package manager.

### 2. Open the project in a terminal

```bash
cd /path/to/IAG-Deepfake
```

### 3. Initialize Conda

Run this once, then close and reopen the terminal:

```bash
conda init
```

### 4. Create and activate the conda environment

From the project root:

```bash
conda create --name facefusion python=3.12 pip=25.0 -y
conda activate facefusion
```

### 5. Install FFmpeg inside the conda environment

```bash
conda install -c conda-forge ffmpeg -y
```

### 6. Install the web app dependency

```bash
pip install -r requirements.txt
```

### 7. Install FaceFusion locally

Clone FaceFusion into this project as a local subdirectory:

```bash
git clone https://github.com/facefusion/facefusion.git ./facefusion
cd facefusion
python install.py --onnxruntime default
cd ..
```

This project auto-detects `./facefusion/facefusion.py`, so cloning FaceFusion into the repo root avoids extra configuration.

### 8. Start the app

```bash
python app.py
```

Open:

```text
http://localhost:5050
```

## Notes

- Uploaded files and generated files are stored only in temporary job folders during processing.
- Temporary files are deleted on failure and removed immediately after the final video is downloaded.
- API endpoint: `POST /swap`.
- The app expects FaceFusion to be cloned into `./facefusion`.
- Target videos in `.mov`, `.mkv`, `.avi`, or `.webm` are converted to `.mp4` with `ffmpeg` before FaceFusion runs.
- Processing runs in the background and the web UI polls job status until the output is ready to download.
