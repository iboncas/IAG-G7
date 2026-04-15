import os
import secrets
import shlex
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _create_job(job_id: str, output_path: Path, temp_dir: Path) -> None:
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "step": "Upload received",
            "detail": "Files uploaded successfully.",
            "progress": 10,
            "output_path": str(output_path),
            "temp_dir": str(temp_dir),
            "download_url": None,
            "error": None,
        }


def _update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)


def _get_job(job_id: str) -> dict | None:
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def _remove_job(job_id: str) -> None:
    with jobs_lock:
        jobs.pop(job_id, None)


def _resolve_facefusion_runtime() -> tuple[str, Path]:
    base = os.getenv("FACEFUSION_BASE_COMMAND")
    cwd = Path(os.getenv("FACEFUSION_CWD", str(BASE_DIR)))
    if base:
        return base, cwd

    root_script = BASE_DIR / "facefusion.py"
    nested_script = BASE_DIR / "facefusion" / "facefusion.py"

    if root_script.exists():
        return "python facefusion.py", BASE_DIR
    if nested_script.exists():
        return "python facefusion.py", nested_script.parent
    if shutil.which("facefusion"):
        return "facefusion", cwd

    raise FileNotFoundError(
        "The processing engine was not found. Install it locally, clone it "
        "into `./facefusion`, place `facefusion.py` in the project root, or "
        "set `FACEFUSION_BASE_COMMAND`."
    )


def _build_facefusion_command(source_path: Path, target_path: Path, output_path: Path) -> tuple[list[str], Path]:
    base, cwd = _resolve_facefusion_runtime()
    command = shlex.split(base)
    command.extend(
        [
            "headless-run",
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--output-path",
            str(output_path),
        ]
    )
    return command, cwd


def _convert_video_to_mp4(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "0",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Unknown ffmpeg conversion error.")


def _prepare_target_video(target_path: Path, job_id: str, temp_dir: Path) -> Path:
    if target_path.suffix.lower() == ".mp4":
        _update_job(
            job_id,
            state="running",
            step="Preparing target video",
            detail="Target is already MP4. Skipping conversion.",
            progress=25,
        )
        return target_path

    converted_path = temp_dir / f"{job_id}_target.mp4"
    _update_job(
        job_id,
        state="running",
        step="Converting target video",
        detail="Converting the uploaded target video to MP4.",
        progress=25,
    )
    _convert_video_to_mp4(target_path, converted_path)
    _update_job(
        job_id,
        state="running",
        step="Target video converted",
        detail="Target video converted to MP4 successfully.",
        progress=45,
    )
    return converted_path


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _validate_extension(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def _run_facefusion_job(job_id: str, source_path: Path, target_path: Path, output_path: Path, temp_dir: Path) -> None:
    cleanup_paths = [source_path, target_path]
    try:
        prepared_target_path = _prepare_target_video(target_path, job_id, temp_dir)
        if prepared_target_path != target_path:
            cleanup_paths.append(prepared_target_path)

        command, facefusion_cwd = _build_facefusion_command(source_path, prepared_target_path, output_path)
        _update_job(
            job_id,
            state="running",
            step="Running video transformation",
            detail="The video transformation is processing the video.",
            progress=65,
        )

        logs = deque(maxlen=20)
        process = subprocess.Popen(
            command,
            cwd=facefusion_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout:
            for line in process.stdout:
                cleaned = line.strip()
                if not cleaned:
                    continue
                logs.append(cleaned)
                _update_job(
                    job_id,
                    state="running",
                    step="Running video transformation",
                    detail=cleaned,
                    progress=75,
                )

        return_code = process.wait()
        if return_code != 0:
            _update_job(
                job_id,
                state="failed",
                step="Video transformation failed",
                detail="\n".join(logs) if logs else "The video transformation exited with a non-zero status.",
                error="Video transformation failed.",
                progress=100,
            )
            return

        if not output_path.exists():
            _update_job(
                job_id,
                state="failed",
                step="Output missing",
                detail="Processing completed but did not create an output file.",
                error="Output file was not created.",
                progress=100,
            )
            return

        _update_job(
            job_id,
            state="completed",
            step="Ready to download",
            detail="Processing finished. The MP4 output is ready.",
            progress=100,
            download_url=f"/download/{job_id}",
        )
    except FileNotFoundError as exc:
        message = str(exc)
        if "ffmpeg" in message.lower():
            error = "ffmpeg executable not found."
        else:
            error = "Processing engine executable not found."
        _update_job(
            job_id,
            state="failed",
            step="Setup error",
            detail=message,
            error=error,
            progress=100,
        )
    except RuntimeError as exc:
        _update_job(
            job_id,
            state="failed",
            step="Video conversion failed",
            detail=str(exc),
            error="Target video conversion to MP4 failed.",
            progress=100,
        )
    finally:
        for path in cleanup_paths:
            _safe_unlink(path)
        job = _get_job(job_id)
        if job and job["state"] != "completed":
            _safe_rmtree(temp_dir)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/swap")
def swap_face():
    source = request.files.get("source")
    target = request.files.get("target")

    if not source or not target:
        return {"error": "Both source image and target video are required."}, 400

    if not _validate_extension(source.filename or "", ALLOWED_IMAGE_EXTENSIONS):
        return {"error": "Invalid source format. Use jpg, jpeg, png, or webp."}, 400

    if not _validate_extension(target.filename or "", ALLOWED_VIDEO_EXTENSIONS):
        return {"error": "Invalid target format. Use mp4, mov, mkv, avi, or webm."}, 400

    job_id = secrets.token_hex(8)
    source_name = secure_filename(source.filename or f"source_{job_id}.jpg")
    target_name = secure_filename(target.filename or f"target_{job_id}.mp4")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"deepfake-{job_id}-"))

    source_path = temp_dir / source_name
    target_path = temp_dir / target_name
    output_path = temp_dir / "output.mp4"

    source.save(source_path)
    target.save(target_path)

    _create_job(job_id, output_path, temp_dir)

    worker = threading.Thread(
        target=_run_facefusion_job,
        args=(job_id, source_path, target_path, output_path, temp_dir),
        daemon=True,
    )
    worker.start()

    return {
        "job_id": job_id,
        "status_url": f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    }, 202


@app.get("/status/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return {"error": "Job not found."}, 404
    return job


@app.get("/download/<job_id>")
def download_output(job_id: str):
    job = _get_job(job_id)
    if not job:
        return {"error": "Job not found."}, 404
    if job["state"] != "completed":
        return {"error": "Output is not ready yet.", "state": job["state"]}, 409

    output_path = Path(job["output_path"])
    if not output_path.exists():
        return {"error": "Output file is no longer available."}, 404

    payload = output_path.read_bytes()
    temp_dir = Path(job["temp_dir"])
    _safe_rmtree(temp_dir)
    _remove_job(job_id)

    return send_file(BytesIO(payload), as_attachment=True, download_name="video.mp4", mimetype="video/mp4")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
