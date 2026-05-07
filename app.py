import os
import secrets
import shlex
import shutil
import subprocess
import sys
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
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _create_job(
    job_id: str,
    output_path: Path,
    temp_dir: Path,
    download_filename: str,
    face_enabled: bool,
    voice_enabled: bool,
) -> None:
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "state": "queued",
            "step": "Upload received",
            "detail": "Files uploaded successfully.",
            "progress": 10,
            "output_path": str(output_path),
            "temp_dir": str(temp_dir),
            "download_filename": download_filename,
            "download_url": None,
            "error": None,
            "face_enabled": face_enabled,
            "voice_enabled": voice_enabled,
            "mode": "face-and-voice" if face_enabled and voice_enabled else "face-only" if face_enabled else "voice-only",
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


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_facefusion_runtime() -> tuple[list[str], Path]:
    base = os.getenv("FACEFUSION_BASE_COMMAND")
    cwd = Path(os.getenv("FACEFUSION_CWD", str(BASE_DIR)))
    if base:
        return shlex.split(base), cwd

    root_script = BASE_DIR / "facefusion.py"
    nested_script = BASE_DIR / "facefusion" / "facefusion.py"

    if root_script.exists():
        return [sys.executable, "facefusion.py"], BASE_DIR
    if nested_script.exists():
        return [sys.executable, "facefusion.py"], nested_script.parent
    if shutil.which("facefusion"):
        return ["facefusion"], cwd

    raise FileNotFoundError(
        "The processing engine was not found. Install it locally, clone it "
        "into `./facefusion`, place `facefusion.py` in the project root, or "
        "set `FACEFUSION_BASE_COMMAND`."
    )


def _build_facefusion_command(source_path: Path, target_path: Path, output_path: Path) -> tuple[list[str], Path]:
    base, cwd = _resolve_facefusion_runtime()
    command = list(base)
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


def _resolve_seed_vc_runtime() -> tuple[list[str], Path]:
    base = os.getenv("SEED_VC_BASE_COMMAND")
    cwd = Path(os.getenv("SEED_VC_CWD", str(BASE_DIR / "seed-vc")))
    if base:
        return shlex.split(base), cwd

    seed_vc_script = BASE_DIR / "seed-vc" / "inference.py"
    if seed_vc_script.exists():
        return [sys.executable, "inference.py"], seed_vc_script.parent

    raise FileNotFoundError(
        "The voice conversion engine was not found. Make sure `./seed-vc` "
        "exists or set `SEED_VC_BASE_COMMAND`."
    )


def _build_seed_vc_command(source_audio_path: Path, reference_audio_path: Path, output_dir: Path) -> tuple[list[str], Path]:
    base, cwd = _resolve_seed_vc_runtime()
    command = list(base)
    command.extend(
        [
            "--source",
            str(source_audio_path),
            "--target",
            str(reference_audio_path),
            "--output",
            str(output_dir),
            "--diffusion-steps",
            os.getenv("SEED_VC_DIFFUSION_STEPS", "20"),
            "--length-adjust",
            os.getenv("SEED_VC_LENGTH_ADJUST", "1.0"),
            "--inference-cfg-rate",
            os.getenv("SEED_VC_INFERENCE_CFG_RATE", "0.7"),
            "--f0-condition",
            os.getenv("SEED_VC_F0_CONDITION", "false"),
            "--auto-f0-adjust",
            os.getenv("SEED_VC_AUTO_F0_ADJUST", "false"),
            "--semi-tone-shift",
            os.getenv("SEED_VC_SEMI_TONE_SHIFT", "0"),
            "--fp16",
            str(_env_flag("SEED_VC_FP16", False)).lower(),
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
    _run_command(command, failure_message="Unknown ffmpeg conversion error.")


def _run_command(command: list[str], cwd: Path | None = None, failure_message: str = "Command failed.") -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or failure_message)
    return result.stdout.strip()


def _run_logged_command(job_id: str, command: list[str], cwd: Path, step: str, progress: int) -> tuple[int, list[str]]:
    logs = deque(maxlen=20)
    process = subprocess.Popen(
        command,
        cwd=cwd,
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
                step=step,
                detail=cleaned,
                progress=progress,
            )

    return process.wait(), list(logs)


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


def _prepare_reference_audio(audio_path: Path, job_id: str, temp_dir: Path) -> Path:
    converted_path = temp_dir / f"{job_id}_reference.wav"
    _update_job(
        job_id,
        state="running",
        step="Preparing voice reference",
        detail="Normalizing the uploaded reference audio for Seed-VC.",
        progress=55,
    )
    _convert_audio_to_wav(audio_path, converted_path)
    return converted_path


def _convert_audio_to_wav(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run_command(command, failure_message="Audio conversion failed.")


def _video_has_audio_track(video_path: Path) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and "audio" in result.stdout.lower()


def _extract_audio_from_video(video_path: Path, output_path: Path, job_id: str) -> None:
    _update_job(
        job_id,
        state="running",
        step="Extracting source speech",
        detail="Extracting the target video's audio track for voice conversion.",
        progress=78,
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run_command(command, failure_message="Audio extraction from the target video failed.")


def _find_latest_wav(output_dir: Path) -> Path | None:
    wav_files = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)
    if wav_files:
        return wav_files[-1]
    return None


def _mux_audio_into_video(video_path: Path, audio_path: Path, output_path: Path, job_id: str) -> None:
    _update_job(
        job_id,
        state="running",
        step="Merging converted voice",
        detail="Muxing the converted voice track into the processed video.",
        progress=94,
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    _run_command(command, failure_message="Merging the converted audio into the final video failed.")


def _finalize_with_target_audio(video_path: Path, target_video_path: Path, output_path: Path, job_id: str) -> None:
    _update_job(
        job_id,
        state="running",
        step="Finalizing video",
        detail="Keeping the target video's original soundtrack.",
        progress=92,
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(target_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    _run_command(command, failure_message="Finalizing the video with the original target audio failed.")


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


def _sanitize_download_filename(filename: str | None) -> str:
    cleaned = secure_filename(filename or "")
    stem = Path(cleaned).stem
    if not stem:
        return "video.mp4"
    return f"{stem}.mp4"


def _has_uploaded_file(file_storage) -> bool:
    if not file_storage or not file_storage.filename:
        return False

    stream = file_storage.stream
    position = stream.tell()
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(position)
    return size > 0


def _move_or_copy_video(source_path: Path, output_path: Path) -> None:
    if source_path.resolve() == output_path.resolve():
        return
    shutil.copy2(source_path, output_path)


def _run_processing_job(
    job_id: str,
    source_path: Path | None,
    target_path: Path,
    audio_path: Path | None,
    output_path: Path,
    temp_dir: Path,
) -> None:
    cleanup_paths = [target_path]
    face_enabled = source_path is not None
    voice_enabled = audio_path is not None
    if source_path:
        cleanup_paths.append(source_path)
    try:
        prepared_target_path = _prepare_target_video(target_path, job_id, temp_dir)
        if prepared_target_path != target_path:
            cleanup_paths.append(prepared_target_path)

        processed_video_path = prepared_target_path

        if source_path:
            facefusion_output_path = temp_dir / "facefusion_output.mp4"
            command, facefusion_cwd = _build_facefusion_command(source_path, prepared_target_path, facefusion_output_path)
            _update_job(
                job_id,
                state="running",
                step="Running face swap",
                detail="FaceFusion is processing the target video.",
                progress=65,
            )

            return_code, logs = _run_logged_command(
                job_id,
                command,
                facefusion_cwd,
                "Running face swap",
                72,
            )
            if return_code != 0:
                _update_job(
                    job_id,
                    state="failed",
                    step="Face swap failed",
                    detail="\n".join(logs) if logs else "The video transformation exited with a non-zero status.",
                    error="Video transformation failed.",
                    progress=100,
                )
                return

            if not facefusion_output_path.exists():
                _update_job(
                    job_id,
                    state="failed",
                    step="Output missing",
                    detail="Processing completed but did not create an output file.",
                    error="Output file was not created.",
                    progress=100,
                )
                return

            processed_video_path = facefusion_output_path
        elif voice_enabled:
            _update_job(
                job_id,
                state="running",
                step="Preparing voice conversion",
                detail="Preparing the target video audio for voice conversion.",
                progress=52,
            )

        if audio_path:
            cleanup_paths.append(audio_path)
            if not _video_has_audio_track(prepared_target_path):
                _update_job(
                    job_id,
                    state="failed",
                    step="Voice conversion unavailable",
                    detail="The target video does not contain an audio track to convert.",
                    error="Target video has no audio track.",
                    progress=100,
                )
                return

            prepared_reference_audio_path = _prepare_reference_audio(audio_path, job_id, temp_dir)
            cleanup_paths.append(prepared_reference_audio_path)

            extracted_audio_path = temp_dir / f"{job_id}_source_audio.wav"
            _extract_audio_from_video(prepared_target_path, extracted_audio_path, job_id)
            cleanup_paths.append(extracted_audio_path)

            seed_vc_output_dir = temp_dir / "seed-vc-output"
            seed_vc_output_dir.mkdir(exist_ok=True)
            seed_vc_command, seed_vc_cwd = _build_seed_vc_command(
                extracted_audio_path,
                prepared_reference_audio_path,
                seed_vc_output_dir,
            )
            _update_job(
                job_id,
                state="running",
                step="Running voice conversion",
                detail="Seed-VC is converting the extracted speech to the reference voice.",
                progress=86,
            )
            seed_vc_return_code, seed_vc_logs = _run_logged_command(
                job_id,
                seed_vc_command,
                seed_vc_cwd,
                "Running voice conversion",
                90,
            )
            if seed_vc_return_code != 0:
                _update_job(
                    job_id,
                    state="failed",
                    step="Voice conversion failed",
                    detail="\n".join(seed_vc_logs) if seed_vc_logs else "Seed-VC exited with a non-zero status.",
                    error="Voice conversion failed.",
                    progress=100,
                )
                return

            converted_audio_path = _find_latest_wav(seed_vc_output_dir)
            if not converted_audio_path or not converted_audio_path.exists():
                _update_job(
                    job_id,
                    state="failed",
                    step="Voice conversion failed",
                    detail="Seed-VC finished without writing an output audio file.",
                    error="Converted audio file was not created.",
                    progress=100,
                )
                return

            _mux_audio_into_video(processed_video_path, converted_audio_path, output_path, job_id)
        else:
            if face_enabled:
                _finalize_with_target_audio(processed_video_path, prepared_target_path, output_path, job_id)
            else:
                _update_job(
                    job_id,
                    state="running",
                    step="Finalizing video",
                    detail="Preparing the processed video for download.",
                    progress=95,
                )
                _move_or_copy_video(processed_video_path, output_path)

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
        lowered = message.lower()
        if "ffmpeg" in lowered or "ffprobe" in lowered:
            error = "ffmpeg/ffprobe executable not found."
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
            step="Processing failed",
            detail=str(exc),
            error="The media processing pipeline failed.",
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
    audio = request.files.get("audio")
    output_name = request.form.get("output_name")

    has_source = _has_uploaded_file(source)
    has_audio = _has_uploaded_file(audio)

    if not target or not target.filename:
        return {"error": "A target video is required."}, 400

    if not has_source and not has_audio:
        return {"error": "Upload a source image, a voice reference, or both."}, 400

    if has_source and not _validate_extension(source.filename or "", ALLOWED_IMAGE_EXTENSIONS):
        return {"error": "Invalid source format. Use jpg, jpeg, png, or webp."}, 400

    if not _validate_extension(target.filename or "", ALLOWED_VIDEO_EXTENSIONS):
        return {"error": "Invalid target format. Use mp4, mov, mkv, avi, or webm."}, 400

    if has_audio and not _validate_extension(audio.filename or "", ALLOWED_AUDIO_EXTENSIONS):
        return {"error": "Invalid voice reference format. Use mp3, wav, m4a, aac, flac, ogg, or opus."}, 400

    job_id = secrets.token_hex(8)
    source_name = secure_filename(source.filename or f"source_{job_id}.jpg") if has_source else None
    target_name = secure_filename(target.filename or f"target_{job_id}.mp4")
    audio_name = secure_filename(audio.filename or f"voice_{job_id}.wav") if has_audio else None
    download_filename = _sanitize_download_filename(output_name)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"deepfake-{job_id}-"))

    source_path = temp_dir / source_name if source_name else None
    target_path = temp_dir / target_name
    audio_path = temp_dir / audio_name if audio_name else None
    output_path = temp_dir / "output.mp4"

    if source and source_path:
        source.save(source_path)
    target.save(target_path)
    if audio and audio_path:
        audio.save(audio_path)

    _create_job(
        job_id,
        output_path,
        temp_dir,
        download_filename,
        face_enabled=bool(source_path),
        voice_enabled=bool(audio_path),
    )

    worker = threading.Thread(
        target=_run_processing_job,
        args=(job_id, source_path, target_path, audio_path, output_path, temp_dir),
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

    return send_file(
        BytesIO(payload),
        as_attachment=True,
        download_name=job.get("download_filename") or "video.mp4",
        mimetype="video/mp4",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
