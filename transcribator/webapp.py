from __future__ import annotations

import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import click
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from .backends import DEFAULT_ENGINE, ENGINE_CHOICES, apply_hf_runtime_settings
from .config import load_config
from .contracts import TranscriptionRequest, TranscriptionResult
from .service import TranscriptionService
from .utils import (
    VIDEO_EXTENSIONS,
    WHISPER_MODEL_CHOICES,
    ensure_project_directories,
    get_default_output_directory,
    get_model_info,
    is_video_file,
    normalize_model_name,
)


ACTIVE_JOB_STATUSES = {"queued", "running", "diarizing", "exporting"}
STATUS_PROGRESS = {
    "queued": 8,
    "running": 52,
    "diarizing": 78,
    "exporting": 92,
    "done": 100,
    "failed": 100,
}
PYANNOTE_ACCESS_REPOSITORIES = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
]


@dataclass
class JobState:
    job_id: str
    filename: str
    input_path: str
    output_dir: str
    status: str = "queued"
    message: str = "Queued"
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None
    progress: int = STATUS_PROGRESS["queued"]


class BusyJobError(RuntimeError):
    def __init__(self, active_job_id: Optional[str]):
        super().__init__("Another transcription job is already running.")
        self.active_job_id = active_job_id


class JobManager:
    def __init__(
        self,
        service: TranscriptionService,
        output_root: Path,
        config: Dict[str, object],
        hf_token: Optional[str],
    ):
        self.service = service
        self.output_root = output_root
        self.config = config
        self.hf_token = hf_token
        self._jobs: Dict[str, JobState] = {}
        self._active_job_id: Optional[str] = None
        self._lock = threading.Lock()

    async def create_job(
        self,
        upload: UploadFile,
        *,
        model: str,
        language: Optional[str],
        quality: str,
        speakers: str,
        engine: str,
        hf_token: Optional[str],
    ) -> JobState:
        with self._lock:
            if self._active_job_id:
                active_job = self._jobs.get(self._active_job_id)
                if active_job and active_job.status in ACTIVE_JOB_STATUSES:
                    raise BusyJobError(self._active_job_id)

            job_id = uuid.uuid4().hex
            upload_name = Path(upload.filename or f"{job_id}.mp4").name
            upload_dir = Path(tempfile.gettempdir()) / "transcribator-web" / job_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            input_path = upload_dir / upload_name

            with input_path.open("wb") as handle:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

            await upload.close()

            output_dir = self.output_root / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            job = JobState(
                job_id=job_id,
                filename=upload_name,
                input_path=str(input_path),
                output_dir=str(output_dir),
            )
            self._jobs[job_id] = job
            self._active_job_id = job_id

        request = TranscriptionRequest(
            input_path=str(input_path),
            model=model,
            engine=engine,
            language=language or None,
            output_formats=["txt", "srt", "vtt"],
            output_dir=str(output_dir),
            quiet=True,
            high_quality=quality == "high",
            no_timestamps=False,
            clean_txt=False,
            diarize={"off": "none", "simple": "simple", "pyannote": "pyannote"}.get(speakers, "none"),
            hf_token=hf_token or self.hf_token,
            beam_size=self.config.get("beam_size"),
            best_of=self.config.get("best_of"),
            preprocess_audio=bool(self.config.get("preprocess_audio")) or quality == "high",
            min_speakers=self.config.get("min_speakers"),
            max_speakers=self.config.get("max_speakers"),
            diarization_threshold=self.config.get("diarization_threshold"),
            pause_threshold=self.config.get("pause_threshold"),
        )

        worker = threading.Thread(
            target=self._run_job,
            args=(job_id, request, upload_name),
            daemon=True,
        )
        worker.start()
        return job

    def get_job(self, job_id: str) -> JobState:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def _run_job(self, job_id: str, request: TranscriptionRequest, output_name: str) -> None:
        try:
            self._update_job(job_id, "running", "Preparing audio")
            result = self.service.transcribe_file(
                request,
                status_callback=lambda status, message: self._update_job(job_id, status, message),
                output_name=output_name,
            )
        except Exception as exc:
            self._fail_job(job_id, str(exc))
        else:
            self._complete_job(job_id, result)
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _update_job(self, job_id: str, status: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.message = message
            job.progress = STATUS_PROGRESS.get(status, job.progress)

    def _fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.message = "Processing failed"
            job.error = error
            job.progress = STATUS_PROGRESS["failed"]

    def _complete_job(self, job_id: str, result: TranscriptionResult) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "done"
            job.message = "Finished"
            job.result = result
            job.progress = STATUS_PROGRESS["done"]


def _build_job_payload(job: JobState) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "filename": job.filename,
        "progress": job.progress,
        "active": job.status in ACTIVE_JOB_STATUSES,
    }

    if job.result:
        payload["language"] = job.result.language
        payload["preview_text"] = job.result.preview_text or job.result.text
        payload["downloads"] = {
            fmt: f"/jobs/{job.job_id}/download/{fmt}"
            for fmt in ("txt", "srt", "vtt")
            if fmt in job.result.artifacts
        }

    return payload


def _resolve_hf_token(config: Dict[str, object]) -> Optional[str]:
    return (
        config.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _web_defaults(config: Dict[str, object]) -> Dict[str, str]:
    diarize = str(config.get("diarize") or "none").lower()
    if diarize == "none":
        speakers = "off"
    elif diarize == "simple":
        speakers = "simple"
    else:
        speakers = "pyannote"

    engine = str(config.get("engine") or DEFAULT_ENGINE).lower()
    if engine not in ENGINE_CHOICES:
        engine = DEFAULT_ENGINE

    return {
        "model": normalize_model_name(str(config.get("model") or "small")) or "small",
        "language": str(config.get("language") or ""),
        "quality": "high" if config.get("high_quality") else "balanced",
        "speakers": speakers,
        "engine": engine,
        "hf_token": str(_resolve_hf_token(config) or ""),
    }


def create_app(
    config_path: Optional[str] = None,
    *,
    service: Optional[TranscriptionService] = None,
) -> FastAPI:
    ensure_project_directories()
    config = load_config(config_path)
    apply_hf_runtime_settings(_resolve_hf_token(config))
    output_root = Path(config.get("output_dir") or get_default_output_directory()) / "web"
    output_root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Transcribator Web")
    app.state.templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app.state.job_manager = JobManager(
        service=service or TranscriptionService(),
        output_root=output_root,
        config=config,
        hf_token=_resolve_hf_token(config),
    )
    app.state.config = config

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return app.state.templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "accept_extensions": ",".join(sorted(VIDEO_EXTENSIONS)),
                "defaults": _web_defaults(config),
                "engine_choices": ENGINE_CHOICES,
                "model_choices": WHISPER_MODEL_CHOICES,
                "model_info": get_model_info(),
                "pyannote_access_repositories": PYANNOTE_ACCESS_REPOSITORIES,
            },
        )

    @app.post("/jobs")
    async def create_job(
        file: UploadFile = File(...),
        model: str = Form("small"),
        language: str = Form(""),
        quality: str = Form("balanced"),
        speakers: str = Form("off"),
        engine: str = Form(DEFAULT_ENGINE),
        hf_token: str = Form(""),
    ):
        filename = Path(file.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="A file must be selected.")
        if not is_video_file(filename):
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        normalized_model = normalize_model_name(model) or "small"
        if normalized_model not in WHISPER_MODEL_CHOICES:
            raise HTTPException(status_code=400, detail="Unsupported model.")
        if engine not in ENGINE_CHOICES:
            raise HTTPException(status_code=400, detail="Unsupported engine.")
        if quality not in {"balanced", "high"}:
            raise HTTPException(status_code=400, detail="Unsupported quality mode.")
        if speakers not in {"off", "simple", "pyannote"}:
            raise HTTPException(status_code=400, detail="Unsupported speaker mode.")
        if speakers == "pyannote" and not hf_token.strip():
            raise HTTPException(
                status_code=400,
                detail="Hugging Face token is required for pyannote diarization.",
            )

        try:
            job = await app.state.job_manager.create_job(
                file,
                model=normalized_model,
                language=language.strip() or None,
                quality=quality,
                speakers=speakers,
                engine=engine,
                hf_token=hf_token.strip() or None,
            )
        except BusyJobError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "active_job_id": exc.active_job_id,
                },
            ) from exc

        return {
            "job_id": job.job_id,
            "status": "queued",
            "message": "Queued",
            "progress": STATUS_PROGRESS["queued"],
            "active": True,
        }

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            job = app.state.job_manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

        return _build_job_payload(job)

    @app.get("/jobs/{job_id}/download/{artifact_name}")
    async def download_artifact(job_id: str, artifact_name: str):
        try:
            job = app.state.job_manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found.") from exc

        if not job.result:
            raise HTTPException(status_code=409, detail="Artifacts are not ready yet.")

        artifact_path = job.result.artifacts.get(artifact_name)
        if not artifact_path:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        path = Path(artifact_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Artifact file is missing.")

        return FileResponse(path)

    return app


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to bind.")
@click.option("--config", default=None, type=click.Path(exists=False), help="Optional config path.")
@click.option(
    "--access-log/--no-access-log",
    default=False,
    show_default=True,
    help="Show one log line per HTTP request.",
)
@click.option(
    "--log-level",
    default="warning",
    show_default=True,
    type=click.Choice(["critical", "error", "warning", "info", "debug", "trace"], case_sensitive=False),
    help="Uvicorn log level.",
)
def main(
    host: str,
    port: int,
    config: Optional[str],
    access_log: bool,
    log_level: str,
) -> None:
    app = create_app(config)
    click.echo(f"Starting Transcribator Web on http://{host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=access_log,
    )


if __name__ == "__main__":
    main()
