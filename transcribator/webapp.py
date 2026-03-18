from __future__ import annotations

import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import click
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from .backends import (
    DEFAULT_DEVICE,
    DEFAULT_ENGINE,
    DEVICE_CHOICES,
    ENGINE_CHOICES,
    apply_hf_runtime_settings,
    get_runtime_acceleration_info,
)
from .config import load_config
from .contracts import TranscriptionRequest, TranscriptionResult
from .diarization import get_pyannote_runtime_info
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
CUDA_TOOLKIT_URL = "https://developer.nvidia.com/cuda/toolkit"
PYANNOTE_ACCESS_REPOSITORIES = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
]
DEFAULT_UI_LANGUAGE = "en"
UI_LANGUAGE_CHOICES = ("en", "ru")
STATUS_LABEL_KEYS = {
    "idle": "status_idle",
    "queued": "status_queued",
    "running": "status_running",
    "diarizing": "status_diarizing",
    "exporting": "status_exporting",
    "done": "status_done",
    "failed": "status_failed",
}
MESSAGE_KEY_BY_TEXT = {
    "Waiting for a file.": "message_waiting",
    "Queued": "message_queued",
    "Preparing audio": "message_preparing_audio",
    "Assigning speakers": "message_assigning_speakers",
    "Writing output files": "message_writing_output",
    "Finished": "message_finished",
    "Processing failed": "message_processing_failed",
}
UI_TRANSLATIONS = {
    "en": {
        "page_title": "Transcribator",
        "hero_badge": "Local transcription",
        "interface_language": "Interface language",
        "hero_title": "Transcribe one file with a faster local pipeline.",
        "hero_subtitle": "Upload a video, pick a quality mode, and let the app generate TXT, SRT, and VTT in one local run.",
        "file_label": "File",
        "model_label": "Model",
        "transcription_language_label": "Language",
        "transcription_language_placeholder": "auto",
        "quality_label": "Quality",
        "quality_balanced": "Balanced",
        "quality_high": "High",
        "speakers_label": "Speakers",
        "speakers_off": "Off",
        "speakers_simple": "Simple (Experimental)",
        "speakers_pyannote": "Pyannote (Recommended)",
        "keep_timestamps_title": "Keep timestamps in TXT",
        "keep_timestamps_help": "Off by default. Turn this on only if you want visible time markers in the text output.",
        "hf_token_label": "Hugging Face Token",
        "hf_token_placeholder": "hf_...",
        "pyannote_summary": "Pyannote needs a token and access approvals",
        "pyannote_status_missing": "Token missing. Pyannote is selected but requests will stay unauthenticated until you paste an HF token.",
        "pyannote_status_ready": "Token detected. Pyannote requests will use it for Hugging Face access.",
        "pyannote_intro": "Do not forget to accept access terms for these repositories before the first run:",
        "pyannote_token_help": "This token is used for Hugging Face requests, including pyannote access and model downloads from the HF Hub.",
        "advanced_summary": "Advanced",
        "acceleration_label": "Acceleration",
        "diarization_acceleration_label": "Diarization acceleration",
        "device_cpu": "CPU",
        "device_cuda": "CUDA",
        "device_auto": "Auto",
        "cuda_available_summary": "CUDA is available for faster-whisper",
        "cuda_unavailable_summary": "CUDA is unavailable, web runs stay on CPU",
        "cuda_detected_devices": "Detected devices: {count}.",
        "cuda_runtime_failed": "Runtime check failed: {message}.",
        "cuda_install_hint": "Install the CUDA runtime/toolkit from the NVIDIA CUDA Toolkit page.",
        "device_note_openai": "`openai-whisper` stays CPU-only in this build.",
        "device_note_faster": "CUDA selection applies to `faster-whisper` when the system exposes accessible CUDA devices.",
        "engine_label": "Engine",
        "engine_help": "`faster-whisper` is the default CPU path. `openai-whisper` stays available as a fallback.",
        "diarization_device_note": "Pyannote runs locally on this machine. Hugging Face is used only for authentication and model downloads.",
        "pyannote_gpu_available_summary": "CUDA is available for pyannote diarization",
        "pyannote_gpu_unavailable_summary": "CUDA is unavailable for pyannote diarization",
        "pyannote_gpu_detected_devices": "Torch sees CUDA devices: {count}.",
        "pyannote_gpu_runtime_failed": "Pyannote GPU check failed: {message}.",
        "pyannote_gpu_install_hint": "GPU diarization needs a CUDA-enabled torch/torchaudio stack. See docs/pyannote-gpu.md.",
        "confirmation_title": "GPU diarization is unavailable",
        "confirmation_switch_cpu": "Switch to CPU and start",
        "confirmation_try_anyway": "Try to start anyway",
        "confirmation_cancel": "Keep current settings",
        "button_start": "Start transcription",
        "button_working": "Working...",
        "submit_note_idle": "Only one local job runs at a time.",
        "submit_note_uploading": "Uploading and starting the local job...",
        "submit_note_active": "Active job",
        "status_idle": "Idle",
        "status_queued": "Queued",
        "status_running": "Running",
        "status_diarizing": "Diarizing",
        "status_exporting": "Exporting",
        "status_done": "Ready",
        "status_failed": "Failed",
        "message_waiting": "Waiting for a file.",
        "message_queued": "Queued",
        "message_preparing_audio": "Preparing audio",
        "message_assigning_speakers": "Assigning speakers",
        "message_writing_output": "Writing output files",
        "message_finished": "Finished",
        "message_processing_failed": "Processing failed",
        "progress_label": "Progress",
        "stage_queued": "Queued",
        "stage_running": "ASR",
        "stage_diarizing": "Speakers",
        "stage_exporting": "Export",
        "stage_done": "Ready",
        "job_error_summary": "Job error",
        "preview_placeholder": "Preview will appear here after the first completed run.",
        "download_prefix": "Download",
        "polling_failed": "Polling failed",
        "could_not_start_job": "Could not start job",
        "could_not_start_transcription": "Could not start transcription.",
        "err_file_required": "A file must be selected.",
        "err_unsupported_file_type": "Unsupported file type.",
        "err_unsupported_model": "Unsupported model.",
        "err_unsupported_device": "Unsupported device.",
        "err_unsupported_engine": "Unsupported engine.",
        "err_unsupported_quality": "Unsupported quality mode.",
        "err_unsupported_speakers": "Unsupported speaker mode.",
        "err_hf_required": "Hugging Face token is required for pyannote diarization.",
        "err_cuda_only_faster": "CUDA acceleration is currently available only for faster-whisper.",
        "err_cuda_unavailable": "CUDA acceleration is not available for faster-whisper on this system.",
        "err_busy_job": "Another transcription job is already running.",
        "err_job_not_found": "Job not found.",
        "err_artifacts_not_ready": "Artifacts are not ready yet.",
        "err_artifact_not_found": "Artifact not found.",
        "err_artifact_missing": "Artifact file is missing.",
        "lang_en": "English",
        "lang_ru": "Russian",
    },
    "ru": {
        "page_title": "Transcribator",
        "hero_badge": "Локальная транскрибация",
        "interface_language": "Язык интерфейса",
        "hero_title": "Транскрибируйте один файл через быстрый локальный пайплайн.",
        "hero_subtitle": "Загрузите видео, выберите режим качества и получите TXT, SRT и VTT за один локальный прогон.",
        "file_label": "Файл",
        "model_label": "Модель",
        "transcription_language_label": "Язык",
        "transcription_language_placeholder": "auto",
        "quality_label": "Качество",
        "quality_balanced": "Сбалансированное",
        "quality_high": "Высокое",
        "speakers_label": "Спикеры",
        "speakers_off": "Выкл",
        "speakers_simple": "Simple (Экспериментальный)",
        "speakers_pyannote": "Pyannote (Рекомендуется)",
        "keep_timestamps_title": "Оставлять таймкоды в TXT",
        "keep_timestamps_help": "По умолчанию выключено. Включайте только если нужны видимые метки времени в текстовом файле.",
        "hf_token_label": "Токен Hugging Face",
        "hf_token_placeholder": "hf_...",
        "pyannote_summary": "Pyannote требует токен и принятые доступы",
        "pyannote_status_missing": "Токен не задан. Режим pyannote выбран, но запросы останутся неавторизованными, пока вы не вставите HF-токен.",
        "pyannote_status_ready": "Токен найден. Запросы pyannote будут выполняться с авторизацией в Hugging Face.",
        "pyannote_intro": "Не забудьте заранее принять условия доступа для этих репозиториев:",
        "pyannote_token_help": "Этот токен используется для запросов к Hugging Face, включая доступ к pyannote и загрузки моделей с HF Hub.",
        "advanced_summary": "Расширенные настройки",
        "acceleration_label": "Ускорение",
        "diarization_acceleration_label": "Ускорение диаризации",
        "device_cpu": "CPU",
        "device_cuda": "CUDA",
        "device_auto": "Авто",
        "cuda_available_summary": "CUDA доступна для faster-whisper",
        "cuda_unavailable_summary": "CUDA недоступна, веб-запуски останутся на CPU",
        "cuda_detected_devices": "Обнаружено устройств: {count}.",
        "cuda_runtime_failed": "Проверка runtime не пройдена: {message}.",
        "cuda_install_hint": "Установите CUDA runtime/toolkit со страницы NVIDIA CUDA Toolkit.",
        "device_note_openai": "`openai-whisper` в этой сборке остаётся только на CPU.",
        "device_note_faster": "Выбор CUDA применяется к `faster-whisper`, если система действительно даёт доступ к CUDA-устройствам.",
        "engine_label": "Движок",
        "engine_help": "`faster-whisper` используется как основной CPU-путь. `openai-whisper` остаётся как fallback.",
        "diarization_device_note": "Pyannote работает локально на этой машине. Hugging Face используется только для авторизации и загрузки моделей.",
        "pyannote_gpu_available_summary": "CUDA доступна для диаризации pyannote",
        "pyannote_gpu_unavailable_summary": "CUDA недоступна для диаризации pyannote",
        "pyannote_gpu_detected_devices": "Torch видит CUDA-устройств: {count}.",
        "pyannote_gpu_runtime_failed": "Проверка GPU для pyannote не пройдена: {message}.",
        "pyannote_gpu_install_hint": "Для GPU-диаризации нужен CUDA-совместимый стек torch/torchaudio. См. docs/pyannote-gpu.md.",
        "confirmation_title": "GPU-диаризация недоступна",
        "confirmation_switch_cpu": "Переключить на CPU и запустить",
        "confirmation_cancel": "Оставить текущие настройки",
        "button_start": "Запустить транскрибацию",
        "button_working": "Идёт обработка...",
        "submit_note_idle": "Одновременно может выполняться только одна локальная задача.",
        "submit_note_uploading": "Загрузка файла и запуск локальной задачи...",
        "submit_note_active": "Активная задача",
        "status_idle": "Ожидание",
        "status_queued": "В очереди",
        "status_running": "Обработка",
        "status_diarizing": "Диаризация",
        "status_exporting": "Экспорт",
        "status_done": "Готово",
        "status_failed": "Ошибка",
        "message_waiting": "Ожидаем файл.",
        "message_queued": "В очереди",
        "message_preparing_audio": "Подготовка аудио",
        "message_assigning_speakers": "Назначение спикеров",
        "message_writing_output": "Запись выходных файлов",
        "message_finished": "Завершено",
        "message_processing_failed": "Обработка завершилась ошибкой",
        "progress_label": "Прогресс",
        "stage_queued": "Очередь",
        "stage_running": "ASR",
        "stage_diarizing": "Спикеры",
        "stage_exporting": "Экспорт",
        "stage_done": "Готово",
        "job_error_summary": "Ошибка задачи",
        "preview_placeholder": "Превью появится здесь после первой завершённой обработки.",
        "download_prefix": "Скачать",
        "polling_failed": "Не удалось получить статус",
        "could_not_start_job": "Не удалось запустить задачу",
        "could_not_start_transcription": "Не удалось запустить транскрибацию.",
        "err_file_required": "Нужно выбрать файл.",
        "err_unsupported_file_type": "Неподдерживаемый тип файла.",
        "err_unsupported_model": "Неподдерживаемая модель.",
        "err_unsupported_device": "Неподдерживаемое устройство.",
        "err_unsupported_engine": "Неподдерживаемый движок.",
        "err_unsupported_quality": "Неподдерживаемый режим качества.",
        "err_unsupported_speakers": "Неподдерживаемый режим спикеров.",
        "err_hf_required": "Для диаризации через pyannote нужен токен Hugging Face.",
        "err_cuda_only_faster": "CUDA-ускорение сейчас доступно только для faster-whisper.",
        "err_cuda_unavailable": "CUDA-ускорение недоступно для faster-whisper на этой системе.",
        "err_busy_job": "Сейчас уже выполняется другая задача транскрибации.",
        "err_job_not_found": "Задача не найдена.",
        "err_artifacts_not_ready": "Артефакты ещё не готовы.",
        "err_artifact_not_found": "Артефакт не найден.",
        "err_artifact_missing": "Файл артефакта отсутствует.",
        "lang_en": "Английский",
        "lang_ru": "Русский",
    },
}


def normalize_ui_language(value: Optional[str]) -> str:
    language = (value or DEFAULT_UI_LANGUAGE).lower()
    if language not in UI_LANGUAGE_CHOICES:
        return DEFAULT_UI_LANGUAGE
    return language


def translate(ui_language: str, key: str, **kwargs: Any) -> str:
    language = normalize_ui_language(ui_language)
    template = UI_TRANSLATIONS.get(language, UI_TRANSLATIONS[DEFAULT_UI_LANGUAGE]).get(
        key,
        UI_TRANSLATIONS[DEFAULT_UI_LANGUAGE].get(key, key),
    )
    return template.format(**kwargs) if kwargs else template


def _ui_dictionary(ui_language: str) -> Dict[str, str]:
    language = normalize_ui_language(ui_language)
    return dict(UI_TRANSLATIONS[DEFAULT_UI_LANGUAGE], **UI_TRANSLATIONS.get(language, {}))


def _translate_status_message(ui_language: str, status: str, message: str) -> str:
    message_key = MESSAGE_KEY_BY_TEXT.get(message)
    if message_key:
        return translate(ui_language, message_key)

    fallback_key = {
        "queued": "message_queued",
        "done": "message_finished",
        "failed": "message_processing_failed",
    }.get(status)
    if fallback_key:
        return translate(ui_language, fallback_key)
    return message


def _translate_status_label(ui_language: str, status: str) -> str:
    return translate(ui_language, STATUS_LABEL_KEYS.get(status, "status_idle"))


@dataclass
class JobState:
    job_id: str
    filename: str
    input_path: str
    output_dir: str
    ui_language: str = DEFAULT_UI_LANGUAGE
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
        device: str,
        diarization_device: str,
        ui_language: str,
        language: Optional[str],
        quality: str,
        speakers: str,
        engine: str,
        keep_timestamps: bool,
        hf_token: Optional[str],
    ) -> JobState:
        ui_language = normalize_ui_language(ui_language)
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
                ui_language=ui_language,
                message=translate(ui_language, "message_queued"),
            )
            self._jobs[job_id] = job
            self._active_job_id = job_id

        request = TranscriptionRequest(
            input_path=str(input_path),
            model=model,
            engine=engine,
            device=device,
            diarization_device=diarization_device,
            ui_language=ui_language,
            language=language or None,
            output_formats=["txt", "srt", "vtt"],
            output_dir=str(output_dir),
            quiet=True,
            high_quality=quality == "high",
            no_timestamps=not keep_timestamps,
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
            job.message = _translate_status_message(job.ui_language, status, message)
            job.progress = STATUS_PROGRESS.get(status, job.progress)

    def _fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.message = translate(job.ui_language, "message_processing_failed")
            job.error = error
            job.progress = STATUS_PROGRESS["failed"]

    def _complete_job(self, job_id: str, result: TranscriptionResult) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "done"
            job.message = translate(job.ui_language, "message_finished")
            job.result = result
            job.progress = STATUS_PROGRESS["done"]


def _build_job_payload(job: JobState) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "job_id": job.job_id,
        "status": job.status,
        "status_label": _translate_status_label(job.ui_language, job.status),
        "message": job.message,
        "error": job.error,
        "filename": job.filename,
        "progress": job.progress,
        "active": job.status in ACTIVE_JOB_STATUSES,
    }

    if job.result:
        payload["language"] = job.result.language
        payload["preview_text"] = job.result.preview_text or job.result.text
        payload["metadata"] = dict(job.result.metadata)
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


def _web_defaults(config: Dict[str, object], ui_language: str) -> Dict[str, Any]:
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
    device = str(config.get("device") or DEFAULT_DEVICE).lower()
    if device not in DEVICE_CHOICES:
        device = DEFAULT_DEVICE
    diarization_device = str(config.get("diarization_device") or DEFAULT_DEVICE).lower()
    if diarization_device not in DEVICE_CHOICES:
        diarization_device = DEFAULT_DEVICE

    config_ui_language = normalize_ui_language(ui_language or str(config.get("ui_language") or DEFAULT_UI_LANGUAGE))
    return {
        "model": normalize_model_name(str(config.get("model") or "small")) or "small",
        "device": device,
        "diarization_device": diarization_device,
        "language": str(config.get("language") or ""),
        "quality": "high" if config.get("high_quality") else "balanced",
        "speakers": speakers,
        "engine": engine,
        "keep_timestamps": bool(config.get("web_keep_timestamps", False)),
        "hf_token": str(_resolve_hf_token(config) or ""),
        "ui_language": config_ui_language,
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
    runtime_info = get_runtime_acceleration_info()
    runtime_info.update(get_pyannote_runtime_info())

    app = FastAPI(title="Transcribator Web")
    app.state.templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    app.state.job_manager = JobManager(
        service=service or TranscriptionService(),
        output_root=output_root,
        config=config,
        hf_token=_resolve_hf_token(config),
    )
    app.state.config = config
    app.state.runtime_info = runtime_info

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, ui_lang: str = Query(DEFAULT_UI_LANGUAGE)):
        normalized_ui_language = normalize_ui_language(ui_lang or str(config.get("ui_language") or DEFAULT_UI_LANGUAGE))
        defaults = _web_defaults(config, normalized_ui_language)
        defaults["ui_language"] = normalized_ui_language
        ui = _ui_dictionary(normalized_ui_language)
        return app.state.templates.TemplateResponse(
            request,
            "index.html",
            {
                "accept_extensions": ",".join(sorted(VIDEO_EXTENSIONS)),
                "defaults": defaults,
                "engine_choices": ENGINE_CHOICES,
                "device_choices": DEVICE_CHOICES,
                "runtime_info": runtime_info,
                "cuda_toolkit_url": CUDA_TOOLKIT_URL,
                "model_choices": WHISPER_MODEL_CHOICES,
                "model_info": get_model_info(),
                "pyannote_access_repositories": PYANNOTE_ACCESS_REPOSITORIES,
                "ui_lang": normalized_ui_language,
                "ui": ui,
                "ui_language_choices": UI_LANGUAGE_CHOICES,
                "tr": lambda key, **kwargs: translate(normalized_ui_language, key, **kwargs),
            },
        )

    @app.post("/jobs")
    async def create_job(
        file: UploadFile = File(...),
        model: str = Form("small"),
        device: str = Form(DEFAULT_DEVICE),
        diarization_device: str = Form(DEFAULT_DEVICE),
        ui_language: str = Form(DEFAULT_UI_LANGUAGE),
        language: str = Form(""),
        quality: str = Form("balanced"),
        speakers: str = Form("off"),
        engine: str = Form(DEFAULT_ENGINE),
        keep_timestamps: bool = Form(False),
        hf_token: str = Form(""),
        allow_unavailable_diarization_cuda: bool = Form(False),
    ):
        ui_language = normalize_ui_language(ui_language)
        filename = Path(file.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_file_required"))
        if not is_video_file(filename):
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_file_type"))

        normalized_model = normalize_model_name(model) or "small"
        if normalized_model not in WHISPER_MODEL_CHOICES:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_model"))
        if device not in DEVICE_CHOICES:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_device"))
        if diarization_device not in DEVICE_CHOICES:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_device"))
        if engine not in ENGINE_CHOICES:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_engine"))
        if quality not in {"balanced", "high"}:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_quality"))
        if speakers not in {"off", "simple", "pyannote"}:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_unsupported_speakers"))
        if speakers == "pyannote" and not hf_token.strip():
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_hf_required"))

        runtime_info = app.state.runtime_info
        if device == "cuda" and engine != DEFAULT_ENGINE:
            raise HTTPException(status_code=400, detail=translate(ui_language, "err_cuda_only_faster"))
        if device == "cuda" and not bool(runtime_info.get("faster_whisper_cuda_available")):
            runtime_error = (
                runtime_info.get("faster_whisper_cuda_runtime_guidance")
                or runtime_info.get("faster_whisper_cuda_runtime_error")
            )
            detail = str(runtime_error) if runtime_error else translate(ui_language, "err_cuda_unavailable")
            detail = f"{detail} {CUDA_TOOLKIT_URL}"
            raise HTTPException(status_code=400, detail=detail)
        if (
            speakers == "pyannote"
            and diarization_device == "cuda"
            and not allow_unavailable_diarization_cuda
            and not bool(runtime_info.get("pyannote_gpu_available"))
        ):
            detail_message = str(
                runtime_info.get("pyannote_runtime_error")
                or runtime_info.get("pyannote_runtime_guidance")
                or translate(ui_language, "pyannote_gpu_install_hint")
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "message": detail_message,
                    "confirmation_required": True,
                    "confirmation_title": translate(ui_language, "confirmation_title"),
                    "switch_label": translate(ui_language, "confirmation_switch_cpu"),
                    "proceed_label": translate(ui_language, "confirmation_try_anyway"),
                    "cancel_label": translate(ui_language, "confirmation_cancel"),
                    "suggested_diarization_device": "cpu",
                },
            )

        try:
            job = await app.state.job_manager.create_job(
                file,
                model=normalized_model,
                device=device,
                diarization_device=diarization_device,
                ui_language=ui_language,
                language=language.strip() or None,
                quality=quality,
                speakers=speakers,
                engine=engine,
                keep_timestamps=keep_timestamps,
                hf_token=hf_token.strip() or None,
            )
        except BusyJobError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": translate(ui_language, "err_busy_job"),
                    "active_job_id": exc.active_job_id,
                },
            ) from exc

        return {
            "job_id": job.job_id,
            "status": "queued",
            "status_label": _translate_status_label(ui_language, "queued"),
            "message": translate(ui_language, "message_queued"),
            "progress": STATUS_PROGRESS["queued"],
            "active": True,
        }

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        try:
            job = app.state.job_manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=translate(DEFAULT_UI_LANGUAGE, "err_job_not_found")) from exc

        return _build_job_payload(job)

    @app.get("/jobs/{job_id}/download/{artifact_name}")
    async def download_artifact(job_id: str, artifact_name: str):
        try:
            job = app.state.job_manager.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=translate(DEFAULT_UI_LANGUAGE, "err_job_not_found")) from exc

        if not job.result:
            raise HTTPException(status_code=409, detail=translate(job.ui_language, "err_artifacts_not_ready"))

        artifact_path = job.result.artifacts.get(artifact_name)
        if not artifact_path:
            raise HTTPException(status_code=404, detail=translate(job.ui_language, "err_artifact_not_found"))

        path = Path(artifact_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=translate(job.ui_language, "err_artifact_missing"))

        return FileResponse(
            path,
            filename=path.name,
            content_disposition_type="attachment",
        )

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
