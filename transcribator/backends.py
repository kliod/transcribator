from __future__ import annotations

import ctypes
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

from .contracts import TranscriptionRequest, TranscriptionResult, TranscriptionSegment
from .transcriber import VideoTranscriber
from .utils import normalize_model_name


ENGINE_CHOICES = ["faster-whisper", "openai-whisper"]
DEFAULT_ENGINE = "faster-whisper"
DEVICE_CHOICES = ["cpu", "cuda", "auto"]
DEFAULT_DEVICE = "cpu"
CUDA_REQUIRED_DLLS = ["cublas64_12.dll", "cudnn64_9.dll"]
_WINDOWS_DLL_HANDLES: list[object] = []
_WINDOWS_RUNTIME_LOCK = threading.Lock()
_WINDOWS_RUNTIME_PREPARED = False


def apply_hf_runtime_settings(hf_token: Optional[str]) -> None:
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


def _version_key(path: Path) -> Tuple[int, ...]:
    parts = []
    for part in path.name.lstrip("v").split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _discover_windows_cuda_dirs() -> list[str]:
    candidates: list[Path] = []
    seen: set[str] = set()

    env_candidates = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDNN_PATH"),
    ]
    env_candidates.extend(
        value
        for key, value in os.environ.items()
        if key.startswith("CUDA_PATH_V") and value
    )

    for raw in env_candidates:
        if not raw:
            continue
        root = Path(raw)
        bin_dir = root / "bin"
        if bin_dir.is_dir():
            candidates.append(bin_dir)
        elif root.is_dir():
            candidates.append(root)

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.is_dir():
        for version_dir in sorted(cuda_root.glob("v*"), key=_version_key, reverse=True):
            bin_dir = version_dir / "bin"
            if bin_dir.is_dir():
                candidates.append(bin_dir)

    cudnn_root = Path(r"C:\Program Files\NVIDIA\CUDNN")
    if cudnn_root.is_dir():
        for version_dir in sorted(cudnn_root.glob("v*"), key=_version_key, reverse=True):
            for arch_dir in sorted(version_dir.glob(r"bin\*\x64"), reverse=True):
                if arch_dir.is_dir():
                    candidates.append(arch_dir)
            fallback_bin = version_dir / "bin"
            if fallback_bin.is_dir():
                candidates.append(fallback_bin)

    resolved: list[str] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        candidate_key = candidate_str.lower()
        if candidate_key in seen or not candidate.is_dir():
            continue
        seen.add(candidate_key)
        resolved.append(candidate_str)

    return resolved


def _prepare_windows_cuda_runtime() -> list[str]:
    global _WINDOWS_RUNTIME_PREPARED

    if os.name != "nt":
        return []

    with _WINDOWS_RUNTIME_LOCK:
        candidate_dirs = _discover_windows_cuda_dirs()
        if not candidate_dirs:
            return []

        path_entries = [entry for entry in os.environ.get("PATH", "").split(";") if entry]
        path_seen = {entry.lower() for entry in path_entries}
        prepend_entries: list[str] = []

        for directory in candidate_dirs:
            if directory.lower() not in path_seen:
                prepend_entries.append(directory)
                path_seen.add(directory.lower())

            if hasattr(os, "add_dll_directory"):
                try:
                    _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(directory))
                except OSError:
                    continue

        if prepend_entries:
            os.environ["PATH"] = ";".join(prepend_entries + path_entries)

        _WINDOWS_RUNTIME_PREPARED = True
        return candidate_dirs


def _probe_windows_cuda_runtime() -> Tuple[bool, list[str]]:
    _prepare_windows_cuda_runtime()
    missing_dlls = []

    for dll_name in CUDA_REQUIRED_DLLS:
        try:
            ctypes.WinDLL(dll_name)
        except OSError:
            missing_dlls.append(dll_name)

    return len(missing_dlls) == 0, missing_dlls


def build_cuda_runtime_guidance(runtime_info: Dict[str, object]) -> str:
    ctranslate2_version = str(runtime_info.get("ctranslate2_version") or "current")
    base = (
        f"Current CTranslate2 {ctranslate2_version} expects CUDA 12 and cuDNN 9 on Windows. "
        "An NVIDIA driver alone is not enough: the CUDA runtime DLLs must be installed and available in PATH."
    )

    missing = runtime_info.get("faster_whisper_cuda_missing_dlls") or []
    if missing:
        base = f"{base} Missing DLLs: {', '.join(str(item) for item in missing)}."

    search_dirs = runtime_info.get("faster_whisper_cuda_search_dirs") or []
    if search_dirs:
        base = f"{base} Probed directories: {', '.join(str(item) for item in search_dirs)}."

    return (
        f"{base} Install CUDA runtime/toolkit: https://developer.nvidia.com/cuda/toolkit "
        "and make sure the CUDA bin folders are in PATH. "
        "If you intentionally stay on CUDA 11, use ctranslate2==3.24.0 instead."
    )


def get_runtime_acceleration_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "faster_whisper_cuda_available": False,
        "faster_whisper_cuda_device_count": 0,
        "faster_whisper_cuda_runtime_ready": True,
        "faster_whisper_cuda_missing_dlls": [],
        "openai_whisper_cuda_available": False,
    }

    if os.name == "nt":
        info["faster_whisper_cuda_search_dirs"] = _prepare_windows_cuda_runtime()
        runtime_ready, missing_dlls = _probe_windows_cuda_runtime()
        info["faster_whisper_cuda_runtime_ready"] = runtime_ready
        info["faster_whisper_cuda_missing_dlls"] = missing_dlls
        if not runtime_ready:
            info["faster_whisper_cuda_runtime_error"] = (
                "CUDA runtime DLLs are missing or cannot be loaded: "
                + ", ".join(missing_dlls)
            )

    try:
        import ctranslate2

        info["ctranslate2_version"] = ctranslate2.__version__
        device_count_getter = getattr(ctranslate2, "get_cuda_device_count", None)
        if callable(device_count_getter):
            device_count = int(device_count_getter() or 0)
            info["faster_whisper_cuda_device_count"] = device_count
            info["faster_whisper_cuda_available"] = (
                device_count > 0 and bool(info["faster_whisper_cuda_runtime_ready"])
            )
    except Exception as exc:
        info["faster_whisper_cuda_error"] = str(exc)

    if not info["faster_whisper_cuda_available"] and (
        info.get("faster_whisper_cuda_runtime_error")
        or info.get("faster_whisper_cuda_missing_dlls")
    ):
        info["faster_whisper_cuda_runtime_guidance"] = build_cuda_runtime_guidance(info)

    try:
        import torch

        info["openai_whisper_cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["openai_whisper_cuda_error"] = str(exc)

    return info


class BaseTranscriber(ABC):
    def __init__(self, request: TranscriptionRequest):
        self.request = request
        self.model_name = normalize_model_name(request.model) or request.model
        apply_hf_runtime_settings(request.hf_token)

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        raise NotImplementedError


class FasterWhisperTranscriber(BaseTranscriber):
    _model_cache: Dict[str, object] = {}
    _cache_lock = threading.Lock()

    def __init__(self, request: TranscriptionRequest):
        super().__init__(request)
        self.runtime_device, self.compute_type = self._resolve_runtime(request)
        self.model = self._get_or_load_model(
            self.model_name,
            self.runtime_device,
            self.compute_type,
        )

    @classmethod
    def _get_or_load_model(cls, model_name: str, runtime_device: str, compute_type: str):
        cache_key = f"{runtime_device}:{compute_type}:{model_name}"
        with cls._cache_lock:
            if cache_key in cls._model_cache:
                return cls._model_cache[cache_key]

            if runtime_device == "cuda" and os.name == "nt":
                _prepare_windows_cuda_runtime()

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install faster-whisper and ctranslate2 first."
                ) from exc

            kwargs = {
                "device": runtime_device,
                "compute_type": compute_type,
            }
            if runtime_device == "cpu":
                kwargs["cpu_threads"] = os.cpu_count() or 4

            model = WhisperModel(
                model_name,
                **kwargs,
            )
            cls._model_cache[cache_key] = model
            return model

    def _resolve_runtime(self, request: TranscriptionRequest) -> Tuple[str, str]:
        requested_device = request.normalized_device()
        runtime_info = get_runtime_acceleration_info()
        cuda_available = bool(runtime_info["faster_whisper_cuda_available"])

        if requested_device == "auto":
            requested_device = "cuda" if cuda_available else "cpu"

        if requested_device == "cuda":
            if not cuda_available:
                guidance = runtime_info.get("faster_whisper_cuda_runtime_guidance")
                runtime_error = runtime_info.get("faster_whisper_cuda_runtime_error")
                if guidance:
                    raise RuntimeError(str(guidance))
                if runtime_error:
                    raise RuntimeError(str(runtime_error))
                raise RuntimeError(
                    "CUDA acceleration is not available for faster-whisper on this system."
                )
            return "cuda", "float16"

        return "cpu", "int8"

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        options = {
            "language": self.request.language,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "word_timestamps": False,
        }

        if self.request.high_quality:
            options["beam_size"] = self.request.beam_size or 5
            if self.request.best_of is not None:
                options["best_of"] = self.request.best_of
        else:
            options["beam_size"] = self.request.beam_size or 1
            if self.request.best_of is not None:
                options["best_of"] = self.request.best_of

        options = {key: value for key, value in options.items() if value is not None}

        segments_iter, info = self.model.transcribe(audio_path, **options)

        segments = []
        text_parts = []
        for segment in segments_iter:
            text = segment.text.strip()
            segments.append(
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )
            if text:
                text_parts.append(text)

        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            segments=segments,
            language=getattr(info, "language", self.request.language),
            engine=DEFAULT_ENGINE,
            model=self.model_name,
            metadata={
                "device": self.runtime_device,
                "compute_type": self.compute_type,
                "language_probability": getattr(info, "language_probability", None),
            },
        )


class OpenAIWhisperTranscriber(BaseTranscriber):
    def __init__(self, request: TranscriptionRequest):
        super().__init__(request)
        requested_device = request.normalized_device()
        if requested_device in {"cuda", "auto"} and requested_device != "cpu":
            runtime_info = get_runtime_acceleration_info()
            if requested_device == "cuda" or bool(runtime_info["openai_whisper_cuda_available"]):
                raise RuntimeError(
                    "CUDA acceleration is currently available only for faster-whisper in this build."
                )
        self.transcriber = VideoTranscriber(
            model_name=self.model_name,
            language=request.language,
            high_quality=request.high_quality,
            beam_size=request.beam_size,
            best_of=request.best_of,
            preprocess_audio_flag=False,
            verbose=not request.quiet,
        )

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        raw_result = self.transcriber.transcribe(audio_path)
        raw_segments = self.transcriber.get_segments_with_timestamps(raw_result)
        segments = [
            TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=segment["text"],
            )
            for segment in raw_segments
        ]

        return TranscriptionResult(
            text=(raw_result.get("text") or "").strip(),
            segments=segments,
            language=raw_result.get("language", self.request.language),
            engine="openai-whisper",
            model=self.model_name,
            metadata={
                "device": "cpu",
            },
        )


def build_transcriber(request: TranscriptionRequest) -> BaseTranscriber:
    engine = (request.engine or DEFAULT_ENGINE).lower()
    if engine == DEFAULT_ENGINE:
        return FasterWhisperTranscriber(request)
    if engine == "openai-whisper":
        return OpenAIWhisperTranscriber(request)
    raise ValueError(f"Unsupported transcription engine: {request.engine}")
