from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional

from .contracts import TranscriptionRequest, TranscriptionResult, TranscriptionSegment
from .transcriber import VideoTranscriber
from .utils import normalize_model_name


ENGINE_CHOICES = ["faster-whisper", "openai-whisper"]
DEFAULT_ENGINE = "faster-whisper"


def apply_hf_runtime_settings(hf_token: Optional[str]) -> None:
    if os.name == "nt":
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


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
        self.model = self._get_or_load_model(self.model_name)

    @classmethod
    def _get_or_load_model(cls, model_name: str):
        cache_key = f"cpu:int8:{model_name}"
        with cls._cache_lock:
            if cache_key in cls._model_cache:
                return cls._model_cache[cache_key]

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install faster-whisper and ctranslate2 first."
                ) from exc

            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=os.cpu_count() or 4,
            )
            cls._model_cache[cache_key] = model
            return model

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
                "language_probability": getattr(info, "language_probability", None),
            },
        )


class OpenAIWhisperTranscriber(BaseTranscriber):
    def __init__(self, request: TranscriptionRequest):
        super().__init__(request)
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
        )


def build_transcriber(request: TranscriptionRequest) -> BaseTranscriber:
    engine = (request.engine or DEFAULT_ENGINE).lower()
    if engine == DEFAULT_ENGINE:
        return FasterWhisperTranscriber(request)
    if engine == "openai-whisper":
        return OpenAIWhisperTranscriber(request)
    raise ValueError(f"Unsupported transcription engine: {request.engine}")
