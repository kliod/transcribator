from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_OUTPUT_FORMATS = ["txt", "srt", "vtt"]


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    speaker: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "start": float(self.start),
            "end": float(self.end),
            "text": self.text,
        }
        if self.speaker is not None:
            payload["speaker"] = int(self.speaker)
        return payload


@dataclass
class TranscriptionRequest:
    input_path: str
    model: str = "small"
    engine: str = "faster-whisper"
    device: str = "cpu"
    diarization_device: str = "cpu"
    ui_language: str = "en"
    language: Optional[str] = None
    output_formats: List[str] = field(default_factory=lambda: list(DEFAULT_OUTPUT_FORMATS))
    output_dir: Optional[str] = None
    quiet: bool = False
    high_quality: bool = False
    no_timestamps: bool = False
    clean_txt: bool = False
    diarize: str = "none"
    hf_token: Optional[str] = None
    beam_size: Optional[int] = None
    best_of: Optional[int] = None
    preprocess_audio: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diarization_threshold: Optional[float] = None
    pause_threshold: Optional[float] = None

    def normalized_output_formats(self) -> List[str]:
        formats = []
        for item in self.output_formats or DEFAULT_OUTPUT_FORMATS:
            normalized = item.lower()
            if normalized not in formats:
                formats.append(normalized)
        return formats or list(DEFAULT_OUTPUT_FORMATS)

    def normalized_diarization_mode(self) -> str:
        mode = (self.diarize or "none").lower()
        if mode == "off":
            return "none"
        return mode

    def normalized_device(self) -> str:
        device = (self.device or "cpu").lower()
        if device not in {"cpu", "cuda", "auto"}:
            return "cpu"
        return device

    def normalized_diarization_device(self) -> str:
        device = (self.diarization_device or "cpu").lower()
        if device not in {"cpu", "cuda", "auto"}:
            return "cpu"
        return device

    def normalized_ui_language(self) -> str:
        language = (self.ui_language or "en").lower()
        if language not in {"en", "ru"}:
            return "en"
        return language


@dataclass
class TranscriptionResult:
    text: str
    segments: List[TranscriptionSegment]
    language: Optional[str]
    engine: str
    model: str
    status: str = "done"
    artifacts: Dict[str, str] = field(default_factory=dict)
    preview_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def segment_dicts(self) -> List[Dict[str, Any]]:
        return [segment.to_dict() for segment in self.segments]
