"""Legacy OpenAI Whisper transcription backend."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import whisper

from .audio_processor import preprocess_audio, validate_audio


class VideoTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        language: Optional[str] = None,
        high_quality: bool = False,
        beam_size: Optional[int] = None,
        best_of: Optional[int] = None,
        preprocess_audio_flag: bool = False,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        condition_on_previous_text: Optional[bool] = None,
        initial_prompt: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.language = language
        self.high_quality = high_quality
        self.beam_size = beam_size
        self.best_of = best_of
        self.preprocess_audio_flag = preprocess_audio_flag
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.verbose = verbose
        self.model = None
        self._load_model()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _load_model(self) -> None:
        try:
            self._log(f"Loading OpenAI Whisper model: {self.model_name}...")
            self.model = whisper.load_model(self.model_name)
            self._log("Model loaded successfully.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load OpenAI Whisper model: {exc}") from exc

    def transcribe(self, video_path: str, **kwargs) -> Dict[str, Any]:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Input file was not found: {video_path}")

        transcribe_options = {
            "language": self.language,
            "word_timestamps": False,
            **kwargs,
        }

        if self.high_quality:
            transcribe_options.update(
                {
                    "temperature": 0,
                    "best_of": self.best_of if self.best_of is not None else 5,
                    "beam_size": self.beam_size if self.beam_size is not None else 5,
                }
            )
        else:
            transcribe_options["temperature"] = 0
            if self.beam_size is not None:
                transcribe_options["beam_size"] = self.beam_size
            if self.best_of is not None:
                transcribe_options["best_of"] = self.best_of

        if self.compression_ratio_threshold is not None:
            transcribe_options["compression_ratio_threshold"] = self.compression_ratio_threshold
        elif self.high_quality:
            transcribe_options["compression_ratio_threshold"] = 2.4

        if self.logprob_threshold is not None:
            transcribe_options["logprob_threshold"] = self.logprob_threshold
        elif self.high_quality:
            transcribe_options["logprob_threshold"] = -1.0

        if self.no_speech_threshold is not None:
            transcribe_options["no_speech_threshold"] = self.no_speech_threshold
        elif self.high_quality:
            transcribe_options["no_speech_threshold"] = 0.6

        if self.condition_on_previous_text is not None:
            transcribe_options["condition_on_previous_text"] = self.condition_on_previous_text
        elif self.high_quality:
            transcribe_options["condition_on_previous_text"] = True

        if self.initial_prompt is not None:
            transcribe_options["initial_prompt"] = self.initial_prompt

        transcribe_options = {
            key: value for key, value in transcribe_options.items() if value is not None
        }

        try:
            self._log(f"Starting OpenAI Whisper transcription: {video_path}")
            audio = whisper.load_audio(video_path)
            sample_rate = 16000

            if self.preprocess_audio_flag:
                self._log("Applying additional audio preprocessing...")
                if validate_audio(audio, sample_rate):
                    audio, sample_rate = preprocess_audio(
                        audio,
                        sample_rate=sample_rate,
                        normalize=True,
                        denoise=True,
                        enhance_quiet=False,
                    )
                else:
                    warnings.warn("Audio validation failed; skipping extra preprocessing.")

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if not audio.flags["C_CONTIGUOUS"]:
                audio = np.ascontiguousarray(audio, dtype=np.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.model.transcribe(audio, **transcribe_options)

            self._log(f"Transcription finished. Language: {result.get('language', 'unknown')}")
            return result
        except Exception as exc:
            raise RuntimeError(f"OpenAI Whisper transcription failed: {exc}") from exc

    def get_segments_with_timestamps(self, transcription_result: Dict[str, Any]) -> list:
        segments = []
        raw_segments = transcription_result.get("segments", [])
        if not raw_segments:
            raise ValueError("Transcription result does not contain timed segments.")

        for segment in raw_segments:
            if "start" not in segment or "end" not in segment or "text" not in segment:
                continue

            start_time = float(segment["start"])
            end_time = float(segment["end"])
            if start_time < 0 or end_time < 0 or start_time >= end_time:
                continue

            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": segment["text"].strip(),
                }
            )

        if not segments:
            raise ValueError("No valid timed segments were produced by OpenAI Whisper.")
        return segments
