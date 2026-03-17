"""Speaker diarization utilities."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="torchcodec is not installed correctly.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*degrees of freedom.*", category=UserWarning)

DIARIZATION_DEVICE_CHOICES = ("cpu", "cuda", "auto")
PYANNOTE_GPU_GUIDANCE = (
    "Pyannote GPU mode requires a CUDA-enabled torch/torchaudio installation. "
    "Keep the default CPU setup or install the optional GPU profile described in docs/pyannote-gpu.md."
)


def get_pyannote_runtime_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "pyannote_available": False,
        "pyannote_gpu_available": False,
        "pyannote_torch_installed": False,
        "pyannote_torchaudio_available": False,
        "pyannote_torch_cuda_available": False,
        "pyannote_torch_device_count": 0,
        "pyannote_runtime_guidance": PYANNOTE_GPU_GUIDANCE,
    }

    try:
        import pyannote.audio  # noqa: F401

        info["pyannote_available"] = True
    except Exception as exc:
        info["pyannote_error"] = str(exc)

    try:
        import torch

        info["pyannote_torch_installed"] = True
        info["pyannote_torch_version"] = torch.__version__
        info["pyannote_torch_cuda_available"] = bool(torch.cuda.is_available())
        info["pyannote_torch_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        info["pyannote_torch_error"] = str(exc)

    try:
        import torchaudio  # noqa: F401

        info["pyannote_torchaudio_available"] = True
    except Exception as exc:
        info["pyannote_torchaudio_error"] = str(exc)

    info["pyannote_gpu_available"] = bool(
        info["pyannote_available"]
        and info["pyannote_torch_installed"]
        and info["pyannote_torchaudio_available"]
        and info["pyannote_torch_cuda_available"]
        and int(info["pyannote_torch_device_count"]) > 0
    )

    if not info["pyannote_available"]:
        info["pyannote_runtime_error"] = (
            "pyannote.audio is not installed. Install pyannote.audio to use speaker diarization."
        )
    elif not info["pyannote_torch_installed"]:
        info["pyannote_runtime_error"] = "torch is not installed."
    elif not info["pyannote_torchaudio_available"]:
        info["pyannote_runtime_error"] = "torchaudio is not installed or cannot be imported."
    elif not info["pyannote_torch_cuda_available"]:
        info["pyannote_runtime_error"] = (
            "Current torch build does not expose CUDA. Install a CUDA-enabled torch/torchaudio build."
        )

    return info


class SpeakerDiarizer:
    """Assign speaker labels to transcription segments."""

    def __init__(
        self,
        method: str = "simple",
        pause_threshold: float = 2.0,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        clustering_threshold: Optional[float] = None,
        device: str = "cpu",
    ):
        self.method = method.lower()
        self.pause_threshold = pause_threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.clustering_threshold = clustering_threshold if clustering_threshold is not None else 0.7
        self.requested_device = self._normalize_device(device)
        self.resolved_device = "cpu"
        self.pyannote_available = False
        self.runtime_info = get_pyannote_runtime_info()

        if self.method in {"pyannote", "auto"}:
            self.pyannote_available = bool(self.runtime_info.get("pyannote_available"))

    def _normalize_device(self, device: str) -> str:
        normalized = (device or "cpu").lower()
        if normalized not in DIARIZATION_DEVICE_CHOICES:
            return "cpu"
        return normalized

    def _resolve_pyannote_device(self) -> str:
        requested_device = self.requested_device
        gpu_available = bool(self.runtime_info.get("pyannote_gpu_available"))

        if requested_device == "auto":
            self.resolved_device = "cuda" if gpu_available else "cpu"
            return self.resolved_device

        if requested_device == "cuda":
            if not gpu_available:
                runtime_error = self.runtime_info.get("pyannote_runtime_error")
                guidance = self.runtime_info.get("pyannote_runtime_guidance")
                if runtime_error and guidance:
                    raise RuntimeError(f"{runtime_error} {guidance}")
                if runtime_error:
                    raise RuntimeError(str(runtime_error))
                raise RuntimeError(PYANNOTE_GPU_GUIDANCE)
            self.resolved_device = "cuda"
            return self.resolved_device

        self.resolved_device = "cpu"
        return self.resolved_device

    def diarize_simple(self, segments: List[Dict]) -> List[Dict]:
        """Approximate diarization by switching speakers on larger pauses."""
        self.resolved_device = "cpu"
        if not segments:
            return []

        pauses = []
        for index in range(1, len(segments)):
            pause = float(segments[index]["start"]) - float(segments[index - 1]["end"])
            if pause > 0:
                pauses.append(pause)

        if pauses:
            pause_median = float(np.median(pauses))
            pause_std = float(np.std(pauses)) if len(pauses) > 1 else pause_median * 0.5
            threshold = max(self.pause_threshold, pause_median + pause_std)
        else:
            threshold = self.pause_threshold

        diarized_segments: List[Dict] = []
        current_speaker = 0

        for index, segment in enumerate(segments):
            segment_copy = segment.copy()
            if index > 0:
                pause = float(segment["start"]) - float(segments[index - 1]["end"])
                if pause >= threshold:
                    current_speaker = 1 - current_speaker
            segment_copy["speaker"] = current_speaker
            diarized_segments.append(segment_copy)

        return self._smooth_assigned_speakers(diarized_segments)

    def diarize_pyannote(
        self,
        audio_path: str,
        hf_token: Optional[str] = None,
    ) -> List[Tuple[float, float, int]]:
        """Run pyannote and return diarization segments as tuples."""
        if not self.pyannote_available:
            raise RuntimeError(
                "pyannote.audio is not installed. Install pyannote.audio before using this mode."
            )
        if not hf_token:
            raise RuntimeError("Hugging Face token is required for pyannote diarization.")

        self._resolve_pyannote_device()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.filterwarnings("ignore", message=".*torchcodec.*")
                warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
                warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
                from pyannote.audio import Pipeline
        except Exception as exc:
            raise RuntimeError(f"Could not import pyannote.audio pipeline: {exc}") from exc

        pipeline = self._load_pipeline(Pipeline, hf_token)
        diarization = self._run_pipeline(pipeline, audio_path)
        raw_segments = self._extract_pyannote_segments(diarization)

        if not raw_segments:
            raise RuntimeError("pyannote returned no speaker segments.")

        return self._merge_adjacent_diarization_segments(raw_segments)

    def _load_pipeline(self, pipeline_class, hf_token: str):
        last_error: Optional[Exception] = None
        for kwargs in (
            {"token": hf_token},
            {"use_auth_token": hf_token},
        ):
            try:
                pipeline = pipeline_class.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    **kwargs,
                )
                self._configure_pipeline(pipeline)
                return pipeline
            except TypeError:
                continue
            except Exception as exc:
                last_error = exc

        message = (
            "Could not load pyannote speaker diarization pipeline. "
            "Check the Hugging Face token and repository access permissions."
        )
        if last_error:
            message = f"{message} Details: {last_error}"
        raise RuntimeError(message)

    def _configure_pipeline(self, pipeline) -> None:
        if hasattr(pipeline, "instantiate"):
            params = {}
            if self.min_speakers is not None:
                params["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                params["max_speakers"] = self.max_speakers
            if params:
                try:
                    pipeline.instantiate(params)
                except Exception:
                    pass

        if hasattr(pipeline, "clustering") and hasattr(pipeline.clustering, "threshold"):
            pipeline.clustering.threshold = self.clustering_threshold

        if self.resolved_device == "cuda":
            try:
                import torch

                if hasattr(pipeline, "to"):
                    pipeline.to(torch.device("cuda"))
            except Exception as exc:
                raise RuntimeError(f"Could not move pyannote pipeline to CUDA: {exc}") from exc

    def _run_pipeline(self, pipeline, audio_path: str):
        try:
            import torch
            import whisper

            waveform = whisper.load_audio(audio_path)
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if self.resolved_device == "cuda":
                waveform = waveform.to(torch.device("cuda"))

            payload = {"waveform": waveform, "sample_rate": 16000}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return pipeline(payload)
        except Exception:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return pipeline(audio_path)

    def _extract_pyannote_segments(self, diarization) -> List[Tuple[float, float, int]]:
        annotation = diarization
        for attribute in ("exclusive_speaker_diarization", "speaker_diarization"):
            candidate = getattr(diarization, attribute, None)
            if candidate is not None:
                annotation = candidate
                break

        speaker_ids: Dict[str, int] = {}
        extracted: List[Tuple[float, float, int]] = []

        if hasattr(annotation, "itertracks"):
            for turn, _, speaker_label in annotation.itertracks(yield_label=True):
                label_key = str(speaker_label)
                speaker_id = speaker_ids.setdefault(label_key, len(speaker_ids))
                extracted.append((float(turn.start), float(turn.end), speaker_id))
            return extracted

        if hasattr(annotation, "items"):
            for turn, speaker_label in annotation.items():
                label_key = str(speaker_label)
                speaker_id = speaker_ids.setdefault(label_key, len(speaker_ids))
                extracted.append((float(turn.start), float(turn.end), speaker_id))
            return extracted

        if hasattr(annotation, "get_timeline") and hasattr(annotation, "__getitem__"):
            for turn in annotation.get_timeline():
                speaker_label = annotation[turn]
                label_key = str(speaker_label)
                speaker_id = speaker_ids.setdefault(label_key, len(speaker_ids))
                extracted.append((float(turn.start), float(turn.end), speaker_id))
            return extracted

        raise RuntimeError(f"Unsupported pyannote diarization output: {type(diarization)}")

    def _merge_adjacent_diarization_segments(
        self,
        diarization_segments: List[Tuple[float, float, int]],
        *,
        max_gap: float = 0.35,
    ) -> List[Tuple[float, float, int]]:
        merged: List[List[float]] = []

        for start, end, speaker in diarization_segments:
            if not merged:
                merged.append([start, end, float(speaker)])
                continue

            previous = merged[-1]
            if int(previous[2]) == speaker and start - previous[1] <= max_gap:
                previous[1] = max(previous[1], end)
            else:
                merged.append([start, end, float(speaker)])

        return [(start, end, int(speaker)) for start, end, speaker in merged]

    def _smooth_assigned_speakers(self, segments: List[Dict]) -> List[Dict]:
        if len(segments) < 3:
            return segments

        smoothed = [segment.copy() for segment in segments]

        for index in range(1, len(smoothed) - 1):
            current = smoothed[index]
            previous = smoothed[index - 1]
            following = smoothed[index + 1]

            current_speaker = current.get("speaker")
            previous_speaker = previous.get("speaker")
            following_speaker = following.get("speaker")
            current_duration = float(current["end"]) - float(current["start"])
            gap_before = float(current["start"]) - float(previous["end"])
            gap_after = float(following["start"]) - float(current["end"])

            if (
                current_speaker is not None
                and previous_speaker == following_speaker
                and current_speaker != previous_speaker
                and current_duration <= 1.2
                and gap_before <= 0.6
                and gap_after <= 0.6
            ):
                current["speaker"] = previous_speaker

        return smoothed

    def assign_speakers_to_segments(
        self,
        segments: List[Dict],
        diarization_result: List[Tuple[float, float, int]],
    ) -> List[Dict]:
        """Assign pyannote speakers to ASR segments using overlap weights."""
        if not diarization_result:
            return segments

        diarized_segments: List[Dict] = []

        for segment in segments:
            segment_copy = segment.copy()
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
            speaker_scores = defaultdict(float)

            for diar_start, diar_end, speaker_id in diarization_result:
                overlap_start = max(seg_start, diar_start)
                overlap_end = min(seg_end, diar_end)
                if overlap_end > overlap_start:
                    speaker_scores[speaker_id] += overlap_end - overlap_start

            if speaker_scores:
                segment_copy["speaker"] = max(speaker_scores, key=speaker_scores.get)
            else:
                midpoint = (seg_start + seg_end) / 2
                assigned_speaker = 0
                for diar_start, diar_end, speaker_id in diarization_result:
                    if diar_start <= midpoint <= diar_end:
                        assigned_speaker = speaker_id
                        break
                segment_copy["speaker"] = assigned_speaker

            diarized_segments.append(segment_copy)

        return self._smooth_assigned_speakers(diarized_segments)

    def diarize(
        self,
        segments: List[Dict],
        audio_path: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> List[Dict]:
        """Run the selected diarization strategy."""
        if self.method == "none":
            self.resolved_device = "cpu"
            return segments

        if self.method == "simple":
            return self.diarize_simple(segments)

        if self.method in {"pyannote", "auto"}:
            if not audio_path:
                raise ValueError("audio_path is required for pyannote diarization.")
            if not self.pyannote_available:
                raise RuntimeError("pyannote.audio is not available. Install pyannote.audio to use this mode.")
            diarization_result = self.diarize_pyannote(audio_path, hf_token)
            return self.assign_speakers_to_segments(segments, diarization_result)

        raise ValueError(f"Unsupported diarization method: {self.method}")
