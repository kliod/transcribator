from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Iterable, List, Optional, Tuple

from .audio_preparation import prepare_audio_file
from .backends import build_transcriber
from .contracts import TranscriptionRequest, TranscriptionResult, TranscriptionSegment
from .diarization import SpeakerDiarizer
from .exporter import (
    export_transcription,
    export_txt,
    render_text_transcript,
    speaker_label_template_for_language,
)
from .utils import ensure_output_directory, get_output_filename


StatusCallback = Optional[Callable[[str, str], None]]


class TranscriptionService:
    def transcribe_file(
        self,
        request: TranscriptionRequest,
        status_callback: StatusCallback = None,
        output_name: Optional[str] = None,
    ) -> TranscriptionResult:
        self._notify(status_callback, "running", "Preparing audio")

        with TemporaryDirectory(prefix="transcribator-") as working_directory:
            prepared_audio_path = prepare_audio_file(
                request.input_path,
                working_directory,
                enhance=request.preprocess_audio or request.high_quality,
            )

            backend = build_transcriber(request)
            result = backend.transcribe(prepared_audio_path)

            diarization_mode = request.normalized_diarization_mode()
            if diarization_mode != "none":
                self._notify(status_callback, "diarizing", "Assigning speakers")
                result.segments, diarization_metadata = self._apply_diarization(
                    result.segments,
                    diarization_mode,
                    prepared_audio_path,
                    request,
                )
                result.metadata.update(diarization_metadata)

            self._notify(status_callback, "exporting", "Writing output files")
            output_directory = ensure_output_directory(request.output_dir, request.input_path)
            output_base_path = self._build_output_base_path(
                output_directory,
                request.input_path,
                output_name=output_name,
            )
            artifacts, preview_text = self._export_outputs(
                result.segments,
                output_base_path,
                request,
            )

        result.artifacts = artifacts
        result.preview_text = preview_text
        result.status = "done"
        result.metadata["output_dir"] = output_directory
        return result

    def _apply_diarization(
        self,
        segments: List[TranscriptionSegment],
        diarization_mode: str,
        audio_path: str,
        request: TranscriptionRequest,
    ) -> Tuple[List[TranscriptionSegment], dict]:
        segment_dicts = [segment.to_dict() for segment in segments]
        diarizer = self._create_diarizer(diarization_mode, request)

        diarized_dicts = diarizer.diarize(
            segment_dicts,
            audio_path=audio_path if diarization_mode in ("pyannote", "auto") else None,
            hf_token=request.hf_token,
        )

        diarized_segments = [
            TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=segment["text"],
                speaker=segment.get("speaker"),
            )
            for segment in diarized_dicts
        ]
        return diarized_segments, {
            "diarization_method": diarization_mode,
            "diarization_requested_device": request.normalized_diarization_device(),
            "diarization_device": diarizer.resolved_device,
        }

    def _create_diarizer(self, diarization_mode: str, request: TranscriptionRequest) -> SpeakerDiarizer:
        return SpeakerDiarizer(
            method=diarization_mode,
            pause_threshold=request.pause_threshold if request.pause_threshold is not None else 2.0,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
            clustering_threshold=request.diarization_threshold,
            device=request.normalized_diarization_device(),
        )

    def _build_output_base_path(
        self,
        output_directory: str,
        input_path: str,
        output_name: Optional[str] = None,
    ) -> str:
        if output_name:
            base_name = Path(output_name).stem
            return str(Path(output_directory) / base_name)
        return get_output_filename(input_path, output_directory, "")

    def _export_outputs(
        self,
        segments: List[TranscriptionSegment],
        output_base_path: str,
        request: TranscriptionRequest,
    ) -> Tuple[dict, str]:
        segment_dicts = [segment.to_dict() for segment in segments]
        include_speakers = any(segment.speaker is not None for segment in segments)
        formats = request.normalized_output_formats()
        speaker_label_template = speaker_label_template_for_language(request.normalized_ui_language())

        export_transcription(
            segment_dicts,
            output_base_path,
            formats,
            include_timestamps_in_txt=not request.no_timestamps,
            include_speakers=include_speakers,
            speaker_label_template=speaker_label_template,
        )

        artifacts = {
            fmt: str(Path(output_base_path).with_suffix(f".{fmt}"))
            for fmt in formats
        }

        preview_text = self._render_preview(
            segment_dicts,
            include_timestamps=not request.no_timestamps,
            ui_language=request.normalized_ui_language(),
        )

        if request.clean_txt:
            clean_path = str(
                Path(output_base_path).with_name(f"{Path(output_base_path).stem}_clean.txt")
            )
            export_txt(
                segment_dicts,
                clean_path,
                include_timestamps=False,
                include_speakers=include_speakers,
                speaker_label_template=speaker_label_template,
            )
            artifacts["clean_txt"] = clean_path

        return artifacts, preview_text

    def _render_preview(
        self,
        segments: Iterable[dict],
        *,
        include_timestamps: bool,
        ui_language: str,
    ) -> str:
        return render_text_transcript(
            segments,
            include_timestamps=include_timestamps,
            include_speakers=any(segment.get("speaker") is not None for segment in segments),
            speaker_label_template=speaker_label_template_for_language(ui_language),
        )

    def _notify(self, status_callback: StatusCallback, status: str, message: str) -> None:
        if status_callback:
            status_callback(status, message)
