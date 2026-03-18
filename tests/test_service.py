from pathlib import Path

import pytest

import transcribator.service as service_module
from transcribator.contracts import TranscriptionRequest, TranscriptionResult, TranscriptionSegment


class _FakeBackend:
    def transcribe(self, audio_path):
        return TranscriptionResult(
            text="alpha beta",
            segments=[
                TranscriptionSegment(start=0.0, end=1.0, text="alpha"),
                TranscriptionSegment(start=1.0, end=2.0, text="beta"),
            ],
            language="ru",
            engine="faster-whisper",
            model="small",
        )


def test_service_fails_when_pyannote_diarization_breaks(monkeypatch, tmp_path):
    diarizer_methods = []

    def fake_prepare_audio_file(input_path, working_directory, enhance):
        return input_path

    def fake_build_transcriber(request):
        return _FakeBackend()

    class FakeDiarizer:
        def __init__(self, method, **kwargs):
            diarizer_methods.append(method)
            self.method = method

        def diarize(self, segments, audio_path=None, hf_token=None):
            raise RuntimeError("pyannote failed")

    monkeypatch.setattr(service_module, "prepare_audio_file", fake_prepare_audio_file)
    monkeypatch.setattr(service_module, "build_transcriber", fake_build_transcriber)
    monkeypatch.setattr(service_module, "SpeakerDiarizer", FakeDiarizer)

    request = TranscriptionRequest(
        input_path=str(tmp_path / "input.mp4"),
        output_dir=str(tmp_path / "out"),
        output_formats=["txt", "srt", "vtt"],
        diarize="pyannote",
        clean_txt=True,
        hf_token="hf_test_token",
    )

    statuses = []
    service = service_module.TranscriptionService()

    with pytest.raises(RuntimeError, match="pyannote failed"):
        service.transcribe_file(
            request,
            status_callback=lambda status, message: statuses.append(status),
            output_name="clip.mp4",
        )

    assert statuses == ["running", "diarizing"]
    assert diarizer_methods == ["pyannote"]


def test_service_skips_diarization_when_disabled(monkeypatch, tmp_path):
    def fake_prepare_audio_file(input_path, working_directory, enhance):
        return input_path

    def fake_build_transcriber(request):
        return _FakeBackend()

    def fake_export_transcription(segments, output_base_path, formats, **kwargs):
        Path(f"{output_base_path}.txt").write_text("txt", encoding="utf-8")

    monkeypatch.setattr(service_module, "prepare_audio_file", fake_prepare_audio_file)
    monkeypatch.setattr(service_module, "build_transcriber", fake_build_transcriber)
    monkeypatch.setattr(service_module, "export_transcription", fake_export_transcription)

    diarizer_called = {"value": False}

    class FakeDiarizer:
        def __init__(self, *args, **kwargs):
            diarizer_called["value"] = True

    monkeypatch.setattr(service_module, "SpeakerDiarizer", FakeDiarizer)

    request = TranscriptionRequest(
        input_path=str(tmp_path / "input.mp4"),
        output_dir=str(tmp_path / "out"),
        output_formats=["txt"],
        diarize="none",
        no_timestamps=True,
    )

    service = service_module.TranscriptionService()
    result = service.transcribe_file(request)

    assert diarizer_called["value"] is False
    assert result.preview_text == "alpha beta"


def test_service_preview_can_include_timestamps(monkeypatch, tmp_path):
    def fake_prepare_audio_file(input_path, working_directory, enhance):
        return input_path

    def fake_build_transcriber(request):
        return _FakeBackend()

    def fake_export_transcription(segments, output_base_path, formats, **kwargs):
        Path(f"{output_base_path}.txt").write_text("txt", encoding="utf-8")

    monkeypatch.setattr(service_module, "prepare_audio_file", fake_prepare_audio_file)
    monkeypatch.setattr(service_module, "build_transcriber", fake_build_transcriber)
    monkeypatch.setattr(service_module, "export_transcription", fake_export_transcription)

    request = TranscriptionRequest(
        input_path=str(tmp_path / "input.mp4"),
        output_dir=str(tmp_path / "out"),
        output_formats=["txt"],
        diarize="none",
        no_timestamps=False,
    )

    service = service_module.TranscriptionService()
    result = service.transcribe_file(request)

    assert result.preview_text == "[00:00] alpha beta"


def test_service_preview_groups_text_by_speaker(monkeypatch, tmp_path):
    class SpeakerBackend:
        def transcribe(self, audio_path):
            return TranscriptionResult(
                text="alpha beta gamma delta",
                segments=[
                    TranscriptionSegment(start=0.0, end=0.8, text="alpha", speaker=0),
                    TranscriptionSegment(start=0.9, end=1.7, text="beta", speaker=0),
                    TranscriptionSegment(start=3.0, end=3.8, text="gamma", speaker=1),
                    TranscriptionSegment(start=3.9, end=4.6, text="delta", speaker=1),
                ],
                language="ru",
                engine="faster-whisper",
                model="small",
            )

    def fake_prepare_audio_file(input_path, working_directory, enhance):
        return input_path

    def fake_build_transcriber(request):
        return SpeakerBackend()

    def fake_export_transcription(segments, output_base_path, formats, **kwargs):
        Path(f"{output_base_path}.txt").write_text("txt", encoding="utf-8")

    monkeypatch.setattr(service_module, "prepare_audio_file", fake_prepare_audio_file)
    monkeypatch.setattr(service_module, "build_transcriber", fake_build_transcriber)
    monkeypatch.setattr(service_module, "export_transcription", fake_export_transcription)

    request = TranscriptionRequest(
        input_path=str(tmp_path / "input.mp4"),
        output_dir=str(tmp_path / "out"),
        output_formats=["txt"],
        diarize="none",
        no_timestamps=True,
    )

    service = service_module.TranscriptionService()
    result = service.transcribe_file(request)

    assert result.preview_text == "[Speaker 1]\nalpha beta\n\n[Speaker 2]\ngamma delta"


def test_service_records_requested_and_resolved_diarization_device(monkeypatch, tmp_path):
    def fake_prepare_audio_file(input_path, working_directory, enhance):
        return input_path

    def fake_build_transcriber(request):
        return _FakeBackend()

    def fake_export_transcription(segments, output_base_path, formats, **kwargs):
        Path(f"{output_base_path}.txt").write_text("txt", encoding="utf-8")

    class FakeDiarizer:
        def __init__(self, method, **kwargs):
            self.method = method
            self.device = kwargs["device"]
            self.resolved_device = "cuda"

        def diarize(self, segments, audio_path=None, hf_token=None):
            return [{**segment, "speaker": 0} for segment in segments]

    monkeypatch.setattr(service_module, "prepare_audio_file", fake_prepare_audio_file)
    monkeypatch.setattr(service_module, "build_transcriber", fake_build_transcriber)
    monkeypatch.setattr(service_module, "export_transcription", fake_export_transcription)
    monkeypatch.setattr(service_module, "SpeakerDiarizer", FakeDiarizer)

    request = TranscriptionRequest(
        input_path=str(tmp_path / "input.mp4"),
        output_dir=str(tmp_path / "out"),
        output_formats=["txt"],
        diarize="pyannote",
        diarization_device="cuda",
        hf_token="hf_test_token",
    )

    service = service_module.TranscriptionService()
    result = service.transcribe_file(request)

    assert result.metadata["diarization_method"] == "pyannote"
    assert result.metadata["diarization_requested_device"] == "cuda"
    assert result.metadata["diarization_device"] == "cuda"
