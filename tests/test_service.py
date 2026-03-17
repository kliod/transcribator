from pathlib import Path

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


def test_service_falls_back_from_pyannote_to_simple(monkeypatch, tmp_path):
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
            if self.method == "pyannote":
                raise RuntimeError("pyannote failed")
            return [
                dict(segment, speaker=index)
                for index, segment in enumerate(segments)
            ]

    def fake_export_transcription(segments, output_base_path, formats, **kwargs):
        for fmt in formats:
            Path(f"{output_base_path}.{fmt}").write_text(fmt, encoding="utf-8")

    def fake_export_txt(segments, output_path, **kwargs):
        Path(output_path).write_text("clean", encoding="utf-8")

    monkeypatch.setattr(service_module, "prepare_audio_file", fake_prepare_audio_file)
    monkeypatch.setattr(service_module, "build_transcriber", fake_build_transcriber)
    monkeypatch.setattr(service_module, "SpeakerDiarizer", FakeDiarizer)
    monkeypatch.setattr(service_module, "export_transcription", fake_export_transcription)
    monkeypatch.setattr(service_module, "export_txt", fake_export_txt)

    request = TranscriptionRequest(
        input_path=str(tmp_path / "input.mp4"),
        output_dir=str(tmp_path / "out"),
        output_formats=["txt", "srt", "vtt"],
        diarize="pyannote",
        clean_txt=True,
    )

    statuses = []
    service = service_module.TranscriptionService()
    result = service.transcribe_file(
        request,
        status_callback=lambda status, message: statuses.append(status),
        output_name="clip.mp4",
    )

    assert statuses == ["running", "diarizing", "exporting"]
    assert diarizer_methods == ["pyannote", "simple"]
    assert sorted(result.artifacts) == ["clean_txt", "srt", "txt", "vtt"]
    assert "[Speaker 1]" in result.preview_text
    assert Path(result.artifacts["txt"]).exists()


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
    )

    service = service_module.TranscriptionService()
    result = service.transcribe_file(request)

    assert diarizer_called["value"] is False
    assert result.preview_text == "alpha\nbeta"
