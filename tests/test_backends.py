import sys
import types

import transcribator.backends as backends
from transcribator.contracts import TranscriptionRequest


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    language = "ru"
    language_probability = 0.98


def test_faster_whisper_backend_normalizes_result_and_reuses_model(monkeypatch):
    init_calls = []

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            init_calls.append((model_name, kwargs))

        def transcribe(self, audio_path, **kwargs):
            assert audio_path == "prepared.wav"
            return iter([
                _FakeSegment(0.0, 1.2, " hello "),
                _FakeSegment(1.2, 2.4, " world "),
            ]), _FakeInfo()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    backends.FasterWhisperTranscriber._model_cache.clear()

    request = TranscriptionRequest(
        input_path="input.mp4",
        engine="faster-whisper",
        model="small",
        language="ru",
    )

    first = backends.FasterWhisperTranscriber(request)
    second = backends.FasterWhisperTranscriber(request)
    result = first.transcribe("prepared.wav")

    assert len(init_calls) == 1
    assert first.model is second.model
    assert result.text == "hello world"
    assert result.language == "ru"
    assert result.segments[0].to_dict() == {"start": 0.0, "end": 1.2, "text": "hello"}


def test_openai_whisper_backend_matches_common_contract(monkeypatch):
    class FakeVideoTranscriber:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def transcribe(self, audio_path):
            assert audio_path == "prepared.wav"
            return {
                "text": "legacy result",
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "legacy"},
                    {"start": 1.0, "end": 2.0, "text": "result"},
                ],
            }

        def get_segments_with_timestamps(self, raw_result):
            return raw_result["segments"]

    monkeypatch.setattr(backends, "VideoTranscriber", FakeVideoTranscriber)

    request = TranscriptionRequest(
        input_path="input.mp4",
        engine="openai-whisper",
        model="small",
    )
    backend = backends.OpenAIWhisperTranscriber(request)
    result = backend.transcribe("prepared.wav")

    assert result.engine == "openai-whisper"
    assert result.text == "legacy result"
    assert [segment.text for segment in result.segments] == ["legacy", "result"]
