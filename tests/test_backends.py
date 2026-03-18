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
    monkeypatch.setattr(
        backends,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": False,
            "faster_whisper_cuda_device_count": 0,
            "openai_whisper_cuda_available": False,
        },
    )
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
    assert init_calls[0][1]["device"] == "cpu"
    assert init_calls[0][1]["compute_type"] == "int8"
    assert result.text == "hello world"
    assert result.language == "ru"
    assert result.segments[0].to_dict() == {"start": 0.0, "end": 1.2, "text": "hello"}


def test_faster_whisper_backend_uses_cuda_when_requested(monkeypatch):
    init_calls = []

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            init_calls.append((model_name, kwargs))

        def transcribe(self, audio_path, **kwargs):
            return iter([]), _FakeInfo()

    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(
        backends,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": True,
            "faster_whisper_cuda_device_count": 1,
            "openai_whisper_cuda_available": False,
        },
    )
    backends.FasterWhisperTranscriber._model_cache.clear()

    request = TranscriptionRequest(
        input_path="input.mp4",
        engine="faster-whisper",
        model="small",
        device="cuda",
    )

    backend = backends.FasterWhisperTranscriber(request)

    assert backend.runtime_device == "cuda"
    assert backend.compute_type == "float16"
    assert init_calls[0][1]["device"] == "cuda"
    assert init_calls[0][1]["compute_type"] == "float16"


def test_faster_whisper_backend_surfaces_runtime_dll_error(monkeypatch):
    monkeypatch.setattr(
        backends,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": False,
            "faster_whisper_cuda_device_count": 1,
            "faster_whisper_cuda_runtime_ready": False,
            "faster_whisper_cuda_missing_dlls": ["cublas64_12.dll"],
            "ctranslate2_version": "4.7.1",
            "faster_whisper_cuda_runtime_guidance": (
                "Current CTranslate2 4.7.1 expects CUDA 12 and cuDNN 9 on Windows. "
                "An NVIDIA driver alone is not enough: the CUDA runtime DLLs must be installed and available in PATH. "
                "Missing DLLs: cublas64_12.dll. "
                "Install CUDA runtime/toolkit: https://developer.nvidia.com/cuda/toolkit "
                "and make sure the CUDA bin folders are in PATH. "
                "If you intentionally stay on CUDA 11, use ctranslate2==3.24.0 instead."
            ),
            "openai_whisper_cuda_available": False,
        },
    )
    backends.FasterWhisperTranscriber._model_cache.clear()

    request = TranscriptionRequest(
        input_path="input.mp4",
        engine="faster-whisper",
        model="small",
        device="cuda",
    )

    try:
        backends.FasterWhisperTranscriber(request)
    except RuntimeError as exc:
        assert "cublas64_12.dll" in str(exc)
        assert "ctranslate2==3.24.0" in str(exc)
    else:
        raise AssertionError("Expected CUDA runtime DLL error to be surfaced")


def test_windows_runtime_probe_adds_standard_cuda_dirs(monkeypatch):
    added_dirs = []
    monkeypatch.setattr(backends.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        backends,
        "_discover_windows_cuda_dirs",
        lambda: [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
            r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64",
        ],
    )
    monkeypatch.setattr(backends, "_WINDOWS_DLL_HANDLES", [])
    monkeypatch.setattr(backends, "_WINDOWS_RUNTIME_PREPARED", False)
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    monkeypatch.setattr(
        backends.os,
        "add_dll_directory",
        lambda directory: added_dirs.append(directory) or object(),
        raising=False,
    )
    monkeypatch.setattr(backends.ctypes, "WinDLL", lambda name: object())

    runtime_ready, missing_dlls = backends._probe_windows_cuda_runtime()

    assert runtime_ready is True
    assert missing_dlls == []
    assert added_dirs == [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9\x64",
    ]
    updated_path = backends.os.environ["PATH"]
    assert updated_path.startswith(added_dirs[0] + ";" + added_dirs[1] + ";")


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


def test_openai_whisper_backend_rejects_cuda_request():
    request = TranscriptionRequest(
        input_path="input.mp4",
        engine="openai-whisper",
        model="small",
        device="cuda",
    )

    try:
        backends.OpenAIWhisperTranscriber(request)
    except RuntimeError as exc:
        assert "CUDA acceleration is currently available only for faster-whisper" in str(exc)
    else:
        raise AssertionError("Expected CUDA request to be rejected for openai-whisper")
