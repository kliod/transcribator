import pytest

import transcribator.diarization as diarization_module
from transcribator.diarization import SpeakerDiarizer


def _runtime_info(**overrides):
    payload = {
        "pyannote_available": True,
        "pyannote_gpu_available": False,
        "pyannote_torch_installed": True,
        "pyannote_torchaudio_available": True,
        "pyannote_torch_cuda_available": False,
        "pyannote_torch_device_count": 0,
        "pyannote_runtime_guidance": "install gpu torch",
    }
    payload.update(overrides)
    return payload


def test_pyannote_cuda_device_resolves_when_gpu_is_available(monkeypatch):
    monkeypatch.setattr(
        diarization_module,
        "get_pyannote_runtime_info",
        lambda: _runtime_info(
            pyannote_gpu_available=True,
            pyannote_torch_cuda_available=True,
            pyannote_torch_device_count=1,
        ),
    )

    diarizer = SpeakerDiarizer(method="pyannote", device="cuda")

    assert diarizer._resolve_pyannote_device() == "cuda"
    assert diarizer.resolved_device == "cuda"


def test_pyannote_auto_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(
        diarization_module,
        "get_pyannote_runtime_info",
        lambda: _runtime_info(
            pyannote_gpu_available=True,
            pyannote_torch_cuda_available=True,
            pyannote_torch_device_count=2,
        ),
    )

    diarizer = SpeakerDiarizer(method="pyannote", device="auto")

    assert diarizer._resolve_pyannote_device() == "cuda"


def test_pyannote_auto_device_falls_back_to_cpu_when_gpu_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        diarization_module,
        "get_pyannote_runtime_info",
        lambda: _runtime_info(
            pyannote_gpu_available=False,
            pyannote_torch_cuda_available=False,
            pyannote_torch_device_count=0,
            pyannote_runtime_error="torch cpu-only",
        ),
    )

    diarizer = SpeakerDiarizer(method="pyannote", device="auto")

    assert diarizer._resolve_pyannote_device() == "cpu"
    assert diarizer.resolved_device == "cpu"


def test_pyannote_cuda_device_raises_with_guidance_when_gpu_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        diarization_module,
        "get_pyannote_runtime_info",
        lambda: _runtime_info(
            pyannote_gpu_available=False,
            pyannote_runtime_error="Current torch build does not expose CUDA.",
            pyannote_runtime_guidance="Install a CUDA-enabled torch build.",
        ),
    )

    diarizer = SpeakerDiarizer(method="pyannote", device="cuda")

    with pytest.raises(RuntimeError, match="Install a CUDA-enabled torch build."):
        diarizer._resolve_pyannote_device()


def test_simple_diarization_stays_on_cpu_even_if_cuda_was_requested(monkeypatch):
    monkeypatch.setattr(
        diarization_module,
        "get_pyannote_runtime_info",
        lambda: _runtime_info(),
    )

    diarizer = SpeakerDiarizer(method="simple", device="cuda")
    result = diarizer.diarize_simple(
        [
            {"start": 0.0, "end": 1.0, "text": "alpha"},
            {"start": 4.0, "end": 5.0, "text": "beta"},
        ]
    )

    assert diarizer.resolved_device == "cpu"
    assert all(item["speaker"] in {0, 1} for item in result)
