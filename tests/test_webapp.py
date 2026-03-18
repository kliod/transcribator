import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from transcribator.contracts import TranscriptionRequest, TranscriptionResult, TranscriptionSegment
import transcribator.webapp as webapp_module
from transcribator.webapp import create_app


class _FakeService:
    last_request = None

    def transcribe_file(self, request: TranscriptionRequest, status_callback=None, output_name=None):
        type(self).last_request = request
        if status_callback:
            status_callback("running", "Preparing audio")
            status_callback("exporting", "Writing output files")

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(output_name or request.input_path).stem
        artifacts = {}
        for fmt in request.output_formats:
            artifact_path = output_dir / f"{base_name}.{fmt}"
            artifact_path.write_text(f"{fmt} content", encoding="utf-8")
            artifacts[fmt] = str(artifact_path)

        return TranscriptionResult(
            text="preview text",
            segments=[TranscriptionSegment(start=0.0, end=1.0, text="preview text")],
            language="en",
            engine=request.engine,
            model=request.model,
            artifacts=artifacts,
            preview_text="preview text",
        )


def _wait_for_completion(client: TestClient, job_id: str):
    for _ in range(100):
        response = client.get(f"/jobs/{job_id}")
        payload = response.json()
        if payload["status"] in {"done", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("Job did not finish in time.")


def test_web_app_upload_poll_and_download(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")

    client = TestClient(create_app(str(config_path), service=_FakeService()))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "language": "en",
            "quality": "balanced",
            "speakers": "off",
            "engine": "faster-whisper",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["status_label"] == "Queued"
    assert body["progress"] == 8
    assert body["active"] is True

    payload = _wait_for_completion(client, body["job_id"])
    assert payload["status"] == "done"
    assert payload["progress"] == 100
    assert payload["active"] is False
    assert payload["preview_text"] == "preview text"
    assert set(payload["downloads"]) == {"txt", "srt", "vtt"}
    assert _FakeService.last_request is not None
    assert _FakeService.last_request.no_timestamps is True
    assert _FakeService.last_request.device == "cpu"
    assert _FakeService.last_request.diarization_device == "cpu"

    download = client.get(payload["downloads"]["txt"])
    assert download.status_code == 200
    assert download.text == "txt content"
    assert "attachment" in download.headers["content-disposition"].lower()
    assert "sample.txt" in download.headers["content-disposition"]


def test_web_app_rejects_unsupported_upload(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")

    client = TestClient(create_app(str(config_path), service=_FakeService()))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "off",
            "engine": "faster-whisper",
        },
        files={"file": ("sample.txt", b"plain-text", "text/plain")},
    )

    assert response.status_code == 400


def test_web_app_renders_default_english_and_optional_russian_locale(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")

    client = TestClient(create_app(str(config_path), service=_FakeService()))

    english = client.get("/")
    assert english.status_code == 200
    assert "Interface language" in english.text
    assert "Start transcription" in english.text

    russian = client.get("/?ui_lang=ru")
    assert russian.status_code == 200
    assert "Язык интерфейса" in russian.text
    assert "Запустить транскрибацию" in russian.text


def test_web_app_requires_hf_token_for_pyannote(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")

    client = TestClient(create_app(str(config_path), service=_FakeService()))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "pyannote",
            "engine": "faster-whisper",
            "hf_token": "",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 400
    assert "Hugging Face token" in response.json()["detail"]


def test_web_app_passes_hf_token_to_request(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")

    fake_service = _FakeService()
    client = TestClient(create_app(str(config_path), service=fake_service))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "pyannote",
            "engine": "faster-whisper",
            "hf_token": "hf_test_token",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    payload = _wait_for_completion(client, response.json()["job_id"])
    assert payload["status"] == "done"
    assert _FakeService.last_request is not None
    assert _FakeService.last_request.hf_token == "hf_test_token"


def test_web_app_can_keep_timestamps_when_requested(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")

    fake_service = _FakeService()
    client = TestClient(create_app(str(config_path), service=fake_service))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "off",
            "engine": "faster-whisper",
            "keep_timestamps": "true",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    payload = _wait_for_completion(client, response.json()["job_id"])
    assert payload["status"] == "done"
    assert _FakeService.last_request is not None
    assert _FakeService.last_request.no_timestamps is False


def test_web_app_passes_cuda_device_to_request(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")
    monkeypatch.setattr(
        webapp_module,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": True,
            "faster_whisper_cuda_device_count": 1,
            "openai_whisper_cuda_available": False,
        },
    )

    fake_service = _FakeService()
    client = TestClient(create_app(str(config_path), service=fake_service))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "off",
            "engine": "faster-whisper",
            "device": "cuda",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    payload = _wait_for_completion(client, response.json()["job_id"])
    assert payload["status"] == "done"
    assert _FakeService.last_request is not None
    assert _FakeService.last_request.device == "cuda"


def test_web_app_passes_pyannote_diarization_device_to_request(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")
    monkeypatch.setattr(
        webapp_module,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": False,
            "faster_whisper_cuda_device_count": 0,
            "openai_whisper_cuda_available": False,
        },
    )
    monkeypatch.setattr(
        webapp_module,
        "get_pyannote_runtime_info",
        lambda: {
            "pyannote_available": True,
            "pyannote_gpu_available": True,
            "pyannote_torch_installed": True,
            "pyannote_torchaudio_available": True,
            "pyannote_torch_cuda_available": True,
            "pyannote_torch_device_count": 1,
            "pyannote_runtime_guidance": "GPU diarization is ready.",
        },
    )

    fake_service = _FakeService()
    client = TestClient(create_app(str(config_path), service=fake_service))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "pyannote",
            "engine": "faster-whisper",
            "hf_token": "hf_test_token",
            "diarization_device": "cuda",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    payload = _wait_for_completion(client, response.json()["job_id"])
    assert payload["status"] == "done"
    assert _FakeService.last_request is not None
    assert _FakeService.last_request.diarization_device == "cuda"


def test_web_app_requires_confirmation_when_pyannote_cuda_is_unavailable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")
    monkeypatch.setattr(
        webapp_module,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": False,
            "faster_whisper_cuda_device_count": 0,
            "openai_whisper_cuda_available": False,
        },
    )
    monkeypatch.setattr(
        webapp_module,
        "get_pyannote_runtime_info",
        lambda: {
            "pyannote_available": True,
            "pyannote_gpu_available": False,
            "pyannote_torch_installed": True,
            "pyannote_torchaudio_available": True,
            "pyannote_torch_cuda_available": False,
            "pyannote_torch_device_count": 0,
            "pyannote_runtime_error": "Current torch build does not expose CUDA.",
            "pyannote_runtime_guidance": "Install a CUDA-enabled torch/torchaudio build.",
        },
    )

    client = TestClient(create_app(str(config_path), service=_FakeService()))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "pyannote",
            "engine": "faster-whisper",
            "hf_token": "hf_test_token",
            "diarization_device": "cuda",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["confirmation_required"] is True
    assert detail["suggested_diarization_device"] == "cpu"
    assert "does not expose CUDA" in detail["message"]
    assert "proceed_label" in detail


def test_web_app_can_force_start_when_pyannote_cuda_is_unavailable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")
    monkeypatch.setattr(
        webapp_module,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": False,
            "faster_whisper_cuda_device_count": 0,
            "openai_whisper_cuda_available": False,
        },
    )
    monkeypatch.setattr(
        webapp_module,
        "get_pyannote_runtime_info",
        lambda: {
            "pyannote_available": True,
            "pyannote_gpu_available": False,
            "pyannote_torch_installed": True,
            "pyannote_torchaudio_available": True,
            "pyannote_torch_cuda_available": False,
            "pyannote_torch_device_count": 0,
            "pyannote_runtime_error": "Current torch build does not expose CUDA.",
            "pyannote_runtime_guidance": "Install a CUDA-enabled torch/torchaudio build.",
        },
    )

    fake_service = _FakeService()
    client = TestClient(create_app(str(config_path), service=fake_service))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "pyannote",
            "engine": "faster-whisper",
            "hf_token": "hf_test_token",
            "diarization_device": "cuda",
            "allow_unavailable_diarization_cuda": "true",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 200
    payload = _wait_for_completion(client, response.json()["job_id"])
    assert payload["status"] == "done"
    assert _FakeService.last_request is not None
    assert _FakeService.last_request.diarization_device == "cuda"


def test_web_app_rejects_cuda_when_unavailable(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "results")}), encoding="utf-8")
    monkeypatch.setattr(
        webapp_module,
        "get_runtime_acceleration_info",
        lambda: {
            "faster_whisper_cuda_available": False,
            "faster_whisper_cuda_device_count": 0,
            "faster_whisper_cuda_runtime_ready": False,
            "faster_whisper_cuda_missing_dlls": ["cublas64_12.dll"],
            "faster_whisper_cuda_runtime_error": (
                "CUDA runtime DLLs are missing or cannot be loaded: cublas64_12.dll"
            ),
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

    client = TestClient(create_app(str(config_path), service=_FakeService()))
    response = client.post(
        "/jobs",
        data={
            "model": "small",
            "quality": "balanced",
            "speakers": "off",
            "engine": "faster-whisper",
            "device": "cuda",
        },
        files={"file": ("sample.mp4", b"video-data", "video/mp4")},
    )

    assert response.status_code == 400
    assert "cublas64_12.dll" in response.json()["detail"]
    assert "https://developer.nvidia.com/cuda/toolkit" in response.json()["detail"]
    assert "ctranslate2==3.24.0" in response.json()["detail"]
