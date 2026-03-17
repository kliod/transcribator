import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from transcribator.contracts import TranscriptionRequest, TranscriptionResult, TranscriptionSegment
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
    assert body["progress"] == 8
    assert body["active"] is True

    payload = _wait_for_completion(client, body["job_id"])
    assert payload["status"] == "done"
    assert payload["progress"] == 100
    assert payload["active"] is False
    assert payload["preview_text"] == "preview text"
    assert set(payload["downloads"]) == {"txt", "srt", "vtt"}

    download = client.get(payload["downloads"]["txt"])
    assert download.status_code == 200
    assert download.text == "txt content"


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
