from transcribator.config import DEFAULT_CONFIG, merge_config_with_cli


def test_default_engine_is_faster_whisper():
    assert DEFAULT_CONFIG["engine"] == "faster-whisper"
    assert DEFAULT_CONFIG["device"] == "cpu"
    assert DEFAULT_CONFIG["diarization_device"] == "cpu"


def test_cli_can_override_engine_and_model():
    merged = merge_config_with_cli(
        DEFAULT_CONFIG.copy(),
        {
            "engine": "openai-whisper",
            "device": "cuda",
            "diarization_device": "auto",
            "model": "medium",
        },
    )

    assert merged["engine"] == "openai-whisper"
    assert merged["device"] == "cuda"
    assert merged["diarization_device"] == "auto"
    assert merged["model"] == "medium"
