from transcribator.config import DEFAULT_CONFIG, merge_config_with_cli


def test_default_engine_is_faster_whisper():
    assert DEFAULT_CONFIG["engine"] == "faster-whisper"


def test_cli_can_override_engine_and_model():
    merged = merge_config_with_cli(
        DEFAULT_CONFIG.copy(),
        {
            "engine": "openai-whisper",
            "model": "medium",
        },
    )

    assert merged["engine"] == "openai-whisper"
    assert merged["model"] == "medium"
