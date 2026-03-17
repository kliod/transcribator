from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from .audio_processor import preprocess_audio, validate_audio


LOUDNESS_FILTER = "loudnorm=I=-16:TP=-1.5:LRA=11"


def _run_ffmpeg(input_path: str, output_path: str) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is required to prepare audio.")

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        LOUDNESS_FILTER,
        output_path,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed while preparing audio: {error}")


def _write_float_wav(output_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
    wavfile.write(output_path, sample_rate, clipped)


def prepare_audio_file(input_path: str, working_directory: str, enhance: bool = False) -> str:
    working_path = Path(working_directory)
    working_path.mkdir(parents=True, exist_ok=True)

    standardized_path = working_path / "prepared.wav"
    _run_ffmpeg(input_path, str(standardized_path))

    if not enhance:
        return str(standardized_path)

    audio, sample_rate = librosa.load(str(standardized_path), sr=16000, mono=True)
    if not validate_audio(audio, sample_rate):
        return str(standardized_path)

    enhanced_audio, enhanced_sample_rate = preprocess_audio(
        audio,
        sample_rate=sample_rate,
        normalize=True,
        denoise=True,
        enhance_quiet=False,
    )

    enhanced_path = working_path / "prepared_enhanced.wav"
    _write_float_wav(enhanced_path, enhanced_audio, enhanced_sample_rate)
    return str(enhanced_path)
