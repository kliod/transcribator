"""CLI interface for Transcribator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import click
from tqdm import tqdm

from .backends import DEFAULT_DEVICE, DEFAULT_ENGINE, DEVICE_CHOICES, ENGINE_CHOICES
from .config import create_default_config, load_config, merge_config_with_cli
from .contracts import TranscriptionRequest
from .service import TranscriptionService
from .utils import (
    WHISPER_MODEL_CHOICES,
    check_ffmpeg,
    clear_model_cache,
    ensure_output_directory,
    ensure_project_directories,
    get_cached_models,
    get_default_input_directory,
    get_default_output_directory,
    get_listed_model_keys,
    get_model_info,
    get_whisper_cache_dir,
    is_model_cached,
    normalize_model_name,
    validate_input_path,
)


@click.command()
@click.argument("input_path", type=click.Path(exists=True), required=False)
@click.option(
    "--engine",
    default=None,
    type=click.Choice(ENGINE_CHOICES, case_sensitive=False),
    help="Engine for local transcription (default: from config or faster-whisper).",
)
@click.option(
    "--model",
    "-m",
    default=None,
    type=click.Choice(WHISPER_MODEL_CHOICES, case_sensitive=False),
    help="Whisper model to use (default: from config or small).",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(DEVICE_CHOICES, case_sensitive=False),
    help="Execution device for faster-whisper: cpu, cuda, or auto.",
)
@click.option(
    "--diarization-device",
    default=None,
    type=click.Choice(DEVICE_CHOICES, case_sensitive=False),
    help="Execution device for pyannote diarization: cpu, cuda, or auto.",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Language code, for example ru or en. Leave empty for auto-detection.",
)
@click.option(
    "--format",
    "-f",
    "output_formats",
    default=None,
    type=click.Choice(["txt", "srt", "vtt", "all"], case_sensitive=False),
    help="Output format: txt, srt, vtt, or all.",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    default=None,
    type=click.Path(),
    help="Directory for result files.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress informational output.",
)
@click.option(
    "--list-models",
    is_flag=True,
    help="Show supported Whisper models and exit.",
)
@click.option(
    "--list-cached",
    is_flag=True,
    help="Show cached openai-whisper model files and exit.",
)
@click.option(
    "--clear-cache",
    "clear_cache_all",
    is_flag=True,
    help="Delete all cached openai-whisper model files.",
)
@click.option(
    "--clear-cache-model",
    "clear_cache_model",
    default=None,
    type=str,
    help="Delete one cached openai-whisper model (for example: small, large-v3, turbo).",
)
@click.option(
    "--high-quality",
    "-hq",
    is_flag=True,
    help="Use slower but more accurate decoding settings.",
)
@click.option(
    "--input",
    "input_dir",
    default=None,
    type=click.Path(),
    help="Directory with input video files (default: ./input).",
)
@click.option(
    "--no-timestamps",
    is_flag=True,
    help="Disable timestamps in TXT output.",
)
@click.option(
    "--clean-txt",
    is_flag=True,
    help="Create an additional _clean.txt file without timestamps.",
)
@click.option(
    "--diarize",
    type=click.Choice(["none", "simple", "pyannote", "auto"], case_sensitive=False),
    default=None,
    help="Speaker diarization mode.",
)
@click.option(
    "--hf-token",
    default=None,
    type=str,
    help="Hugging Face token for pyannote diarization.",
)
@click.option(
    "--beam-size",
    default=None,
    type=int,
    help="Beam size override for the selected engine.",
)
@click.option(
    "--best-of",
    default=None,
    type=int,
    help="Best-of override for the selected engine.",
)
@click.option(
    "--preprocess-audio",
    is_flag=True,
    help="Apply additional audio enhancement before transcription.",
)
@click.option(
    "--min-speakers",
    default=None,
    type=int,
    help="Minimum speaker count for pyannote.",
)
@click.option(
    "--max-speakers",
    default=None,
    type=int,
    help="Maximum speaker count for pyannote.",
)
@click.option(
    "--diarization-threshold",
    default=None,
    type=float,
    help="Clustering threshold for pyannote.",
)
@click.option(
    "--pause-threshold",
    default=None,
    type=float,
    help="Pause threshold in seconds for simple diarization.",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=False),
    help="Optional config path (JSON).",
)
@click.option(
    "--create-config",
    is_flag=True,
    help="Create a default config file and exit.",
)
def main(
    input_path: str,
    engine: str,
    model: str,
    device: str,
    diarization_device: str,
    language: str,
    output_formats: str,
    output_dir: str,
    quiet: bool,
    list_models: bool,
    high_quality: bool,
    input_dir: str,
    list_cached: bool,
    clear_cache_all: bool,
    clear_cache_model: str,
    no_timestamps: bool,
    clean_txt: bool,
    diarize: str,
    hf_token: str,
    beam_size: int,
    best_of: int,
    preprocess_audio: bool,
    min_speakers: int,
    max_speakers: int,
    diarization_threshold: float,
    pause_threshold: float,
    config: str,
    create_config: bool,
) -> None:
    if create_config:
        try:
            config_file = create_default_config(config)
            click.echo(f"Configuration file created: {config_file}")
            return
        except Exception as exc:
            click.echo(f"Could not create config file: {exc}", err=True)
            return

    try:
        file_config = load_config(config)
    except ValueError as exc:
        click.echo(f"Warning: {exc}", err=True)
        click.echo("Using default configuration values.", err=True)
        file_config = {}

    cli_params: Dict[str, object] = {}
    for key, value in {
        "engine": engine,
        "model": model,
        "device": device,
        "diarization_device": diarization_device,
        "language": language,
        "output_formats": output_formats,
        "output_dir": output_dir,
        "input_dir": input_dir,
        "diarize": diarize,
        "hf_token": hf_token,
        "beam_size": beam_size,
        "best_of": best_of,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "diarization_threshold": diarization_threshold,
        "pause_threshold": pause_threshold,
    }.items():
        if value is not None:
            cli_params[key] = value

    for key, value in {
        "quiet": quiet,
        "high_quality": high_quality,
        "no_timestamps": no_timestamps,
        "clean_txt": clean_txt,
        "preprocess_audio": preprocess_audio,
    }.items():
        if value:
            cli_params[key] = True

    cfg = merge_config_with_cli(file_config, cli_params)

    engine = str(cfg.get("engine") or DEFAULT_ENGINE).lower()
    model = str(cfg["model"])
    device = str(cfg.get("device") or DEFAULT_DEVICE).lower()
    diarization_device = str(cfg.get("diarization_device") or DEFAULT_DEVICE).lower()
    language = cfg["language"]
    output_formats = cfg["output_formats"]
    output_dir = cfg["output_dir"]
    quiet = bool(cfg["quiet"])
    high_quality = bool(cfg["high_quality"])
    input_dir = cfg["input_dir"]
    no_timestamps = bool(cfg["no_timestamps"])
    clean_txt = bool(cfg["clean_txt"])
    diarize = cfg["diarize"]
    hf_token = cfg["hf_token"]
    beam_size = cfg["beam_size"]
    best_of = cfg["best_of"]
    preprocess_audio = bool(cfg["preprocess_audio"])
    min_speakers = cfg["min_speakers"]
    max_speakers = cfg["max_speakers"]
    diarization_threshold = cfg["diarization_threshold"]
    pause_threshold = cfg["pause_threshold"]

    if clear_cache_all or clear_cache_model is not None:
        _handle_cache_commands(clear_cache_all, clear_cache_model, quiet)
        return

    if list_cached:
        _list_cached_models()
        return

    if list_models:
        _list_models(model)
        return

    ffmpeg_available, ffmpeg_error = check_ffmpeg()
    if not ffmpeg_available:
        click.echo(f"Error: {ffmpeg_error}", err=True)
        return

    if input_path is None:
        if input_dir:
            input_path = input_dir
        else:
            ensure_project_directories()
            input_path = get_default_input_directory()
            if not quiet:
                click.echo(f"Using default input directory: {input_path}")

    if input_path is None:
        click.echo("Error: input path is required.", err=True)
        return

    formats = ["txt", "srt", "vtt"] if output_formats == "all" else [output_formats]

    if output_dir is None:
        ensure_project_directories()
        output_dir = get_default_output_directory()
        if not quiet:
            click.echo(f"Results will be saved to: {output_dir}")

    is_valid, error_message, file_list = validate_input_path(input_path)
    if not is_valid:
        click.echo(f"Error: {error_message}", err=True)
        return

    if not quiet:
        model_info = get_model_info().get(normalize_model_name(model) or model, {})
        click.echo(f"Files found: {len(file_list)}")
        click.echo(f"Engine: {engine}")
        click.echo(f"Model: {model} ({model_info.get('description', '')})")
        click.echo(f"Device: {device}")
        click.echo(f"Diarization device: {diarization_device}")
        click.echo(f"Quality: {'high' if high_quality else 'balanced'}")
        click.echo(f"Language: {language or 'auto'}")
        click.echo(f"Speaker diarization: {diarize}")
        click.echo(f"Output formats: {', '.join(formats)}")
        click.echo("")

    service = TranscriptionService()
    successful = 0
    failed = 0

    for video_file in tqdm(file_list, disable=quiet, desc="Processing files"):
        try:
            per_file_output_dir = ensure_output_directory(output_dir, video_file)
            request = TranscriptionRequest(
                input_path=video_file,
                model=model,
                engine=engine,
                device=device,
                diarization_device=diarization_device,
                language=language,
                output_formats=formats,
                output_dir=per_file_output_dir,
                quiet=quiet,
                high_quality=high_quality,
                no_timestamps=no_timestamps,
                clean_txt=clean_txt,
                diarize=diarize,
                hf_token=hf_token,
                beam_size=beam_size,
                best_of=best_of,
                preprocess_audio=preprocess_audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                diarization_threshold=diarization_threshold,
                pause_threshold=pause_threshold,
            )

            status_callback = None if quiet else _make_status_callback(video_file)
            result = service.transcribe_file(request, status_callback=status_callback)

            if not quiet and result.artifacts:
                click.echo(f"Completed: {video_file}")
                for artifact_name, artifact_path in result.artifacts.items():
                    click.echo(f"  {artifact_name}: {artifact_path}")

            successful += 1
        except Exception as exc:
            failed += 1
            if not quiet:
                click.echo(f"\nError while processing {video_file}: {exc}", err=True)

    if not quiet:
        click.echo("")
        click.echo(f"Finished. Successful: {successful}, Failed: {failed}")


def _make_status_callback(video_file: str):
    label = Path(video_file).name
    last_status = {"value": None}

    def callback(status: str, message: str) -> None:
        if status != last_status["value"]:
            click.echo(f"[{label}] {status}: {message}")
            last_status["value"] = status

    return callback


def _handle_cache_commands(clear_cache_all: bool, clear_cache_model: str, quiet: bool) -> None:
    cached_models = get_cached_models()

    if clear_cache_all:
        if not cached_models:
            click.echo("Cache is empty.")
            return

        click.echo(f"Cached models found: {len(cached_models)}")
        total_size = sum(item["size_bytes"] for item in cached_models.values())
        click.echo(f"Total size: {round(total_size / (1024 * 1024), 2)} MB")
        if not quiet:
            confirm = click.prompt("Delete all cached models? (yes/no)", default="no")
            if confirm.lower() not in {"yes", "y"}:
                click.echo("Cancelled.")
                return

        deleted_count, freed_mb = clear_model_cache()
        click.echo(f"Deleted models: {deleted_count}")
        click.echo(f"Freed space: {freed_mb} MB")
        return

    normalized_name = normalize_model_name(clear_cache_model)
    if normalized_name not in cached_models:
        click.echo(f"Model '{clear_cache_model}' was not found in cache.", err=True)
        if cached_models:
            click.echo("Available cached models:")
            for model_name in sorted(cached_models):
                click.echo(f"  - {model_name}")
        return

    info = cached_models[normalized_name]
    click.echo(f"Model: {clear_cache_model}")
    click.echo(f"Size: {info['size_mb']} MB")
    click.echo(f"File: {info['file']}")
    if not quiet:
        confirm = click.prompt("Delete this cached model? (yes/no)", default="no")
        if confirm.lower() not in {"yes", "y"}:
            click.echo("Cancelled.")
            return

    deleted_count, freed_mb = clear_model_cache(clear_cache_model)
    if deleted_count > 0:
        click.echo(f"Deleted '{clear_cache_model}'.")
        click.echo(f"Freed space: {freed_mb} MB")
    else:
        click.echo(f"Could not delete '{clear_cache_model}'.", err=True)


def _list_cached_models() -> None:
    cache_dir = get_whisper_cache_dir()
    cached_models = get_cached_models()

    click.echo(f"OpenAI Whisper cache: {cache_dir}\n")
    if not cached_models:
        click.echo("No cached openai-whisper models found.")
        return

    for model_name, info in sorted(cached_models.items()):
        click.echo(f"  {model_name:15} - {info['size_mb']} MB")
        click.echo(f"           File: {info['file']}")

    total_size = sum(item["size_bytes"] for item in cached_models.values())
    click.echo(f"\nTotal models: {len(cached_models)}")
    click.echo(f"Total size: {round(total_size / (1024 * 1024), 2)} MB")


def _list_models(selected_model: str) -> None:
    click.echo("Supported Whisper models:\n")
    models_info = get_model_info()
    cached_models = get_cached_models()
    selected_key = selected_model if selected_model in get_listed_model_keys() else normalize_model_name(selected_model)

    for model_key in get_listed_model_keys():
        info = models_info[model_key]
        cached_status = " [CACHED]" if is_model_cached(model_key, cached_models) else ""
        click.echo(f"  {info['name'].upper():14} - {info['description']}{cached_status}")
        click.echo(
            f"                 Size: {info['size']:10} | "
            f"Speed: {info['speed']:15} | Accuracy: {info['accuracy']}"
        )
        if model_key == selected_key:
            click.echo("                 (current default model)")
        click.echo("")


if __name__ == "__main__":
    main()
