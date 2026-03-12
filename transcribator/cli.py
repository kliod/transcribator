"""CLI интерфейс для транскрибации видео."""

import click
from pathlib import Path
from typing import List
from tqdm import tqdm
import os

from .transcriber import VideoTranscriber
from .exporter import export_transcription, export_txt
from .diarization import SpeakerDiarizer
from .config import load_config, merge_config_with_cli, get_config_path, create_default_config
from .utils import (
    validate_input_path, 
    ensure_output_directory, 
    get_output_filename,
    check_ffmpeg,
    get_model_info,
    ensure_project_directories,
    get_default_input_directory,
    get_default_output_directory,
    get_cached_models,
    get_whisper_cache_dir,
    clear_model_cache
)


@click.command()
@click.argument('input_path', type=click.Path(exists=True), required=False)
@click.option(
    '--model',
    '-m',
    default=None,
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'turbo'], case_sensitive=False),
    help='Модель Whisper для использования (по умолчанию из конфига или small)'
)
@click.option(
    '--language',
    '-l',
    default=None,
    help='Язык для транскрибации (например, ru, en). Если не указан, язык определяется автоматически'
)
@click.option(
    '--format',
    '-f',
    'output_formats',
    default=None,
    type=click.Choice(['txt', 'srt', 'vtt', 'all'], case_sensitive=False),
    help='Формат вывода (txt, srt, vtt, all). По умолчанию из конфига или all'
)
@click.option(
    '--output',
    '-o',
    'output_dir',
    default=None,
    type=click.Path(),
    help='Выходная директория для сохранения результатов. По умолчанию: рядом с исходным файлом'
)
@click.option(
    '--quiet',
    '-q',
    is_flag=True,
    help='Подавить вывод прогресса и информационных сообщений'
)
@click.option(
    '--list-models',
    is_flag=True,
    help='Показать информацию о доступных моделях и выйти'
)
@click.option(
    '--list-cached',
    is_flag=True,
    help='Показать закэшированные модели и выйти'
)
@click.option(
    '--clear-cache',
    'clear_cache_all',
    is_flag=True,
    help='Удалить все закэшированные модели'
)
@click.option(
    '--clear-cache-model',
    'clear_cache_model',
    default=None,
    type=str,
    help='Удалить конкретную модель из кэша (укажите имя модели: tiny, base, small, medium, large, turbo)'
)
@click.option(
    '--high-quality',
    '-hq',
    is_flag=True,
    help='Использовать максимальные параметры качества (медленнее, но точнее)'
)
@click.option(
    '--input',
    'input_dir',
    default=None,
    type=click.Path(),
    help='Директория с входными видео файлами (по умолчанию: ./input/)'
)
@click.option(
    '--no-timestamps',
    is_flag=True,
    help='Отключить тайминги в TXT формате'
)
@click.option(
    '--clean-txt',
    is_flag=True,
    help='Создать дополнительный clean.txt без таймингов'
)
@click.option(
    '--diarize',
    type=click.Choice(['none', 'simple', 'pyannote', 'auto'], case_sensitive=False),
    default=None,
    help='Метод разбивки по спикерам (none, simple, pyannote, auto). По умолчанию из конфига или none'
)
@click.option(
    '--hf-token',
    default=None,
    type=str,
    help='Hugging Face токен для pyannote (опционально, требуется только для некоторых моделей)'
)
@click.option(
    '--beam-size',
    default=None,
    type=int,
    help='Размер луча для Whisper (1-10, по умолчанию 5 в режиме высокого качества)'
)
@click.option(
    '--best-of',
    default=None,
    type=int,
    help='Количество попыток для выбора лучшего результата Whisper (1-10, по умолчанию 5 в режиме высокого качества)'
)
@click.option(
    '--preprocess-audio',
    is_flag=True,
    help='Включить предобработку аудио (нормализация, шумоподавление) для улучшения качества'
)
@click.option(
    '--min-speakers',
    default=None,
    type=int,
    help='Минимальное количество спикеров для pyannote (опционально)'
)
@click.option(
    '--max-speakers',
    default=None,
    type=int,
    help='Максимальное количество спикеров для pyannote (опционально)'
)
@click.option(
    '--diarization-threshold',
    default=None,
    type=float,
    help='Порог кластеризации для pyannote (0.0-1.0, по умолчанию 0.7)'
)
@click.option(
    '--pause-threshold',
    default=None,
    type=float,
    help='Порог паузы в секундах для простого метода диаризации (по умолчанию 2.0)'
)
@click.option(
    '--config',
    default=None,
    type=click.Path(exists=False),
    help='Путь к файлу конфигурации (JSON). Если не указан, используется transcribator.json в текущей директории или ~/.transcribator.json'
)
@click.option(
    '--create-config',
    is_flag=True,
    help='Создать файл конфигурации с дефолтными значениями и выйти'
)
def main(input_path: str, model: str, language: str, output_formats: str, output_dir: str, quiet: bool, list_models: bool, high_quality: bool, input_dir: str, list_cached: bool, clear_cache_all: bool, clear_cache_model: str, no_timestamps: bool, clean_txt: bool, diarize: str, hf_token: str, beam_size: int, best_of: int, preprocess_audio: bool, min_speakers: int, max_speakers: int, diarization_threshold: float, pause_threshold: float, config: str, create_config: bool):
    """
    Транскрибирует видео файл или все видео файлы в директории.
    
    INPUT_PATH: Путь к видео файлу или директории с видео файлами
    """
    # Создание конфигурационного файла
    if create_config:
        try:
            config_file = create_default_config(config)
            click.echo(f"Конфигурационный файл создан: {config_file}")
            click.echo("Отредактируйте его для настройки параметров по умолчанию.")
            return
        except Exception as e:
            click.echo(f"Ошибка при создании конфигурационного файла: {e}", err=True)
            return
    
    # Загрузка конфигурации
    try:
        file_config = load_config(config)
    except ValueError as e:
        click.echo(f"Предупреждение: {e}", err=True)
        click.echo("Используются значения по умолчанию.")
        file_config = {}
    
    # Объединение конфигурации с параметрами CLI (CLI имеет приоритет)
    # Для флагов (is_flag=True) None означает что флаг не был установлен
    cli_params = {}
    
    # Параметры со значениями
    if model is not None:
        cli_params['model'] = model
    if language is not None:
        cli_params['language'] = language
    if output_formats is not None:
        cli_params['output_formats'] = output_formats
    if output_dir is not None:
        cli_params['output_dir'] = output_dir
    if input_dir is not None:
        cli_params['input_dir'] = input_dir
    if diarize is not None:
        cli_params['diarize'] = diarize
    if hf_token is not None:
        cli_params['hf_token'] = hf_token
    if beam_size is not None:
        cli_params['beam_size'] = beam_size
    if best_of is not None:
        cli_params['best_of'] = best_of
    if min_speakers is not None:
        cli_params['min_speakers'] = min_speakers
    if max_speakers is not None:
        cli_params['max_speakers'] = max_speakers
    if diarization_threshold is not None:
        cli_params['diarization_threshold'] = diarization_threshold
    if pause_threshold is not None:
        cli_params['pause_threshold'] = pause_threshold
    
    # Флаги (если установлены, переопределяют конфиг)
    if quiet:
        cli_params['quiet'] = True
    if high_quality:
        cli_params['high_quality'] = True
    if no_timestamps:
        cli_params['no_timestamps'] = True
    if clean_txt:
        cli_params['clean_txt'] = True
    if preprocess_audio:
        cli_params['preprocess_audio'] = True
    
    # Объединяем конфигурацию (файл -> CLI)
    cfg = merge_config_with_cli(file_config, cli_params)
    
    # Используем значения из объединенной конфигурации
    model = cfg['model']
    language = cfg['language']
    output_formats = cfg['output_formats']
    output_dir = cfg['output_dir']
    quiet = cfg['quiet']
    high_quality = cfg['high_quality']
    input_dir = cfg['input_dir']
    no_timestamps = cfg['no_timestamps']
    clean_txt = cfg['clean_txt']
    diarize = cfg['diarize']
    hf_token = cfg['hf_token']
    beam_size = cfg['beam_size']
    best_of = cfg['best_of']
    preprocess_audio = cfg['preprocess_audio']
    min_speakers = cfg['min_speakers']
    max_speakers = cfg['max_speakers']
    diarization_threshold = cfg['diarization_threshold']
    pause_threshold = cfg['pause_threshold']
    
    # Очистка кэша моделей
    if clear_cache_all or clear_cache_model is not None:
        cache_dir = get_whisper_cache_dir()
        cached_models = get_cached_models()
        
        if clear_cache_all:
            # Удалить все модели
            if not cached_models:
                click.echo("Кэш пуст. Нет моделей для удаления.")
                return
            
            click.echo(f"Найдено моделей в кэше: {len(cached_models)}")
            total_size = sum(m['size_bytes'] for m in cached_models.values())
            click.echo(f"Общий размер: {round(total_size / (1024 * 1024), 2)} MB")
            click.echo("\nВНИМАНИЕ: Будут удалены ВСЕ закэшированные модели!")
            
            if not quiet:
                confirm = click.prompt("Продолжить? (yes/no)", default="no")
                if confirm.lower() not in ['yes', 'y', 'да']:
                    click.echo("Отменено.")
                    return
            
            deleted_count, freed_mb = clear_model_cache()
            click.echo(f"\nУдалено моделей: {deleted_count}")
            click.echo(f"Освобождено места: {freed_mb} MB")
        elif clear_cache_model:
            # Удалить конкретную модель
            if clear_cache_model not in cached_models:
                click.echo(f"Модель '{clear_cache_model}' не найдена в кэше.", err=True)
                if cached_models:
                    click.echo("\nДоступные модели в кэше:")
                    for model_name in sorted(cached_models.keys()):
                        click.echo(f"  - {model_name}")
                return
            
            model_info = cached_models[clear_cache_model]
            click.echo(f"Модель: {clear_cache_model}")
            click.echo(f"Размер: {model_info['size_mb']} MB")
            click.echo(f"Файл: {model_info['file']}")
            
            if not quiet:
                confirm = click.prompt("\nУдалить эту модель? (yes/no)", default="no")
                if confirm.lower() not in ['yes', 'y', 'да']:
                    click.echo("Отменено.")
                    return
            
            deleted_count, freed_mb = clear_model_cache(clear_cache_model)
            if deleted_count > 0:
                click.echo(f"\nМодель '{clear_cache_model}' успешно удалена.")
                click.echo(f"Освобождено места: {freed_mb} MB")
            else:
                click.echo(f"\nНе удалось удалить модель '{clear_cache_model}'.", err=True)
        return
    
    # Показать информацию о закэшированных моделях и выйти
    if list_cached:
        cache_dir = get_whisper_cache_dir()
        cached_models = get_cached_models()
        
        click.echo(f"Кэш моделей Whisper: {cache_dir}\n")
        
        if cached_models:
            click.echo("Закэшированные модели:\n")
            for model_name, info in sorted(cached_models.items()):
                click.echo(f"  {model_name:15} - {info['size_mb']} MB")
                click.echo(f"           Файл: {info['file']}")
            click.echo(f"\nВсего моделей: {len(cached_models)}")
            total_size = sum(m['size_bytes'] for m in cached_models.values())
            click.echo(f"Общий размер: {round(total_size / (1024 * 1024), 2)} MB")
        else:
            click.echo("Нет закэшированных моделей.")
            click.echo("Модели будут загружены автоматически при первом использовании.")
        return
    
    # Показать информацию о моделях и выйти
    if list_models:
        click.echo("Доступные модели Whisper:\n")
        models_info = get_model_info()
        cached_models = get_cached_models()
        
        for model_key in ['tiny', 'base', 'small', 'medium', 'large', 'turbo']:
            info = models_info[model_key]
            cached_status = " [ЗАКЭШИРОВАНА]" if model_key in cached_models else ""
            click.echo(f"  {info['name'].upper():8} - {info['description']}{cached_status}")
            click.echo(f"           Размер: {info['size']:10} | Скорость: {info['speed']:15} | Точность: {info['accuracy']}")
            if model_key == model:
                click.echo(f"           (текущая модель по умолчанию)")
            click.echo("")
        return
    
    # Проверка FFmpeg перед началом работы
    ffmpeg_available, ffmpeg_error = check_ffmpeg()
    if not ffmpeg_available:
        click.echo(f"Ошибка: {ffmpeg_error}", err=True)
        return
    
    # Определяем входной путь
    if input_path is None:
        # Если не указан input_path, используем input_dir или папку input по умолчанию
        if input_dir:
            input_path = input_dir
        else:
            # Создаем папки input/ и output/ если не существуют
            ensure_project_directories()
            input_path = get_default_input_directory()
            if not quiet:
                click.echo(f"Используется папка по умолчанию: {input_path}")
    
    # Проверка наличия input_path
    if input_path is None:
        click.echo("Ошибка: требуется указать путь к видео файлу или директории", err=True)
        click.echo("Используйте --help для справки", err=True)
        return
    
    # Определяем форматы вывода
    if output_formats == 'all':
        formats = ['txt', 'srt', 'vtt']
    else:
        formats = [output_formats]
    
    # Определяем выходную директорию по умолчанию если не указана
    if output_dir is None:
        ensure_project_directories()
        output_dir = get_default_output_directory()
        if not quiet:
            click.echo(f"Результаты будут сохранены в: {output_dir}")
    
    # Валидация входного пути
    is_valid, error_message, file_list = validate_input_path(input_path)
    
    if not is_valid:
        click.echo(f"Ошибка: {error_message}", err=True)
        return
    
    if not quiet:
        models_info = get_model_info()
        model_info = models_info.get(model, {})
        click.echo(f"Найдено файлов для обработки: {len(file_list)}")
        click.echo(f"Модель: {model} ({model_info.get('description', '')})")
        if high_quality:
            click.echo("Режим: высокое качество (медленнее, но точнее)")
        if language:
            click.echo(f"Язык: {language}")
        else:
            click.echo("Язык: автоопределение")
        click.echo(f"Форматы вывода: {', '.join(formats)}")
        click.echo("")
    
    # Инициализация транскрибатора
    try:
        transcriber = VideoTranscriber(
            model_name=model,
            language=language,
            high_quality=high_quality,
            beam_size=beam_size,
            best_of=best_of,
            preprocess_audio_flag=preprocess_audio
        )
    except Exception as e:
        click.echo(f"Ошибка при инициализации транскрибатора: {e}", err=True)
        return
    
    # Инициализация диаризатора если требуется
    diarizer = None
    if diarize != 'none':
        try:
            diarizer = SpeakerDiarizer(
                method=diarize,
                pause_threshold=pause_threshold if pause_threshold is not None else 2.0,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                clustering_threshold=diarization_threshold
            )
            if not quiet:
                click.echo(f"Диаризация: метод {diarize}")
                if pause_threshold is not None and diarize == 'simple':
                    click.echo(f"Порог паузы: {pause_threshold} сек")
                if min_speakers is not None or max_speakers is not None:
                    click.echo(f"Количество спикеров: {min_speakers or '?'}-{max_speakers or '?'}")
                if diarization_threshold is not None:
                    click.echo(f"Порог кластеризации: {diarization_threshold}")
        except Exception as e:
            click.echo(f"Предупреждение: не удалось инициализировать диаризатор: {e}", err=True)
            if diarize == 'pyannote':
                click.echo("Используется простой метод диаризации", err=True)
                diarizer = SpeakerDiarizer(
                    method='simple',
                    pause_threshold=pause_threshold if pause_threshold is not None else 2.0
                )
    
    # Обработка файлов
    successful = 0
    failed = 0
    
    for video_file in tqdm(file_list, disable=quiet, desc="Обработка файлов"):
        try:
            # Определяем выходную директорию
            output_directory = ensure_output_directory(output_dir, video_file)
            
            # Транскрибация
            result = transcriber.transcribe(video_file)
            segments = transcriber.get_segments_with_timestamps(result)
            
            # Применяем диаризацию если требуется
            if diarizer:
                try:
                    # Для pyannote нужен путь к аудио файлу
                    audio_path = video_file if diarize in ('pyannote', 'auto') else None
                    segments = diarizer.diarize(segments, audio_path=audio_path, hf_token=hf_token)
                except Exception as e:
                    if not quiet:
                        click.echo(f"Предупреждение: ошибка диаризации: {e}", err=True)
                    # Если pyannote не работает, пробуем простой метод как fallback
                    if diarize in ('pyannote', 'auto'):
                        try:
                            if not quiet:
                                click.echo("Переключение на простой метод диаризации...", err=True)
                            simple_diarizer = SpeakerDiarizer(method='simple')
                            segments = simple_diarizer.diarize(segments, audio_path=None, hf_token=None)
                        except Exception as e2:
                            if not quiet:
                                click.echo(f"Предупреждение: простой метод также не сработал: {e2}", err=True)
                    # Продолжаем без диаризации
            
            # Экспорт результатов
            output_base_path = get_output_filename(video_file, output_directory, '')
            
            # Определяем, включать ли тайминги в TXT
            include_timestamps = not no_timestamps
            
            # Определяем, включать ли метки спикеров
            include_speakers = diarizer is not None and any('speaker' in seg for seg in segments)
            
            # Экспорт в основные форматы
            export_transcription(
                segments, 
                output_base_path, 
                formats, 
                include_timestamps_in_txt=include_timestamps,
                include_speakers=include_speakers
            )
            
            # Создаем clean.txt если запрошено
            if clean_txt:
                clean_path = str(Path(output_base_path).with_name(Path(output_base_path).stem + '_clean.txt'))
                export_txt(segments, clean_path, include_timestamps=False, include_speakers=include_speakers)
            
            successful += 1
            
        except Exception as e:
            failed += 1
            if not quiet:
                click.echo(f"\nОшибка при обработке {video_file}: {e}", err=True)
    
    # Итоговая статистика
    if not quiet:
        click.echo("")
        click.echo(f"Обработка завершена. Успешно: {successful}, Ошибок: {failed}")


if __name__ == '__main__':
    main()
