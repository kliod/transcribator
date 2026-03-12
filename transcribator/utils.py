"""Вспомогательные функции для валидации и обработки файлов."""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict


# Поддерживаемые форматы видео
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp', '.ogv'}


def is_video_file(file_path: str) -> bool:
    """Проверяет, является ли файл видео файлом."""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS


def find_video_files(directory: str) -> List[str]:
    """Находит все видео файлы в указанной директории."""
    video_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Директория не найдена: {directory}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Путь не является директорией: {directory}")
    
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and is_video_file(str(file_path)):
            video_files.append(str(file_path))
    
    return sorted(video_files)


def validate_input_path(input_path: str) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """
    Валидирует входной путь и возвращает информацию о типе входа.
    
    Returns:
        tuple: (is_valid, error_message, file_list)
        - is_valid: True если путь валиден
        - error_message: Сообщение об ошибке или None
        - file_list: Список файлов для обработки или None
    """
    path = Path(input_path)
    
    if not path.exists():
        return False, f"Путь не существует: {input_path}", None
    
    if path.is_file():
        if not is_video_file(input_path):
            return False, f"Файл не является поддерживаемым видео форматом: {input_path}", None
        return True, None, [str(path.absolute())]
    
    if path.is_dir():
        video_files = find_video_files(input_path)
        if not video_files:
            return False, f"В директории не найдено видео файлов: {input_path}", None
        return True, None, video_files
    
    return False, f"Путь не является файлом или директорией: {input_path}", None


def ensure_output_directory(output_path: Optional[str], input_file: str) -> str:
    """
    Создает выходную директорию если необходимо.
    
    Args:
        output_path: Указанная пользователем выходная директория
        input_file: Путь к входному файлу
    
    Returns:
        str: Путь к выходной директории
    """
    if output_path:
        output_dir = Path(output_path)
    else:
        # По умолчанию сохраняем рядом с исходным файлом
        output_dir = Path(input_file).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def get_output_filename(input_file: str, output_dir: str, extension: str) -> str:
    """
    Генерирует имя выходного файла на основе входного файла.
    
    Args:
        input_file: Путь к входному файлу
        output_dir: Выходная директория
        extension: Расширение выходного файла (например, '.txt')
    
    Returns:
        str: Полный путь к выходному файлу
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_filename = output_path / f"{input_path.stem}{extension}"
    return str(output_filename)


def check_ffmpeg() -> Tuple[bool, Optional[str]]:
    """
    Проверяет наличие FFmpeg в системе.
    
    Returns:
        tuple: (is_available, error_message)
        - is_available: True если FFmpeg доступен
        - error_message: Сообщение об ошибке или None
    """
    ffmpeg_path = shutil.which('ffmpeg')
    
    if ffmpeg_path is None:
        error_msg = (
            "FFmpeg не найден в системе. FFmpeg необходим для обработки видео.\n"
            "Для установки на Windows запустите: .\\check_ffmpeg.ps1\n"
            "Или установите вручную с https://ffmpeg.org/download.html"
        )
        return False, error_msg
    
    return True, None


def ensure_project_directories(project_root: Optional[str] = None) -> Tuple[str, str]:
    """
    Создает папки input/ и output/ в корне проекта если они не существуют.
    
    Args:
        project_root: Корневая директория проекта (если None, определяется автоматически)
    
    Returns:
        tuple: (input_dir, output_dir) - пути к папкам input и output
    """
    if project_root is None:
        # Определяем корень проекта как директорию, содержащую transcribator/
        current_file = Path(__file__)
        # Ищем корень проекта (где находится setup.py или requirements.txt)
        project_root = current_file.parent.parent
    
    project_path = Path(project_root)
    input_dir = project_path / "input"
    output_dir = project_path / "output"
    
    # Создаем папки если не существуют
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return str(input_dir), str(output_dir)


def get_default_input_directory(project_root: Optional[str] = None) -> str:
    """
    Возвращает путь к папке input по умолчанию.
    
    Args:
        project_root: Корневая директория проекта
    
    Returns:
        str: Путь к папке input
    """
    input_dir, _ = ensure_project_directories(project_root)
    return input_dir


def get_default_output_directory(project_root: Optional[str] = None) -> str:
    """
    Возвращает путь к папке output по умолчанию.
    
    Args:
        project_root: Корневая директория проекта
    
    Returns:
        str: Путь к папке output
    """
    _, output_dir = ensure_project_directories(project_root)
    return output_dir


def get_whisper_cache_dir() -> Path:
    """
    Возвращает путь к директории кэша Whisper.
    
    Returns:
        Path: Путь к директории кэша
    """
    cache_dir = Path.home() / ".cache" / "whisper"
    return cache_dir


def clear_model_cache(model_name: Optional[str] = None) -> Tuple[int, float]:
    """
    Удаляет закэшированные модели Whisper.
    
    Args:
        model_name: Имя модели для удаления (None - удалить все модели)
    
    Returns:
        tuple: (deleted_count, freed_space_mb)
        - deleted_count: Количество удаленных файлов
        - freed_space_mb: Освобожденное место в MB
    """
    cache_dir = get_whisper_cache_dir()
    
    if not cache_dir.exists():
        return 0, 0.0
    
    deleted_count = 0
    freed_space_bytes = 0
    
    if model_name:
        # Удаляем конкретную модель
        cached_models = get_cached_models()
        if model_name in cached_models:
            model_file = Path(cached_models[model_name]['file'])
            if model_file.exists():
                freed_space_bytes = model_file.stat().st_size
                model_file.unlink()
                deleted_count = 1
    else:
        # Удаляем все модели
        for model_file in cache_dir.glob("*.pt"):
            freed_space_bytes += model_file.stat().st_size
            model_file.unlink()
            deleted_count += 1
    
    freed_space_mb = round(freed_space_bytes / (1024 * 1024), 2)
    return deleted_count, freed_space_mb


def get_cached_models() -> Dict[str, Dict[str, any]]:
    """
    Возвращает информацию о закэшированных моделях Whisper.
    
    Returns:
        dict: Словарь с информацией о закэшированных моделях
              Ключ - имя модели, значение - словарь с информацией о файле
    """
    cache_dir = get_whisper_cache_dir()
    cached_models = {}
    
    if not cache_dir.exists():
        return cached_models
    
    # Маппинг имен файлов на имена моделей
    model_files = {
        'tiny.pt': 'tiny',
        'tiny.en.pt': 'tiny.en',
        'base.pt': 'base',
        'base.en.pt': 'base.en',
        'small.pt': 'small',
        'small.en.pt': 'small.en',
        'medium.pt': 'medium',
        'medium.en.pt': 'medium.en',
        'large-v1.pt': 'large-v1',
        'large-v2.pt': 'large-v2',
        'large-v3.pt': 'large',
        'large-v3-turbo.pt': 'turbo',
    }
    
    for file_path in cache_dir.glob("*.pt"):
        file_name = file_path.name
        if file_name in model_files:
            model_name = model_files[file_name]
            file_size = file_path.stat().st_size
            cached_models[model_name] = {
                'file': str(file_path),
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'exists': True
            }
    
    return cached_models


def get_model_info() -> dict:
    """
    Возвращает информацию о доступных моделях Whisper.
    
    Returns:
        dict: Словарь с информацией о моделях
    """
    return {
        'tiny': {
            'name': 'tiny',
            'size': '39 MB',
            'speed': 'Очень быстрая',
            'accuracy': 'Низкая',
            'description': 'Самая быстрая модель, подходит для быстрой транскрибации'
        },
        'base': {
            'name': 'base',
            'size': '74 MB',
            'speed': 'Быстрая',
            'accuracy': 'Средняя',
            'description': 'Баланс скорости и точности (рекомендуется по умолчанию)'
        },
        'small': {
            'name': 'small',
            'size': '244 MB',
            'speed': 'Средняя',
            'accuracy': 'Хорошая',
            'description': 'Хорошая точность при приемлемой скорости'
        },
        'medium': {
            'name': 'medium',
            'size': '769 MB',
            'speed': 'Медленная',
            'accuracy': 'Высокая',
            'description': 'Высокая точность, требует больше времени'
        },
        'large': {
            'name': 'large',
            'size': '1550 MB',
            'speed': 'Очень медленная',
            'accuracy': 'Максимальная',
            'description': 'Максимальная точность, требует много времени и памяти'
        },
        'turbo': {
            'name': 'turbo',
            'size': '1550 MB',
            'speed': 'Быстрая',
            'accuracy': 'Максимальная',
            'description': 'Турбо-версия large-v3: максимальная точность с улучшенной скоростью'
        }
    }
