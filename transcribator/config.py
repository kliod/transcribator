"""Модуль для работы с конфигурацией."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import os


DEFAULT_CONFIG = {
    "model": "small",
    "language": None,
    "output_formats": "all",
    "output_dir": None,
    "quiet": False,
    "high_quality": False,
    "input_dir": None,
    "no_timestamps": False,
    "clean_txt": False,
    "diarize": "none",
    "hf_token": None,
    "beam_size": None,
    "best_of": None,
    "preprocess_audio": False,
    "min_speakers": None,
    "max_speakers": None,
    "diarization_threshold": None,
    "pause_threshold": None
}


def get_config_path() -> Path:
    """
    Возвращает путь к файлу конфигурации по умолчанию.
    
    Returns:
        Path: Путь к файлу конфигурации
    """
    # Ищем конфиг в текущей директории или домашней директории пользователя
    current_dir_config = Path.cwd() / "transcribator.json"
    home_dir_config = Path.home() / ".transcribator.json"
    
    # Приоритет: текущая директория > домашняя директория
    if current_dir_config.exists():
        return current_dir_config
    elif home_dir_config.exists():
        return home_dir_config
    else:
        # По умолчанию используем текущую директорию
        return current_dir_config


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Загружает конфигурацию из файла.
    
    Args:
        config_path: Путь к файлу конфигурации. Если None, используется путь по умолчанию.
    
    Returns:
        dict: Словарь с параметрами конфигурации
    """
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = get_config_path()
    
    if not config_file.exists():
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Объединяем с дефолтными значениями (дефолты используются для отсутствующих ключей)
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        
        return merged_config
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Ошибка при загрузке конфигурации из {config_file}: {e}")


def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> Path:
    """
    Сохраняет конфигурацию в файл.
    
    Args:
        config: Словарь с параметрами конфигурации
        config_path: Путь к файлу конфигурации. Если None, используется путь по умолчанию.
    
    Returns:
        Path: Путь к сохраненному файлу конфигурации
    """
    if config_path:
        config_file = Path(config_path)
    else:
        config_file = get_config_path()
    
    # Создаем директорию если нужно
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем только не-None значения для читаемости
    clean_config = {k: v for k, v in config.items() if v is not None}
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(clean_config, f, indent=2, ensure_ascii=False)
    
    return config_file


def create_default_config(config_path: Optional[str] = None) -> Path:
    """
    Создает файл конфигурации с дефолтными значениями.
    
    Args:
        config_path: Путь к файлу конфигурации. Если None, используется путь по умолчанию.
    
    Returns:
        Path: Путь к созданному файлу конфигурации
    """
    return save_config(DEFAULT_CONFIG.copy(), config_path)


def merge_config_with_cli(config: Dict[str, Any], cli_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Объединяет конфигурацию из файла с параметрами из CLI.
    Параметры CLI имеют приоритет над конфигурацией.
    
    Args:
        config: Конфигурация из файла
        cli_params: Параметры из CLI
    
    Returns:
        dict: Объединенная конфигурация
    """
    merged = config.copy()
    
    # Обновляем только те параметры, которые были явно указаны в CLI
    for key, value in cli_params.items():
        # Для флагов (bool) обновляем только если True (флаг был установлен)
        # Для остальных параметров обновляем если не None
        if isinstance(value, bool):
            if value:  # Флаг установлен
                merged[key] = value
        elif value is not None:
            merged[key] = value
    
    return merged
