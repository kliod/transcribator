"""Модуль экспорта транскрипций в различные форматы."""

from typing import List, Dict
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """
    Форматирует время в секундах в формат HH:MM:SS,mmm для SRT.
    
    Args:
        seconds: Время в секундах
    
    Returns:
        str: Отформатированная строка времени
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    Форматирует время в секундах в формат HH:MM:SS.mmm для VTT.
    
    Args:
        seconds: Время в секундах
    
    Returns:
        str: Отформатированная строка времени
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_readable(seconds: float) -> str:
    """
    Форматирует время в секундах в читаемый формат [MM:SS] для TXT.
    
    Args:
        seconds: Время в секундах
    
    Returns:
        str: Отформатированная строка времени [MM:SS]
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def export_txt(segments: List[Dict], output_path: str, include_timestamps: bool = True, include_speakers: bool = False):
    """
    Экспортирует транскрипцию в текстовый формат (.txt).

    Args:
        segments: Список сегментов с ключами 'start', 'end', 'text', и опционально 'speaker'
        output_path: Путь к выходному файлу
        include_timestamps: Включать ли временные метки в текстовый формат
        include_speakers: Включать ли метки спикеров [Спикер 1], [Спикер 2] и т.д.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        prev_speaker = None

        for i, segment in enumerate(segments):
            text = segment['text']
            speaker = segment.get('speaker')

            # Добавляем пустую строку между разными спикерами для читаемости
            if include_speakers and speaker is not None and prev_speaker is not None:
                if speaker != prev_speaker and i > 0:
                    f.write("\n")

            # Формируем строку для записи
            parts = []

            # Добавляем метку спикера если нужно и если спикер изменился
            if include_speakers and speaker is not None:
                speaker_label = f"[Спикер {speaker + 1}]"
                # Добавляем метку если это первый сегмент или спикер изменился
                if prev_speaker is None or speaker != prev_speaker:
                    parts.append(speaker_label)
                prev_speaker = speaker

            # Добавляем таймстамп если нужно
            if include_timestamps and 'start' in segment:
                timestamp = format_timestamp_readable(segment['start'])
                parts.append(timestamp)

            # Добавляем текст
            parts.append(text)

            # Записываем строку
            line = ' '.join(parts)
            f.write(f"{line}\n")

    print(f"Текст сохранен: {output_path}")


def export_srt(segments: List[Dict], output_path: str):
    """
    Экспортирует транскрипцию в формат субтитров SRT (.srt).
    
    Args:
        segments: Список сегментов с ключами 'start', 'end', 'text'
        output_path: Путь к выходному файлу
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text']
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    print(f"SRT субтитры сохранены: {output_path}")


def export_vtt(segments: List[Dict], output_path: str):
    """
    Экспортирует транскрипцию в формат WebVTT (.vtt).
    
    Args:
        segments: Список сегментов с ключами 'start', 'end', 'text'
        output_path: Путь к выходному файлу
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        
        for segment in segments:
            start_time = format_timestamp_vtt(segment['start'])
            end_time = format_timestamp_vtt(segment['end'])
            text = segment['text']
            
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")
    
    print(f"VTT субтитры сохранены: {output_path}")


def export_transcription(
    segments: List[Dict],
    output_base_path: str,
    formats: List[str],
    include_timestamps_in_txt: bool = True,
    include_speakers: bool = False
):
    """
    Экспортирует транскрипцию в указанные форматы.
    
    Args:
        segments: Список сегментов с ключами 'start', 'end', 'text', и опционально 'speaker'
        output_base_path: Базовый путь к выходному файлу (без расширения)
        formats: Список форматов для экспорта ('txt', 'srt', 'vtt')
        include_timestamps_in_txt: Включать ли временные метки в TXT формат
        include_speakers: Включать ли метки спикеров в TXT формат
    """
    # Валидация сегментов перед экспортом
    if not segments:
        raise ValueError("Нет сегментов для экспорта")
    
    for i, segment in enumerate(segments):
        if 'start' not in segment or 'end' not in segment or 'text' not in segment:
            raise ValueError(f"Сегмент {i} не содержит обязательных полей: start, end, text")
    
    output_path = Path(output_base_path)
    
    if 'txt' in formats:
        export_txt(
            segments, 
            str(output_path.with_suffix('.txt')), 
            include_timestamps=include_timestamps_in_txt,
            include_speakers=include_speakers
        )
    
    if 'srt' in formats:
        export_srt(segments, str(output_path.with_suffix('.srt')))
    
    if 'vtt' in formats:
        export_vtt(segments, str(output_path.with_suffix('.vtt')))
