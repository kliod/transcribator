"""Модуль транскрибации видео с использованием OpenAI Whisper."""

import whisper
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
import numpy as np

from .audio_processor import preprocess_audio, validate_audio


class VideoTranscriber:
    """Класс для транскрибации видео файлов с использованием Whisper."""
    
    def __init__(
        self,
        model_name: str = "small",
        language: Optional[str] = None,
        high_quality: bool = False,
        beam_size: Optional[int] = None,
        best_of: Optional[int] = None,
        preprocess_audio_flag: bool = False,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        condition_on_previous_text: Optional[bool] = None,
        initial_prompt: Optional[str] = None
    ):
        """
        Инициализирует транскрибатор.
        
        Args:
            model_name: Название модели Whisper (tiny, base, small, medium, large, turbo)
            language: Язык для транскрибации (None для автоопределения)
            high_quality: Использовать максимальные параметры качества
            beam_size: Размер луча для поиска (1-10, None для авто)
            best_of: Количество попыток для выбора лучшего результата (1-10, None для авто)
            preprocess_audio_flag: Включить предобработку аудио
            compression_ratio_threshold: Порог коэффициента сжатия для фильтрации (по умолчанию 2.4)
            logprob_threshold: Минимальный логарифм вероятности (по умолчанию -1.0)
            no_speech_threshold: Порог для определения отсутствия речи (по умолчанию 0.6)
            condition_on_previous_text: Использовать контекст предыдущих сегментов (True для лучшего качества)
            initial_prompt: Начальный промпт для улучшения распознавания специфической терминологии
        """
        self.model_name = model_name
        self.language = language
        self.high_quality = high_quality
        self.beam_size = beam_size
        self.best_of = best_of
        self.preprocess_audio_flag = preprocess_audio_flag
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель Whisper."""
        try:
            print(f"Загрузка модели Whisper: {self.model_name}...")
            self.model = whisper.load_model(self.model_name)
            print("Модель загружена успешно.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке модели Whisper: {e}")
    
    def transcribe(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Транскрибирует видео файл.
        
        Args:
            video_path: Путь к видео файлу
            **kwargs: Дополнительные параметры для whisper.transcribe()
        
        Returns:
            dict: Результат транскрибации с ключами:
                - 'text': Полный текст транскрипции
                - 'segments': Список сегментов с временными метками
                - 'language': Определенный язык
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")
        
        # Поддерживаемые форматы: mp4, avi, mov, mkv, webm и другие (через ffmpeg)
        
        # Базовые параметры для гарантии наличия сегментов с временными метками
        transcribe_options = {
            "language": self.language,
            "word_timestamps": False,  # Отключаем временные метки слов, но сегменты остаются
            **kwargs
        }
        
        # Параметры качества для улучшения точности
        if self.high_quality:
            transcribe_options.update({
                "temperature": 0,  # Детерминированный вывод
                "best_of": self.best_of if self.best_of is not None else 5,  # Выбор лучшего результата из 5 попыток
                "beam_size": self.beam_size if self.beam_size is not None else 5,  # Размер луча для поиска
            })
        else:
            # Базовые параметры качества
            transcribe_options.update({
                "temperature": 0,  # Детерминированный вывод для стабильности
            })
        
        # Дополнительные параметры качества (если указаны)
        if self.compression_ratio_threshold is not None:
            transcribe_options["compression_ratio_threshold"] = self.compression_ratio_threshold
        elif self.high_quality:
            transcribe_options["compression_ratio_threshold"] = 2.4  # Значение по умолчанию для высокого качества
        
        if self.logprob_threshold is not None:
            transcribe_options["logprob_threshold"] = self.logprob_threshold
        elif self.high_quality:
            transcribe_options["logprob_threshold"] = -1.0  # Значение по умолчанию для высокого качества
        
        if self.no_speech_threshold is not None:
            transcribe_options["no_speech_threshold"] = self.no_speech_threshold
        elif self.high_quality:
            transcribe_options["no_speech_threshold"] = 0.6  # Значение по умолчанию для высокого качества
        
        if self.condition_on_previous_text is not None:
            transcribe_options["condition_on_previous_text"] = self.condition_on_previous_text
        elif self.high_quality:
            transcribe_options["condition_on_previous_text"] = True  # Использовать контекст для лучшего качества
        
        if self.initial_prompt is not None:
            transcribe_options["initial_prompt"] = self.initial_prompt
        
        # Если указаны beam_size или best_of без high_quality, применяем их
        if not self.high_quality:
            if self.beam_size is not None:
                transcribe_options["beam_size"] = self.beam_size
            if self.best_of is not None:
                transcribe_options["best_of"] = self.best_of
        
        # Удаляем None значения
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
        
        try:
            print(f"Начало транскрибации: {video_path}")
            print("Извлечение аудио из видео...")
            
            # Явно извлекаем аудио из видео перед транскрибацией
            # Это предотвращает зависание внутри whisper.transcribe()
            audio = whisper.load_audio(video_path)
            sample_rate = 16000  # Whisper всегда использует 16kHz
            
            # Предобработка аудио если включена
            if self.preprocess_audio_flag:
                print("Предобработка аудио...")
                if validate_audio(audio, sample_rate):
                    audio, sample_rate = preprocess_audio(
                        audio,
                        sample_rate=sample_rate,
                        normalize=True,
                        denoise=True,
                        enhance_quiet=False
                    )
                    print("Предобработка аудио завершена.")
                else:
                    warnings.warn("Аудио не прошло валидацию, предобработка пропущена.")
            
            # Убеждаемся что массив float32 и непрерывный перед передачей в Whisper
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if not audio.flags['C_CONTIGUOUS']:
                audio = np.ascontiguousarray(audio, dtype=np.float32)
            
            print("Аудио извлечено, начало транскрибации...")
            
            # Подавляем предупреждения от whisper
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Используем уже извлеченное аудио вместо пути к видео
                result = self.model.transcribe(audio, **transcribe_options)
            
            print(f"Транскрибация завершена. Определенный язык: {result.get('language', 'неизвестно')}")
            return result
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при транскрибации видео: {e}")
    
    def get_segments_with_timestamps(self, transcription_result: Dict[str, Any]) -> list:
        """
        Извлекает сегменты с временными метками из результата транскрибации.
        
        Args:
            transcription_result: Результат транскрибации от Whisper
        
        Returns:
            list: Список словарей с ключами:
                - 'start': Время начала (секунды)
                - 'end': Время окончания (секунды)
                - 'text': Текст сегмента
        """
        segments = []
        raw_segments = transcription_result.get('segments', [])
        
        if not raw_segments:
            raise ValueError("Результат транскрибации не содержит сегментов с временными метками. Возможно, проблема с моделью или аудио файлом.")
        
        for segment in raw_segments:
            # Валидация наличия обязательных полей
            if 'start' not in segment or 'end' not in segment or 'text' not in segment:
                continue  # Пропускаем невалидные сегменты
            
            # Валидация значений временных меток
            start_time = float(segment['start'])
            end_time = float(segment['end'])
            
            if start_time < 0 or end_time < 0 or start_time >= end_time:
                continue  # Пропускаем невалидные временные метки
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': segment['text'].strip()
            })
        
        if not segments:
            raise ValueError("Не удалось извлечь валидные сегменты с временными метками из результата транскрибации.")
        
        return segments
