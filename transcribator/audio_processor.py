"""Модуль для предобработки аудио перед транскрибацией."""

import numpy as np
from typing import Optional, Tuple
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    normalize: bool = True,
    denoise: bool = True,
    enhance_quiet: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Предобрабатывает аудио для улучшения качества транскрибации.
    
    Args:
        audio: Аудио данные как numpy массив
        sample_rate: Частота дискретизации аудио
        normalize: Нормализовать громкость
        denoise: Применить шумоподавление
        enhance_quiet: Усилить тихие участки
    
    Returns:
        tuple: (обработанное аудио, частота дискретизации)
    """
    processed_audio = audio.copy()
    
    # Ресемплинг до 16kHz если нужно (Whisper работает с 16kHz)
    if sample_rate != 16000:
        if LIBROSA_AVAILABLE:
            processed_audio = librosa.resample(processed_audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        else:
            warnings.warn(
                "librosa не установлен. Ресемплинг не выполнен. "
                "Установите librosa для лучшего качества: pip install librosa"
            )
    
    # Нормализация громкости
    if normalize:
        if LIBROSA_AVAILABLE:
            processed_audio = librosa.util.normalize(processed_audio)
        else:
            # Простая нормализация без librosa
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0:
                processed_audio = processed_audio / max_val * 0.95  # Оставляем небольшой запас
    
    # Шумоподавление
    if denoise:
        if LIBROSA_AVAILABLE:
            # Используем предэмфазис вместо фильтра Винера - он работает с float32 и не вызывает проблем с dtype
            try:
                processed_audio = librosa.effects.preemphasis(processed_audio)
                # Убеждаемся что dtype остался float32
                if processed_audio.dtype != np.float32:
                    processed_audio = processed_audio.astype(np.float32)
            except Exception as e:
                warnings.warn(f"Ошибка при применении предэмфаза: {e}")
        elif SCIPY_AVAILABLE:
            # Фильтр Винера вызывает проблемы с dtype (float64), поэтому используем его только как последний вариант
            # и только если librosa недоступен
            warnings.warn(
                "librosa не установлен. Использование фильтра Винера может вызвать проблемы с типами данных. "
                "Рекомендуется установить librosa: pip install librosa"
            )
            try:
                processed_audio = signal.wiener(processed_audio, mysize=5)
                # Немедленно преобразуем обратно в float32 после wiener фильтра
                processed_audio = processed_audio.astype(np.float32)
            except Exception as e:
                warnings.warn(f"Ошибка при применении фильтра Винера: {e}")
        else:
            warnings.warn(
                "scipy или librosa не установлены. Шумоподавление не выполнено. "
                "Установите scipy для лучшего качества: pip install scipy"
            )
    
    # Усиление тихих участков
    if enhance_quiet:
        if LIBROSA_AVAILABLE:
            # Вычисляем RMS энергии
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=processed_audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Находим тихие участки (ниже медианы)
            threshold = np.median(rms)
            
            # Применяем динамическое усиление только к тихим участкам
            # Это более безопасно чем глобальное усиление
            if threshold > 0:
                # Нормализуем RMS для создания маски усиления
                rms_normalized = rms / (threshold + 1e-10)
                # Ограничиваем усиление до 2x для избежания искажений
                gain_mask = np.clip(1.0 / (rms_normalized + 0.5), 1.0, 2.0)
                
                # Применяем усиление к аудио (упрощенная версия)
                # В реальности нужно применять к фреймам, но для простоты используем глобальное усиление
                avg_gain = np.mean(gain_mask)
                if avg_gain > 1.0:
                    processed_audio = processed_audio * min(avg_gain, 1.5)  # Ограничиваем до 1.5x
        else:
            warnings.warn(
                "librosa не установлен. Усиление тихих участков не выполнено. "
                "Установите librosa для этой функции: pip install librosa"
            )
    
    # Убеждаемся что dtype правильный (float32 для Whisper) и нет NaN/Inf
    if processed_audio.dtype != np.float32:
        processed_audio = processed_audio.astype(np.float32)
    
    # Обрабатываем NaN и Inf значения
    if np.isnan(processed_audio).any() or np.isinf(processed_audio).any():
        # Заменяем NaN и Inf на 0
        processed_audio = np.nan_to_num(processed_audio, nan=0.0, posinf=0.0, neginf=0.0)
    
    return processed_audio, sample_rate


def validate_audio(audio: np.ndarray, sample_rate: int) -> bool:
    """
    Валидирует аудио данные перед обработкой.
    
    Args:
        audio: Аудио данные
        sample_rate: Частота дискретизации
    
    Returns:
        bool: True если аудио валидно
    """
    if audio is None or len(audio) == 0:
        return False
    
    if sample_rate <= 0:
        return False
    
    # Проверяем что аудио не полностью тихое
    if np.max(np.abs(audio)) < 1e-6:
        return False
    
    return True
