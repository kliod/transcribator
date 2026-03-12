"""Модуль для разбивки транскрипции по спикерам (speaker diarization)."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
import numpy as np

# Подавляем предупреждения от pyannote.audio и torchcodec глобально
warnings.filterwarnings("ignore", category=UserWarning, module='pyannote.audio')
warnings.filterwarnings("ignore", category=UserWarning, module='torchcodec')
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*degrees of freedom.*", category=UserWarning)


class SpeakerDiarizer:
    """Класс для разбивки транскрипции по спикерам."""
    
    def __init__(
        self,
        method: str = 'simple',
        pause_threshold: float = 2.0,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        clustering_threshold: Optional[float] = None
    ):
        """
        Инициализирует диаризатор.
        
        Args:
            method: Метод диаризации ('simple', 'pyannote', 'auto')
            pause_threshold: Порог паузы в секундах для простого метода
            min_speakers: Минимальное количество спикеров для pyannote (опционально)
            max_speakers: Максимальное количество спикеров для pyannote (опционально)
            clustering_threshold: Порог кластеризации для pyannote (0.0-1.0, по умолчанию 0.7)
        """
        self.method = method
        self.pause_threshold = pause_threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.clustering_threshold = clustering_threshold if clustering_threshold is not None else 0.7
        self.pyannote_available = False
        
        # Проверяем доступность pyannote
        if method in ('pyannote', 'auto'):
            try:
                import pyannote.audio
                self.pyannote_available = True
            except ImportError:
                self.pyannote_available = False
                if method == 'pyannote':
                    warnings.warn(
                        "pyannote.audio не установлен. Используйте 'pip install pyannote.audio' "
                        "для более точной разбивки по спикерам. Используется простой метод.",
                        UserWarning
                    )
    
    def diarize_simple(self, segments: List[Dict]) -> List[Dict]:
        """
        Улучшенный простой метод разбивки по спикерам на основе пауз между сегментами.
        
        Этот метод анализирует паузы между сегментами Whisper и группирует
        сегменты по временным интервалам, предполагая что длинные паузы
        означают смену спикера. Использует динамический порог паузы и анализ
        длины сегментов для улучшения точности.
        
        Args:
            segments: Список сегментов с ключами 'start', 'end', 'text'
        
        Returns:
            Список сегментов с добавленным ключом 'speaker' (0, 1, 2, ...)
        """
        if not segments:
            return []
        
        # Вычисляем статистику пауз для динамического порога
        pauses = []
        segment_durations = []
        for i in range(1, len(segments)):
            pause = segments[i]['start'] - segments[i - 1]['end']
            if pause > 0:
                pauses.append(pause)
            segment_durations.append(segments[i - 1]['end'] - segments[i - 1]['start'])
        
        # Добавляем длительность последнего сегмента
        if segments:
            segment_durations.append(segments[-1]['end'] - segments[-1]['start'])
        
        # Динамический порог: медиана пауз + стандартное отклонение
        # Это адаптируется к характеристикам конкретного аудио
        if pauses:
            pause_median = np.median(pauses)
            pause_std = np.std(pauses) if len(pauses) > 1 else pause_median * 0.5
            dynamic_threshold = pause_median + pause_std
            # Используем максимум из заданного порога и динамического
            effective_threshold = max(self.pause_threshold, dynamic_threshold)
        else:
            effective_threshold = self.pause_threshold
        
        # Вычисляем среднюю длину сегмента для анализа
        avg_segment_duration = np.mean(segment_durations) if segment_durations else 3.0
        
        diarized_segments = []
        current_speaker = 0
        
        for i, segment in enumerate(segments):
            segment_copy = segment.copy()
            segment_duration = segment['end'] - segment['start']
            
            # Для первого сегмента всегда спикер 0
            if i == 0:
                segment_copy['speaker'] = current_speaker
            else:
                # Вычисляем паузу между текущим и предыдущим сегментом
                prev_end = segments[i - 1]['end']
                curr_start = segment['start']
                pause_duration = curr_start - prev_end
                
                # Улучшенная логика определения смены спикера:
                # 1. Пауза больше эффективного порога
                # 2. ИЛИ (пауза больше базового порога И предыдущий сегмент был длинным)
                #    Длинные сегменты часто означают монолог одного спикера
                should_switch = False
                
                if pause_duration >= effective_threshold:
                    should_switch = True
                elif pause_duration >= self.pause_threshold:
                    # Проверяем длину предыдущего сегмента
                    prev_duration = segments[i - 1]['end'] - segments[i - 1]['start']
                    # Если предыдущий сегмент был длинным (больше средней длины), 
                    # это может указывать на окончание монолога
                    if prev_duration > avg_segment_duration * 1.5:
                        should_switch = True
                
                if should_switch:
                    current_speaker = 1 - current_speaker  # Переключаем между 0 и 1
                
                segment_copy['speaker'] = current_speaker
            
            diarized_segments.append(segment_copy)
        
        return diarized_segments
    
    def diarize_pyannote(
        self, 
        audio_path: str, 
        hf_token: Optional[str] = None
    ) -> List[Tuple[float, float, int]]:
        """
        Разбивка по спикерам с использованием pyannote.audio.
        
        Args:
            audio_path: Путь к аудио файлу
            hf_token: Токен Hugging Face (опционально)
        
        Returns:
            Список кортежей (start, end, speaker_id) для каждого сегмента речи
        """
        if not self.pyannote_available:
            raise RuntimeError(
                "pyannote.audio не установлен. "
                "Установите: pip install pyannote.audio"
            )
        
        try:
            import warnings
            # Подавляем все предупреждения от pyannote и torchcodec
            # Мы используем предзагруженное аудио, поэтому torchcodec не нужен
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.filterwarnings("ignore", message=".*torchcodec.*")
                warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
                warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
                from pyannote.audio import Pipeline
            
            # Загружаем pipeline для диаризации
            # В новых версиях pyannote используется параметр 'token' вместо 'use_auth_token'
            if hf_token:
                try:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=hf_token
                    )
                except TypeError:
                    # Fallback: пробуем старый параметр для совместимости
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
            else:
                # Пробуем использовать модель без токена
                try:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                except Exception as e:
                    # Если не получилось, пробуем альтернативную модель
                    try:
                        pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization"
                        )
                    except Exception as e2:
                        raise RuntimeError(
                            f"Не удалось загрузить модель диаризации. "
                            f"Модель требует токен Hugging Face. "
                            f"Получите токен на https://hf.co/settings/tokens и используйте --hf-token. "
                            f"Ошибки: {e}, {e2}"
                        )
            
            # Применяем диаризацию
            # Из-за проблем с torchcodec на Windows, нужно использовать предзагруженное аудио
            # Загружаем аудио через whisper.load_audio для совместимости
            try:
                import whisper
                audio_data = whisper.load_audio(audio_path)
                # Преобразуем в формат, который ожидает pyannote: {'waveform': tensor, 'sample_rate': int}
                import torch
                # whisper.load_audio возвращает numpy array, нужно преобразовать в torch tensor
                if not isinstance(audio_data, torch.Tensor):
                    audio_data = torch.from_numpy(audio_data)
                # Добавляем размерность канала если нужно: (time,) -> (1, time)
                if audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)
                # Получаем sample_rate (Whisper использует 16000)
                sample_rate = 16000
                audio_dict = {'waveform': audio_data, 'sample_rate': sample_rate}
                
                # Настраиваем параметры pipeline если указаны
                if hasattr(pipeline, 'instantiate'):
                    # Для новых версий pyannote используем instantiate для настройки параметров
                    pipeline_params = {}
                    if self.min_speakers is not None:
                        pipeline_params['min_speakers'] = self.min_speakers
                    if self.max_speakers is not None:
                        pipeline_params['max_speakers'] = self.max_speakers
                    if self.clustering_threshold is not None:
                        # Порог кластеризации обычно настраивается через параметры сегментации
                        # Пробуем найти соответствующий параметр в pipeline
                        if hasattr(pipeline, 'segmentation'):
                            if hasattr(pipeline.segmentation, 'threshold'):
                                pipeline.segmentation.threshold = self.clustering_threshold
                        if hasattr(pipeline, 'clustering'):
                            if hasattr(pipeline.clustering, 'threshold'):
                                pipeline.clustering.threshold = self.clustering_threshold
                    
                    if pipeline_params:
                        try:
                            pipeline.instantiate(pipeline_params)
                        except Exception as e:
                            warnings.warn(f"Не удалось настроить параметры pipeline: {e}")
                
                # Подавляем предупреждения при применении диаризации
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.filterwarnings("ignore", message=".*torchcodec.*")
                    warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
                    warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
                    diarization = pipeline(audio_dict)
            except Exception as e:
                # Fallback: пробуем использовать путь к файлу напрямую
                # Подавляем предупреждения при применении диаризации
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.filterwarnings("ignore", message=".*torchcodec.*")
                    warnings.filterwarnings("ignore", message=".*std\\(\\): degrees of freedom.*")
                    warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
                    diarization = pipeline(audio_path)
            
            # Преобразуем результат в список кортежей
            # В новых версиях pyannote API изменился - DiarizeOutput это Annotation объект
            segments_list = []
            try:
                # В новых версиях pyannote DiarizeOutput это Annotation объект
                # Используем правильный API для Annotation
                from pyannote.core import Annotation
                
                # Проверяем тип объекта и доступные методы
                diarization_type = type(diarization).__name__
                is_annotation = isinstance(diarization, Annotation)
                has_itertracks = hasattr(diarization, 'itertracks')
                
                # DiarizeOutput содержит результаты в атрибутах speaker_diarization или exclusive_speaker_diarization
                # Пробуем извлечь Annotation из этих атрибутов
                annotation_result = None
                if hasattr(diarization, 'speaker_diarization'):
                    annotation_result = diarization.speaker_diarization
                elif hasattr(diarization, 'exclusive_speaker_diarization'):
                    annotation_result = diarization.exclusive_speaker_diarization
                
                # Если нашли Annotation в атрибутах, используем его
                if annotation_result is not None:
                    if isinstance(annotation_result, Annotation):
                        diarization = annotation_result
                        is_annotation = True
                        has_itertracks = hasattr(diarization, 'itertracks')
                    else:
                        # Пробуем использовать найденный атрибут как Annotation
                        if hasattr(annotation_result, 'itertracks'):
                            diarization = annotation_result
                            is_annotation = isinstance(annotation_result, Annotation)
                            has_itertracks = True
                
                # Пробуем разные способы итерации
                if is_annotation or has_itertracks:
                    try:
                        # Пробуем itertracks с yield_label=True (новый API)
                        try:
                            for segment, track, speaker_label in diarization.itertracks(yield_label=True):
                                speaker_id = int(speaker_label.split('_')[-1]) if isinstance(speaker_label, str) and '_' in speaker_label else 0
                                segments_list.append((segment.start, segment.end, speaker_id))
                        except (TypeError, ValueError):
                            # Пробуем старый формат (turn, _, speaker)
                            for turn, _, speaker in diarization.itertracks(yield_label=True):
                                speaker_id = int(speaker.split('_')[-1]) if isinstance(speaker, str) and '_' in speaker else 0
                                segments_list.append((turn.start, turn.end, speaker_id))
                    except AttributeError as e:
                        # Пробуем альтернативные методы
                        if hasattr(diarization, 'get_timeline'):
                            timeline = diarization.get_timeline()
                            for segment in timeline:
                                speaker_label = diarization[segment]
                                speaker_id = int(speaker_label.split('_')[-1]) if isinstance(speaker_label, str) and '_' in speaker_label else 0
                                segments_list.append((segment.start, segment.end, speaker_id))
                        elif hasattr(diarization, 'items'):
                            for segment, speaker_label in diarization.items():
                                speaker_id = int(speaker_label.split('_')[-1]) if isinstance(speaker_label, str) and '_' in speaker_label else 0
                                segments_list.append((segment.start, segment.end, speaker_id))
                        else:
                            raise RuntimeError(f"Не удалось обработать результат диаризации. Тип: {diarization_type}, ошибка: {e}")
                else:
                    # Пробуем итерацию как словарь или через другие методы
                    if hasattr(diarization, 'get_timeline'):
                        timeline = diarization.get_timeline()
                        for segment in timeline:
                            speaker_label = diarization[segment]
                            speaker_id = int(speaker_label.split('_')[-1]) if isinstance(speaker_label, str) and '_' in speaker_label else 0
                            segments_list.append((segment.start, segment.end, speaker_id))
                    elif hasattr(diarization, 'items'):
                        for segment, speaker_label in diarization.items():
                            speaker_id = int(speaker_label.split('_')[-1]) if isinstance(speaker_label, str) and '_' in speaker_label else 0
                            segments_list.append((segment.start, segment.end, speaker_id))
                    else:
                        raise RuntimeError(f"Не удалось определить API для обработки результата диаризации. Тип: {diarization_type}")
            except Exception as e:
                raise RuntimeError(f"Не удалось обработать результат диаризации: {e}. Тип объекта: {type(diarization)}")
            
            return segments_list
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при диаризации с pyannote: {e}")
    
    def _postprocess_diarization(self, segments: List[Dict], min_segment_duration: float = 0.5) -> List[Dict]:
        """
        Постобработка результатов диаризации для улучшения точности.
        
        Args:
            segments: Список сегментов с метками спикеров
            min_segment_duration: Минимальная длительность сегмента в секундах для фильтрации
        
        Returns:
            Обработанный список сегментов
        """
        if not segments:
            return segments
        
        # Фильтрация очень коротких сегментов
        filtered_segments = []
        for seg in segments:
            duration = seg['end'] - seg['start']
            if duration >= min_segment_duration:
                filtered_segments.append(seg)
            # Очень короткие сегменты пропускаем или объединяем с соседними
        
        if not filtered_segments:
            return segments  # Если все сегменты были отфильтрованы, возвращаем исходные
        
        # Сглаживание переключений спикеров
        # Если короткий сегмент одного спикера находится между двумя длинными сегментами другого,
        # вероятно это ошибка - меняем спикера короткого сегмента
        smoothed_segments = []
        for i, seg in enumerate(filtered_segments):
            seg_copy = seg.copy()
            duration = seg['end'] - seg['start']
            
            # Проверяем соседние сегменты для сглаживания
            if i > 0 and i < len(filtered_segments) - 1:
                prev_seg = filtered_segments[i - 1]
                next_seg = filtered_segments[i + 1]
                
                prev_duration = prev_seg['end'] - prev_seg['start']
                next_duration = next_seg['end'] - next_seg['start']
                
                # Если текущий сегмент короткий, а соседние длинные и одного спикера
                if (duration < 1.0 and 
                    prev_duration > 2.0 and 
                    next_duration > 2.0 and
                    prev_seg.get('speaker') == next_seg.get('speaker') and
                    seg.get('speaker') != prev_seg.get('speaker')):
                    # Вероятно ошибка - меняем спикера на спикера соседних сегментов
                    seg_copy['speaker'] = prev_seg.get('speaker', 0)
            
            smoothed_segments.append(seg_copy)
        
        return smoothed_segments
    
    def assign_speakers_to_segments(
        self, 
        segments: List[Dict], 
        diarization_result: List[Tuple[float, float, int]]
    ) -> List[Dict]:
        """
        Присваивает спикеров сегментам транскрипции на основе результата диаризации.
        Использует взвешенное присвоение на основе пересечения временных интервалов.
        
        Args:
            segments: Список сегментов транскрипции с 'start', 'end', 'text'
            diarization_result: Результат диаризации - список (start, end, speaker_id)
        
        Returns:
            Список сегментов с добавленным ключом 'speaker'
        """
        if not diarization_result:
            return segments
        
        diarized_segments = []
        
        for segment in segments:
            segment_copy = segment.copy()
            seg_start = segment['start']
            seg_end = segment['end']
            
            # Взвешенное присвоение: находим диаризационный сегмент с наибольшим пересечением
            best_speaker = 0  # По умолчанию
            max_overlap = 0.0
            
            for diar_start, diar_end, speaker_id in diarization_result:
                # Вычисляем пересечение интервалов
                overlap_start = max(seg_start, diar_start)
                overlap_end = min(seg_end, diar_end)
                
                if overlap_start < overlap_end:
                    # Есть пересечение
                    overlap_duration = overlap_end - overlap_start
                    seg_duration = seg_end - seg_start
                    
                    # Вычисляем долю пересечения от длительности сегмента транскрипции
                    overlap_ratio = overlap_duration / seg_duration if seg_duration > 0 else 0
                    
                    # Выбираем спикера с наибольшим пересечением
                    if overlap_ratio > max_overlap:
                        max_overlap = overlap_ratio
                        best_speaker = speaker_id
            
            # Если пересечение слишком мало (< 20%), используем центр сегмента как fallback
            if max_overlap < 0.2:
                seg_mid = (seg_start + seg_end) / 2
                for diar_start, diar_end, speaker_id in diarization_result:
                    if diar_start <= seg_mid <= diar_end:
                        best_speaker = speaker_id
                        break
            
            segment_copy['speaker'] = best_speaker
            diarized_segments.append(segment_copy)
        
        # Применяем постобработку для улучшения точности
        diarized_segments = self._postprocess_diarization(diarized_segments)
        
        return diarized_segments
    
    def diarize(
        self, 
        segments: List[Dict], 
        audio_path: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> List[Dict]:
        """
        Выполняет диаризацию сегментов транскрипции.
        
        Args:
            segments: Список сегментов транскрипции
            audio_path: Путь к аудио файлу (требуется для pyannote)
            hf_token: Токен Hugging Face (опционально, для pyannote)
        
        Returns:
            Список сегментов с добавленным ключом 'speaker'
        """
        if self.method == 'none':
            return segments
        
        if self.method == 'simple':
            return self.diarize_simple(segments)
        
        if self.method == 'pyannote':
            if not audio_path:
                raise ValueError("audio_path требуется для метода pyannote")
            
            if not self.pyannote_available:
                warnings.warn(
                    "pyannote.audio недоступен, используется простой метод",
                    UserWarning
                )
                return self.diarize_simple(segments)
            
            diarization_result = self.diarize_pyannote(audio_path, hf_token)
            return self.assign_speakers_to_segments(segments, diarization_result)
        
        if self.method == 'auto':
            # Автоматический выбор: пробуем pyannote, если доступен, иначе simple
            if self.pyannote_available and audio_path:
                try:
                    diarization_result = self.diarize_pyannote(audio_path, hf_token)
                    return self.assign_speakers_to_segments(segments, diarization_result)
                except Exception as e:
                    warnings.warn(
                        f"Не удалось использовать pyannote: {e}. Используется простой метод.",
                        UserWarning
                    )
                    return self.diarize_simple(segments)
            else:
                return self.diarize_simple(segments)
        
        raise ValueError(f"Неизвестный метод диаризации: {self.method}")
