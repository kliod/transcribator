# Transcribator

Небольшой эксперимент для транскрибации видео в текст с использованием OpenAI Whisper.
Если ищете некий схожий функицонал есть [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI)

## Возможности

- Транскрибация видео файлов в текст с временными метками
- Поддержка пакетной обработки нескольких файлов
- Экспорт результатов в различных форматах:
  - TXT - текстовый формат с временными метками
  - SRT - формат субтитров
  - VTT - формат WebVTT
- Автоматическое определение языка или указание языка вручную
- Выбор модели Whisper (tiny, base, small, medium, large, turbo)
- Режим высокого качества для максимальной точности
- Структурированная организация файлов (папки input/ и output/)
- Отображение прогресса обработки
- **Разбивка по спикерам** - автоматическое определение кто говорит в диалоге
- **Опциональные тайминги** - возможность отключить тайминги в TXT формате
- **Чистый формат** - создание дополнительного файла без таймингов для удобного чтения

## Требования

- Python 3.8 или выше
- FFmpeg (требуется для обработки видео)

### Опциональные зависимости

Для более точной разбивки по спикерам можно установить pyannote.audio:

```bash
pip install pyannote.audio
```

**Примечание:** pyannote.audio опционален. Без него доступен простой метод разбивки по спикерам на основе пауз.

#### Разрешения Hugging Face для pyannote

При использовании метода диаризации `pyannote` или `auto` необходимо принять условия использования моделей на Hugging Face и указать токен доступа:

1. Зарегистрируйтесь на [Hugging Face](https://huggingface.co/) и создайте [Access Token](https://huggingface.co/settings/tokens).
2. Примите условия использования для следующих моделей (кнопка «Agree and access repository» на каждой странице):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Укажите токен в конфиге (`hf_token`) или через опцию `--hf-token` при запуске.

Без принятия условий и без валидного токена загрузка моделей pyannote завершится ошибкой.

### Установка FFmpeg

**Windows (рекомендуется):**
Используйте встроенный скрипт для автоматической проверки и установки:
```powershell
.\check_ffmpeg.ps1
```

Скрипт автоматически:
- Проверяет наличие FFmpeg в системе
- Предлагает установку через Chocolatey (если установлен)
- Предоставляет инструкции для ручной установки

**Windows (ручная установка):**
1. Скачайте FFmpeg с [официального сайта](https://ffmpeg.org/download.html)
2. Распакуйте архив (например, в `C:\ffmpeg`)
3. Добавьте `C:\ffmpeg\bin` в переменную окружения PATH
4. Перезапустите терминал

**Linux:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## Установка

### Установка через uv (современный вариант)

1. Установите `uv` по инструкции из официальной документации.
2. В корне проекта выполните:
```bash
uv sync
```
3. Запуск из среды проекта:
```bash
uv run transcribator --list-models
uv run transcribator video.mp4
```

### Быстрая установка (рекомендуется, Windows)

Из корня проекта выполните скрипт — он проверит Python 3.8+, создаст `.venv`, установит зависимости и пакет, опционально FFmpeg и pyannote.audio:

```powershell
.\install.ps1
```

После установки: `.\.venv\Scripts\Activate.ps1`, затем `transcribator --list-models` или `transcribator video.mp4`.

Параметры: `-InstallPyannote`, `-SkipPyannote`, `-NoVenv`, `-SkipFfmpegCheck`, `-Quiet`. Пример: `.\install.ps1 -InstallPyannote`

### Ручная установка

1. Клонируйте репозиторий или скачайте проект:
```bash
cd transcribator
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. (Опционально) Установите пакет для использования из любой директории:
```bash
pip install -e .
```

**Важно:** После установки команда `transcribator` может быть недоступна, если директория со скриптами Python не добавлена в PATH. 

**Варианты использования:**

1. **Через Python модуль (рекомендуется):**
```bash
python -m transcribator.cli --list-models
python -m transcribator.cli video.mp4
```

2. **Через батник (Windows):**
```bash
.\transcribator.bat --list-models
.\transcribator.bat video.mp4
```

3. **Если команда `transcribator` доступна:**
```bash
transcribator --list-models
transcribator video.mp4
```

**Примечание:** Если команда `transcribator` не работает после установки, добавьте директорию со скриптами Python в PATH или используйте вариант 1 или 2.

## Структура проекта

Проект автоматически создает папки для организации файлов:

```
transcribator/
├── input/          # Входные видео файлы (создается автоматически)
├── output/         # Выходные транскрипции (создается автоматически)
├── transcribator/  # Код проекта
└── ...
```

По умолчанию инструмент ищет видео файлы в папке `input/` и сохраняет результаты в папку `output/`. Вы можете указать свои пути через опции командной строки.

## Конфигурация

Для удобства можно создать файл конфигурации с дефолтными параметрами, чтобы не указывать их каждый раз в командной строке.

### Создание конфигурационного файла

```bash
transcribator --create-config
```

Это создаст файл `transcribator.json` в текущей директории с дефолтными значениями. Вы можете отредактировать его для настройки параметров по умолчанию.

### Расположение конфигурационного файла

Инструмент ищет конфигурацию в следующем порядке:
1. `transcribator.json` в текущей директории
2. `~/.transcribator.json` в домашней директории пользователя

Также можно указать путь к конфигу вручную:
```bash
transcribator video.mp4 --config /path/to/config.json
```

### Пример конфигурационного файла

**Базовый конфиг (минимальные настройки):**
```json
{
  "model": "small",
  "output_formats": "all",
  "diarize": "none"
}
```

**Расширенный конфиг для максимального качества:**
```json
{
  "model": "medium",
  "language": "ru",
  "output_formats": "all",
  "high_quality": true,
  "preprocess_audio": true,
  "diarize": "auto",
  "beam_size": 5,
  "best_of": 5,
  "min_speakers": 2,
  "max_speakers": 4,
  "diarization_threshold": 0.7,
  "pause_threshold": 2.0
}
```

**Все доступные параметры:**
- `model` - модель Whisper (tiny, base, small, medium, large, turbo)
- `language` - язык транскрибации (ru, en, и т.д.) или null для автоопределения
- `output_formats` - форматы вывода (txt, srt, vtt, all)
- `output_dir` - выходная директория (null для автоматической)
- `quiet` - тихий режим (true/false)
- `high_quality` - режим высокого качества (true/false)
- `input_dir` - входная директория (null для автоматической)
- `no_timestamps` - отключить тайминги в TXT (true/false)
- `clean_txt` - создавать clean.txt без таймингов (true/false)
- `diarize` - метод диаризации (none, simple, pyannote, auto)
- `hf_token` - токен Hugging Face для pyannote (null если не нужен)
- `beam_size` - размер луча для Whisper (1-10, null для авто)
- `best_of` - количество попыток для выбора лучшего результата (1-10, null для авто)
- `preprocess_audio` - предобработка аудио (true/false)
- `min_speakers` - минимальное количество спикеров (null для авто)
- `max_speakers` - максимальное количество спикеров (null для авто)
- `diarization_threshold` - порог кластеризации для pyannote (0.0-1.0, null для 0.7)
- `pause_threshold` - порог паузы для простого метода (секунды, null для 2.0)

### Приоритет параметров

Параметры из командной строки имеют приоритет над конфигурационным файлом. Это позволяет переопределить настройки из конфига при необходимости:

```bash
# Использует модель из конфига
transcribator video.mp4

# Переопределяет модель из конфига
transcribator video.mp4 --model large
```

## Использование

### Базовое использование

**Использование папок по умолчанию (рекомендуется):**
1. Поместите видео файлы в папку `input/`
2. Запустите транскрибацию:
```bash
transcribator
# или
python -m transcribator.cli
```
Результаты будут сохранены в папку `output/`

**Транскрибация конкретного файла или папки:**
```bash
transcribator video.mp4
transcribator ./videos/
```

**Если пакет не установлен:**
```bash
python -m transcribator.cli video.mp4
```

### Пакетная обработка

Обработка всех видео файлов в директории:
```bash
transcribator ./videos/
```

Или используйте папку по умолчанию:
```bash
# Поместите все видео в input/, затем:
transcribator
```

### Выбор модели

Просмотр информации о доступных моделях:
```bash
transcribator --list-models
# или
python -m transcribator.cli --list-models
```

Просмотр закэшированных моделей (уже загруженных):
```bash
transcribator --list-cached
```

Команда `--list-models` также показывает, какие модели уже закэшированы (помечены как `[ЗАКЭШИРОВАНА]`).

Использование более точной модели:
```bash
transcribator video.mp4 --model large
```

Использование turbo модели (максимальная точность с улучшенной скоростью):
```bash
transcribator video.mp4 --model turbo
```

Доступные модели (от быстрой к медленной):
- `tiny` - самая быстрая (39 MB), наименьшая точность
- `base` - баланс скорости и точности (74 MB)
- `small` - хорошая точность (244 MB, **по умолчанию**)
- `medium` - высокая точность (769 MB)
- `large` - максимальная точность (1550 MB), очень медленная
- `turbo` - турбо-версия large-v3: максимальная точность с улучшенной скоростью (1550 MB)

### Разбивка по спикерам

Инструмент поддерживает автоматическое определение спикеров в диалоге:

**Методы диаризации:**
- `none` - без разбивки по спикерам (по умолчанию)
- `simple` - простой метод на основе пауз между сегментами (не требует зависимостей)
- `pyannote` - точный метод с использованием pyannote.audio (требует установки)
- `auto` - автоматический выбор метода (пытается использовать pyannote, если доступен)

**Примеры:**
```bash
# Простой метод (работает всегда)
transcribator video.mp4 --diarize simple

# Pyannote метод (более точный, требует установки и разрешений HF)
pip install pyannote.audio
transcribator video.mp4 --diarize pyannote

# Автоматический выбор
transcribator video.mp4 --diarize auto
```

Результат будет содержать метки `[Спикер 1]`, `[Спикер 2]` и т.д. перед каждой репликой.

### Управление таймингами

По умолчанию TXT файлы содержат временные метки в формате `[MM:SS]`. Вы можете управлять этим:

```bash
# Отключить тайминги в основном файле
transcribator video.mp4 --no-timestamps

# Создать дополнительный clean файл без таймингов
transcribator video.mp4 --clean-txt
```

Файл `video_clean.txt` будет содержать только текст без временных меток, что удобно для чтения.

При запуске транскрибации автоматически отображается информация о выбранной модели.

### Режим высокого качества

Для максимальной точности используйте опцию `--high-quality`:
```bash
transcribator video.mp4 --high-quality
```

Этот режим включает:
- `temperature=0` - детерминированный вывод
- `best_of=5` - выбор лучшего результата из 5 попыток
- `beam_size=5` - размер луча для поиска
- `compression_ratio_threshold=2.4` - фильтрация низкокачественных сегментов
- `logprob_threshold=-1.0` - минимальный логарифм вероятности
- `no_speech_threshold=0.6` - порог для определения отсутствия речи
- `condition_on_previous_text=True` - использование контекста предыдущих сегментов

### Настройка качества транскрибации

Для тонкой настройки качества транскрибации доступны следующие опции:

**Параметры Whisper:**
```bash
# Настройка размера луча (1-10, больше = точнее, но медленнее)
transcribator video.mp4 --beam-size 7

# Количество попыток для выбора лучшего результата (1-10)
transcribator video.mp4 --best-of 7

# Предобработка аудио (нормализация, шумоподавление)
transcribator video.mp4 --preprocess-audio

# Комбинация параметров для максимального качества
transcribator video.mp4 --high-quality --beam-size 7 --best-of 7 --preprocess-audio --model large
```

**Рекомендации по выбору модели:**
- `tiny`, `base` - для быстрой обработки, низкая точность
- `small` - баланс скорости и качества (по умолчанию)
- `medium` - хорошая точность для большинства случаев
- `large` - максимальная точность, требует много памяти и времени
- `turbo` - максимальная точность с улучшенной скоростью (рекомендуется для production)

**Примеры для разных сценариев:**

```bash
# Быстрая обработка с приемлемым качеством
transcribator video.mp4 --model small

# Максимальное качество (медленно)
transcribator video.mp4 --model large --high-quality --preprocess-audio

# Оптимальный баланс скорости и качества
transcribator video.mp4 --model medium --high-quality

# Production качество с turbo моделью
transcribator video.mp4 --model turbo --high-quality --preprocess-audio --beam-size 5 --best-of 5
```

### Настройка диаризации

Для улучшения точности разбивки по спикерам доступны следующие опции:

**Параметры pyannote:**

Перед использованием примите условия использования моделей на Hugging Face (см. раздел «Разрешения Hugging Face для pyannote» выше).

```bash
# Указать количество спикеров (улучшает точность если известно)
transcribator video.mp4 --diarize pyannote --min-speakers 2 --max-speakers 2

# Настройка порога кластеризации (0.0-1.0, по умолчанию 0.7)
# Меньше значение = больше спикеров, больше значение = меньше спикеров
transcribator video.mp4 --diarize pyannote --diarization-threshold 0.6

# Комбинация параметров
transcribator video.mp4 --diarize pyannote --min-speakers 2 --max-speakers 4 --diarization-threshold 0.65 --hf-token YOUR_TOKEN
```

**Параметры простого метода:**
```bash
# Настройка порога паузы в секундах (по умолчанию 2.0)
# Меньше значение = больше переключений спикеров
transcribator video.mp4 --diarize simple --pause-threshold 1.5
```

**Рекомендации:**
- Используйте `pyannote` для максимальной точности (требует установки `pyannote.audio`)
- Если количество спикеров известно, укажите `--min-speakers` и `--max-speakers` для улучшения точности
- Для диалогов с четкими паузами простой метод может работать достаточно хорошо
- Простой метод использует динамический порог паузы, адаптирующийся к характеристикам аудио
- Постобработка автоматически фильтрует короткие сегменты и сглаживает переключения спикеров

### Указание языка

Если язык известен заранее, можно указать его для улучшения точности:
```bash
transcribator video.mp4 --language ru
```

Коды языков: `ru` (русский), `en` (английский), `de` (немецкий) и т.д.

### Выбор формата вывода

Экспорт только в текстовый формат:
```bash
transcribator video.mp4 --format txt
```

Экспорт в формат субтитров SRT:
```bash
transcribator video.mp4 --format srt
```

Экспорт во все форматы (по умолчанию):
```bash
transcribator video.mp4 --format all
```

### Указание входной и выходной директории

Сохранение результатов в указанную директорию:
```bash
transcribator video.mp4 --output ./transcripts/
```

Указание входной директории:
```bash
transcribator --input ./my_videos/ --output ./my_transcripts/
```

Если не указаны пути, используются папки `input/` и `output/` по умолчанию.

### Тихий режим

Подавление вывода прогресса:
```bash
transcribator video.mp4 --quiet
```

### Полный пример

Комплексный пример с несколькими параметрами:
```bash
transcribator ./videos/ --model medium --language ru --format all --output ./transcripts/ --high-quality
```

Или используя папки по умолчанию:
```bash
# Поместите видео в input/
transcribator --model medium --language ru --high-quality
# Результаты будут в output/
```

## Поддерживаемые форматы видео

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- FLV (.flv)
- WMV (.wmv)
- M4V (.m4v)
- 3GP (.3gp)
- OGV (.ogv)

И другие форматы, поддерживаемые FFmpeg.

## Структура проекта

```
transcribator/
├── input/                  # Входные видео файлы (создается автоматически)
├── output/                 # Выходные транскрипции (создается автоматически)
├── transcribator/
│   ├── __init__.py
│   ├── cli.py              # CLI интерфейс
│   ├── transcriber.py      # Основная логика транскрибации
│   ├── exporter.py         # Экспорт в различные форматы
│   └── utils.py            # Вспомогательные функции
├── requirements.txt        # Зависимости
├── setup.py                # Установка пакета
├── check_ffmpeg.ps1        # Скрипт проверки/установки FFmpeg (Windows)
└── README.md               # Документация
```

## Примеры использования

### Пример 1: Быстрая транскрибация

```bash
transcribator presentation.mp4
```

Создаст файлы:
- `presentation.txt`
- `presentation.srt`
- `presentation.vtt`

### Пример 2: Высококачественная транскрибация на русском

```bash
transcribator interview.mp4 --model large --language ru
```

### Пример 3: Пакетная обработка с сохранением в отдельную папку

```bash
transcribator ./recordings/ --output ./transcripts/ --format srt
```

### Пример 4: Просмотр информации о моделях

```bash
transcribator --list-models
```

### Пример 5: Просмотр закэшированных моделей

```bash
transcribator --list-cached
```

Показывает какие модели уже загружены и их размер.

### Пример 6: Очистка кэша моделей

Удалить все модели:
```bash
transcribator --clear-cache
```

Удалить конкретную модель:
```bash
transcribator --clear-cache-model base
```

### Пример 7: Использование turbo модели

```bash
transcribator video.mp4 --model turbo
```

Turbo модель обеспечивает максимальную точность с улучшенной скоростью по сравнению с обычной large моделью.

## Устранение неполадок

### Ошибка: "FFmpeg не найден"

Инструмент автоматически проверяет наличие FFmpeg перед началом работы.

**Windows:**
Запустите скрипт проверки и установки:
```powershell
.\check_ffmpeg.ps1
```

**Проверка вручную:**
```bash
ffmpeg -version
```

Если команда не найдена, установите FFmpeg согласно инструкциям выше.

### Ошибка: "Модель не найдена"

При первом использовании модели Whisper она автоматически загружается из интернета. Убедитесь, что у вас есть подключение к интернету.

### Кэширование моделей

Whisper автоматически кэширует загруженные модели для повторного использования. Модели хранятся в:

- **Windows:** `C:\Users\<username>\.cache\whisper\`
- **Linux/macOS:** `~/.cache/whisper/`

**Размеры моделей:**
- `tiny.pt` - ~39 MB
- `base.pt` - ~74 MB
- `small.pt` - ~244 MB
- `medium.pt` - ~769 MB
- `large-v3.pt` - ~1550 MB
- `large-v3-turbo.pt` - ~1550 MB (турбо-версия с улучшенной скоростью)

**Управление кэшем:**

Проверить загруженные модели:
```bash
transcribator --list-cached
# или
python -m transcribator.cli --list-cached
```

Удалить все модели из кэша:
```bash
transcribator --clear-cache
# или
python -m transcribator.cli --clear-cache
```

Удалить конкретную модель:
```bash
transcribator --clear-cache-model base
transcribator --clear-cache-model small
transcribator --clear-cache-model turbo
```

**Примечание:** После очистки кэша модели будут загружены заново при следующем использовании. Команды запрашивают подтверждение перед удалением (кроме режима `--quiet`).

### Медленная обработка

Для ускорения обработки:
- Используйте меньшую модель (например, `tiny` или `base`)
- Обрабатывайте файлы по одному вместо пакетной обработки

## Лицензия

MIT License

## Благодарности

Проект использует [OpenAI Whisper](https://github.com/openai/whisper) для транскрибации.
