# Transcribator

Локальный инструмент для транскрибации видео и аудио с минимальным web UI, CPU-first установкой и опциональным GPU-ускорением для `faster-whisper` и `pyannote`.

## Что умеет

- Локальная транскрибация через `faster-whisper` по умолчанию
- Legacy fallback через `openai-whisper`
- TXT, SRT и VTT на выходе
- Минимальный web UI для загрузки одного файла, статуса, preview и скачивания результатов
- Разметка спикеров:
  - `off` — выключено
  - `simple` — approximate / experimental
  - `pyannote` — основной качественный режим
- Отдельные устройства для:
  - ASR (`device`)
  - diarization (`diarization_device`)
- Консервативная аудиоподготовка через FFmpeg
- Опциональные тайминги в TXT

## Актуальные требования

- Windows 11
- `uv`
- Python `3.12`
- FFmpeg
- Для `pyannote + CUDA`:
  - NVIDIA GPU
  - совместимый CUDA runtime
  - CUDA-сборки `torch` и `torchaudio`

## Быстрый старт

Основной рекомендуемый путь установки:

```powershell
.\install.ps1
```

Скрипт:

- проверит и при необходимости поставит `uv`
- обеспечит Python `3.12`
- пересоздаст `.venv`
- выполнит `uv sync`
- предложит CPU-only или GPU overlay для `pyannote`
- подскажет готовые команды запуска

После установки каноничные команды запуска такие:

```powershell
.\transcribator-web.bat
.\transcribator.bat --list-models
.\transcribator.bat "D:\path\to\video.mp4"
```

Эти wrapper-скрипты используют проектный `.venv` напрямую и не делают лишний `sync`, поэтому не ломают GPU overlay для `pyannote`.

## Альтернативный путь через uv

Если bootstrap не нужен:

```powershell
uv python install 3.12
uv venv --python 3.12 --clear .venv
uv sync
```

Запуск после этого рекомендуется такой:

```powershell
.\.venv\Scripts\python.exe -m transcribator.webapp
.\.venv\Scripts\python.exe -m transcribator.cli --help
```

Если у тебя уже настроен GPU overlay для `pyannote`, не используй обычный:

```powershell
uv run transcribator-web
```

Потому что `uv` может вернуть `torch/torchaudio` к CPU-версии из lock-файла. Безопасные варианты:

```powershell
uv run --no-sync transcribator-web
```

или wrapper-команды из корня проекта.

## Web UI

Запуск:

```powershell
.\transcribator-web.bat
```

После этого открой:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

В web UI доступны:

- загрузка одного файла
- выбор модели
- выбор языка
- quality mode
- speaker mode
- отдельные настройки ускорения для ASR и diarization
- preview текста
- скачивание `TXT`, `SRT`, `VTT`

Ограничения web UI:

- одна активная задача за раз
- локальный single-user сценарий
- без истории задач и без auth

## CLI

Основные примеры:

```powershell
.\transcribator.bat --list-models
.\transcribator.bat "D:\path\to\video.mp4"
.\transcribator.bat "D:\path\to\video.mp4" --engine faster-whisper --model small
.\transcribator.bat "D:\path\to\video.mp4" --device cuda
.\transcribator.bat "D:\path\to\video.mp4" --diarize pyannote --diarization-device cuda --hf-token hf_xxx
```

Полезные флаги:

- `--engine [faster-whisper|openai-whisper]`
- `--device [cpu|cuda|auto]`
- `--diarization-device [cpu|cuda|auto]`
- `--diarize [none|simple|pyannote|auto]`
- `--hf-token`
- `--no-timestamps`
- `--high-quality`

## ASR device и Diarization device

Это две разные настройки.

- `device` управляет ASR backend для `faster-whisper`
- `diarization_device` управляет `pyannote`

Типичный сценарий:

- ASR на `cuda`
- diarization на `cpu`

или наоборот, если нужен именно GPU для `pyannote`.

## Speaker modes

- `off`
  - без diarization
- `simple`
  - approximate режим по паузам
  - experimental, для финального качества не рекомендуется
- `pyannote`
  - основной качественный режим
  - требует Hugging Face token и доступ к моделям
  - если выбран и не может отработать, задача падает явно
  - silent fallback в `simple` больше нет

## Hugging Face для pyannote

Для `pyannote` нужен:

- Hugging Face account
- access token
- принятые условия доступа для репозиториев:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

Важно:

- Hugging Face используется только для токена, проверки доступа и загрузки моделей
- сама diarization выполняется локально на твоей машине

## Конфиг

Пример актуального `transcribator.json`:

```json
{
  "engine": "faster-whisper",
  "device": "cpu",
  "diarization_device": "cpu",
  "model": "small",
  "language": null,
  "output_formats": "all",
  "high_quality": false,
  "no_timestamps": false,
  "clean_txt": true,
  "diarize": "none",
  "hf_token": null,
  "beam_size": null,
  "best_of": null,
  "preprocess_audio": false,
  "min_speakers": null,
  "max_speakers": null,
  "diarization_threshold": 0.7,
  "pause_threshold": 2.0
}
```

Ключевые поля:

- `engine`
- `device`
- `diarization_device`
- `diarize`
- `hf_token`

## FFmpeg

FFmpeg обязателен для аудиоподготовки.

Проверка:

```powershell
ffmpeg -version
```

Если нужен helper-скрипт:

```powershell
.\check_ffmpeg.ps1
```

## GPU для pyannote

Базовая установка проекта остаётся CPU-first.

Если нужен `pyannote + CUDA`, см. отдельную памятку:

[docs/pyannote-gpu.md](docs/pyannote-gpu.md)

Коротко:

- базовый `uv sync` ставит CPU-first стек
- GPU overlay для `pyannote` накатывается отдельным шагом
- после этого запускать проект нужно через wrapper-скрипты или через `.venv\Scripts\python.exe`, а не через обычный `uv run`

## Wrapper-команды

В репозитории есть стабильные Windows-launchers:

- `transcribator.bat` — CLI
- `transcribator-web.bat` — web UI

Они запускают проект через локальный `.venv` и подходят как для CPU, так и для GPU overlay-сценария.

## Проверка установки

CLI:

```powershell
.\transcribator.bat --list-models
```

Web:

```powershell
.\transcribator-web.bat
```

GPU для `pyannote`:

```powershell
.\.venv\Scripts\python.exe -c "import transcribator.diarization as d; print(d.get_pyannote_runtime_info())"
```

Если всё настроено правильно, в выводе должно быть:

- `pyannote_gpu_available: True`
- `pyannote_torch_cuda_available: True`

## Лицензия

MIT
