# Pyannote GPU в Transcribator

`pyannote` в проекте остаётся opt-in режимом для speaker diarization, а GPU-путь для него — отдельным overlay поверх базовой CPU-first установки.

## Базовая идея

- `uv sync` ставит проект в CPU-first конфигурации
- если нужен `pyannote + CUDA`, поверх неё нужно поставить CUDA-сборки `torch` и `torchaudio`
- после этого запускать проект лучше через wrapper-скрипты или через `.venv\Scripts\python.exe`, а не через обычный `uv run`

## Рекомендуемый путь

Самый простой сценарий:

```powershell
.\install.ps1
```

Во время установки скрипт предложит:

- нужен ли `pyannote`
- нужен ли GPU overlay для `pyannote`

Если выбрать GPU overlay, bootstrap сам установит CUDA-сборки `torch/torchaudio`.

## Ручной GPU overlay

Если базовая установка уже сделана:

```powershell
uv pip install --python .\.venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 --reinstall torch==2.10.0 torchaudio==2.10.0
```

## Проверка

Проверить сам `torch`:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Ожидаемый результат:

- версия `torch` не оканчивается на `+cpu`
- `torch.version.cuda` не `None`
- `torch.cuda.is_available()` возвращает `True`

Проверить готовность `pyannote`:

```powershell
.\.venv\Scripts\python.exe -c "import transcribator.diarization as d; print(d.get_pyannote_runtime_info())"
```

Ожидаемые поля:

- `pyannote_available: True`
- `pyannote_gpu_available: True`
- `pyannote_torch_cuda_available: True`

## Как запускать после GPU overlay

Каноничные варианты:

```powershell
.\transcribator-web.bat
.\transcribator.bat "D:\path\to\video.mp4" --diarize pyannote --diarization-device cuda --hf-token hf_xxx
```

Или напрямую:

```powershell
.\.venv\Scripts\python.exe -m transcribator.webapp
.\.venv\Scripts\python.exe -m transcribator.cli --diarize pyannote --diarization-device cuda
```

## Почему не обычный uv run

Сейчас lock-файл проекта остаётся CPU-first. Поэтому обычный:

```powershell
uv run transcribator-web
```

может вернуть `torch/torchaudio` к CPU-версии из lock-файла.

Если нужен запуск через `uv`, используй:

```powershell
uv run --no-sync transcribator-web
```

Но для Windows-практики проще и надёжнее использовать wrapper-скрипты из корня проекта.
