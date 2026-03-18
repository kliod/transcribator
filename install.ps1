#Requires -Version 5.1
<#
.SYNOPSIS
    One-command bootstrap для установки Transcribator на Windows.

.DESCRIPTION
    Скрипт использует uv как основной менеджер окружения:
    - проверяет и при необходимости устанавливает uv
    - обеспечивает Python 3.12
    - пересоздаёт .venv
    - выполняет uv sync
    - опционально настраивает pyannote GPU overlay
    - печатает каноничные команды запуска через локальный .venv

.PARAMETER CpuOnly
    Установить только CPU-first вариант без GPU overlay для pyannote.

.PARAMETER WithPyannote
    Считать, что pyannote планируется использовать, и показать финальные шаги для HF token/access.

.PARAMETER WithPyannoteGpu
    Поверх базовой установки накатить CUDA-сборки torch/torchaudio для pyannote diarization.
    Имплицитно включает WithPyannote.

.PARAMETER NoPrompt
    Не задавать интерактивные вопросы. Все решения берутся из флагов.

.PARAMETER SkipFfmpegCheck
    Не запускать check_ffmpeg.ps1 в конце установки.

.PARAMETER Quiet
    Минимум служебного вывода.

.EXAMPLE
    .\install.ps1

.EXAMPLE
    .\install.ps1 -NoPrompt -CpuOnly

.EXAMPLE
    .\install.ps1 -NoPrompt -WithPyannote -WithPyannoteGpu
#>

[CmdletBinding()]
param(
    [switch] $CpuOnly,
    [switch] $WithPyannote,
    [switch] $WithPyannoteGpu,
    [switch] $NoPrompt,
    [switch] $SkipFfmpegCheck,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step {
    param([string] $Message)
    if (-not $Quiet) { Write-Host $Message -ForegroundColor Cyan }
}

function Write-Ok {
    param([string] $Message)
    if (-not $Quiet) { Write-Host $Message -ForegroundColor Green }
}

function Write-Warn {
    param([string] $Message)
    if (-not $Quiet) { Write-Host $Message -ForegroundColor Yellow }
}

function Write-Err {
    param([string] $Message)
    Write-Host $Message -ForegroundColor Red
}

function Confirm-Choice {
    param(
        [string] $Prompt,
        [bool] $Default = $false
    )

    if ($NoPrompt) {
        return $Default
    }

    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $raw = Read-Host "$Prompt $suffix"
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $Default
    }
    return $raw -match "^[YyАа]"
}

function Ensure-Uv {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCommand) {
        return $uvCommand.Source
    }

    Write-Step "`n=== Установка uv ==="
    Write-Step "uv не найден. Пробую установить через официальный install script..."

    try {
        & powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    } catch {
        Write-Err "Не удалось установить uv автоматически. Установите uv вручную и повторите запуск."
        Write-Err "Сайт: https://docs.astral.sh/uv/"
        exit 1
    }

    $candidatePaths = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            $candidateDir = Split-Path $candidate -Parent
            $pathItems = $env:PATH -split ';'
            if ($pathItems -notcontains $candidateDir) {
                $env:PATH = "$candidateDir;$env:PATH"
            }
            return $candidate
        }
    }

    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCommand) {
        return $uvCommand.Source
    }

    Write-Err "uv установлен, но не найден в PATH. Перезапустите PowerShell и повторите запуск."
    exit 1
}

function Get-TorchBaseVersion {
    param([string] $PythonExe)
    return (& $PythonExe -c "import torch; print(torch.__version__.split('+')[0])").Trim()
}

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

if ($WithPyannoteGpu) {
    $WithPyannote = $true
}
if ($CpuOnly) {
    $WithPyannoteGpu = $false
}

if (-not $NoPrompt) {
    if (-not $WithPyannote -and -not $WithPyannoteGpu) {
        $WithPyannote = Confirm-Choice "Планируете использовать pyannote для diarization?" $false
    }
    if ($WithPyannote -and -not $CpuOnly -and -not $WithPyannoteGpu) {
        $WithPyannoteGpu = Confirm-Choice "Нужен GPU overlay для pyannote diarization?" $false
    }
}

Write-Step "`n=== 1. uv ==="
$uvExe = Ensure-Uv
Write-Ok "uv готов: $uvExe"

Write-Step "`n=== 2. Python 3.12 ==="
& $uvExe python install 3.12
if ($LASTEXITCODE -ne 0) {
    Write-Err "Не удалось подготовить Python 3.12 через uv."
    exit 1
}
Write-Ok "Python 3.12 подготовлен."

Write-Step "`n=== 3. Виртуальное окружение ==="
& $uvExe venv --python 3.12 --clear .venv
if ($LASTEXITCODE -ne 0) {
    Write-Err "Не удалось пересоздать .venv на Python 3.12."
    exit 1
}

$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Err "Не найден $pythonExe"
    exit 1
}
Write-Ok ".venv создан."

Write-Step "`n=== 4. Базовая установка проекта ==="
& $uvExe sync
if ($LASTEXITCODE -ne 0) {
    Write-Err "uv sync завершился ошибкой."
    exit 1
}
Write-Ok "Базовая CPU-first установка завершена."

if ($WithPyannoteGpu) {
    Write-Step "`n=== 5. GPU overlay для pyannote ==="
    $torchVersion = Get-TorchBaseVersion -PythonExe $pythonExe
    Write-Step "Устанавливаю CUDA-сборки torch/torchaudio версии $torchVersion..."
    & $uvExe pip install `
        --python $pythonExe `
        --index-url https://download.pytorch.org/whl/cu128 `
        --reinstall `
        "torch==$torchVersion" `
        "torchaudio==$torchVersion"

    if ($LASTEXITCODE -ne 0) {
        Write-Err "Не удалось установить CUDA-сборки torch/torchaudio."
        exit 1
    }

    $cudaCheck = & $pythonExe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
    Write-Ok "GPU overlay установлен."
    if (-not $Quiet) {
        Write-Host $cudaCheck
    }
} else {
    Write-Step "`n=== 5. GPU overlay ==="
    Write-Ok "Пропущено. Остаёмся на CPU-first варианте."
}

if (-not $SkipFfmpegCheck) {
    Write-Step "`n=== 6. FFmpeg ==="
    $ffmpegScript = Join-Path $ProjectRoot "check_ffmpeg.ps1"
    if (Test-Path $ffmpegScript) {
        & $ffmpegScript
    } else {
        Write-Warn "check_ffmpeg.ps1 не найден. Проверьте FFmpeg вручную: ffmpeg -version"
    }
} else {
    Write-Step "`n=== 6. FFmpeg ==="
    Write-Ok "Проверка FFmpeg пропущена."
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Установка завершена" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

Write-Host "Каноничные команды запуска:" -ForegroundColor Cyan
Write-Host "  .\transcribator-web.bat" -ForegroundColor White
Write-Host "  .\transcribator.bat --list-models" -ForegroundColor White
Write-Host "  .\transcribator.bat `"D:\path\to\video.mp4`"" -ForegroundColor White
Write-Host ""

Write-Host "Fallback без wrapper-скриптов:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe -m transcribator.webapp" -ForegroundColor White
Write-Host "  .\.venv\Scripts\python.exe -m transcribator.cli --help" -ForegroundColor White
Write-Host ""

Write-Warn "Для GPU overlay не используйте обычный 'uv run ...' — он может вернуть torch/torchaudio к CPU-версии из lock-файла."
Write-Host "Безопасная альтернатива: uv run --no-sync transcribator-web" -ForegroundColor White

if ($WithPyannote) {
    Write-Host ""
    Write-Warn "Для pyannote заранее примите доступы на Hugging Face и используйте token:"
    Write-Host "  https://huggingface.co/pyannote/speaker-diarization-3.1" -ForegroundColor White
    Write-Host "  https://huggingface.co/pyannote/segmentation-3.0" -ForegroundColor White
    Write-Host "  https://huggingface.co/pyannote/speaker-diarization-community-1" -ForegroundColor White
    Write-Host ""
    Write-Host "Токен можно передать через web UI, config (hf_token) или --hf-token." -ForegroundColor White
}

Write-Host ""
