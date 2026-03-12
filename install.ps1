#Requires -Version 5.1
<#
.SYNOPSIS
    Первичная полная установка Transcribator: Python-окружение, зависимости, FFmpeg.

.DESCRIPTION
    Скрипт проверяет Python 3.8+, создаёт виртуальное окружение (по умолчанию),
    устанавливает зависимости из requirements.txt и пакет в режиме разработки,
    опционально — pyannote.audio, проверяет/устанавливает FFmpeg.

.PARAMETER NoVenv
    Не создавать venv, использовать системный Python (не рекомендуется).

.PARAMETER InstallPyannote
    Установить pyannote.audio для разбивки по спикерам (без запроса).

.PARAMETER SkipPyannote
    Не предлагать установку pyannote.audio.

.PARAMETER SkipFfmpegCheck
    Не проверять и не устанавливать FFmpeg.

.PARAMETER Quiet
    Минимум вывода, только ошибки и итог.

.EXAMPLE
    .\install.ps1
    Полная установка с запросом по venv и pyannote.

.EXAMPLE
    .\install.ps1 -InstallPyannote
    Установка с pyannote.audio без запроса.

.EXAMPLE
    .\install.ps1 -NoVenv -SkipPyannote
    Установка в системный Python без pyannote.
#>

[CmdletBinding()]
param(
    [switch] $NoVenv,
    [switch] $InstallPyannote,
    [switch] $SkipPyannote,
    [switch] $SkipFfmpegCheck,
    [switch] $Quiet
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Step { param($Message) if (-not $Quiet) { Write-Host $Message -ForegroundColor Cyan } }
function Write-Ok    { param($Message) if (-not $Quiet) { Write-Host $Message -ForegroundColor Green } }
function Write-Warn  { param($Message) if (-not $Quiet) { Write-Host $Message -ForegroundColor Yellow } }
function Write-Err   { param($Message) Write-Host $Message -ForegroundColor Red }

$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# ----- 1. Python -----
Write-Step "`n=== 1. Проверка Python ==="
$py = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $info = & $cmd -c "import sys; print(sys.version_info.major, sys.version_info.minor)" 2>$null
        if ($info) {
            $py = $cmd
            $major, $minor = $info.Trim().Split()
            break
        }
    } catch { continue }
}

if (-not $py) {
    Write-Err "Python не найден. Установите Python 3.8+ с https://www.python.org/downloads/ или через winget: winget install Python.Python.3.12"
    exit 1
}

if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 8)) {
    Write-Err "Требуется Python 3.8 или выше. Найдено: $major.$minor"
    exit 1
}
Write-Ok "Python $major.$minor найден: $py"

# ----- 2. Виртуальное окружение -----
$venvPath = Join-Path $ProjectRoot ".venv"
$useVenv = -not $NoVenv

if ($useVenv) {
    Write-Step "`n=== 2. Виртуальное окружение ==="
    if (Test-Path $venvPath) {
        Write-Ok "Каталог .venv уже существует."
    } else {
        Write-Step "Создание .venv..."
        & $py -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { Write-Err "Не удалось создать venv."; exit 1 }
        Write-Ok "Создано: $venvPath"
    }
    $pythonExe = Join-Path $venvPath "Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) { Write-Err "Не найден $pythonExe"; exit 1 }
} else {
    Write-Step "`n=== 2. Системный Python (venv пропущен) ==="
    $pythonExe = $py
}

# ----- 3. pip и зависимости -----
Write-Step "`n=== 3. Зависимости ==="
& $pythonExe -m pip install --upgrade pip --quiet
$reqPath = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $reqPath)) { Write-Err "Не найден requirements.txt"; exit 1 }
& $pythonExe -m pip install -r $reqPath
if ($LASTEXITCODE -ne 0) { Write-Err "Ошибка установки зависимостей."; exit 1 }
Write-Ok "Установлены пакеты из requirements.txt"

# ----- 4. Pyannote (опционально) -----
$doPyannote = $InstallPyannote
if (-not $SkipPyannote -and -not $InstallPyannote -and -not $Quiet) {
    $r = Read-Host "Установить pyannote.audio для разбивки по спикерам? (y/N)"
    $doPyannote = ($r -match '^[yY]')
}
if ($doPyannote) {
    Write-Step "Установка pyannote.audio..."
    & $pythonExe -m pip install pyannote.audio
    if ($LASTEXITCODE -eq 0) { Write-Ok "pyannote.audio установлен." }
    else { Write-Warn "Ошибка установки pyannote.audio. Диаризация pyannote будет недоступна." }
}

# ----- 5. Установка пакета в режиме разработки -----
Write-Step "`n=== 4. Установка Transcribator ==="
& $pythonExe -m pip install -e .
if ($LASTEXITCODE -ne 0) { Write-Err "Ошибка установки пакета."; exit 1 }
Write-Ok "Пакет transcribator установлен (editable)."

# ----- 6. FFmpeg -----
if (-not $SkipFfmpegCheck) {
    Write-Step "`n=== 5. FFmpeg ==="
    $ffmpegScript = Join-Path $ProjectRoot "check_ffmpeg.ps1"
    if (Test-Path $ffmpegScript) {
        & $ffmpegScript
    } else {
        $hasFfmpeg = $false
        try { $null = Get-Command ffmpeg -ErrorAction Stop; $hasFfmpeg = $true } catch {}
        if (-not $hasFfmpeg) {
            Write-Warn "FFmpeg не найден. Установите вручную или выполните: winget install ffmpeg"
        } else {
            Write-Ok "FFmpeg найден."
        }
    }
} else {
    Write-Step "`n=== 5. FFmpeg (проверка пропущена) ==="
}

# ----- Итог -----
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Установка завершена." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if ($useVenv) {
    Write-Host "Активация окружения:" -ForegroundColor Cyan
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Запуск (из этой папки после активации):" -ForegroundColor Cyan
    Write-Host "  transcribator --list-models" -ForegroundColor White
    Write-Host "  transcribator video.mp4" -ForegroundColor White
    Write-Host ""
    Write-Host "Или без активации:" -ForegroundColor Cyan
    Write-Host "  .\.venv\Scripts\python.exe -m transcribator.cli --list-models" -ForegroundColor White
} else {
    Write-Host "Запуск:" -ForegroundColor Cyan
    Write-Host "  transcribator --list-models" -ForegroundColor White
    Write-Host "  python -m transcribator.cli video.mp4" -ForegroundColor White
}

if ($doPyannote) {
    Write-Host ""
    Write-Warn "Для диаризации pyannote примите условия на Hugging Face и укажите токен:"
    Write-Host "  https://huggingface.co/pyannote/speaker-diarization-3.1"
    Write-Host "  https://huggingface.co/pyannote/segmentation-3.0"
    Write-Host "  https://huggingface.co/pyannote/speaker-diarization-community-1"
    Write-Host "  Токен: в transcribator.json (hf_token) или --hf-token при запуске."
}
Write-Host ""
