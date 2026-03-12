# Скрипт для проверки и установки FFmpeg на Windows
# Использование: .\check_ffmpeg.ps1
# Установка кодировки UTF-8 для корректного отображения русских символов
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=== Проверка FFmpeg для Transcribator ===" -ForegroundColor Cyan
Write-Host ""

# Функция для проверки наличия команды в PATH
function Test-Command {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

# Проверка наличия FFmpeg
Write-Host "Проверка наличия FFmpeg..." -ForegroundColor Yellow
$ffmpegExists = Test-Command "ffmpeg"

if ($ffmpegExists) {
    Write-Host "[OK] FFmpeg найден в системе!" -ForegroundColor Green
    $version = & ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "  Версия: $version" -ForegroundColor Gray
    Write-Host ""
    Write-Host "FFmpeg готов к использованию с Transcribator." -ForegroundColor Green
    exit 0
}

Write-Host "[X] FFmpeg не найден в системе." -ForegroundColor Red
Write-Host ""

# Проверка наличия Chocolatey
Write-Host "Проверка наличия Chocolatey..." -ForegroundColor Yellow
$chocoExists = Test-Command "choco"

if ($chocoExists) {
    Write-Host "[OK] Chocolatey найден!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Хотите установить FFmpeg через Chocolatey? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq "Y" -or $response -eq "y" -or $response -eq "Yes" -or $response -eq "yes") {
        Write-Host ""
        Write-Host "Установка FFmpeg через Chocolatey..." -ForegroundColor Yellow
        
        # Проверка прав администратора
        $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        
        if (-not $isAdmin) {
            Write-Host "[!] Требуются права администратора для установки." -ForegroundColor Yellow
            Write-Host "Запустите PowerShell от имени администратора и выполните:" -ForegroundColor Yellow
            Write-Host "  choco install ffmpeg -y" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Или скачайте FFmpeg вручную с https://ffmpeg.org/download.html" -ForegroundColor Yellow
            exit 1
        }
        
        try {
            & choco install ffmpeg -y
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "[OK] FFmpeg успешно установлен!" -ForegroundColor Green
                Write-Host ""
                Write-Host "[!] Может потребоваться перезапустить терминал для обновления PATH." -ForegroundColor Yellow
                Write-Host ""
                
                # Проверка после установки
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
                Start-Sleep -Seconds 2
                
                $ffmpegExistsAfter = Test-Command "ffmpeg"
                if ($ffmpegExistsAfter) {
                    Write-Host "[OK] FFmpeg доступен и готов к использованию!" -ForegroundColor Green
                    exit 0
                }
                else {
                    Write-Host "[!] FFmpeg установлен, но не найден в PATH. Перезапустите терминал." -ForegroundColor Yellow
                    exit 0
                }
            }
            else {
                Write-Host "[X] Ошибка при установке FFmpeg через Chocolatey." -ForegroundColor Red
                exit 1
            }
        }
        catch {
            Write-Host "[X] Ошибка при установке: $_" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host ""
        Write-Host "Установка отменена." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Для ручной установки:" -ForegroundColor Yellow
        Write-Host "1. Скачайте FFmpeg с https://ffmpeg.org/download.html" -ForegroundColor Cyan
        Write-Host "2. Распакуйте архив" -ForegroundColor Cyan
        Write-Host "3. Добавьте папку 'bin' в переменную окружения PATH" -ForegroundColor Cyan
        Write-Host "4. Перезапустите терминал" -ForegroundColor Cyan
        exit 1
    }
}
else {
    Write-Host "[X] Chocolatey не найден." -ForegroundColor Red
    Write-Host ""
    Write-Host "Варианты установки FFmpeg:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Вариант 1: Установить через Chocolatey (рекомендуется)" -ForegroundColor Cyan
    Write-Host "  1. Установите Chocolatey: https://chocolatey.org/install" -ForegroundColor Gray
    Write-Host "  2. Запустите этот скрипт снова" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Вариант 2: Ручная установка" -ForegroundColor Cyan
    Write-Host "  1. Скачайте FFmpeg с https://ffmpeg.org/download.html" -ForegroundColor Gray
    Write-Host "  2. Распакуйте архив (например, в C:\ffmpeg)" -ForegroundColor Gray
    Write-Host "  3. Добавьте C:\ffmpeg\bin в переменную PATH:" -ForegroundColor Gray
    Write-Host "     - Откройте 'Система' -> 'Дополнительные параметры системы'" -ForegroundColor Gray
    Write-Host "     - Нажмите 'Переменные среды'" -ForegroundColor Gray
    Write-Host "     - Найдите 'Path' в системных переменных и отредактируйте" -ForegroundColor Gray
    Write-Host "     - Добавьте путь к папке bin FFmpeg" -ForegroundColor Gray
    Write-Host "  4. Перезапустите терминал" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Вариант 3: Использовать winget (Windows 10/11)" -ForegroundColor Cyan
    Write-Host "  winget install ffmpeg" -ForegroundColor Gray
    Write-Host ""
    exit 1
}
