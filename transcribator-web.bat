@echo off
setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Transcribator is not installed yet.
  echo Run .\install.ps1 from the project root first.
  exit /b 1
)

"%PYTHON%" -m transcribator.webapp %*
