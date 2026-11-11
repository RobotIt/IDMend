@echo off
setlocal
set "TRACK_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "TRACK_PYTHON=%~dp0.venv\Scripts\python.exe"
"%TRACK_PYTHON%" -I "%~dp0smoke_test.py" %*
endlocal & exit /b %ERRORLEVEL%
