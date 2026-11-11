@echo off
setlocal
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>&1
if errorlevel 1 goto use_python
py -3.10 -m venv .venv
if errorlevel 1 goto failed
goto install
:use_python
python -c "import sys; sys.exit(sys.version_info[:2] != (3, 10))"
if errorlevel 1 goto failed
python -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
call run.cmd --check
if errorlevel 1 goto failed
popd
exit /b 0
:failed
popd
echo Setup failed. Check that Python 3.10 is installed and review the error above.
exit /b 1
