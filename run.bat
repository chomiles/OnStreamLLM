@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

if not exist ".venv\Scripts\python.exe" (
    echo [Live Translate Studio]
    echo First-time setup is starting.
    echo A private Python 3.12 runtime will be installed in this folder.
    echo Your existing Python installation will not be changed.
    echo This may take several minutes.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
    if errorlevel 1 (
        echo.
        echo Setup failed.
        echo Check your internet connection and the error shown above.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m live_translate.main

if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
)

endlocal
