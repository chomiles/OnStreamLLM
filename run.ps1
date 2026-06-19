$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "먼저 .\setup.ps1 을 실행하세요."
}

& .\.venv\Scripts\python.exe -m live_translate.main

