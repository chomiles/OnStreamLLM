$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$LocalPython = Join-Path $ProjectRoot ".python312\python.exe"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Installer = Join-Path $env:TEMP "python-3.12.10-amd64.exe"
$InstallerUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"

if (-not (Test-Path $LocalPython)) {
    Write-Host "Downloading Python 3.12.10 from python.org..."
    Invoke-WebRequest -Uri $InstallerUrl -OutFile $Installer

    Write-Host "Installing a private Python 3.12 runtime in this project..."
    $Install = Start-Process -FilePath $Installer -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "Include_launcher=0",
        "Include_test=0",
        "Include_doc=0",
        "Include_pip=1",
        "Include_tcltk=1",
        "Shortcuts=0",
        "PrependPath=0",
        "TargetDir=$ProjectRoot\.python312"
    ) -Wait -PassThru

    if ($Install.ExitCode -ne 0 -or -not (Test-Path $LocalPython)) {
        throw "Private Python 3.12 installation failed with exit code $($Install.ExitCode)."
    }
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the virtual environment..."
    & $LocalPython -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        throw "Virtual environment creation failed."
    }
}

Write-Host "Installing application packages..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}
& $VenvPython -m pip install -e ".[ocr,dev]"
if ($LASTEXITCODE -ne 0) {
    throw "Application package installation failed."
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU detected. Installing CUDA-enabled PyTorch..."
    & $VenvPython -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA-enabled PyTorch installation failed."
    }

    Write-Host "Installing CUDA-enabled sherpa-onnx for SenseVoice..."
    & $VenvPython -m pip install --upgrade --force-reinstall "sherpa-onnx>=1.13.2"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA-enabled sherpa-onnx installation failed."
    }

    Write-Host "Installing CUDA-enabled llama-cpp-python for GGUF translation..."
    & $VenvPython -m pip install --upgrade --force-reinstall "llama-cpp-python==0.3.30" --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA-enabled llama-cpp-python installation failed."
    }

    & $VenvPython -m pip install --force-reinstall "numpy>=1.26,<2.4"
    if ($LASTEXITCODE -ne 0) {
        throw "NumPy compatibility pin failed."
    }
}

Write-Host ""
Write-Host "Setup complete. Run run.bat to start the application."
