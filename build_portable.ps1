param(
    [ValidateSet("full", "slim", "sensevoice-hymt2", "download-runtime")]
    [string]$Profile = "download-runtime"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Run setup.ps1 before building."
}

& $Python -m pip install "pyinstaller>=6.13,<7"
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller installation failed."
}

$env:LTS_BUILD_PROFILE = $Profile
& $Python -m PyInstaller --noconfirm --clean live_translate.spec
if ($LASTEXITCODE -ne 0) {
    throw "Portable build failed."
}

$DistDir = Join-Path $ProjectRoot "dist\OnStreamLLM"

if ($Profile -eq "download-runtime") {
    $Internal = Resolve-Path (Join-Path $DistDir "_internal")
    foreach ($Name in @(
        "torch",
        "transformers",
        "qwen_asr",
        "paddle",
        "paddleocr",
        "paddlex",
        "cv2",
        "llama_cpp",
        "sherpa_onnx",
        "onnxruntime",
        "scipy",
        "scipy.libs",
        "pandas",
        "pandas.libs",
        "Cython",
        "hf_xet",
        "PIL",
        "lxml",
        "gradio",
        "librosa",
        "sklearn",
        "soynlp",
        "tokenizers",
        "safetensors"
    )) {
        $Target = Join-Path $Internal $Name
        if (Test-Path $Target) {
            $Resolved = Resolve-Path $Target
            if (-not $Resolved.Path.StartsWith($Internal.Path)) {
                throw "Refusing to clean unexpected path: $Resolved"
            }
            Remove-Item -LiteralPath $Resolved.Path -Recurse -Force
        }
    }
}

$ConfigDist = Join-Path $DistDir "Config"
New-Item -ItemType Directory -Force -Path $ConfigDist | Out-Null
$BundledSettings = Join-Path $ConfigDist "settings.json"
if (Test-Path $BundledSettings) {
    Remove-Item $BundledSettings -Force
}

Copy-Item README_FIELD_TEST.md (Join-Path $DistDir "README_FIELD_TEST.md") -Force
Copy-Item Readme.txt (Join-Path $DistDir "Readme.txt") -Force
$DistInfo = Join-Path $DistDir "info.txt"
if (Test-Path $DistInfo) {
    Remove-Item $DistInfo -Force
}
Copy-Item "Config\download_sources.json" (Join-Path $ConfigDist "download_sources.json") -Force
Copy-Item "Config\runtime_manifest.example.json" (Join-Path $ConfigDist "runtime_manifest.example.json") -Force
Write-Host "Runtime libraries are installed from official pip routes inside the app UI."
if (Test-Path "Config\runtime_manifest.json") {
    Copy-Item "Config\runtime_manifest.json" (Join-Path $ConfigDist "runtime_manifest.json") -Force
}
Write-Host "Build complete: dist\OnStreamLLM\OnStreamLLM.exe"
