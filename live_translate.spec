# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_all, copy_metadata


datas = []
binaries = []
hiddenimports = []
build_profile = os.environ.get("LTS_BUILD_PROFILE", "full").lower()
slim_build = build_profile in {"slim", "sensevoice-hymt2"}
download_runtime_build = build_profile in {"download-runtime", "runtime-download"}

runtime_stdlib = Path(".python312/Lib")
if runtime_stdlib.is_dir():
    datas += [
        (str(path), "runtime_stdlib")
        for path in runtime_stdlib.glob("*.py")
    ]
    datas += [
        (str(path), f"runtime_stdlib/{path.name}")
        for path in runtime_stdlib.iterdir()
        if path.is_dir() and path.name not in {"site-packages", "__pycache__"}
    ]

packages = [
    "soundcard",
]

if not download_runtime_build:
    packages += [
        "sherpa_onnx",
        "llama_cpp",
    ]

if not slim_build and not download_runtime_build:
    packages += [
        "transformers",
        "qwen_asr",
        "paddleocr",
        "paddlex",
        "Cython",
        "imagesize",
        "pyclipper",
        "pypdfium2",
        "bidi",
        "shapely",
        "openpyxl",
        "premailer",
        "bs4",
        "cssselect",
        "cssutils",
    ]

for package in packages:
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

if not slim_build and not download_runtime_build:
    for distribution in (
        "imagesize",
        "opencv-contrib-python",
        "pyclipper",
        "pypdfium2",
        "python-bidi",
        "shapely",
        "openpyxl",
        "premailer",
        "beautifulsoup4",
        "cssselect",
        "cssutils",
    ):
        datas += copy_metadata(distribution)

if not slim_build and not download_runtime_build:
    import paddle

    paddle_libs = Path(paddle.__file__).resolve().parent / "libs"
    if paddle_libs.exists():
        binaries += [(str(path), "paddle/libs") for path in paddle_libs.glob("*.dll")]

llama_stq_bin = Path("runtime/llama.cpp/build-stq/bin/Release")
if llama_stq_bin.exists() and not download_runtime_build:
    binaries += [
        (str(path), "runtime/llama.cpp/build-stq/bin/Release")
        for path in llama_stq_bin.glob("*")
        if path.suffix.lower() in {".exe", ".dll"}
    ]

hiddenimports += [
    "mss",
    "pickletools",
    "uvicorn.loops.asyncio",
    "uvicorn.lifespan.on",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "websockets.legacy.server",
    "h11",
]

if not slim_build and not download_runtime_build:
    hiddenimports += [
        "torch",
        "paddle",
        "cv2",
        "qwen_asr",
        "qwen_asr.inference.qwen3_asr",
        "qwen_asr.inference.utils",
        "qwen_asr.core.transformers_backend.configuration_qwen3_asr",
        "qwen_asr.core.transformers_backend.modeling_qwen3_asr",
        "qwen_asr.core.transformers_backend.processing_qwen3_asr",
    ]

excludes = ["pytest", "ruff"]
if slim_build:
    excludes += [
        "torch",
        "transformers",
        "qwen_asr",
        "paddle",
        "paddleocr",
        "paddlex",
        "cv2",
    ]
if download_runtime_build:
    excludes += [
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
        "numba",
        "llvmlite",
        "scipy",
        "pandas",
        "Cython",
        "hf_xet",
        "PIL",
        "lxml",
        "gradio",
        "librosa",
        "scikit-learn",
        "soynlp",
        "tokenizers",
        "safetensors",
        "openpyxl",
        "premailer",
        "beautifulsoup4",
        "cssselect",
        "cssutils",
        "imagesize",
        "pyclipper",
        "pypdfium2",
        "bidi",
        "shapely",
    ]

a = Analysis(
    ["portable_main.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

_icon_file = Path("icon.ico")
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OnStreamLLM",
    console=False,
    icon=str(_icon_file) if _icon_file.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="OnStreamLLM",
)
