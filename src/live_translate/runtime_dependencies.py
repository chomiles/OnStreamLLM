from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeDependency:
    name: str
    description: str
    packages: tuple[str, ...]
    required_for_slim: bool = False
    optional: bool = False
    target_dir: str = "python"


@dataclass(frozen=True, slots=True)
class RuntimeInstallSpec:
    packages: tuple[str, ...]
    index_url: str = ""
    extra_index_url: str = ""
    find_links: str = ""
    follow_up: tuple[str, ...] = ()
    follow_up_no_deps: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeDownloadInfo:
    url: str = ""
    repo_id: str = ""
    filename: str = ""
    repo_type: str = "model"
    sha256: str = ""
    size: int = 0


# Shared runtime_libraries/python folder: install in this order to avoid numpy
# downgrades, wheel source builds, and Windows file-lock conflicts.
RUNTIME_INSTALL_ORDER: tuple[str, ...] = (
    "Torch",
    "Qwen ASR",
    "llama",
    "SenseVoice",
    "PaddleOCR",
)

RUNTIME_DEPENDENCIES: tuple[RuntimeDependency, ...] = (
    RuntimeDependency(
        name="Torch",
        description="QWEN 계열 모델과 NVIDIA RTX GPU 가속을 위한 라이브러리",
        packages=("torch",),
    ),
    RuntimeDependency(
        name="Qwen ASR",
        description="Qwen3-ASR 음성 인식과 번역 파이프라인을 구동하기 위한 라이브러리",
        packages=("qwen_asr", "transformers"),
    ),
    RuntimeDependency(
        name="llama",
        description="Hy-MT2 같은 GGUF 언어 모델을 구동하기 위한 라이브러리",
        packages=("llama_cpp", "diskcache"),
        required_for_slim=True,
    ),
    RuntimeDependency(
        name="SenseVoice",
        description="SenseVoice 음성 인식을 구동하기 위한 CPU 전용 라이브러리",
        packages=("sherpa_onnx",),
        required_for_slim=True,
    ),
    RuntimeDependency(
        name="PaddleOCR",
        description="화면 번역 감지를 위한 OCR 라이브러리",
        packages=("paddleocr", "paddle"),
        optional=True,
    ),
)

# Official package sources mirrored from setup.ps1.
_RUNTIME_INSTALL_SPECS_CUDA: dict[str, RuntimeInstallSpec] = {
    "Torch": RuntimeInstallSpec(
        packages=("torch",),
        index_url="https://download.pytorch.org/whl/cu128",
    ),
    "Qwen ASR": RuntimeInstallSpec(
        packages=("transformers==4.57.6",),
        follow_up=(
            "accelerate==1.12.0",
            "soynlp==0.0.493",
            "qwen-omni-utils",
            "librosa",
            "soundfile",
        ),
        follow_up_no_deps=("qwen-asr>=0.0.4",),
    ),
    "llama": RuntimeInstallSpec(
        packages=(),
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
        follow_up=(
            "diskcache>=5.6.1",
            "jinja2>=2.11.3",
        ),
        follow_up_no_deps=("llama-cpp-python==0.3.30",),
    ),
    "SenseVoice": RuntimeInstallSpec(packages=("sherpa-onnx>=1.13.2",)),
    "PaddleOCR": RuntimeInstallSpec(
        packages=("paddlepaddle>=3.2", "paddleocr>=3.7"),
        follow_up=(
            "pillow>=10.4",
            "lxml>=5.3",
            "openpyxl>=3.1",
            "premailer>=3.10",
            "python-docx>=1.1",
            "beautifulsoup4>=4.12",
        ),
    ),
}

_RUNTIME_INSTALL_SPECS_CPU: dict[str, RuntimeInstallSpec] = {
    "Torch": RuntimeInstallSpec(packages=("torch>=2.5",)),
    "Qwen ASR": RuntimeInstallSpec(
        packages=("transformers==4.57.6",),
        follow_up=(
            "accelerate==1.12.0",
            "soynlp==0.0.493",
            "qwen-omni-utils",
            "librosa",
            "soundfile",
        ),
        follow_up_no_deps=("qwen-asr>=0.0.4",),
    ),
    "llama": RuntimeInstallSpec(
        packages=(),
        follow_up=(
            "diskcache>=5.6.1",
            "jinja2>=2.11.3",
        ),
        follow_up_no_deps=("llama-cpp-python==0.3.30",),
    ),
    "SenseVoice": RuntimeInstallSpec(packages=("sherpa-onnx>=1.13.2",)),
    "PaddleOCR": RuntimeInstallSpec(
        packages=("paddlepaddle>=3.2", "paddleocr>=3.7"),
        follow_up=(
            "pillow>=10.4",
            "lxml>=5.3",
            "openpyxl>=3.1",
            "premailer>=3.10",
            "python-docx>=1.1",
            "beautifulsoup4>=4.12",
        ),
    ),
}


def runtime_dependencies_in_install_order() -> tuple[RuntimeDependency, ...]:
    by_name = {dependency.name: dependency for dependency in RUNTIME_DEPENDENCIES}
    ordered: list[RuntimeDependency] = []
    for name in RUNTIME_INSTALL_ORDER:
        dependency = by_name.get(name)
        if dependency is not None:
            ordered.append(dependency)
    for dependency in RUNTIME_DEPENDENCIES:
        if dependency.name not in RUNTIME_INSTALL_ORDER:
            ordered.append(dependency)
    return tuple(ordered)


def runtime_dependencies_pending_install(
    *,
    include_optional: bool = False,
) -> tuple[RuntimeDependency, ...]:
    return tuple(
        dependency
        for dependency in runtime_dependencies_in_install_order()
        if include_optional or not dependency.optional
        if not runtime_dependency_installed(dependency)
        and runtime_install_available(dependency.name)
    )


def all_runtime_dependencies_installed() -> bool:
    return all(
        runtime_dependency_installed(dependency)
        for dependency in RUNTIME_DEPENDENCIES
        if not dependency.optional
    )


def runtime_dependency_installed(dependency: RuntimeDependency) -> bool:
    available = _runtime_packages_available(dependency)
    marker = _runtime_marker_path(dependency.name)
    if not available:
        if marker.is_file():
            marker.unlink(missing_ok=True)
        return False
    return True


def installed_runtime_dependency_names(*, exclude: str = "") -> tuple[str, ...]:
    names: list[str] = []
    for dependency in RUNTIME_DEPENDENCIES:
        if dependency.name == exclude:
            continue
        if runtime_dependency_installed(dependency):
            names.append(dependency.name)
    return tuple(names)


def _runtime_packages_available(dependency: RuntimeDependency) -> bool:
    from .runtime_bootstrap import configure_runtime_paths

    configure_runtime_paths()
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(package) is not None for package in dependency.packages)


def runtime_install_spec(name: str, *, cuda: bool | None = None) -> RuntimeInstallSpec | None:
    if cuda is None:
        from .runtime_installer import nvidia_gpu_available

        cuda = nvidia_gpu_available()
    specs = _RUNTIME_INSTALL_SPECS_CUDA if cuda else _RUNTIME_INSTALL_SPECS_CPU
    return specs.get(name)


def runtime_install_available(name: str) -> bool:
    if runtime_install_spec(name) is not None:
        return True
    info = runtime_download_info(name)
    return bool(info.url or (info.repo_id and info.filename))


def runtime_download_info(name: str) -> RuntimeDownloadInfo:
    entry = _runtime_manifest().get(name, {})
    if not isinstance(entry, dict):
        return RuntimeDownloadInfo()
    return RuntimeDownloadInfo(
        url=str(entry.get("url", "")).strip(),
        repo_id=str(entry.get("repo_id", "")).strip(),
        filename=str(entry.get("filename", "")).strip(),
        repo_type=str(entry.get("repo_type", "model") or "model").strip(),
        sha256=str(entry.get("sha256", "")).strip().lower(),
        size=int(entry.get("size", 0) or 0),
    )


def portable_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def runtime_library_root() -> Path:
    return portable_app_root() / "runtime_libraries"


def runtime_install_root(dependency: RuntimeDependency) -> Path:
    return runtime_library_root() / dependency.target_dir


def mark_runtime_installed(name: str) -> None:
    marker = _runtime_marker_path(name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("installed\n", encoding="utf-8")


def clear_runtime_install_state(
    dependency: RuntimeDependency,
    *,
    reinstall: bool = False,
) -> None:
    marker = _runtime_marker_path(dependency.name)
    if marker.is_file():
        marker.unlink(missing_ok=True)
    if not reinstall:
        return
    from .runtime_bootstrap import release_runtime_libraries
    from .runtime_installer import terminate_active_install_processes

    terminate_active_install_processes()
    release_runtime_libraries()
    target = runtime_install_root(dependency)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if dependency.target_dir == "python":
        _clear_all_runtime_markers()


def _clear_all_runtime_markers() -> None:
    marker_dir = runtime_library_root() / ".installed"
    if not marker_dir.is_dir():
        return
    for marker in marker_dir.glob("*.txt"):
        marker.unlink(missing_ok=True)


def _runtime_marker_path(name: str) -> Path:
    return runtime_library_root() / ".installed" / f"{name.lower()}.txt"


def _runtime_manifest() -> dict[str, object]:
    root = portable_app_root()
    config_root = root / "Config"
    legacy_manifest = root / "runtime_manifest.json"
    if not (config_root / "runtime_manifest.json").exists() and legacy_manifest.is_file():
        config_root.mkdir(parents=True, exist_ok=True)
        legacy_manifest.replace(config_root / "runtime_manifest.json")
    candidates = [
        config_root / "runtime_manifest.json",
        legacy_manifest,
        Path.cwd() / "Config" / "runtime_manifest.json",
        Path.cwd() / "runtime_manifest.json",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}
