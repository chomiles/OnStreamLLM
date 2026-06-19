from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)
_EXCLUDED_STDLIB_DIRS = {"site-packages", "__pycache__"}
_SHADOWABLE_STDLIB_PACKAGES = {
    "email",
    "html",
    "http",
    "json",
    "logging",
    "urllib",
    "xml",
    "xmlrpc",
}


def configure_runtime_paths() -> None:
    ensure_runtime_stdlib_shims()
    root = _runtime_library_root()
    python_root = root / "python"
    if python_root.is_dir():
        os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
        resolved_python_root = python_root.resolve()
        sys.path[:] = [
            entry
            for entry in sys.path
            if Path(entry).resolve() != resolved_python_root
        ]
        sys.path.insert(0, str(python_root))
        _release_shadowed_runtime_imports(resolved_python_root)
    if hasattr(os, "add_dll_directory"):
        for candidate in _dll_candidates(root):
            try:
                os.add_dll_directory(str(candidate))
            except OSError:
                continue
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"


def _release_shadowed_runtime_imports(python_root: Path) -> None:
    removed: list[str] = []
    for name, module in list(sys.modules.items()):
        if not (
            name in {"setuptools", "pkg_resources", "distutils", "_distutils_hack"}
            or name.startswith("setuptools.")
            or name.startswith("pkg_resources.")
            or name.startswith("distutils.")
            or name.startswith("_distutils_hack.")
            or name.split(".", 1)[0] in _SHADOWABLE_STDLIB_PACKAGES
        ):
            continue
        if module is not None and _module_is_under(module, python_root):
            continue
        removed.append(name)
        del sys.modules[name]
    if removed:
        LOGGER.info(
            "Released %d shadowed runtime import(s): %s",
            len(removed),
            ", ".join(sorted(removed)[:8]),
        )


def release_runtime_libraries(*, wait_seconds: float = 1.0) -> None:
    root = _runtime_library_root()
    python_root = (root / "python").resolve()
    python_root_text = str(python_root)

    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry).resolve() != python_root
    ]

    removed: list[str] = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        module_paths: list[str] = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            module_paths.append(module_file)
        module_path = getattr(module, "__path__", None)
        if module_path is not None:
            try:
                module_paths.extend(str(path) for path in module_path)
            except Exception:
                pass
        if any(path and Path(path).resolve().is_relative_to(python_root) for path in module_paths):
            removed.append(name)
            del sys.modules[name]

    gc.collect()
    if wait_seconds > 0 and sys.platform == "win32":
        time.sleep(wait_seconds)
    if removed:
        LOGGER.info(
            "Released %d runtime module(s) from %s",
            len(removed),
            python_root_text,
        )


def ensure_runtime_stdlib_shims() -> None:
    python_root = _runtime_library_root() / "python"
    if not python_root.is_dir():
        return
    lib_dir = _portable_python_lib_dir()
    if lib_dir is None:
        return
    python_root.mkdir(parents=True, exist_ok=True)
    for source in lib_dir.glob("*.py"):
        destination = python_root / source.name
        if destination.is_file():
            continue
        shutil.copy2(source, destination)
        LOGGER.info("Copied runtime stdlib shim: %s", source.name)
    for source in lib_dir.iterdir():
        if not source.is_dir() or source.name in _EXCLUDED_STDLIB_DIRS:
            continue
        destination = python_root / source.name
        if destination.is_dir():
            continue
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
        LOGGER.info("Copied runtime stdlib package shim: %s", source.name)
    _ensure_runtime_distutils_shim(python_root)


def _ensure_runtime_distutils_shim(python_root: Path) -> None:
    source = python_root / "setuptools" / "_distutils"
    destination = python_root / "distutils"
    if not source.is_dir() or destination.is_dir():
        return
    shutil.copytree(source, destination)
    LOGGER.info("Copied runtime distutils shim: %s", destination)


def _module_is_under(module: object, root: Path) -> bool:
    module_paths: list[str] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        module_paths.append(str(module_file))
    module_path = getattr(module, "__path__", None)
    if module_path is not None:
        try:
            module_paths.extend(str(path) for path in module_path)
        except Exception:
            pass
    for path in module_paths:
        try:
            if Path(path).resolve().is_relative_to(root):
                return True
        except OSError:
            continue
    return False


def _portable_python_lib_dir() -> Path | None:
    from .runtime_dependencies import portable_app_root

    app_root = portable_app_root()
    for relative in (
        "_internal/runtime_stdlib",
        "runtime_stdlib",
        "python312/Lib",
        ".python312/Lib",
    ):
        candidate = app_root / relative
        if candidate.is_dir():
            return candidate
    if not getattr(sys, "frozen", False):
        import sysconfig

        return Path(sysconfig.get_path("stdlib"))
    return None


def _runtime_library_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "runtime_libraries"
    return Path.cwd() / "runtime_libraries"


def _dll_candidates(root: Path) -> list[Path]:
    candidates = [
        root / "python",
        root / "python" / "torch" / "lib",
        root / "python" / "llama_cpp" / "lib",
        root / "python" / "sherpa_onnx" / "lib",
        root / "python" / "paddle" / "libs",
        root / "runtime" / "llama.cpp" / "build-cuda" / "bin" / "Release",
        root / "runtime" / "llama.cpp" / "build-stq" / "bin" / "Release",
        root / "runtime" / "llama.cpp" / "build" / "bin" / "Release",
    ]
    return [candidate for candidate in candidates if candidate.is_dir()]
