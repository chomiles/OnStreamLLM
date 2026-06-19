from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _models_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        if executable_dir.parent.name.lower() == "dist":
            return executable_dir.parent.parent / "models"
        return executable_dir / "models"
    return Path.cwd() / "models"


MODELS_ROOT = _models_root()
STT_ROOT = MODELS_ROOT / "STT"
LLM_ROOT = MODELS_ROOT / "LLM"
MT_ROOT = MODELS_ROOT / "MT"
OCR_ROOT = MODELS_ROOT / "OCR" / "models"


def ensure_model_folders() -> None:
    STT_ROOT.mkdir(parents=True, exist_ok=True)
    LLM_ROOT.mkdir(parents=True, exist_ok=True)
    MT_ROOT.mkdir(parents=True, exist_ok=True)
    OCR_ROOT.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_models()


def model_root(kind: str) -> Path:
    if kind == "asr":
        return STT_ROOT
    if kind == "translation":
        return LLM_ROOT
    if kind == "ocr":
        return OCR_ROOT
    return MT_ROOT


def model_download_path(kind: str, model_id: str) -> Path:
    return model_root(kind) / model_id.replace("/", "--")


def discover_models(kind: str) -> list[Path]:
    ensure_model_folders()
    return sorted(
        (path.resolve() for path in model_root(kind).iterdir() if is_model_complete(path)),
        key=lambda path: path.name.lower(),
    )


def display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def is_model_complete(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    if next(path.rglob("*.gguf"), None):
        return True
    if next(path.rglob("*.onnx"), None):
        return True
    if next(path.glob("inference.pdiparams"), None) or next(path.rglob("inference.pdiparams"), None):
        return True
    if not (path / "config.json").is_file():
        return False
    weight_patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    return any(next(path.glob(pattern), None) for pattern in weight_patterns)


def repository_id_from_path(path: str | Path) -> str | None:
    name = Path(path).name
    if "--" not in name:
        return None
    owner, repository = name.split("--", 1)
    return f"{owner}/{repository}" if owner and repository else None


def delete_model(kind: str, path: str | Path) -> None:
    root = model_root(kind).resolve()
    target = Path(path).resolve()
    if target.parent != root or not target.is_dir():
        raise ValueError("Only downloaded models inside the model folder can be deleted.")
    shutil.rmtree(target)


def _migrate_legacy_models() -> None:
    if not MODELS_ROOT.exists():
        return
    for path in tuple(MODELS_ROOT.iterdir()):
        if not path.is_dir() or path in (STT_ROOT, LLM_ROOT, MT_ROOT):
            continue
        name = path.name.lower()
        if "asr" in name:
            destination = STT_ROOT / path.name
        elif is_model_complete(path):
            destination = LLM_ROOT / path.name
        else:
            continue
        if not destination.exists():
            shutil.move(str(path), str(destination))
