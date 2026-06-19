from __future__ import annotations

import json
from pathlib import Path

from .config import config_dir


def runtime_install_queue_path() -> Path:
    return config_dir() / "runtime_install_queue.json"


def save_runtime_install_queue(queue: list[tuple[str, bool]]) -> None:
    path = runtime_install_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"name": name, "reinstall": reinstall} for name, reinstall in queue]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_runtime_install_queue() -> list[tuple[str, bool]]:
    path = runtime_install_queue_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    queue: list[tuple[str, bool]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        queue.append((name, bool(item.get("reinstall", False))))
    return queue


def clear_runtime_install_queue() -> None:
    path = runtime_install_queue_path()
    if path.is_file():
        path.unlink(missing_ok=True)