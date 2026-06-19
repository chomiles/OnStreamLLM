from __future__ import annotations

import sys
from pathlib import Path

from live_translate.config import app_icon_path, app_root


def test_app_icon_path_points_to_project_icon() -> None:
    root = app_root()
    icon = app_icon_path()
    assert icon is not None
    assert icon == root / "icon.ico"
    assert icon.is_file()


def test_app_icon_path_uses_executable_icon_when_frozen(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    icon = app_icon_path()
    assert icon is not None
    assert icon == Path(sys.executable).resolve()
