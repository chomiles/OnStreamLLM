from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from live_translate.ui import SearchableFontCombo, windows_font_families


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_windows_font_families_includes_common_fonts(qt_app: QApplication) -> None:
    families = windows_font_families()
    assert families
    assert "Malgun Gothic" in families or "Arial" in families


def test_searchable_font_combo_keeps_unknown_saved_font(qt_app: QApplication) -> None:
    combo = SearchableFontCombo("Custom Saved Font")
    assert combo.current_font_family() == "Custom Saved Font"
    assert combo.findText("Custom Saved Font") == 0