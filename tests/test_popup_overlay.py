from PySide6.QtWidgets import QApplication

from live_translate.models import AppSettings
from live_translate.ui import POPUP_OPACITY_OPTIONS, TransparentCaptionPopup


def test_transparent_popup_has_overlay_controls(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    settings = AppSettings()
    popup = TransparentCaptionPopup(
        settings,
        translate=lambda key, **kwargs: key,
        on_settings_changed=lambda: None,
    )

    assert popup.lock_checkbox.isChecked() is False
    assert popup.font_combo.current_font_family() == "Malgun Gothic"
    assert popup.font_size_combo.currentData() == 32
    assert popup.opacity_combo.currentData() == 80
    assert 80 in POPUP_OPACITY_OPTIONS
    assert popup.windowOpacity() == 0.8
    popup.close()


def test_popup_lock_disables_resize(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QApplication.instance() or QApplication([])
    settings = AppSettings()
    popup = TransparentCaptionPopup(
        settings,
        translate=lambda key, **kwargs: key,
        on_settings_changed=lambda: None,
    )
    popup.show()

    popup.resize(500, 300)
    locked_size = popup.size()
    popup.lock_checkbox.setChecked(True)
    assert popup._grip_row.isVisible() is False
    assert popup._size_grip.isEnabled() is False
    assert popup.font_combo.isEnabled() is False
    assert popup.font_size_combo.isEnabled() is False
    assert popup.opacity_combo.isEnabled() is False
    assert popup.lock_checkbox.isEnabled() is True
    assert popup.size() == locked_size

    popup.lock_checkbox.setChecked(False)
    QApplication.processEvents()
    assert popup._grip_row.isVisible() is True
    assert popup._size_grip.isEnabled() is True
    assert popup.font_combo.isEnabled() is True
    assert popup.font_size_combo.isEnabled() is True
    assert popup.opacity_combo.isEnabled() is True
    assert popup.minimumSize().width() == 360
    popup.close()