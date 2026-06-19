from PySide6.QtWidgets import QApplication

from live_translate.models import AppSettings
from live_translate.ui import MainWindow


def test_inline_translator_toggles_body(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    assert window.inline_body.isHidden()
    window.inline_translator_enabled.setChecked(True)
    assert not window.inline_body.isHidden()
    window.close()


def test_inline_translation_result_renders_cross_check(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    window._inline_translation_finished(
        {
            "request_id": 0,
            "source": "안녕하세요?",
            "translated": "HELLO?",
            "verified": "안녕하세요?",
            "error": "",
        }
    )

    assert window.inline_translated_text.text() == "HELLO?"
    assert window.inline_verified_text.text() == "안녕하세요?"
    window.close()


def test_ocr_runs_on_cpu_to_avoid_gpu_runtime_conflicts(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings(translation_device="cuda:0"))

    assert window.ocr.device == "cpu"
    window.close()
