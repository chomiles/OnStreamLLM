from PySide6.QtWidgets import QApplication

from live_translate.config import SENSEVOICE_ASR_MODEL
from live_translate.models import AppSettings, Caption, SourceKind
from live_translate.ui import AutoScrollTextEdit, MainWindow, _html_preserve_lines


def test_source_status_keeps_last_three_messages(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    monkeypatch.setattr("live_translate.ui.runtime_dependency_installed", lambda _dependency: False)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())
    for number in range(1, 5):
        window._set_source_status(SourceKind.INPUT, f"상태 {number}")
    assert window.source_status_history[SourceKind.INPUT] == ["상태 2", "상태 3", "상태 4"]
    window.close()


def test_caption_monitor_keeps_last_three_captions(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    monkeypatch.setattr("live_translate.ui.runtime_dependency_installed", lambda _dependency: False)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())
    for number in range(1, 5):
        window._show_caption(
            Caption(SourceKind.INPUT, f"original {number}", f"translated {number}", float(number))
        )
    assert [item.original for item in window.unified_caption_history] == [
        "original 2",
        "original 3",
        "original 4",
    ]
    assert "original 1" not in window.unified_caption_view.toPlainText()
    assert window.unified_caption_view.toPlainText().count("[원문]") == 3
    window.close()


def test_caption_monitor_can_omit_original(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings(omit_original_text=True))
    window._show_caption(Caption(SourceKind.SCREEN, "original", "translated", 1.0))
    rendered = window.unified_caption_view.toPlainText()
    assert "[화면 텍스트 번역]" in rendered
    assert "original" not in rendered
    assert "translated" in rendered
    window.close()


def test_caption_view_pauses_and_resumes_auto_scroll(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    view = AutoScrollTextEdit()
    view.resize(300, 100)
    view.show()
    view.set_auto_scroll_html("<br>".join(f"line {number}" for number in range(30)))
    app.processEvents()
    scrollbar = view.verticalScrollBar()
    scrollbar.setValue(scrollbar.maximum())
    view.set_auto_scroll_html("<br>".join(f"line {number}" for number in range(40)))
    assert scrollbar.value() == scrollbar.maximum()

    scrollbar.setValue(0)
    view.set_auto_scroll_html("<br>".join(f"line {number}" for number in range(50)))
    assert scrollbar.value() == 0

    scrollbar.setValue(scrollbar.maximum())
    view.set_auto_scroll_html("<br>".join(f"line {number}" for number in range(60)))
    assert scrollbar.value() == scrollbar.maximum()
    view.close()


def test_dashboard_shows_remote_models(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())
    window._update_remote_info(
        {
            "connected": True,
            "asr_model": "Qwen/Qwen3-ASR-1.7B",
            "translation_model": "Qwen/Qwen3-4B",
        }
    )
    assert "Qwen3-ASR-1.7B" in window.remote_models_status.text()
    assert "Qwen3-4B" in window.remote_models_status.text()
    window.close()


def test_model_combo_shows_compute_tags(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    asr_labels = {
        window.asr_model.itemData(index): window.asr_model.itemText(index)
        for index in range(window.asr_model.count())
        if window.asr_model.itemData(index)
    }
    translation_labels = {
        window.translation_model.itemData(index): window.translation_model.itemText(index)
        for index in range(window.translation_model.count())
        if window.translation_model.itemData(index)
    }
    assert "[CPU/GPU]" in asr_labels["Qwen/Qwen3-ASR-0.6B"]
    assert "[CPU/GPU]" in asr_labels["Qwen/Qwen3-ASR-1.7B"]
    assert "[CPU 전용]" in asr_labels[SENSEVOICE_ASR_MODEL]
    assert "[CPU/GPU]" in translation_labels["tencent/Hy-MT2-1.8B-GGUF"]
    assert "[CPU 전용]" in translation_labels["tencent/Hy-MT2-1.8B-2bit-GGUF"]
    assert "[CPU/GPU]" in translation_labels["Qwen/Qwen3-4B"]
    assert "무거움 12GB 이상 VRAM" in translation_labels["Qwen/Qwen3-4B"]
    assert "매우 무거움 16GB 이상 VRAM" in translation_labels["Qwen/Qwen3-8B-AWQ"]
    assert "[GPU 전용]" in translation_labels["Qwen/Qwen3-8B-AWQ"]
    window.close()


def test_lightweight_models_can_be_selected_without_performance_presets(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    asr_items = [window.asr_model.itemData(index) for index in range(window.asr_model.count())]
    translation_items = [window.translation_model.itemData(index) for index in range(window.translation_model.count())]
    assert SENSEVOICE_ASR_MODEL in asr_items
    assert "tencent/Hy-MT2-1.8B-GGUF" in translation_items
    assert "tencent/Hy-MT2-1.8B-2bit-GGUF" in translation_items
    assert not hasattr(window, "performance_preset")
    assert window._local_engine_supported(
        SENSEVOICE_ASR_MODEL,
        "tencent/Hy-MT2-1.8B-2bit-GGUF",
    )
    assert window._local_engine_supported("Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-4B")
    assert window._local_engine_supported("Qwen/Qwen3-ASR-0.6B", "tencent/Hy-MT2-1.8B-GGUF")
    window.close()


def test_main_tabs_are_consolidated(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "번역기",
        "모델 관리",
        "설정",
        "정보",
    ]
    window.close()


def test_status_bar_elides_long_errors(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())
    long_error = "오류 - " + ("very long error " * 80)

    window._set_status(long_error)

    assert window.status.toolTip() == f"상태: {long_error}"
    assert len(window.status.text()) < len(window.status.toolTip())
    assert window.status.height() == 28
    window.close()


def test_ocr_button_requires_engine(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    window.local_running = False
    window.remote_connected = False
    window._refresh_dashboard()
    assert not window.ocr_toggle.isEnabled()
    assert "먼저" in window.ocr_toggle.text()

    window.local_running = True
    window.pipeline._ready.set()
    window._refresh_dashboard()
    assert window.ocr_toggle.isEnabled()
    assert window.ocr_toggle.text() == "화면 감지"
    window.close()


def test_late_runtime_progress_does_not_overwrite_completed_status(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    monkeypatch.setattr("live_translate.ui.all_runtime_dependencies_installed", lambda: True)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())
    window._runtime_install_active = "PaddleOCR"
    window._runtime_downloads["PaddleOCR"] = object()
    window._runtime_install_batch_total = 1

    window._runtime_download_finished("PaddleOCR", "C:/runtime/python")
    completed_status = window.download_status.toolTip()
    completed_bar = window.status.toolTip()
    window._runtime_download_progress("PaddleOCR", 100, "(10/10) install complete")

    assert window.download_status.toolTip() == completed_status
    assert window.status.toolTip() == completed_bar
    window.close()


def test_optional_runtime_install_can_start_paddleocr_directly(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    monkeypatch.setattr("live_translate.ui.runtime_dependency_installed", lambda _dependency: False)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())
    started: list[str] = []
    monkeypatch.setattr(window, "_validate_runtime_cuda_requirements", lambda _names: True)
    monkeypatch.setattr(
        window,
        "_begin_runtime_install",
        lambda dependency, *, reinstall=False: started.append(dependency.name),
    )

    window._install_optional_runtime_dependency("PaddleOCR")

    assert started == ["PaddleOCR"]
    window.close()


def test_caption_html_preserves_line_breaks() -> None:
    assert _html_preserve_lines("first\nsecond") == "first<br>second"
