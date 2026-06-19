from PySide6.QtWidgets import QApplication

from live_translate.models import AppSettings, SourceKind
from live_translate.ui import MainWindow


def test_language_change_keeps_local_engine_running(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    window.local_running = True
    _source, output_target = window.channel_languages[SourceKind.OUTPUT]
    output_target.setCurrentText("English")

    assert window.settings.output_target_language == "English"
    assert window.local_running
    assert not window.local_starting
    window.close()


def test_stale_pipeline_status_does_not_mark_engine_running(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    old_generation = window._pipeline_generation
    window._reload_pipeline()
    window.local_starting = True
    window._emit_pipeline_source_status(
        old_generation,
        SourceKind.SCREEN,
        window._t("engine.model_load_complete", devices="old"),
    )

    assert window.local_starting
    assert not window.local_running
    window.close()


def test_model_activation_preserves_screen_target_language(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(
        AppSettings(
            input_target_language="Korean",
            target_language="English",
        )
    )

    _screen_source, screen_target = window.channel_languages[SourceKind.SCREEN]
    screen_target.setCurrentText("English")
    window._activate_selected_models(start_engine=False)

    assert window.settings.input_target_language == "Korean"
    assert window.settings.target_language == "English"
    assert window.settings.languages_for(SourceKind.SCREEN) == ("auto", "English")
    window.close()


def test_screen_language_change_uses_ocr_cooldown(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings(source_language="auto", target_language="Korean"))
    monkeypatch.setattr(window, "_engine_ready", lambda: True)
    stops: list[bool] = []
    starts: list[bool] = []
    window.ocr.stop = lambda: stops.append(True)  # type: ignore[method-assign]
    window.ocr.start = lambda: starts.append(True)  # type: ignore[method-assign]
    window.ocr.set_source_language = lambda _language: None  # type: ignore[method-assign]
    window.settings.ocr_enabled = True
    window._set_toggle_checked(window.ocr_toggle, True)

    screen_source, screen_target = window.channel_languages[SourceKind.SCREEN]
    screen_source.setCurrentText("Japanese")

    assert stops == [True]
    assert starts == []
    assert not window.settings.ocr_enabled
    assert window._ocr_cooldown_remaining == 10
    assert not window.ocr_toggle.isEnabled()
    assert not screen_source.isEnabled()
    assert not screen_target.isEnabled()
    assert "전환 대기중" in window.ocr_toggle.text()

    for _ in range(10):
        window._tick_ocr_cooldown()

    assert window._ocr_cooldown_remaining == 0
    assert window.ocr_toggle.isEnabled()
    assert screen_source.isEnabled()
    assert screen_target.isEnabled()
    window.close()
