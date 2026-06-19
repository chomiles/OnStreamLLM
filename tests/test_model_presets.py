from PySide6.QtWidgets import QApplication

from live_translate.compute import CpuCore
from live_translate.config import SENSEVOICE_ASR_MODEL
from live_translate.models import AppSettings
from live_translate.ui import MainWindow


def test_game_preset_selects_sensevoice_hymt2_cpu_gpu(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    monkeypatch.setattr(
        "live_translate.ui.list_cpu_cores",
        lambda: [CpuCore(0, 1), CpuCore(1, 1), CpuCore(2, 0), CpuCore(3, 0)],
    )
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    window._apply_model_preset("game_light")

    assert window.asr_model.currentData() == SENSEVOICE_ASR_MODEL
    assert window.translation_model.currentData() == "tencent/Hy-MT2-1.8B-GGUF"
    assert window.asr_device.currentData() == "cpu"
    assert str(window.translation_device.currentData()).startswith(("cuda:", "cpu"))
    assert window.asr_cpu_threads.currentData() == 2
    assert window.settings.asr_cpu_core_ids == ""
    window.close()


def test_legacy_qwen_settings_keep_qwen_light_preset(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(
        AppSettings(
            asr_model="Qwen/Qwen3-ASR-0.6B",
            translation_model="Qwen/Qwen3-4B",
            model_preset="",
        )
    )

    assert window.model_preset.currentData() == "qwen_light"
    window.close()


def test_fresh_settings_default_to_game_light_preset(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    assert window.model_preset.currentData() == "game_light"
    assert window.asr_cpu_threads.currentData() == 2
    window.close()


def test_qwen_high_preset_selects_larger_qwen_models(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr("live_translate.ui.save_settings", lambda _settings: None)
    QApplication.instance() or QApplication([])
    window = MainWindow(AppSettings())

    window._apply_model_preset("qwen_high")

    assert window.asr_model.currentData() == "Qwen/Qwen3-ASR-1.7B"
    assert window.translation_model.currentData() == "Qwen/Qwen3-8B-AWQ"
    assert str(window.asr_device.currentData()).startswith(("cuda:", "cpu"))
    assert str(window.translation_device.currentData()).startswith(("cuda:", "cpu"))
    window.close()
