import json

from live_translate.config import (
    GAME_LIGHT_ASR_MODEL,
    GAME_LIGHT_TRANSLATION_MODEL,
    load_settings,
)
from live_translate.models import AppSettings, Caption, SourceKind


def test_caption_round_trip() -> None:
    caption = Caption(SourceKind.INPUT, "hello", "안녕하세요", 123.0)
    assert Caption.from_dict(caption.to_dict()) == caption


def test_public_settings_hide_password() -> None:
    settings = AppSettings(password="secret")
    assert "password" not in settings.public_dict()


def test_screen_uses_ocr_languages() -> None:
    settings = AppSettings(source_language="Japanese", target_language="Korean")
    assert settings.languages_for(SourceKind.SCREEN) == ("Japanese", "Korean")


def test_popup_overlay_defaults() -> None:
    settings = AppSettings()
    assert settings.popup_opacity_percent == 80
    assert settings.popup_font_size == 32
    assert settings.popup_locked is False
    assert settings.popup_width == 900
    assert settings.popup_height == 400


def test_legacy_qwen_defaults_migrate_to_game_light(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    settings_file = config_dir / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "asr_model": "Qwen/Qwen3-ASR-0.6B",
                "translation_model": "Qwen/Qwen3-4B",
                "password": "change-me",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("live_translate.config.config_dir", lambda: config_dir)
    monkeypatch.setattr("live_translate.config.settings_path", lambda: settings_file)

    settings = load_settings()

    assert settings.asr_model == GAME_LIGHT_ASR_MODEL
    assert settings.translation_model == GAME_LIGHT_TRANSLATION_MODEL
    assert settings.model_preset == "game_light"
    assert settings.settings_schema_version == 3
    saved = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved["model_preset"] == "game_light"
