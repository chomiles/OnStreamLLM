from __future__ import annotations

import base64
import ctypes
import json
import secrets
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from .models import AppSettings, CaptionStyle


APP_DISPLAY_NAME = "OnStreamLLM v0.1 - chomiles"
APP_USER_MODEL_ID = "chomiles.OnStreamLLM"
LEGACY_CONFIG_PATH = Path.home() / ".live_translate_studio.json"
SENSEVOICE_ASR_MODEL = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


class _DataBlob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_char))]


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def app_icon_path() -> Path | None:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).resolve()
        if executable.is_file():
            return executable
    candidate = app_root() / "icon.ico"
    return candidate if candidate.is_file() else None


def config_dir() -> Path:
    return app_root() / "Config"


def settings_path() -> Path:
    directory = config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    preferred = directory / "settings.json"
    legacy_root = app_root() / "settings.json"
    if not preferred.exists() and legacy_root.is_file():
        legacy_root.replace(preferred)
    return preferred


CONFIG_PATH = settings_path()

GAME_LIGHT_ASR_MODEL = SENSEVOICE_ASR_MODEL
GAME_LIGHT_TRANSLATION_MODEL = "tencent/Hy-MT2-1.8B-GGUF"
CURRENT_SETTINGS_SCHEMA_VERSION = 3


def set_windows_app_user_model_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def _protect_password(password: str) -> str:
    if sys.platform != "win32":
        return password
    raw = password.encode("utf-8")
    buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(source), APP_DISPLAY_NAME, None, None, None, 0, ctypes.byref(target)
    ):
        raise OSError("Windows password protection failed")
    try:
        return base64.b64encode(ctypes.string_at(target.data, target.size)).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


def _unprotect_password(value: str) -> str:
    if sys.platform != "win32":
        return value
    raw = base64.b64decode(value)
    buffer = ctypes.create_string_buffer(raw)
    source = _DataBlob(len(raw), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    target = _DataBlob()
    if not ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(target)
    ):
        raise OSError("Windows password recovery failed")
    try:
        return ctypes.string_at(target.data, target.size).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


def load_settings() -> AppSettings:
    path = settings_path()
    source = path if path.exists() else LEGACY_CONFIG_PATH
    if not source.exists():
        return AppSettings()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
        if "password_protected" in data:
            data["password"] = _unprotect_password(str(data.pop("password_protected")))
        data.setdefault("asr_device", data.get("device", "auto"))
        data.setdefault("translation_device", data.get("device", "auto"))
        remote = urlparse(data.get("remote_url", "ws://127.0.0.1:8765/ws/client"))
        data.setdefault("remote_host", remote.hostname or "127.0.0.1")
        data.setdefault("remote_port", remote.port or 8765)
        data.pop("translation_backend", None)
        data.pop("vllm_url", None)
        data.pop("vllm_model", None)
        data.setdefault("input_source_language", data.get("source_language", "auto"))
        data.setdefault("input_target_language", data.get("target_language", "Korean"))
        data.setdefault("output_source_language", data.get("source_language", "auto"))
        data.setdefault("output_target_language", data.get("target_language", "Korean"))
        data.setdefault("source_language", "auto")
        data.setdefault("target_language", "Korean")
        data.setdefault("ui_language", "ko")
        data.setdefault("client_ip_whitelist_enabled", False)
        data.setdefault("client_ip_whitelist_catalog", "")
        data.setdefault("client_ip_whitelist_allowed", "")
        data.setdefault("popup_font_family", "Malgun Gothic")
        data.setdefault("popup_font_size", 32)
        data.setdefault("popup_opacity_percent", 80)
        data.setdefault("popup_locked", False)
        data.setdefault("popup_x", -1)
        data.setdefault("popup_y", -1)
        data.setdefault("popup_width", 900)
        data.setdefault("popup_height", 400)
        if "model_preset" not in data:
            data["model_preset"] = ""
        migrated = False
        schema_version = int(data.get("settings_schema_version", 0) or 0)
        if schema_version < CURRENT_SETTINGS_SCHEMA_VERSION:
            asr = str(data.get("asr_model", "")).replace("\\", "/")
            translation = str(data.get("translation_model", "")).replace("\\", "/")
            uses_legacy_qwen_default = (
                "Qwen3-ASR-0.6B" in asr and "Qwen3-4B" in translation
            )
            if uses_legacy_qwen_default:
                data["asr_model"] = GAME_LIGHT_ASR_MODEL
                data["translation_model"] = GAME_LIGHT_TRANSLATION_MODEL
                data["asr_device"] = "cpu"
                data["translation_device"] = "cuda:0"
                data["asr_cpu_threads"] = 2
                data["asr_cpu_core_ids"] = ""
                data["model_preset"] = "game_light"
                migrated = True
            if schema_version < 2:
                asr_name = str(data.get("asr_model", "")).replace("\\", "/").lower()
                translation_name = str(
                    data.get("translation_model", "")
                ).replace("\\", "/").lower()
                uses_game_light = data.get("model_preset") == "game_light" or (
                    ("sense-voice" in asr_name or "sensevoice" in asr_name)
                    and "hy-mt2" in translation_name
                )
                if uses_game_light and str(data.get("translation_device", "cpu")) == "cpu":
                    data["translation_device"] = "cuda:0"
                    migrated = True
            if schema_version < 3:
                asr_name = str(data.get("asr_model", "")).replace("\\", "/")
                if "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09" in asr_name:
                    data["asr_model"] = GAME_LIGHT_ASR_MODEL
                    data["asr_device"] = "cpu"
                    data["model_preset"] = "game_light"
                    migrated = True
            data["settings_schema_version"] = CURRENT_SETTINGS_SCHEMA_VERSION
        if len(str(data.get("password", ""))) < 12 or data.get("password") == "change-me":
            data["password"] = secrets.token_urlsafe(24)
        for key in ("input_style", "output_style", "screen_style"):
            if key in data:
                data[key] = CaptionStyle(**data[key])
        settings = AppSettings(**data)
        if migrated:
            save_settings(settings)
        return settings
    except (OSError, ValueError, TypeError):
        return AppSettings()


def save_settings(settings: AppSettings) -> None:
    path = settings_path()
    temporary = path.with_suffix(".tmp")
    data = asdict(settings)
    data["password_protected"] = _protect_password(str(data.pop("password")))
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
