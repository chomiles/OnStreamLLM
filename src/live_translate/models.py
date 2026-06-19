from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import secrets
from typing import Any


class SourceKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    SCREEN = "screen"


@dataclass(slots=True)
class Caption:
    source: SourceKind
    original: str
    translated: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = self.source.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Caption":
        return cls(
            source=SourceKind(data["source"]),
            original=str(data.get("original", "")),
            translated=str(data.get("translated", "")),
            timestamp=float(data.get("timestamp", 0)),
        )


@dataclass(slots=True)
class CaptionStyle:
    font_family: str = "Malgun Gothic"
    font_size: int = 42
    color: str = "#ffffff"
    outline_color: str = "#000000"
    outline_width: int = 3


@dataclass(slots=True)
class AppSettings:
    host: str = "0.0.0.0"
    port: int = 8765
    overlay_port: int = 17865
    password: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    remote_host: str = "127.0.0.1"
    remote_port: int = 8765
    remote_url: str = "ws://127.0.0.1:8765/ws/client"
    device: str = "auto"
    asr_device: str = "cpu"
    translation_device: str = "cuda:0"
    asr_model: str = (
        "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    )
    translation_model: str = "tencent/Hy-MT2-1.8B-GGUF"
    model_preset: str = "game_light"
    settings_schema_version: int = 3
    source_language: str = "auto"
    target_language: str = "Korean"
    input_source_language: str = "auto"
    input_target_language: str = "Korean"
    output_source_language: str = "auto"
    output_target_language: str = "Korean"
    llm_rules: str = (
        "Preserve the speaker's meaning and tone. "
        "Return only the translated text, with no explanation."
    )
    demo_mode: bool = False
    asr_cpu_threads: int = 2
    asr_cpu_core_ids: str = ""
    translation_cpu_threads: int = 0
    translation_cpu_core_ids: str = ""
    always_run_obs_overlay: bool = False
    input_device_id: str = ""
    input_device_name: str = ""
    output_device_id: str = ""
    output_device_name: str = ""
    input_capture_enabled: bool = False
    output_capture_enabled: bool = False
    ocr_enabled: bool = False
    ocr_auto_refresh: bool = True
    ocr_interval: float = 1.5
    omit_original_text: bool = False
    transparent_popup_enabled: bool = False
    popup_font_family: str = "Malgun Gothic"
    popup_font_size: int = 32
    popup_opacity_percent: int = 80
    popup_locked: bool = False
    popup_x: int = -1
    popup_y: int = -1
    popup_width: int = 900
    popup_height: int = 400
    ocr_left: int = 0
    ocr_top: int = 0
    ocr_width: int = 0
    ocr_height: int = 0
    min_speech_seconds: float = 0.5
    remote_enabled: bool = False
    host_server_enabled: bool = False
    client_ip_whitelist_enabled: bool = False
    client_ip_whitelist_catalog: str = ""
    client_ip_whitelist_allowed: str = ""
    input_style: CaptionStyle = field(default_factory=CaptionStyle)
    output_style: CaptionStyle = field(
        default_factory=lambda: CaptionStyle(color="#8ed8ff", font_size=42)
    )
    screen_style: CaptionStyle = field(
        default_factory=lambda: CaptionStyle(color="#ffe28a", font_size=36)
    )
    ui_language: str = "ko"

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("password", None)
        return data

    def languages_for(self, source: SourceKind) -> tuple[str, str]:
        if source == SourceKind.INPUT:
            return self.input_source_language, self.input_target_language
        if source == SourceKind.OUTPUT:
            return self.output_source_language, self.output_target_language
        return self.source_language, self.target_language
