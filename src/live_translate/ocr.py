from __future__ import annotations

import threading
import time
import os
import re
import sys
import logging
from collections import deque
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path

import numpy as np

OCR_STATUS_LOADING = "__ocr_status_loading__"
OCR_STATUS_READY = "__ocr_status_ready__"

KOREAN_OCR_MODEL_ID = "PaddlePaddle/korean_PP-OCRv5_mobile_rec"
KOREAN_OCR_MODEL_NAME = "korean_PP-OCRv5_mobile_rec"
DEFAULT_OCR_REC_MODEL = "PP-OCRv5_mobile_rec"
DEFAULT_OCR_DET_MODEL = "PP-OCRv6_small_det"


LOGGER = logging.getLogger(__name__)


def _ocr_cache_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        if executable_dir.parent.name.lower() == "dist":
            return executable_dir.parent.parent / "models" / "OCR"
        return executable_dir / "models" / "OCR"
    return Path.cwd() / "models" / "OCR"


OCR_CACHE = _ocr_cache_root()
PADDLE_OCR_CREATE_LOCK = threading.Lock()


def _configure_paddle_dll_paths() -> None:
    if not getattr(sys, "frozen", False) or not hasattr(os, "add_dll_directory"):
        return
    executable_dir = Path(sys.executable).resolve().parent
    candidates = (
        executable_dir / "_internal" / "paddle" / "libs",
        executable_dir / "paddle" / "libs",
    )
    for candidate in candidates:
        if candidate.exists():
            os.add_dll_directory(str(candidate))
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"


@contextmanager
def paddle_cache_environment():
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    old_profile = os.environ.get("USERPROFILE")
    old_source_check = os.environ.get("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK")
    os.environ["USERPROFILE"] = str(OCR_CACHE)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(OCR_CACHE / "paddlex")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    try:
        yield
    finally:
        if old_profile is None:
            os.environ.pop("USERPROFILE", None)
        else:
            os.environ["USERPROFILE"] = old_profile
        if old_source_check is None:
            os.environ.pop("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", None)
        else:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = old_source_check


def resolve_paddle_device(requested: str) -> str:
    # On Windows, loading CUDA PyTorch after Paddle can cause a DLL conflict.
    import torch  # noqa: F401
    import paddle

    if requested.startswith("cuda") and paddle.device.is_compiled_with_cuda():
        return requested.replace("cuda", "gpu")
    return "cpu"


def korean_ocr_model_dir() -> Path:
    return OCR_CACHE / "paddlex" / "official_models" / f"{KOREAN_OCR_MODEL_NAME}_safetensors"


def is_korean_ocr_model_ready() -> bool:
    model_dir = korean_ocr_model_dir()
    return (model_dir / "model.safetensors").is_file() and (model_dir / "config.json").is_file()


def download_korean_ocr_model() -> Path:
    from .runtime_bootstrap import configure_runtime_paths

    configure_runtime_paths()
    if is_korean_ocr_model_ready():
        return korean_ocr_model_dir()
    with paddle_cache_environment():
        import torch  # noqa: F401
        from paddlex.inference.utils.official_models import official_models

        return Path(
            official_models.get_model_path(
                KOREAN_OCR_MODEL_NAME,
                model_formats=["safetensors"],
            )
        )


def _use_korean_recognition_model(source_language: str) -> bool:
    return source_language.strip().casefold() in {"korean", "ko"}


def _recognition_model_for_language(source_language: str) -> str:
    normalized = source_language.strip().casefold()
    if normalized in {"korean", "ko"}:
        return KOREAN_OCR_MODEL_NAME
    if normalized in {"japanese", "ja", "jpn"}:
        return DEFAULT_OCR_REC_MODEL
    if normalized in {"english", "en"}:
        return "en_PP-OCRv5_mobile_rec"
    return DEFAULT_OCR_REC_MODEL


def _engine_for_recognition_model(recognition_model_name: str) -> str:
    return "paddle_dynamic"


def create_paddle_ocr(device: str, source_language: str = "") -> object:
    from .runtime_bootstrap import configure_runtime_paths

    configure_runtime_paths()
    _configure_paddle_dll_paths()
    with paddle_cache_environment():
        # Windows must load CUDA PyTorch DLLs before Paddle/PaddleX.
        import torch  # noqa: F401
        import paddle
        from paddleocr import PaddleOCR

        use_korean = _use_korean_recognition_model(source_language)
        if use_korean and not is_korean_ocr_model_ready():
            raise RuntimeError("한국어 OCR 모델이 설치되어 있지 않습니다.")
        recognition_model_name = _recognition_model_for_language(source_language)
        engine_name = _engine_for_recognition_model(recognition_model_name)
        if engine_name == "paddle_static":
            paddle.enable_static()
        else:
            paddle.disable_static()
        LOGGER.info(
            "Creating PaddleOCR: device=%s source_language=%s rec_model=%s det_model=%s engine=%s",
            device,
            source_language or "auto",
            recognition_model_name,
            DEFAULT_OCR_DET_MODEL,
            engine_name,
        )
        engine = PaddleOCR(
            device=resolve_paddle_device(device),
            engine=engine_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=DEFAULT_OCR_DET_MODEL,
            text_recognition_model_name=recognition_model_name,
        )
        setattr(engine, "_live_translate_paddle_engine", engine_name)
        return engine


def extract_paddle_text(results: object) -> str:
    texts: list[str] = []
    for result in results or []:
        payload = getattr(result, "json", result)
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            payload = payload["res"]
        if isinstance(payload, dict):
            values = payload.get("rec_texts", [])
            if isinstance(values, (list, tuple)):
                texts.extend(str(value).strip() for value in values if str(value).strip())
    return "\n".join(texts).strip()


def normalize_ocr_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


CHAT_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d")
CHAT_NAME_PATTERN = re.compile(r"^[^\s:：]{2,40}[:：]\s*")
CHAT_BAD_PREFIX_PATTERN = re.compile(r"^(?:now\??|no[wv]\??)\s*", re.IGNORECASE)
KOREAN_CHAT_WELCOME_PATTERN = re.compile(r"\S+\s*채팅방에\s+오신\s+것을\s+환영합니다[!.！]*")

KOREAN_TEXT_PATTERN = re.compile(r"[가-힣]")
ASCII_CHAT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{2,32}\s+")


def _split_ocr_chat_segments(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    matches = list(CHAT_TIMESTAMP_PATTERN.finditer(compact))
    if not matches:
        return [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[str] = []
    if matches[0].start() > 0:
        prefix = compact[: matches[0].start()].strip()
        if prefix:
            segments.append(prefix)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(compact)
        segment = compact[match.end() : end].strip()
        if segment:
            segments.append(segment)
    return segments


def clean_ocr_chat_line(line: str, known_names: set[str] | None = None) -> str:
    cleaned = re.sub(r"\s+", " ", line).strip()
    cleaned = CHAT_BAD_PREFIX_PATTERN.sub("", cleaned).strip()
    cleaned = CHAT_TIMESTAMP_PATTERN.sub("", cleaned).strip()
    cleaned = CHAT_NAME_PATTERN.sub("", cleaned).strip()
    if match := ASCII_CHAT_NAME_PATTERN.match(cleaned):
        remainder = cleaned[match.end() :].lstrip()
        if KOREAN_TEXT_PATTERN.match(remainder):
            cleaned = remainder
    for name in sorted(known_names or set(), key=len, reverse=True):
        if cleaned.startswith(name):
            cleaned = cleaned[len(name) :].strip(" :：")
            break
    cleaned = KOREAN_CHAT_WELCOME_PATTERN.sub("", cleaned).strip()
    cleaned = re.sub(r"^[^\w가-힣ぁ-んァ-ン一-龥]+", "", cleaned).strip()
    if ":" in cleaned[:45]:
        before, after = cleaned.split(":", 1)
        if 1 < len(before.strip()) <= 40:
            cleaned = after.strip()
    cleaned = re.sub(r"\bI(?=(?:doubt|dont|don't|am|have|will|can|would|could|should)\b)", "I ", cleaned)
    return cleaned


def clean_ocr_text(text: str) -> str:
    lines: list[str] = []
    segments = _split_ocr_chat_segments(text)
    known_names = {
        match.group(1).strip()
        for segment in segments
        if (match := re.match(r"^([^\s:：]{2,40})[:：]", segment.strip()))
    }
    for segment in segments:
        cleaned = clean_ocr_chat_line(segment, known_names)
        if len(cleaned) < 2:
            continue
        if re.fullmatch(r"[\W_]+", cleaned):
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def prepare_game_frame(image: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(2.0, max_dimension / max(height, width))
    if scale <= 1.05:
        return image
    import cv2

    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def paddle_predict(engine: object, image: np.ndarray) -> object:
    import paddle

    engine_name = getattr(engine, "_live_translate_paddle_engine", "paddle_dynamic")
    if engine_name == "paddle_static":
        paddle.enable_static()
    else:
        paddle.disable_static()
    return engine.predict(image)


def format_ocr_error(exc: Exception) -> str:
    message = str(exc)
    cause = getattr(exc, "__cause__", None)
    cause_message = str(cause) if cause else ""
    details = cause_message or message
    lowered_details = details.lower()
    lowered = message.lower()
    if (
        "no available model hosting" in lowered
        or "no model source is available" in lowered
        or "network connection" in lowered
    ):
        return (
            "오류 - PaddleOCR 모델 다운로드 실패: 인터넷 연결을 확인한 뒤 "
            "화면 번역 엔진을 다시 시작해주세요."
        )
    if "dependency error" in lowered or "required dependencies" in lowered:
        if "following dependencies are not available" in lowered_details:
            missing = details.split("following dependencies are not available:", 1)[-1]
            missing = ", ".join(part.strip() for part in missing.splitlines() if part.strip())
            return f"오류 - PaddleOCR 의존성 누락: {missing}"
        return (
            "오류 - PaddleOCR 의존성 누락: setup.ps1을 다시 실행하거나 "
            "새 배포본으로 실행해주세요."
        )
    if "access is denied" in lowered or "permissionerror" in lowered or "액세스가 거부" in message:
        return (
            "오류 - PaddleOCR 모델 캐시 권한 문제: models/OCR 폴더를 정리한 뒤 "
            "화면 번역 엔진을 다시 시작해주세요."
        )
    if "json.exception.parse_error" in lowered or "parse error" in lowered:
        return (
            "오류 - PaddleOCR 모델 캐시 손상: 화면 감지를 끄고 "
            "models/OCR/paddlex/official_models 폴더를 정리한 뒤 다시 시도해주세요."
        )
    return f"오류 - PaddleOCR 실행 실패: {message}"


class ScreenOcr:
    def __init__(
        self,
        callback: Callable[[str], None],
        interval: float = 1.5,
        auto_refresh: bool = True,
        region: dict[str, int] | None = None,
        device: str = "cpu",
        source_language: str = "",
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.callback = callback
        self.interval = interval
        self.auto_refresh = auto_refresh
        self.source_language = source_language
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_text = ""
        self._recent_lines: deque[str] = deque(maxlen=200)
        self._recent_line_keys: set[str] = set()
        self._region = region
        self.device = device
        self.status_callback = status_callback or (lambda _message: None)
        self._region_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._recent_lock = threading.Lock()
        self._language_lock = threading.Lock()

    def set_region(self, region: dict[str, int] | None) -> None:
        with self._region_lock:
            self._region = region
            self._last_text = ""
        with self._recent_lock:
            self._recent_lines.clear()
            self._recent_line_keys.clear()
        LOGGER.info("OCR region updated: %s", region or "full screen")

    def set_refresh(self, auto_refresh: bool, interval: float) -> None:
        with self._refresh_lock:
            self.auto_refresh = auto_refresh
            self.interval = max(0.5, interval)
        LOGGER.info("OCR refresh updated: auto=%s interval=%.1fs", auto_refresh, self.interval)

    def set_source_language(self, source_language: str) -> None:
        with self._language_lock:
            self.source_language = source_language

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                LOGGER.warning("OCR worker did not stop before timeout")

    def filter_new_text(self, text: str) -> str:
        new_lines: list[str] = []
        text = clean_ocr_text(text)
        with self._recent_lock:
            for line in text.splitlines():
                cleaned = line.strip()
                key = normalize_ocr_line(cleaned)
                if not key or key in self._recent_line_keys:
                    continue
                if len(self._recent_lines) == self._recent_lines.maxlen:
                    self._recent_line_keys.discard(self._recent_lines[0])
                self._recent_lines.append(key)
                self._recent_line_keys.add(key)
                new_lines.append(cleaned)
        return "\n".join(new_lines)

    def _run(self) -> None:
        try:
            import mss

            LOGGER.info(
                "OCR worker starting: device=%s interval=%.1fs auto=%s region=%s",
                self.device,
                self.interval,
                self.auto_refresh,
                self._region or "full screen",
            )
            self.status_callback(OCR_STATUS_LOADING)
            with self._language_lock:
                source_language = self.source_language
            with PADDLE_OCR_CREATE_LOCK:
                engine = create_paddle_ocr(self.device, source_language)
            if self._stop.is_set():
                return
            self.status_callback(OCR_STATUS_READY)
            with mss.mss() as capture:
                while not self._stop.is_set():
                    started = time.monotonic()
                    with self._region_lock:
                        monitor = self._region or capture.monitors[1]
                    image = prepare_game_frame(np.asarray(capture.grab(monitor))[:, :, :3])
                    text = extract_paddle_text(paddle_predict(engine, image))
                    if text and text != self._last_text:
                        self._last_text = text
                        new_text = self.filter_new_text(text)
                        if new_text:
                            LOGGER.info("OCR detected %d chars", len(new_text))
                            self.callback(new_text)
                    self._wait_for_next_scan(time.monotonic() - started)
        except Exception as exc:
            LOGGER.exception("OCR worker failed")
            self.status_callback(format_ocr_error(exc))

    def _wait_for_next_scan(self, scan_elapsed: float = 0.0) -> None:
        elapsed = scan_elapsed
        while not self._stop.is_set():
            with self._refresh_lock:
                auto_refresh = self.auto_refresh
                interval = self.interval
            if auto_refresh and elapsed >= interval:
                return
            time.sleep(0.1)
            if auto_refresh:
                elapsed += 0.1
