import numpy as np
import sys
from types import SimpleNamespace
import threading

from live_translate.ocr import (
    DEFAULT_OCR_REC_MODEL,
    KOREAN_OCR_MODEL_ID,
    KOREAN_OCR_MODEL_NAME,
    _engine_for_recognition_model,
    _recognition_model_for_language,
    _use_korean_recognition_model,
    clean_ocr_text,
    extract_paddle_text,
    is_korean_ocr_model_ready,
    korean_ocr_model_dir,
    ScreenOcr,
    paddle_predict,
    prepare_game_frame,
    resolve_paddle_device,
)


class FakeResult:
    json = {"res": {"rec_texts": ["first line", "second line"]}}


def test_extract_paddle_text() -> None:
    assert extract_paddle_text([FakeResult()]) == "first line\nsecond line"


def test_screen_ocr_only_returns_new_lines() -> None:
    ocr = ScreenOcr(lambda _text: None)
    assert ocr.filter_new_text("Hello\nWorld") == "Hello\nWorld"
    assert ocr.filter_new_text("Hello\nWorld\nNew message") == "New message"
    assert ocr.filter_new_text("  new   MESSAGE  ") == ""


def test_screen_ocr_region_change_clears_recent_lines() -> None:
    ocr = ScreenOcr(lambda _text: None)
    assert ocr.filter_new_text("Repeated") == "Repeated"
    assert ocr.filter_new_text("Repeated") == ""
    ocr.set_region({"left": 0, "top": 0, "width": 100, "height": 100})
    assert ocr.filter_new_text("Repeated") == "Repeated"


def test_screen_ocr_stop_waits_for_worker() -> None:
    ocr = ScreenOcr(lambda _text: None)
    finished = threading.Event()

    def worker() -> None:
        ocr._stop.wait(0.5)
        finished.set()

    ocr._thread = threading.Thread(target=worker)
    ocr._thread.start()

    ocr.stop()

    assert finished.is_set()
    assert not ocr.is_running()


def test_screen_ocr_cleans_game_chat_noise() -> None:
    sample = (
        "now? 6:14ZekkenNightstarprobably "
        "6:15ZekkenNightstar:maybe "
        "6:15ZekkenNightstar:Idoubt though those posts got thousands in hours "
        "pinksheep 채팅방에 오신 것을 환영합니다!"
    )
    assert clean_ocr_text(sample) == (
        "probably\n"
        "maybe\n"
        "I doubt though those posts got thousands in hours"
    )


def test_screen_ocr_removes_ascii_name_before_korean_chat() -> None:
    assert clean_ocr_text("kapaconty 우리도 똑같이 씁니다") == "우리도 똑같이 씁니다"


def test_paddle_device_falls_back_to_cpu(monkeypatch) -> None:
    fake_paddle = SimpleNamespace(
        device=SimpleNamespace(is_compiled_with_cuda=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    assert resolve_paddle_device("cuda:1") == "cpu"


def test_small_game_frame_is_upscaled() -> None:
    image = np.zeros((200, 600, 3), dtype=np.uint8)
    assert prepare_game_frame(image).shape[:2] == (400, 1200)


def test_korean_ocr_model_id_is_paddle_korean_rec() -> None:
    assert KOREAN_OCR_MODEL_ID == "PaddlePaddle/korean_PP-OCRv5_mobile_rec"


def test_use_korean_recognition_model_matches_korean_labels() -> None:
    assert _use_korean_recognition_model("Korean")
    assert _use_korean_recognition_model("ko")
    assert not _use_korean_recognition_model("Japanese")


def test_recognition_model_matches_source_language() -> None:
    assert _recognition_model_for_language("Korean") == KOREAN_OCR_MODEL_NAME
    assert _recognition_model_for_language("Japanese") == DEFAULT_OCR_REC_MODEL
    assert _recognition_model_for_language("JPN") == DEFAULT_OCR_REC_MODEL
    assert _recognition_model_for_language("English") == "en_PP-OCRv5_mobile_rec"
    assert _recognition_model_for_language("auto") == DEFAULT_OCR_REC_MODEL


def test_ocr_engine_matches_recognition_model() -> None:
    assert _engine_for_recognition_model(DEFAULT_OCR_REC_MODEL) == "paddle_dynamic"
    assert _engine_for_recognition_model(KOREAN_OCR_MODEL_NAME) == "paddle_dynamic"


def test_korean_ocr_model_dir_uses_paddlex_safetensors_cache() -> None:
    assert korean_ocr_model_dir().name == f"{KOREAN_OCR_MODEL_NAME}_safetensors"


def test_korean_ocr_model_ready_requires_safetensors(monkeypatch) -> None:
    from pathlib import Path

    model_dir = Path("tests") / "_korean_ocr_ready_probe"
    if model_dir.exists():
        for child in model_dir.iterdir():
            child.unlink()
    else:
        model_dir.mkdir(parents=True)
    monkeypatch.setattr("live_translate.ocr.korean_ocr_model_dir", lambda: model_dir)
    try:
        assert not is_korean_ocr_model_ready()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        assert not is_korean_ocr_model_ready()
        (model_dir / "model.safetensors").write_bytes(b"weights")
        assert is_korean_ocr_model_ready()
    finally:
        for child in model_dir.iterdir():
            child.unlink()
        model_dir.rmdir()


def test_paddle_predict_enables_dynamic_mode(monkeypatch) -> None:
    calls = []
    fake_paddle = SimpleNamespace(
        disable_static=lambda: calls.append("dynamic"),
        enable_static=lambda: calls.append("static"),
    )
    fake_engine = SimpleNamespace(predict=lambda image: image.shape)
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    assert paddle_predict(fake_engine, np.zeros((10, 20, 3))) == (10, 20, 3)
    assert calls == ["dynamic"]


def test_paddle_predict_keeps_static_engine(monkeypatch) -> None:
    calls = []
    fake_paddle = SimpleNamespace(
        disable_static=lambda: calls.append("dynamic"),
        enable_static=lambda: calls.append("static"),
    )
    fake_engine = SimpleNamespace(
        _live_translate_paddle_engine="paddle_static",
        predict=lambda image: image.shape,
    )
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)
    assert paddle_predict(fake_engine, np.zeros((10, 20, 3))) == (10, 20, 3)
    assert calls == ["static"]
