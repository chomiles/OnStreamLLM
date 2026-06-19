from pathlib import Path

import pytest

from live_translate.model_store import (
    delete_model,
    display_path,
    is_model_complete,
    repository_id_from_path,
)


def test_display_path_uses_forward_slashes() -> None:
    displayed = display_path(Path.cwd() / "models" / "LLM" / "sample")
    assert "\\" not in displayed
    assert displayed == "models/LLM/sample"


def test_repository_id_from_download_folder() -> None:
    assert repository_id_from_path("models/STT/Qwen--Qwen3-ASR-0.6B") == "Qwen/Qwen3-ASR-0.6B"


def test_model_complete_accepts_gguf_and_onnx(tmp_path) -> None:
    gguf_model = tmp_path / "hy-mt2"
    gguf_model.mkdir()
    (gguf_model / "model.gguf").write_bytes(b"model")
    onnx_model = tmp_path / "sensevoice"
    onnx_model.mkdir()
    (onnx_model / "model.int8.onnx").write_bytes(b"model")
    paddle_model = tmp_path / "korean-ocr"
    paddle_model.mkdir()
    (paddle_model / "config.json").write_text("{}", encoding="utf-8")
    (paddle_model / "inference.pdiparams").write_bytes(b"model")

    assert is_model_complete(gguf_model)
    assert is_model_complete(onnx_model)
    assert is_model_complete(paddle_model)


def test_delete_model_rejects_path_outside_model_folder(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError):
        delete_model("asr", outside)
