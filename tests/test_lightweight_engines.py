from pathlib import Path

import numpy as np

from live_translate.config import SENSEVOICE_ASR_MODEL
from live_translate.engines import (
    _missing_python_module_name,
    _missing_runtime_module_message,
    HyMt2Translator,
    QwenTranslator,
    QwenAsr,
    SenseVoiceAsr,
    TranslationPipeline,
    _relative_path,
    _requires_stq_llama_cpp,
    _to_mono_float32,
)
from live_translate.event_bus import CaptionBus
from live_translate.models import AppSettings, SourceKind


def test_lightweight_engine_classes_can_be_constructed() -> None:
    settings = AppSettings(
        asr_model=SENSEVOICE_ASR_MODEL,
        translation_model="tencent/Hy-MT2-1.8B-2bit-GGUF",
    )

    assert isinstance(SenseVoiceAsr(settings), SenseVoiceAsr)
    assert isinstance(HyMt2Translator(settings), HyMt2Translator)
    assert _requires_stq_llama_cpp(settings.translation_model)


def test_audio_is_contiguous_mono_float32() -> None:
    stereo = np.array([[1.0, -1.0], [0.5, 0.0]], dtype=np.float64)
    mono = _to_mono_float32(stereo)

    assert mono.dtype == np.float32
    assert mono.flags["C_CONTIGUOUS"]
    assert mono.tolist() == [0.0, 0.25]


def test_relative_path_prefers_workspace_relative_paths() -> None:
    path = Path.cwd() / "models" / "example.gguf"

    assert _relative_path(path) == str(Path("models") / "example.gguf")


def test_qwen_asr_is_forced_to_cpu_when_hymt2_uses_gpu(monkeypatch) -> None:
    monkeypatch.setattr("live_translate.engines._hy_mt2_uses_gpu", lambda _model: True)
    settings = AppSettings(
        asr_device="cuda:0",
        translation_device="cuda:0",
        asr_model="Qwen/Qwen3-ASR-0.6B",
        translation_model="tencent/Hy-MT2-1.8B-GGUF",
    )

    pipeline = TranslationPipeline(settings, CaptionBus())

    assert isinstance(pipeline.asr, QwenAsr)
    assert pipeline.asr_settings.asr_device == "cpu"


def test_sensevoice_asr_is_forced_to_cpu_when_hymt2_uses_gpu(monkeypatch) -> None:
    monkeypatch.setattr("live_translate.engines._hy_mt2_uses_gpu", lambda _model: True)
    settings = AppSettings(
        asr_device="cuda:0",
        translation_device="cuda:0",
        asr_model=SENSEVOICE_ASR_MODEL,
        translation_model="tencent/Hy-MT2-1.8B-GGUF",
    )

    pipeline = TranslationPipeline(settings, CaptionBus())

    assert isinstance(pipeline.asr, SenseVoiceAsr)
    assert pipeline.asr_settings.asr_device == "cpu"
    assert "LLM GPU 충돌 방지로 CPU 실행" in pipeline.runtime_device_report()


def test_missing_python_module_name_detects_module_not_found() -> None:
    assert _missing_python_module_name(ModuleNotFoundError("No module named 'diskcache'")) == "diskcache"
    assert _missing_python_module_name(ImportError("No module named 'jinja2'")) == "jinja2"
    assert _missing_python_module_name(RuntimeError("boom")) is None


def test_missing_runtime_module_message_distinguishes_stdlib_modules() -> None:
    stdlib_message = _missing_runtime_module_message("pickletools")
    package_message = _missing_runtime_module_message("diskcache")
    assert "최신 OnStreamLLM" in stdlib_message
    assert "llama" in package_message


def test_released_pipeline_rejects_audio_without_touching_released_model() -> None:
    statuses: list[tuple[SourceKind, str]] = []
    pipeline = TranslationPipeline(
        AppSettings(demo_mode=True),
        CaptionBus(),
        lambda source, message: statuses.append((source, message)),
    )

    pipeline._ready.set()
    pipeline.release()
    pipeline.asr = None
    pipeline._process_audio(SourceKind.OUTPUT, np.zeros(1600, dtype=np.float32), 16000)

    assert statuses
    assert statuses[-1][0] == SourceKind.OUTPUT


def test_pipeline_release_calls_engine_release(monkeypatch) -> None:
    calls: list[str] = []
    pipeline = TranslationPipeline(AppSettings(demo_mode=True), CaptionBus())
    pipeline.asr = type("FakeAsr", (), {"release": lambda self: calls.append("asr")})()
    pipeline.translator = type(
        "FakeTranslator",
        (),
        {"release": lambda self: calls.append("translator")},
    )()
    monkeypatch.setattr("live_translate.engines._release_cuda_memory", lambda: calls.append("cuda"))

    pipeline.release()

    assert calls == ["asr", "translator", "cuda"]


def test_qwen_prompt_forces_target_language() -> None:
    translator = QwenTranslator(AppSettings())

    system_prompt = translator._messages("우리도 똑같이 씁니다", "Korean", "English")[0]["content"]

    assert "Translate from Korean into English" in system_prompt
    assert "output language must be English" in system_prompt
    assert "Do not paraphrase in the source language" in system_prompt
