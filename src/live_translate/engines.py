from __future__ import annotations

import threading
import time
import sys
import types
import re
import subprocess
import hashlib
import shutil
import tempfile
import os
import logging
import gc
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable

import numpy as np

from .event_bus import CaptionBus
from .i18n import Translator
from .models import AppSettings, Caption, SourceKind
from .compute import cpu_execution_context, effective_cpu_thread_count, normalize_device
from .model_store import is_model_complete, model_download_path


LANGUAGE_ALIASES = {
    "auto": "",
    "chinese": "zh",
    "cantonese": "yue",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "zh": "zh",
    "yue": "yue",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
}

_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORY_PATHS: set[str] = set()
_LLAMA_GPU_DEVICE_CACHE: dict[str, bool] = {}
_LLAMA_CPP_PYTHON_HAS_GPU: bool | None = None
LOGGER = logging.getLogger(__name__)

_MISSING_RUNTIME_MODULE_NAMES = {
    "sherpa_onnx": "SenseVoice",
    "qwen_asr": "Qwen ASR",
    "torch": "Torch",
    "llama_cpp": "llama",
    "paddleocr": "PaddleOCR",
    "paddle": "PaddleOCR",
}


def _disable_unused_nagisa_forced_alignment() -> None:
    """Avoid qwen-asr's eager Windows-only nagisa/DyNet failure.

    qwen-asr imports its optional forced aligner even when timestamps are not
    requested. This app does not use forced alignment, so a lightweight module
    placeholder keeps normal ASR available without loading nagisa's DyNet model.
    """
    if "nagisa" in sys.modules:
        return
    module = types.ModuleType("nagisa")

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("Japanese forced alignment is disabled in OnStreamLLM.")

    module.tagging = unavailable  # type: ignore[attr-defined]
    sys.modules["nagisa"] = module


class QwenTranslator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._pipe = None

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            return ""
        if self.settings.demo_mode:
            return f"[{target_language}] {text}"
        messages = self._messages(text, source_language, target_language)
        if self._pipe is None:
            from transformers import pipeline

            device = self._resolve_device()
            model_path = _local_model_path(self.settings.translation_model)
            self._pipe = pipeline(
                "text-generation",
                model=model_path,
                device_map=device,
                dtype="auto",
            )
        return self._translate_local(messages, multiline="\n" in text)

    def _messages(self, text: str, source_language: str, target_language: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a real-time subtitle translator. "
                    f"Translate from {source_language} into {target_language}. "
                    f"The output language must be {target_language}. "
                    "Do not paraphrase in the source language. "
                    "Output exactly one translation. If the input has multiple lines, "
                    "translate each line in the same order. Do not continue the source text, "
                    "repeat phrases, explain, or add labels. Ignore OCR artifacts such as "
                    "timestamps, usernames, icons, and UI system messages.\n"
                    f"Additional rules:\n{self.settings.llm_rules}"
                ),
            },
            {"role": "user", "content": text},
        ]

    def _translate_local(self, messages: list[dict], multiline: bool = False) -> str:
        try:
            prompt = self._pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self._pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        result = self._pipe(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.12,
            return_full_text=False,
        )
        return _clean_translation(str(result[0]["generated_text"]), multiline=multiline)

    def _resolve_device(self) -> str:
        return normalize_device(self.settings.translation_device or self.settings.device)

    def release(self) -> None:
        self._pipe = None


class HyMt2Translator:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._llama = None

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        if not text.strip():
            return ""
        if self.settings.demo_mode:
            return f"[{target_language}] {text}"
        prompt = (
            "You are a real-time subtitle translator. "
            f"Translate from {source_language} into {target_language}. "
            f"The output language must be {target_language}. "
            "Do not paraphrase in the source language. "
            "Preserve line breaks and translate each line in the same order. "
            "Return only the translated result without explanations, labels, or speaker names.\n\n"
            f"{text}"
        )
        external_error: Exception | None = None
        prefer_external = _requires_stq_llama_cpp(self.settings.translation_model) or (
            _is_cuda_device(self.settings.translation_device)
            and _llama_has_gpu_device(self.settings.translation_model)
            and not _llama_cpp_python_has_gpu()
        )
        if prefer_external:
            try:
                external = self._translate_with_llama_completion(prompt, multiline="\n" in text)
                if external and not _looks_unusable_translation(external, text):
                    return external
                if external:
                    raise RuntimeError(_hymt2_unusable_output_message())
            except Exception as exc:
                external_error = exc
        try:
            translated = self._translate_with_llama_cpp_python(prompt, multiline="\n" in text)
            if _looks_unusable_translation(translated, text):
                raise RuntimeError(_hymt2_unusable_output_message())
            return translated
        except Exception as exc:
            missing_module = _missing_python_module_name(exc)
            if missing_module is not None:
                raise RuntimeError(_missing_runtime_module_message(missing_module)) from exc
            try:
                external = self._translate_with_llama_completion(prompt, multiline="\n" in text)
                if external and not _looks_unusable_translation(external, text):
                    return external
                if external:
                    raise RuntimeError(_hymt2_unusable_output_message())
            except Exception as retry_exc:
                external_error = retry_exc
            detail = f" 상세: {external_error}" if external_error else ""
            raise RuntimeError(
                "Hy-MT2 GGUF 실행 실패. 2bit/1.25bit 모델은 STQ 커널이 포함된 "
                f"llama-completion.exe가 필요합니다.{detail}"
            ) from exc

    def _translate_with_llama_cpp_python(self, prompt: str, multiline: bool) -> str:
        if self._llama is None:
            from llama_cpp import Llama

            gpu_layers = (
                -1
                if _is_cuda_device(self.settings.translation_device)
                and _llama_cpp_python_has_gpu()
                and not _requires_stq_llama_cpp(self.settings.translation_model)
                else 0
            )
            self._llama = Llama(
                model_path=_gguf_model_path(self.settings.translation_model),
                n_ctx=1024,
                n_threads=max(
                    1,
                    effective_cpu_thread_count(
                        self.settings.translation_cpu_threads,
                        self.settings.translation_cpu_core_ids,
                        default=4,
                    ),
                ),
                n_gpu_layers=gpu_layers,
                verbose=False,
            )
        result = self._llama(
            prompt,
            max_tokens=128,
            temperature=0.2,
            top_p=0.6,
            top_k=20,
            repeat_penalty=1.05,
            echo=False,
        )
        return _clean_translation(str(result["choices"][0]["text"]), multiline=multiline)

    def _translate_with_llama_completion(self, prompt: str, multiline: bool) -> str:
        executable = _find_llama_completion(self.settings.translation_model)
        if executable is None:
            return ""
        model_path, working_dir = _gguf_model_path_for_cli(self.settings.translation_model)
        gpu_layers = (
            "all"
            if _is_cuda_device(self.settings.translation_device)
            and _llama_has_gpu_device(self.settings.translation_model)
            else "0"
        )
        command = [
            str(executable),
            "--model",
            model_path,
            "-f",
            "",
            "--jinja",
            "--no-display-prompt",
            "-ngl",
            gpu_layers,
            "-n",
            "128",
            "-c",
            "1024",
            "--temp",
            "0.2",
            "--top-p",
            "0.6",
            "--top-k",
            "20",
            "--no-warmup",
            "-st",
        ]
        prompt_path = _write_llama_prompt_file(prompt)
        command[command.index("-f") + 1] = str(prompt_path)
        gpu_index = _cuda_device_index(self.settings.translation_device)
        if gpu_layers != "0" and gpu_index is not None:
            command += ["-mg", str(gpu_index)]
        cpu_threads = effective_cpu_thread_count(
            self.settings.translation_cpu_threads,
            self.settings.translation_cpu_core_ids,
        )
        if gpu_layers == "0" and cpu_threads > 0:
            command += ["-t", str(cpu_threads)]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=False,
                timeout=120,
                cwd=str(working_dir) if working_dir is not None else None,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        finally:
            try:
                prompt_path.unlink(missing_ok=True)
            except OSError:
                pass
        stdout = _decode_process_output(completed.stdout)
        stderr = _decode_process_output(completed.stderr)
        if completed.returncode != 0:
            detail = (stderr or stdout or "").strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            raise RuntimeError(
                f"llama-completion 실행 실패(code {completed.returncode}): {detail}"
            )
        return _clean_llama_completion_output(stdout, multiline=multiline)

    def release(self) -> None:
        close = getattr(self._llama, "close", None)
        if callable(close):
            close()
        self._llama = None


class QwenAsr:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._pipe = None

    def preload(self, language: str = "auto") -> None:
        self._ensure_model()

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto") -> str:
        if self.settings.demo_mode:
            return "오디오 감지됨"
        self._ensure_model()
        language = None if language == "auto" else language
        with cpu_execution_context(self.settings.asr_cpu_threads, self.settings.asr_cpu_core_ids):
            result = self._pipe.transcribe(audio=(audio, sample_rate), language=language)
        return str(result[0].text).strip()

    def _ensure_model(self) -> None:
        if self._pipe is not None:
            return
        import torch

        _disable_unused_nagisa_forced_alignment()
        from qwen_asr import Qwen3ASRModel

        device = self._device_name()
        model_path = _local_model_path(self.settings.asr_model)
        self._pipe = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            device_map=device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def _device_name(self) -> str:
        return normalize_device(self.settings.asr_device or self.settings.device)

    def release(self) -> None:
        self._pipe = None


class SenseVoiceAsr:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._recognizers: dict[str, object] = {}

    def preload(self, language: str = "auto") -> None:
        self._recognizer_for(language)

    def transcribe(self, audio: np.ndarray, sample_rate: int, language: str = "auto") -> str:
        if self.settings.demo_mode:
            return "오디오 감지됨"
        recognizer = self._recognizer_for(language)
        audio = _to_mono_float32(audio)
        if sample_rate != 16000:
            audio = _resample_linear(audio, sample_rate, 16000)
            sample_rate = 16000
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        with cpu_execution_context(self.settings.asr_cpu_threads, self.settings.asr_cpu_core_ids):
            recognizer.decode_stream(stream)
        text = str(stream.result.text).strip()
        return "" if _is_filler_transcript(text) else text

    def _recognizer_for(self, language: str) -> object:
        language_code = LANGUAGE_ALIASES.get(language.casefold(), "")
        provider = "cuda" if _is_cuda_device(self.settings.asr_device) else "cpu"
        if provider == "cuda":
            _ensure_cuda_dll_directories()
        import sherpa_onnx

        cache_key = f"{provider}:{language_code}"
        if cache_key in self._recognizers:
            return self._recognizers[cache_key]
        model_dir = _ascii_safe_sensevoice_dir(_model_dir_path(self.settings.asr_model))
        errors: list[str] = []
        providers = [provider]
        if provider != "cpu":
            providers.append("cpu")
        for selected_provider in providers:
            cache_key = f"{selected_provider}:{language_code}"
            for model_path in _sensevoice_model_paths(model_dir):
                try:
                    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                        model=_sherpa_path(model_path),
                        tokens=_sherpa_path(model_dir / "tokens.txt"),
                        provider=selected_provider,
                        language=language_code,
                        use_itn=False,
                    )
                    self._recognizers[cache_key] = recognizer
                    return recognizer
                except Exception as exc:
                    errors.append(f"{selected_provider}/{model_path.name}: {exc}")
        else:
            raise RuntimeError("SenseVoice 모델 로드 실패: " + " / ".join(errors))

    def release(self) -> None:
        self._recognizers.clear()


def _local_model_path(model: str) -> str:
    path = Path(model)
    if path.exists():
        return str(path.resolve())
    for root in _runtime_roots():
        candidate = root / path
        if candidate.exists():
            return str(candidate.resolve())
    downloaded = _downloaded_model_path(model)
    if downloaded is not None:
        return str(downloaded.resolve())
    return model


def _model_dir_path(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path
    for root in _runtime_roots():
        candidate = root / path
        if candidate.exists():
            return candidate
    downloaded = _downloaded_model_path(model)
    if downloaded is not None:
        return downloaded
    return Path(model)


def _downloaded_model_path(model: str) -> Path | None:
    if "/" not in model:
        return None
    for kind in ("asr", "translation"):
        candidate = model_download_path(kind, model)
        if is_model_complete(candidate):
            return candidate
    return None


def _is_sensevoice_model(model: str) -> bool:
    return "sense-voice" in model.lower() or "sensevoice" in model.lower()


def _is_hymt2_model(model: str) -> bool:
    return "hy-mt2" in model.lower()


def _is_filler_transcript(text: str) -> bool:
    normalized = re.sub(r"[\s。.!！?？,，、~…]+", "", text).casefold()
    return normalized in {
        "",
        "嗯",
        "呃",
        "啊",
        "え",
        "ええ",
        "あ",
        "ああ",
        "어",
        "음",
        "응",
        "아",
        "uh",
        "um",
        "ah",
        "er",
    }


def _requires_stq_llama_cpp(model: str) -> bool:
    normalized = model.lower()
    return "hy-mt2" in normalized and ("2bit" in normalized or "1.25bit" in normalized)


def _is_cuda_device(device: str) -> bool:
    return normalize_device(device).startswith("cuda")


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        LOGGER.debug("CUDA memory cleanup skipped", exc_info=True)


def _cuda_device_index(device: str) -> int | None:
    normalized = normalize_device(device)
    if not normalized.startswith("cuda"):
        return None
    if ":" not in normalized:
        return 0
    try:
        return int(normalized.split(":", 1)[1])
    except ValueError:
        return 0


def _ensure_cuda_dll_directories() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    candidates: list[Path] = []
    try:
        import torch

        candidates.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    candidates.extend(
        [
            Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
            Path(sys.prefix) / "Library" / "bin",
        ]
    )
    for candidate in candidates:
        if not candidate.exists() or not (candidate / "cudnn64_9.dll").exists():
            continue
        candidate_key = str(candidate.resolve())
        if candidate_key in _DLL_DIRECTORY_PATHS:
            continue
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(candidate_key))
        _DLL_DIRECTORY_PATHS.add(candidate_key)
        os.environ["PATH"] = candidate_key + os.pathsep + os.environ.get("PATH", "")


def _sensevoice_model_path(model_dir: Path) -> Path:
    return _sensevoice_model_paths(model_dir)[0]


def _sensevoice_model_paths(model_dir: Path) -> list[Path]:
    paths: list[Path] = []
    int8 = model_dir / "model.int8.onnx"
    if int8.exists():
        paths.append(int8)
    model = model_dir / "model.onnx"
    if model.exists():
        paths.append(model)
    if paths:
        return paths
    raise FileNotFoundError(f"SenseVoice ONNX 모델 파일을 찾을 수 없습니다: {model_dir}")


def _sherpa_path(path: Path) -> str:
    return str(path.resolve())


def _ascii_safe_sensevoice_dir(model_dir: Path) -> Path:
    resolved = model_dir.resolve()
    if _is_ascii_path(resolved):
        return resolved
    cache_root = Path(tempfile.gettempdir()) / "OnStreamLLM" / "sensevoice"
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    destination = cache_root / digest
    model_file = resolved / "model.int8.onnx"
    if not model_file.exists():
        model_file = resolved / "model.onnx"
    source_files = [path for path in (model_file, resolved / "tokens.txt") if path.exists()]
    if not source_files:
        return resolved
    destination.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        target = destination / source.name
        if _copy_needed(source, target):
            _copy_model_file(source, target)
    return destination


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _copy_needed(source: Path, target: Path) -> bool:
    if not target.exists():
        return True
    try:
        source_stat = source.stat()
        target_stat = target.stat()
    except OSError:
        return True
    return source_stat.st_size != target_stat.st_size or int(source_stat.st_mtime) != int(target_stat.st_mtime)


def _copy_model_file(source: Path, target: Path) -> None:
    temp_target = target.with_suffix(target.suffix + ".tmp")
    try:
        shutil.copy2(source, temp_target)
        temp_target.replace(target)
    except PermissionError:
        if target.exists() and target.stat().st_size > 0:
            return
        raise
    finally:
        try:
            if temp_target.exists():
                temp_target.unlink()
        except OSError:
            pass


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    for root in _runtime_roots():
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            continue
    return str(path)


def _gguf_model_path(model: str) -> str:
    path = Path(_local_model_path(model))
    if path.is_file() and path.suffix.lower() == ".gguf":
        return str(path.resolve())
    if path.is_dir():
        candidate = next(path.glob("*.gguf"), None)
        if candidate:
            return str(candidate.resolve())
    raise FileNotFoundError(f"GGUF 모델 파일을 찾을 수 없습니다: {model}")


def _gguf_model_path_for_cli(model: str) -> tuple[str, Path | None]:
    resolved = Path(_gguf_model_path(model)).resolve()
    for root in _runtime_roots():
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        if "models" in relative.parts and (root / relative).exists():
            return str(relative), root.resolve()
    return str(resolved), None


def _find_llama_completion(model: str = "") -> Path | None:
    needs_stq = _requires_stq_llama_cpp(model)
    for root in _runtime_roots():
        gpu_candidates = [
            root / "runtime" / "llama.cpp" / "build-cuda" / "bin" / "Release" / "llama-completion.exe",
            root / "runtime" / "llama.cpp" / "build-cuda" / "bin" / "llama-completion.exe",
            root / "runtime" / "llama.cpp" / "build-cuda-ninja" / "bin" / "llama-completion.exe",
            root / "runtime" / "llama.cpp" / "build-cuda-vs-ninja" / "bin" / "llama-completion.exe",
        ]
        stq_candidates = [
            root / "runtime" / "llama.cpp" / "build-stq" / "bin" / "Release" / "llama-completion.exe",
        ]
        cpu_candidates = [
            root / "runtime" / "llama.cpp" / "llama-completion.exe",
            root / "runtime" / "llama.cpp" / "build" / "bin" / "Release" / "llama-completion.exe",
            root / "runtime" / "llama.cpp" / "build" / "bin" / "llama-completion.exe",
        ]
        candidates = stq_candidates + gpu_candidates + cpu_candidates if needs_stq else gpu_candidates + cpu_candidates + stq_candidates
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    return None


def _llama_has_gpu_device(model: str = "") -> bool:
    cache_key = "stq" if _requires_stq_llama_cpp(model) else "default"
    if cache_key in _LLAMA_GPU_DEVICE_CACHE:
        return _LLAMA_GPU_DEVICE_CACHE[cache_key]
    executable = _find_llama_completion(model)
    if executable is None:
        _LLAMA_GPU_DEVICE_CACHE[cache_key] = False
        return False
    try:
        completed = subprocess.run(
            [str(executable), "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception:
        _LLAMA_GPU_DEVICE_CACHE[cache_key] = False
        return False
    output = f"{completed.stdout}\n{completed.stderr}".casefold()
    device_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and "available devices" not in line
    ]
    _LLAMA_GPU_DEVICE_CACHE[cache_key] = completed.returncode == 0 and bool(device_lines)
    return _LLAMA_GPU_DEVICE_CACHE[cache_key]


def _llama_cpp_python_has_gpu() -> bool:
    global _LLAMA_CPP_PYTHON_HAS_GPU
    if _LLAMA_CPP_PYTHON_HAS_GPU is not None:
        return _LLAMA_CPP_PYTHON_HAS_GPU
    try:
        import llama_cpp

        supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        _LLAMA_CPP_PYTHON_HAS_GPU = bool(supports and supports())
    except Exception:
        _LLAMA_CPP_PYTHON_HAS_GPU = False
    return _LLAMA_CPP_PYTHON_HAS_GPU


def _hy_mt2_uses_gpu(model: str) -> bool:
    if _requires_stq_llama_cpp(model):
        return _llama_has_gpu_device(model)
    return _llama_cpp_python_has_gpu() or _llama_has_gpu_device(model)


def _should_force_asr_cpu_for_hymt2_gpu(settings: AppSettings) -> bool:
    if _is_sensevoice_model(settings.asr_model):
        return _is_cuda_device(settings.asr_device or settings.device)
    return (
        _is_hymt2_model(settings.translation_model)
        and _is_cuda_device(settings.asr_device or settings.device)
        and _is_cuda_device(settings.translation_device or settings.device)
        and _hy_mt2_uses_gpu(settings.translation_model)
    )


def _runtime_roots() -> list[Path]:
    roots = [Path.cwd()]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        roots.append(exe_dir / "runtime_libraries")
        roots.append(exe_dir)
        roots.append(exe_dir.parent)
        roots.append(exe_dir.parent.parent)
        bundle_dir = getattr(sys, "_MEIPASS", "")
        if bundle_dir:
            roots.append(Path(bundle_dir))
            roots.append(Path(bundle_dir) / "runtime_libraries")
    module_root = Path(__file__).resolve().parents[2]
    roots.append(Path.cwd() / "runtime_libraries")
    roots.append(module_root)
    unique_roots: list[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _clean_llama_completion_output(text: str, multiline: bool = False) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("build:", "main:", "llama_", "ggml_")):
            continue
        lines.append(stripped)
    return _clean_translation("\n".join(lines), multiline=multiline)


def _write_llama_prompt_file(prompt: str) -> Path:
    prompt_root = Path(tempfile.gettempdir()) / "OnStreamLLM" / "prompts"
    prompt_root.mkdir(parents=True, exist_ok=True)
    name = f"prompt-{threading.get_ident()}-{time.time_ns()}.txt"
    path = prompt_root / name
    path.write_text(prompt, encoding="utf-8")
    return path


def _decode_process_output(data: bytes) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _looks_unusable_translation(translated: str, source: str) -> bool:
    if not _contains_cjk(source):
        return False
    compact = re.sub(r"\s+", "", translated)
    if len(compact) < 3:
        return False
    question_count = compact.count("?") + compact.count("�")
    return question_count >= 3 and question_count / len(compact) >= 0.5


def _missing_runtime_dependency_name(exc: ModuleNotFoundError) -> str:
    return _MISSING_RUNTIME_MODULE_NAMES.get(exc.name or "", exc.name or "runtime")


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3040" <= char <= "\u30ff"
        or "\u3400" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7af"
        for char in text
    )


_RUNTIME_STDLIB_MODULES = frozenset({"pickletools", "sqlite3"})


def _missing_python_module_name(exc: BaseException) -> str | None:
    if isinstance(exc, ModuleNotFoundError):
        name = str(getattr(exc, "name", "") or "").strip()
        if name:
            return name
    if isinstance(exc, ImportError):
        message = str(exc)
        marker = "No module named "
        if marker in message:
            raw = message.split(marker, 1)[1].strip()
            return raw.strip("'\"")
    return None


def _missing_runtime_module_message(module_name: str) -> str:
    if module_name in _RUNTIME_STDLIB_MODULES:
        return (
            "Hy-MT2 실행에 필요한 Python 구성 요소가 앱에 포함되어 있지 않습니다: "
            f"{module_name}. 최신 OnStreamLLM 빌드로 업데이트한 뒤 다시 시도해 주세요."
        )
    return (
        "Hy-MT2 실행에 필요한 라이브러리가 누락되었습니다: "
        f"{module_name}. 모델 관리 탭에서 llama 라이브러리를 재설치해 주세요."
    )


def _hymt2_unusable_output_message() -> str:
    return (
        "Hy-MT2 GGUF가 현재 llama.cpp 실행파일에서 한중일 원문을 제대로 읽지 못했습니다. "
        "한국어/일본어/중국어 원문 번역은 Qwen 번역 모델을 사용하거나, "
        "Hy-MT2용 최신 UTF-8/STQ llama.cpp 빌드가 필요합니다."
    )


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / source_rate
    target_length = max(1, int(duration * target_rate))
    source_positions = np.linspace(0, len(audio) - 1, num=len(audio), dtype=np.float32)
    target_positions = np.linspace(0, len(audio) - 1, num=target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _clean_translation(text: str, multiline: bool = False) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "").strip()
    text = text.replace("[end of text]", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    if not multiline:
        return lines[0]
    deduplicated: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(line)
    return "\n".join(deduplicated)


class TranslationPipeline:
    def __init__(
        self,
        settings: AppSettings,
        bus: CaptionBus,
        status_callback: Callable[[SourceKind, str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.tr = Translator(settings.ui_language)
        self.bus = bus
        self.asr_settings = (
            replace(settings, asr_device="cpu")
            if _should_force_asr_cpu_for_hymt2_gpu(settings)
            else settings
        )
        self.translation_settings = settings
        self.asr = (
            SenseVoiceAsr(self.asr_settings)
            if _is_sensevoice_model(settings.asr_model)
            else QwenAsr(self.asr_settings)
        )
        self.translator = (
            HyMt2Translator(self.translation_settings)
            if _is_hymt2_model(settings.translation_model)
            else QwenTranslator(self.translation_settings)
        )
        self.status_callback = status_callback or (lambda _source, _message: None)
        self._asr_lock = threading.Lock()
        self._translation_lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = threading.Event()

    def release(self) -> None:
        self._closed.set()
        self._ready.clear()
        with self._asr_lock:
            release = getattr(self.asr, "release", None)
            if callable(release):
                release()
        with self._translation_lock:
            release = getattr(self.translator, "release", None)
            if callable(release):
                release()
        _release_cuda_memory()

    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._closed.is_set()

    def runtime_device_report(self) -> str:
        requested_asr_device = normalize_device(self.settings.asr_device or self.settings.device)
        asr_device = normalize_device(self.asr_settings.asr_device or self.asr_settings.device)
        translation_device = normalize_device(
            self.settings.translation_device or self.settings.device
        )
        asr_status = f"STT {asr_device}"
        if isinstance(self.asr, SenseVoiceAsr) and _is_cuda_device(asr_device):
            asr_status += " (SenseVoice CUDA 사용)"
        elif (
            _is_cuda_device(requested_asr_device)
            and asr_device == "cpu"
            and _is_hymt2_model(self.settings.translation_model)
        ):
            asr_status += " (LLM GPU 충돌 방지로 CPU 실행)"
        translation_status = f"LLM {translation_device}"
        if isinstance(self.translator, HyMt2Translator) and _is_cuda_device(translation_device):
            if _hy_mt2_uses_gpu(self.settings.translation_model):
                translation_status += " (llama.cpp GPU 사용 가능)"
            elif _requires_stq_llama_cpp(self.settings.translation_model):
                translation_status += " (2bit/STQ 모델은 현재 CPU 실행)"
            else:
                translation_status += " (llama.cpp GPU 미지원, CPU로 실행)"
        return f"{asr_status} / {translation_status}"

    def asr_work_label(self) -> str:
        device = normalize_device(self.asr_settings.asr_device or self.asr_settings.device)
        model_name = "SenseVoice" if isinstance(self.asr, SenseVoiceAsr) else "Qwen-ASR"
        if _is_cuda_device(device):
            return f"GPU ({model_name}) 연산중"
        return f"CPU ({model_name}) 연산중"

    def translation_work_label(self) -> str:
        device = normalize_device(self.settings.translation_device or self.settings.device)
        model_name = "Hy-MT2" if isinstance(self.translator, HyMt2Translator) else "Qwen"
        if _is_cuda_device(device) and not (
            isinstance(self.translator, HyMt2Translator)
            and not _hy_mt2_uses_gpu(self.settings.translation_model)
        ):
            return f"GPU ({model_name}) 연산중"
        if isinstance(self.translator, HyMt2Translator) and _requires_stq_llama_cpp(
            self.settings.translation_model
        ):
            return f"CPU ({model_name} 2bit/STQ) 연산중"
        return f"CPU ({model_name}) 연산중"

    def process_audio(self, source: SourceKind, audio: np.ndarray, sample_rate: int) -> None:
        threading.Thread(
            target=self._process_audio,
            args=(source, audio, sample_rate),
            daemon=True,
        ).start()

    def process_remote_audio(
        self,
        source: SourceKind,
        audio: np.ndarray,
        sample_rate: int,
        source_language: str,
        target_language: str,
    ) -> None:
        threading.Thread(
            target=self._process_remote_audio,
            args=(source, audio, sample_rate, source_language, target_language),
            daemon=True,
        ).start()

    def process_text(self, source: SourceKind, text: str) -> None:
        threading.Thread(target=self._process_text, args=(source, text), daemon=True).start()

    def translate_text(self, text: str, source_language: str, target_language: str) -> str:
        with self._translation_lock:
            with cpu_execution_context(
                self.settings.translation_cpu_threads,
                self.settings.translation_cpu_core_ids,
            ):
                return self.translator.translate(text, source_language, target_language)

    def process_remote_text(
        self,
        source: SourceKind,
        text: str,
        source_language: str,
        target_language: str,
    ) -> None:
        threading.Thread(
            target=self._process_remote_text,
            args=(source, text, source_language, target_language),
            daemon=True,
        ).start()

    def preload(self) -> None:
        threading.Thread(target=self._preload, daemon=True).start()

    def _preload(self) -> None:
        if self.settings.demo_mode:
            self._ready.set()
            return
        self._ready.clear()
        try:
            import numpy as np

            if self._closed.is_set():
                return
            LOGGER.info("Pipeline preload starting: %s", self.runtime_device_report())
            self.status_callback(
                SourceKind.SCREEN,
                self.tr.t(
                    "engine.model_loading",
                    devices=self.runtime_device_report(),
                ),
            )
            with self._asr_lock:
                if self._closed.is_set():
                    return
                LOGGER.info("Pipeline preload ASR starting")
                with cpu_execution_context(self.settings.asr_cpu_threads, self.settings.asr_cpu_core_ids):
                    if hasattr(self.asr, "preload"):
                        self.asr.preload("English")
                    else:
                        self.asr.transcribe(np.zeros(1600, dtype=np.float32), 16000, "English")
                LOGGER.info("Pipeline preload ASR complete")
            with self._translation_lock:
                if self._closed.is_set():
                    return
                LOGGER.info("Pipeline preload translator starting")
                with cpu_execution_context(
                    self.settings.translation_cpu_threads,
                    self.settings.translation_cpu_core_ids,
                ):
                    self.translator.translate("Hello", "English", "Korean")
                LOGGER.info("Pipeline preload translator complete")
            if self._closed.is_set():
                return
            self._ready.set()
            self.status_callback(
                SourceKind.SCREEN,
                self.tr.t(
                    "engine.model_load_complete",
                    devices=self.runtime_device_report(),
                ),
            )
            LOGGER.info("Pipeline preload complete")
        except ModuleNotFoundError as exc:
            self._ready.clear()
            LOGGER.exception("Pipeline preload failed")
            dependency_name = _missing_runtime_dependency_name(exc)
            self.status_callback(
                SourceKind.SCREEN,
                self.tr.t("engine.missing_runtime_library", name=dependency_name),
            )
        except Exception as exc:
            self._ready.clear()
            LOGGER.exception("Pipeline preload failed")
            self.status_callback(
                SourceKind.SCREEN,
                self.tr.t("engine.model_preload_failed", error=exc),
            )

    def _process_audio(self, source: SourceKind, audio: np.ndarray, sample_rate: int) -> None:
        try:
            if not self.is_ready():
                self.status_callback(
                    source,
                    self.tr.t(
                        "engine.model_loading",
                        devices=self.runtime_device_report(),
                    ),
                )
                return
            source_language, target_language = self.settings.languages_for(source)
            self.status_callback(
                source,
                self.tr.t("engine.asr_processing", label=self.asr_work_label()),
            )
            with self._asr_lock:
                with cpu_execution_context(self.settings.asr_cpu_threads, self.settings.asr_cpu_core_ids):
                    text = self.asr.transcribe(audio, sample_rate, source_language)
            self._publish_translation(source, text, source_language, target_language)
        except Exception as exc:
            LOGGER.exception("Audio model processing failed: source=%s", source.value)
            self.status_callback(source, self.tr.t("engine.model_process_failed", error=exc))

    def _process_remote_audio(
        self,
        source: SourceKind,
        audio: np.ndarray,
        sample_rate: int,
        source_language: str,
        target_language: str,
    ) -> None:
        try:
            self.status_callback(
                source,
                self.tr.t("engine.remote_audio_processing", label=self.asr_work_label()),
            )
            with self._asr_lock:
                with cpu_execution_context(self.settings.asr_cpu_threads, self.settings.asr_cpu_core_ids):
                    text = self.asr.transcribe(audio, sample_rate, source_language)
            if text.strip():
                self.status_callback(
                    source,
                    self.tr.t(
                        "engine.remote_translation_processing",
                        label=self.translation_work_label(),
                    ),
                )
                with self._translation_lock:
                    with cpu_execution_context(
                        self.settings.translation_cpu_threads,
                        self.settings.translation_cpu_core_ids,
                    ):
                        translated = self.translator.translate(text, source_language, target_language)
                self.bus.publish(Caption(source, text, translated, time.time()))
                self.status_callback(source, self.tr.t("engine.remote_translation_complete"))
        except Exception as exc:
            LOGGER.exception("Remote audio model processing failed: source=%s", source.value)
            self.status_callback(source, self.tr.t("engine.remote_process_failed", error=exc))

    def _process_text(self, source: SourceKind, text: str) -> None:
        try:
            if not self.is_ready():
                self.status_callback(
                    source,
                    self.tr.t(
                        "engine.model_loading",
                        devices=self.runtime_device_report(),
                    ),
                )
                return
            source_language, target_language = self.settings.languages_for(source)
            self._publish_translation(source, text, source_language, target_language)
        except Exception as exc:
            LOGGER.exception("Text translation failed: source=%s", source.value)
            self.status_callback(source, self.tr.t("engine.translation_failed", error=exc))

    def _process_remote_text(
        self,
        source: SourceKind,
        text: str,
        source_language: str,
        target_language: str,
    ) -> None:
        try:
            self.status_callback(
                source,
                self.tr.t(
                    "engine.remote_text_processing",
                    label=self.translation_work_label(),
                ),
            )
            with self._translation_lock:
                with cpu_execution_context(
                    self.settings.translation_cpu_threads,
                    self.settings.translation_cpu_core_ids,
                ):
                    translated = self.translator.translate(text, source_language, target_language)
            self.bus.publish(Caption(source, text, translated, time.time()))
            self.status_callback(source, self.tr.t("engine.remote_text_complete"))
        except Exception as exc:
            LOGGER.exception("Remote text translation failed: source=%s", source.value)
            self.status_callback(source, self.tr.t("engine.remote_text_failed", error=exc))

    def _publish_translation(
        self,
        source: SourceKind,
        text: str,
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> None:
        if not text.strip():
            return
        if source_language is None or target_language is None:
            source_language, target_language = self.settings.languages_for(source)
        self.status_callback(
            source,
            self.tr.t("engine.translation_processing", label=self.translation_work_label()),
        )
        with self._translation_lock:
            with cpu_execution_context(
                self.settings.translation_cpu_threads,
                self.settings.translation_cpu_core_ids,
            ):
                translated = self.translator.translate(text, source_language, target_language)
        self.bus.publish(Caption(source, text, translated, time.time()))
        self.status_callback(source, self.tr.t("engine.translation_complete"))
