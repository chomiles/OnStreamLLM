from __future__ import annotations

import tempfile
from pathlib import Path

from live_translate.i18n import Translator
from live_translate.runtime_dependencies import (
    RUNTIME_DEPENDENCIES,
    RUNTIME_INSTALL_ORDER,
    RuntimeDependency,
    clear_runtime_install_state,
    installed_runtime_dependency_names,
    runtime_dependencies_in_install_order,
    runtime_dependencies_pending_install,
    runtime_dependency_installed,
    runtime_install_available,
    runtime_install_root,
    runtime_install_spec,
)
from live_translate.runtime_installer import (
    GET_PIP_URL,
    PYTHON_EMBED_URL,
    InstallProgressTracker,
    _decode_subprocess_bytes,
    _enable_embedded_site_imports,
    _is_retryable_install_error,
    build_pip_install_args,
    build_pip_install_command,
    build_pip_target_install_args,
    count_install_steps,
    format_install_error,
    format_install_step,
    pip_python_executable,
)


def test_runtime_description_marks_slow_dependencies() -> None:
    tr = Translator("ko")
    torch_desc = tr.runtime_description("Torch", "")
    qwen_desc = tr.runtime_description("Qwen ASR", "")
    llama_desc = tr.runtime_description("llama", "")
    assert "※ 설치 시간이 오래 걸립니다." in torch_desc
    assert "※ 설치 시간이 오래 걸립니다." in qwen_desc
    assert "※ 설치 시간이 오래 걸립니다." not in llama_desc


def test_runtime_install_order_matches_safe_sequence() -> None:
    ordered_names = [dependency.name for dependency in runtime_dependencies_in_install_order()]
    assert ordered_names == list(RUNTIME_INSTALL_ORDER)
    assert ordered_names == [dependency.name for dependency in RUNTIME_DEPENDENCIES]


def test_runtime_install_specs_exist_for_all_dependencies() -> None:
    for dependency in RUNTIME_DEPENDENCIES:
        assert runtime_install_available(dependency.name)
        assert runtime_install_spec(dependency.name, cuda=True) is not None
        assert runtime_install_spec(dependency.name, cuda=False) is not None


def test_paddleocr_is_optional_and_excluded_from_required_install(monkeypatch) -> None:
    monkeypatch.setattr(
        "live_translate.runtime_dependencies.runtime_dependency_installed",
        lambda _dependency: False,
    )
    pending_required = [dependency.name for dependency in runtime_dependencies_pending_install()]
    pending_all = [
        dependency.name
        for dependency in runtime_dependencies_pending_install(include_optional=True)
    ]
    paddle = next(dependency for dependency in RUNTIME_DEPENDENCIES if dependency.name == "PaddleOCR")

    assert paddle.optional
    assert "PaddleOCR" not in pending_required
    assert "PaddleOCR" in pending_all


def test_torch_cuda_spec_uses_official_pytorch_index() -> None:
    spec = runtime_install_spec("Torch", cuda=True)
    assert spec is not None
    assert spec.index_url == "https://download.pytorch.org/whl/cu128"
    assert spec.packages == ("torch",)


def test_llama_spec_uses_prebuilt_wheel_index() -> None:
    spec = runtime_install_spec("llama", cuda=True)
    assert spec is not None
    assert spec.packages == ()
    assert spec.follow_up == ("diskcache>=5.6.1", "jinja2>=2.11.3")
    assert spec.follow_up_no_deps == ("llama-cpp-python==0.3.30",)
    assert spec.extra_index_url == "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    assert count_install_steps(spec) == 6
    target = Path("C:/runtime/python")
    args = build_pip_target_install_args(
        "llama-cpp-python==0.3.30",
        target,
        no_deps=True,
        spec=spec,
        only_binary=True,
    )
    assert "--only-binary=:all:" in args
    assert "--extra-index-url" in args
    assert "https://abetlen.github.io/llama-cpp-python/whl/cu124" in args


def test_qwen_asr_spec_uses_official_packages() -> None:
    spec = runtime_install_spec("Qwen ASR", cuda=True)
    assert spec is not None
    assert spec.packages == ("transformers==4.57.6",)
    assert "accelerate==1.12.0" in spec.follow_up
    assert "librosa" in spec.follow_up
    assert "sox" not in spec.follow_up
    assert spec.follow_up_no_deps == ("qwen-asr>=0.0.4",)
    assert count_install_steps(spec) == 10


def test_format_install_step_and_progress_tracker() -> None:
    assert format_install_step(3, 10, "librosa 설치 중...") == "(3/10) librosa 설치 중..."
    events: list[tuple[int, str]] = []

    tracker = InstallProgressTracker(4, lambda percent, detail: events.append((percent, detail)))
    tracker.advance("pip 업그레이드 중...")
    tracker.update_detail("Downloading librosa")
    tracker.advance("librosa 설치 중...")

    assert events[0][1].startswith("(1/4)")
    assert events[1][1].startswith("(1/4)")
    assert events[2][1].startswith("(2/4)")


def test_build_pip_target_install_args_supports_no_deps() -> None:
    target = Path("C:/runtime/python")
    args = build_pip_target_install_args("qwen-asr>=0.0.4", target, no_deps=True)
    assert args[:6] == [
        "install",
        "--no-deps",
        "--no-cache-dir",
        "--no-warn-script-location",
        "--target",
        str(target),
    ]
    assert "qwen-asr>=0.0.4" in args


def test_sensevoice_cuda_spec_uses_cpu_package_to_avoid_gpu_conflicts() -> None:
    spec = runtime_install_spec("SenseVoice", cuda=True)
    assert spec is not None
    assert spec.find_links == ""
    assert spec.packages == ("sherpa-onnx>=1.13.2",)


def test_build_pip_install_command_includes_target_and_index() -> None:
    spec = runtime_install_spec("Torch", cuda=True)
    assert spec is not None
    python = Path("C:/python/python.exe")
    target = Path("C:/runtime/python")
    args = build_pip_install_args(spec, target)
    command = build_pip_install_command(python, spec, target)
    assert args[0] == "install"
    assert "--upgrade" not in args
    assert "--no-cache-dir" in args
    assert args.count("pip") == 0
    assert command[:4] == [str(python), "-m", "pip", "install"]
    assert command[4:] == args[1:]
    assert "--target" in command
    assert str(target) in command
    assert "--index-url" in command
    assert "https://download.pytorch.org/whl/cu128" in command


def test_decode_subprocess_bytes_handles_utf8_and_cp949() -> None:
    assert _decode_subprocess_bytes("한국어 설치 완료\n".encode("utf-8")) == "한국어 설치 완료"
    assert _decode_subprocess_bytes("日本語テスト\n".encode("utf-8")) == "日本語テスト"
    assert _decode_subprocess_bytes("中文测试\n".encode("utf-8")) == "中文测试"
    assert _decode_subprocess_bytes("english ok\n".encode("ascii")) == "english ok"


def test_portable_python_bootstrap_uses_official_embed_and_get_pip_urls() -> None:
    assert PYTHON_EMBED_URL.endswith("python-3.12.10-embed-amd64.zip")
    assert GET_PIP_URL.endswith("get-pip.py")


def test_enable_embedded_site_imports_uncomments_import_site() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pth_path = root / "python312._pth"
        pth_path.write_text("python312.zip\n.\n#import site\n", encoding="utf-8")
        _enable_embedded_site_imports(root)
        content = pth_path.read_text(encoding="utf-8")
        assert "import site" in content
        assert "#import site" not in content


def test_paddleocr_spec_installs_core_packages_together() -> None:
    spec = runtime_install_spec("PaddleOCR", cuda=False)
    assert spec is not None
    assert spec.packages == ("paddlepaddle>=3.2", "paddleocr>=3.7")
    assert "pillow>=10.4" in spec.follow_up
    assert count_install_steps(spec) == 10


def test_is_retryable_install_error_detects_windows_permission_denied() -> None:
    assert _is_retryable_install_error("PermissionError: [WinError 5] 액세스가 거부되었습니다")
    assert not _is_retryable_install_error("No matching distribution found")


def test_format_install_error_replaces_permission_denied() -> None:
    raw = "PermissionError: [WinError 5] 액세스가 거부되었습니다: numpy.pyd"
    message = format_install_error(raw)
    assert "잠겨" in message
    assert "종료" in message


def test_format_install_error_replaces_build_wheel_trace() -> None:
    raw = "return super().get_requires_for_build_wheel(config_settings=cs)"
    message = format_install_error(raw)
    assert "get_requires_for_build_wheel" not in message
    assert "다시 시도" in message


def test_format_install_error_replaces_scikit_build_core() -> None:
    raw = "BackendUnavailable: Cannot import 'scikit_build_core.build'"
    message = format_install_error(raw)
    assert "llama-cpp-python" in message
    assert "wheel" in message


def test_clear_runtime_install_state_preserves_shared_folder_on_fresh_install(
    monkeypatch,
) -> None:
    dependency = next(item for item in RUNTIME_DEPENDENCIES if item.name == "Qwen ASR")
    with tempfile.TemporaryDirectory() as temp_dir:
        runtime_root = Path(temp_dir) / "runtime_libraries"
        monkeypatch.setattr(
            "live_translate.runtime_dependencies.runtime_library_root",
            lambda: runtime_root,
        )
        target = runtime_install_root(dependency)
        target.mkdir(parents=True)
        (target / "torch").mkdir()
        (target / "torch" / "marker.txt").write_text("keep", encoding="utf-8")
        marker = runtime_root / ".installed" / "qwen asr.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("installed\n", encoding="utf-8")
        other_marker = marker.parent / "torch.txt"
        other_marker.write_text("installed\n", encoding="utf-8")

        clear_runtime_install_state(dependency)

        assert target.is_dir()
        assert (target / "torch" / "marker.txt").read_text(encoding="utf-8") == "keep"
        assert not marker.exists()
        assert other_marker.is_file()


def test_clear_runtime_install_state_removes_target_and_marker_on_reinstall(
    monkeypatch,
) -> None:
    dependency = next(item for item in RUNTIME_DEPENDENCIES if item.name == "Qwen ASR")
    with tempfile.TemporaryDirectory() as temp_dir:
        runtime_root = Path(temp_dir) / "runtime_libraries"
        monkeypatch.setattr(
            "live_translate.runtime_dependencies.runtime_library_root",
            lambda: runtime_root,
        )
        target = runtime_install_root(dependency)
        target.mkdir(parents=True)
        (target / "partial.txt").write_text("leftover", encoding="utf-8")
        marker = runtime_root / ".installed" / "qwen asr.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("installed\n", encoding="utf-8")
        other_marker = marker.parent / "sensevoice.txt"
        other_marker.write_text("installed\n", encoding="utf-8")

        clear_runtime_install_state(dependency, reinstall=True)

        assert not target.exists()
        assert not marker.exists()
        assert not other_marker.exists()


def test_installed_runtime_dependency_names_excludes_requested_dependency(monkeypatch) -> None:
    dependency = next(item for item in RUNTIME_DEPENDENCIES if item.name == "SenseVoice")

    def fake_installed(candidate: RuntimeDependency) -> bool:
        return candidate.name in {"Torch", "SenseVoice"}

    monkeypatch.setattr(
        "live_translate.runtime_dependencies.runtime_dependency_installed",
        fake_installed,
    )

    assert installed_runtime_dependency_names(exclude=dependency.name) == ("Torch",)
    assert installed_runtime_dependency_names() == ("Torch", "SenseVoice")


def test_runtime_dependency_installed_clears_stale_marker(monkeypatch) -> None:
    dependency = next(item for item in RUNTIME_DEPENDENCIES if item.name == "SenseVoice")
    with tempfile.TemporaryDirectory() as temp_dir:
        runtime_root = Path(temp_dir) / "runtime_libraries"
        monkeypatch.setattr(
            "live_translate.runtime_dependencies.runtime_library_root",
            lambda: runtime_root,
        )
        monkeypatch.setattr(
            "live_translate.runtime_dependencies._runtime_packages_available",
            lambda _dependency: False,
        )
        marker = runtime_root / ".installed" / "sensevoice.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("installed\n", encoding="utf-8")

        assert runtime_dependency_installed(dependency) is False
        assert not marker.exists()


def test_pip_python_executable_uses_current_interpreter_in_dev_mode() -> None:
    import sys

    if getattr(sys, "frozen", False):
        return
    assert pip_python_executable() == Path(sys.executable).resolve()
