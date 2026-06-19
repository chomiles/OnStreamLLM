from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime_dependencies import RuntimeDependency, RuntimeInstallSpec

logger = logging.getLogger(__name__)

_ACTIVE_INSTALL_LOCK = threading.Lock()
_ACTIVE_INSTALL_PROCESSES: list[subprocess.Popen[bytes]] = []

PYTHON_VERSION = "3.12.10"
PYTHON_EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

_PREPARE_INSTALL_STEPS = 3


def count_install_package_steps(spec: RuntimeInstallSpec) -> int:
    package_steps = 1 if spec.packages else 0
    return package_steps + len(spec.follow_up) + len(spec.follow_up_no_deps)


def count_install_steps(spec: RuntimeInstallSpec) -> int:
    return _PREPARE_INSTALL_STEPS + count_install_package_steps(spec)


def format_install_step(current: int, total: int, detail: str) -> str:
    return f"({current}/{total}) {detail}"


class InstallProgressTracker:
    def __init__(
        self,
        total_steps: int,
        emit: Callable[[int, str], None],
    ) -> None:
        self.total_steps = max(1, total_steps)
        self._emit = emit
        self.current_step = 0
        self._step_base_percent = 0
        self._step_span = 0

    def advance(self, detail: str) -> None:
        self.current_step = min(self.total_steps, self.current_step + 1)
        self._step_base_percent = int(((self.current_step - 1) / self.total_steps) * 95)
        self._step_span = max(1, int(95 / self.total_steps))
        self._emit(
            self._step_base_percent,
            format_install_step(self.current_step, self.total_steps, detail),
        )

    def update_detail(self, detail: str) -> None:
        if self.current_step <= 0:
            self._emit(0, detail)
            return
        percent = min(99, self._step_base_percent + self._step_span // 2)
        self._emit(
            percent,
            format_install_step(self.current_step, self.total_steps, detail),
        )


def nvidia_gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = _decode_subprocess_bytes(result.stdout)
    return bool(output.strip())


def pip_python_executable() -> Path:
    from .runtime_dependencies import portable_app_root

    if not getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()

    app_root = portable_app_root()
    for relative in ("python312/python.exe", ".python312/python.exe"):
        candidate = app_root / relative
        if candidate.is_file():
            return candidate
    return app_root / "python312" / "python.exe"


def ensure_pip_python(
    *,
    progress: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    python = pip_python_executable()
    if python.is_file():
        return python
    if not getattr(sys, "frozen", False):
        raise RuntimeError("현재 Python 환경을 찾을 수 없습니다.")
    _bootstrap_portable_python(python, progress=progress, cancel_event=cancel_event)
    if not python.is_file():
        raise RuntimeError(
            "포터블 Python 설치에 실패했습니다. "
            f"다음 경로에 python.exe가 없습니다: {python}"
        )
    return python


def install_runtime_dependency(
    dependency: RuntimeDependency,
    *,
    progress: Callable[[int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    reinstall: bool = False,
) -> Path:
    from .gpu_driver import check_cuda_driver_compatibility
    from .runtime_bootstrap import release_runtime_libraries
    from .runtime_dependencies import (
        clear_runtime_install_state,
        mark_runtime_installed,
        runtime_install_root,
        runtime_install_spec,
    )

    logger.info("Starting runtime install: %s", dependency.name)
    release_runtime_libraries()
    logger.info("Released runtime libraries before install: %s", dependency.name)
    terminate_active_install_processes()
    logger.info("Terminated stale installer processes before install: %s", dependency.name)

    spec = runtime_install_spec(dependency.name)
    if spec is None:
        raise RuntimeError(f"{dependency.name} 공식 설치 정보를 찾을 수 없습니다.")

    compatible, driver_message = check_cuda_driver_compatibility(dependency.name)
    if not compatible:
        raise RuntimeError(driver_message)

    def emit(percent: int, detail: str) -> None:
        if progress is not None:
            progress(percent, detail)

    tracker = InstallProgressTracker(count_install_steps(spec), emit)

    tracker.advance("Python 환경 준비 중...")
    python = ensure_pip_python(
        progress=tracker.update_detail,
        cancel_event=cancel_event,
    )
    _check_cancel(cancel_event)

    clear_runtime_install_state(dependency, reinstall=reinstall)
    target = runtime_install_root(dependency)
    target.mkdir(parents=True, exist_ok=True)

    tracker.advance("pip 업그레이드 중...")
    _run_pip(
        python,
        ["install", "--upgrade", "pip"],
        cancel_event=cancel_event,
        on_line=tracker.update_detail,
    )
    _check_cancel(cancel_event)

    tracker.advance("빌드 도구 업그레이드 중...")
    _run_pip(
        python,
        [
            "install",
            "--upgrade",
            "--no-cache-dir",
            "--no-warn-script-location",
            "setuptools>=75",
            "wheel",
        ],
        cancel_event=cancel_event,
        on_line=tracker.update_detail,
    )
    _check_cancel(cancel_event)

    _pip_install_spec(
        python,
        spec,
        target,
        tracker=tracker,
        cancel_event=cancel_event,
    )
    _check_cancel(cancel_event)

    if dependency.name == "llama":
        from .runtime_bootstrap import ensure_runtime_stdlib_shims

        ensure_runtime_stdlib_shims()

    mark_runtime_installed(dependency.name)
    emit(100, format_install_step(tracker.total_steps, tracker.total_steps, "설치 완료"))
    logger.info("Finished runtime install: %s", dependency.name)
    return target


def build_pip_install_args(
    spec: RuntimeInstallSpec,
    target: Path,
) -> list[str]:
    args = [
        "install",
        "--no-cache-dir",
        "--no-warn-script-location",
        "--target",
        str(target),
        *spec.packages,
    ]
    if spec.index_url:
        args.extend(["--index-url", spec.index_url])
    if spec.extra_index_url:
        args.extend(["--extra-index-url", spec.extra_index_url])
    if spec.find_links:
        args.extend(["--find-links", spec.find_links])
    return args


def build_pip_install_command(
    python: Path,
    spec: RuntimeInstallSpec,
    target: Path,
) -> list[str]:
    return [str(python), "-m", "pip", *build_pip_install_args(spec, target)]


def _pip_install_cooldown() -> None:
    if sys.platform == "win32":
        time.sleep(2.0)


def _pip_install_spec(
    python: Path,
    spec: RuntimeInstallSpec,
    target: Path,
    *,
    tracker: InstallProgressTracker,
    cancel_event: threading.Event | None,
) -> None:
    if spec.packages:
        _check_cancel(cancel_event)
        if len(spec.packages) == 1:
            detail = f"{spec.packages[0]} 설치 중..."
        else:
            detail = f"{', '.join(spec.packages)} 설치 중..."
        tracker.advance(detail)
        _run_pip(
            python,
            build_pip_install_args(spec, target),
            cancel_event=cancel_event,
            on_line=tracker.update_detail,
        )
        _pip_install_cooldown()
    for package in spec.follow_up:
        _check_cancel(cancel_event)
        tracker.advance(f"{package} 설치 중...")
        _run_pip(
            python,
            build_pip_target_install_args(package, target, spec=spec),
            cancel_event=cancel_event,
            on_line=tracker.update_detail,
        )
        _pip_install_cooldown()
    for package in spec.follow_up_no_deps:
        _check_cancel(cancel_event)
        tracker.advance(f"{package} 설치 중...")
        _run_pip(
            python,
            build_pip_target_install_args(
                package,
                target,
                no_deps=True,
                spec=spec,
                only_binary=_requires_prebuilt_wheel(package),
            ),
            cancel_event=cancel_event,
            on_line=tracker.update_detail,
        )
        _pip_install_cooldown()


def _requires_prebuilt_wheel(package: str) -> bool:
    lowered = package.casefold()
    return "llama-cpp-python" in lowered or "sherpa-onnx" in lowered


def build_pip_target_install_args(
    package: str,
    target: Path,
    *,
    no_deps: bool = False,
    spec: RuntimeInstallSpec | None = None,
    only_binary: bool = False,
) -> list[str]:
    args = [
        "install",
        "--no-cache-dir",
        "--no-warn-script-location",
        "--target",
        str(target),
        package,
    ]
    if no_deps:
        args.insert(1, "--no-deps")
    if only_binary:
        args.insert(1, "--only-binary=:all:")
    if spec is not None:
        if spec.index_url:
            args.extend(["--index-url", spec.index_url])
        if spec.extra_index_url:
            args.extend(["--extra-index-url", spec.extra_index_url])
        if spec.find_links:
            args.extend(["--find-links", spec.find_links])
    return args


def _bootstrap_portable_python(
    python: Path,
    *,
    progress: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    target_dir = python.parent.resolve()
    _prepare_embed_python_directory(target_dir)
    archive_path = Path(tempfile.gettempdir()) / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    get_pip_path = target_dir / "get-pip.py"

    if progress is not None:
        progress("python.org에서 Python 3.12 embed 패키지 다운로드 중...")
    _check_cancel(cancel_event)
    _download_file(PYTHON_EMBED_URL, archive_path, cancel_event=cancel_event)

    if progress is not None:
        progress("앱 폴더에 독립 Python 3.12 압축 해제 중...")
    _check_cancel(cancel_event)
    _extract_embed_python_archive(archive_path, target_dir, cancel_event=cancel_event)
    archive_path.unlink(missing_ok=True)
    _enable_embedded_site_imports(target_dir)

    if progress is not None:
        progress("pip 부트스트랩 다운로드 및 설치 중...")
    _check_cancel(cancel_event)
    _download_file(GET_PIP_URL, get_pip_path, cancel_event=cancel_event)
    _bootstrap_embedded_pip(python, get_pip_path, cancel_event=cancel_event)
    get_pip_path.unlink(missing_ok=True)

    if not python.is_file():
        raise RuntimeError(f"Python 부트스트랩 후에도 실행 파일을 찾을 수 없습니다: {python}")


def _prepare_embed_python_directory(target_dir: Path) -> None:
    if target_dir.exists() and not (target_dir / "python.exe").is_file():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)


def _extract_embed_python_archive(
    archive_path: Path,
    target_dir: Path,
    cancel_event: threading.Event | None,
) -> None:
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            _check_cancel(cancel_event)
            destination = (target_dir / member.filename).resolve()
            if target_root != destination and target_root not in destination.parents:
                raise RuntimeError("Python embed 압축 파일 경로가 올바르지 않습니다.")
            archive.extract(member, target_dir)


def _enable_embedded_site_imports(target_dir: Path) -> None:
    matches = list(target_dir.glob("python*._pth"))
    if not matches:
        raise RuntimeError("Python embed 설정 파일(.pth)을 찾을 수 없습니다.")
    pth_path = matches[0]
    lines = pth_path.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    enabled_site = False
    for line in lines:
        stripped = line.strip()
        if stripped == "#import site":
            updated.append("import site")
            enabled_site = True
        elif stripped == "import site":
            updated.append(line)
            enabled_site = True
        else:
            updated.append(line)
    if not enabled_site:
        updated.append("import site")
    pth_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _bootstrap_embedded_pip(
    python: Path,
    get_pip_path: Path,
    cancel_event: threading.Event | None,
) -> None:
    env = _utf8_subprocess_env()
    completed = subprocess.run(
        [str(python), str(get_pip_path), "--no-warn-script-location"],
        capture_output=True,
        check=False,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _check_cancel(cancel_event)
    if completed.returncode != 0:
        output = _decode_subprocess_bytes(completed.stdout) or _decode_subprocess_bytes(
            completed.stderr
        )
        raise RuntimeError(output or "pip 부트스트랩에 실패했습니다.")


def _download_file(
    url: str,
    destination: Path,
    *,
    cancel_event: threading.Event | None,
) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=(15, 120))
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            _check_cancel(cancel_event)
            if chunk:
                handle.write(chunk)


def terminate_active_install_processes() -> None:
    with _ACTIVE_INSTALL_LOCK:
        processes = list(_ACTIVE_INSTALL_PROCESSES)
    for process in processes:
        _terminate_process_tree(process)


def _register_install_process(process: subprocess.Popen[bytes]) -> None:
    with _ACTIVE_INSTALL_LOCK:
        _ACTIVE_INSTALL_PROCESSES.append(process)


def _unregister_install_process(process: subprocess.Popen[bytes]) -> None:
    with _ACTIVE_INSTALL_LOCK:
        if process in _ACTIVE_INSTALL_PROCESSES:
            _ACTIVE_INSTALL_PROCESSES.remove(process)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timed out while terminating installer process tree: %s", process.pid)
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _run_pip(
    python: Path,
    pip_args: list[str],
    *,
    cancel_event: threading.Event | None,
    on_line: Callable[[str], None] | None,
) -> None:
    last_error = ""
    for attempt in range(3):
        _check_cancel(cancel_event)
        try:
            last_error = _run_pip_once(
                python,
                pip_args,
                cancel_event=cancel_event,
                on_line=on_line,
            )
            return
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt >= 2 or not _is_retryable_install_error(last_error):
                raise
            if on_line is not None:
                on_line("파일 잠금으로 설치가 지연되었습니다. 잠시 후 다시 시도합니다...")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(format_install_error(last_error))


def _is_retryable_install_error(message: str) -> bool:
    lowered = message.casefold()
    return (
        "permissionerror" in lowered
        or "winerror 5" in lowered
        or "액세스가 거부" in lowered
        or "access is denied" in lowered
    )


def _run_pip_once(
    python: Path,
    pip_args: list[str],
    *,
    cancel_event: threading.Event | None,
    on_line: Callable[[str], None] | None,
) -> str:
    command = [str(python), "-m", "pip", *pip_args]
    env = _utf8_subprocess_env()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creationflags,
    )
    _register_install_process(process)
    output_lines: list[str] = []
    try:
        assert process.stdout is not None
        while True:
            _check_cancel(cancel_event, process)
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            decoded = _decode_subprocess_bytes(line)
            if decoded:
                output_lines.append(decoded)
                logger.info("pip: %s", decoded)
                if on_line is not None:
                    on_line(decoded)
        if process.returncode != 0:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("설치가 취소되었습니다.")
            tail = "\n".join(output_lines[-8:]).strip()
            raise RuntimeError(format_install_error(tail))
        return ""
    finally:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
        _unregister_install_process(process)


def _utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_PROGRESS_BAR"] = "off"
    return env


def _decode_subprocess_bytes(data: bytes) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").strip()


def format_install_error(message: str) -> str:
    normalized = " ".join(message.split())
    if not normalized:
        return "pip 설치에 실패했습니다."
    lowered = normalized.casefold()
    if (
        "permissionerror" in lowered
        or "winerror 5" in lowered
        or "액세스가 거부" in lowered
        or "access is denied" in lowered
    ):
        return (
            "설치 중 파일이 다른 프로그램에 의해 잠겨 있습니다. "
            "앱을 완전히 종료한 뒤 같은 라이브러리 설치를 다시 시도해 주세요."
        )
    if "scikit_build_core" in lowered or "scikit-build-core" in lowered:
        return (
            "llama-cpp-python 사전 빌드 wheel을 찾지 못했습니다. "
            "CUDA용 wheel 인덱스를 확인한 뒤 llama 설치를 다시 시도해 주세요."
        )
    if (
        "get_requires_for_build_wheel" in lowered
        or "build_wheel" in lowered
        or "setuptools.build_meta" in lowered
        or "backendunavailable" in lowered
    ):
        return (
            "패키지 빌드 준비 중 오류가 발생했습니다. "
            "이전 설치가 중단된 경우일 수 있으니 다시 시도해 주세요."
        )
    if len(normalized) > 240:
        return normalized[:240] + "..."
    return normalized


def _check_cancel(
    cancel_event: threading.Event | None,
    process: subprocess.Popen[bytes] | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        if process is not None:
            _terminate_process_tree(process)
        terminate_active_install_processes()
        raise InterruptedError("설치가 취소되었습니다.")
