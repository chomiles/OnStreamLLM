from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NvidiaDriverInfo:
    driver_version: tuple[int, ...]
    driver_version_raw: str
    cuda_version: tuple[int, ...] | None
    cuda_version_raw: str
    gpu_names: tuple[str, ...]


# Windows minimum driver versions for the CUDA wheels used by this app.
# Torch cu128 is the strictest requirement in the current stack.
CUDA_DRIVER_MINIMUMS: dict[str, tuple[int, ...]] = {
    "12.8": (570, 65),
    "12.4": (550, 54),
    "12.0": (527, 41),
}

RUNTIME_CUDA_REQUIREMENT = "12.8"
CUDA_RUNTIME_DEPENDENCIES: frozenset[str] = frozenset(
    {"Torch", "Qwen ASR", "llama", "SenseVoice"}
)


def parse_version(text: str) -> tuple[int, ...]:
    numbers = [int(part) for part in re.findall(r"\d+", text or "")]
    return tuple(numbers) if numbers else (0,)


def format_version(version: tuple[int, ...]) -> str:
    if not version:
        return "0"
    return ".".join(str(part) for part in version)


def minimum_driver_for_cuda(cuda_version: str = RUNTIME_CUDA_REQUIREMENT) -> tuple[int, ...]:
    return CUDA_DRIVER_MINIMUMS.get(cuda_version, CUDA_DRIVER_MINIMUMS[RUNTIME_CUDA_REQUIREMENT])


def query_nvidia_driver() -> NvidiaDriverInfo | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        header = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None

    rows = _decode_subprocess_bytes(result.stdout).splitlines()
    if not rows:
        return None

    driver_version_raw = ""
    gpu_names: list[str] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 2:
            continue
        if not driver_version_raw:
            driver_version_raw = parts[0]
        gpu_names.append(parts[1])

    if not driver_version_raw:
        return None

    cuda_version_raw = ""
    cuda_match = re.search(
        r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)",
        _decode_subprocess_bytes(header.stdout),
    )
    if cuda_match:
        cuda_version_raw = cuda_match.group(1)

    return NvidiaDriverInfo(
        driver_version=parse_version(driver_version_raw),
        driver_version_raw=driver_version_raw,
        cuda_version=parse_version(cuda_version_raw) if cuda_version_raw else None,
        cuda_version_raw=cuda_version_raw,
        gpu_names=tuple(gpu_names),
    )


def driver_meets_cuda_requirement(
    info: NvidiaDriverInfo,
    cuda_version: str = RUNTIME_CUDA_REQUIREMENT,
) -> bool:
    required = minimum_driver_for_cuda(cuda_version)
    if info.driver_version < required:
        return False
    if info.cuda_version is not None:
        required_cuda = parse_version(cuda_version)
        if info.cuda_version < required_cuda:
            return False
    return True


def driver_upgrade_message(
    info: NvidiaDriverInfo | None = None,
    *,
    cuda_version: str = RUNTIME_CUDA_REQUIREMENT,
) -> str:
    resolved = info or query_nvidia_driver()
    required = minimum_driver_for_cuda(cuda_version)
    required_text = format_version(required)
    if resolved is None:
        return (
            f"NVIDIA GPU 드라이버를 확인할 수 없습니다. "
            f"CUDA {cuda_version} 라이브러리 사용을 위해 "
            f"NVIDIA 드라이버 {required_text} 이상이 필요합니다."
        )

    current_text = resolved.driver_version_raw or format_version(resolved.driver_version)
    lines = [
        "NVIDIA 드라이버 버전이 낮아 CUDA 라이브러리와 호환되지 않습니다.",
        f"현재 드라이버: {current_text}",
        f"필요 드라이버: {required_text} 이상",
        f"필요 CUDA 런타임: {cuda_version}",
    ]
    if resolved.cuda_version_raw:
        lines.append(f"현재 드라이버가 지원하는 CUDA: {resolved.cuda_version_raw}")
    if resolved.gpu_names:
        lines.append(f"감지된 GPU: {', '.join(resolved.gpu_names)}")
    lines.append(
        "GeForce Experience 또는 NVIDIA 공식 사이트에서 드라이버를 업데이트한 뒤 다시 시도하세요."
    )
    return "\n".join(lines)


def check_cuda_driver_compatibility(
    dependency_name: str | None = None,
) -> tuple[bool, str]:
    if dependency_name is not None and dependency_name not in CUDA_RUNTIME_DEPENDENCIES:
        return True, ""
    info = query_nvidia_driver()
    if info is None:
        return True, ""
    if driver_meets_cuda_requirement(info):
        return True, ""
    return False, driver_upgrade_message(info)


def runtime_driver_status_message() -> str:
    info = query_nvidia_driver()
    if info is None:
        return "NVIDIA GPU가 없거나 드라이버를 확인할 수 없습니다. CPU 설치 경로를 사용합니다."
    if driver_meets_cuda_requirement(info):
        gpu_label = ", ".join(info.gpu_names) if info.gpu_names else "NVIDIA GPU"
        return (
            f"GPU 호환 확인됨: {gpu_label} "
            f"(드라이버 {info.driver_version_raw}, CUDA {info.cuda_version_raw or RUNTIME_CUDA_REQUIREMENT} 이상 필요)"
        )
    return driver_upgrade_message(info)


def _decode_subprocess_bytes(data: bytes) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
