from __future__ import annotations

import ctypes
import os
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True)
class ComputeDevice:
    label: str
    value: str
    available: bool = True


_PROCESS_CPU_LOCK = threading.RLock()


@dataclass(slots=True)
class CpuCore:
    index: int
    efficiency_class: int | None = None

    @property
    def label(self) -> str:
        suffix = "E" if self.efficiency_class == 0 else ""
        return f"{self.index}{suffix}"


def list_compute_devices() -> list[ComputeDevice]:
    devices = [ComputeDevice("CPU", "cpu")]
    torch_names = _torch_cuda_names()
    system_names = _nvidia_smi_names()
    names = torch_names or system_names
    for index, name in enumerate(names):
        available = True
        suffix = "" if index < len(torch_names) else " (llama GPU용)"
        devices.append(ComputeDevice(f"CUDA {index + 1}: {name}{suffix}", f"cuda:{index}", available))
    return devices


def list_cpu_cores() -> list[CpuCore]:
    cores = _windows_cpu_cores()
    if cores:
        return cores
    return [CpuCore(index) for index in range(os.cpu_count() or 1)]


def parse_cpu_core_ids(value: str) -> list[int]:
    core_ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            core_ids.append(int(part))
        except ValueError:
            continue
    available = {core.index for core in list_cpu_cores()}
    return [core_id for core_id in dict.fromkeys(core_ids) if core_id in available]


def format_cpu_core_ids(core_ids: str) -> str:
    selected = set(parse_cpu_core_ids(core_ids))
    if not selected:
        return "선택 없음"
    labels = [core.label for core in list_cpu_cores() if core.index in selected]
    return ", ".join(labels)


def effective_cpu_thread_count(thread_count: int = 0, core_ids: str = "", default: int = 0) -> int:
    if thread_count > 0:
        return thread_count
    selected_cores = parse_cpu_core_ids(core_ids)
    if thread_count < 0 and selected_cores:
        return len(selected_cores)
    return default


@contextmanager
def cpu_execution_context(thread_count: int = 0, core_ids: str = "") -> Iterator[None]:
    selected_cores = parse_cpu_core_ids(core_ids)
    selected_threads = effective_cpu_thread_count(thread_count, core_ids)
    if not selected_cores and selected_threads <= 0:
        yield
        return
    with _PROCESS_CPU_LOCK:
        previous_affinity = _get_process_affinity()
        if selected_cores:
            _set_process_affinity(selected_cores)
        _set_library_thread_count(selected_threads)
        try:
            yield
        finally:
            if previous_affinity:
                _set_process_affinity(previous_affinity)


def normalize_device(device: str) -> str:
    if device in ("auto", "cuda"):
        return "cuda:0" if _torch_cuda_names() else "cpu"
    return device


def validate_device(device: str) -> tuple[bool, str]:
    device = normalize_device(device)
    if device == "cpu":
        return True, ""
    names = _torch_cuda_names()
    if not names:
        names = _nvidia_smi_names()
    try:
        index = int(device.split(":", 1)[1])
    except (IndexError, ValueError):
        return False, f"잘못된 연산 장치입니다: {device}"
    if index >= len(names):
        return False, "선택한 GPU를 사용할 수 없습니다."
    return True, ""


def _torch_cuda_names() -> list[str]:
    try:
        import torch

        return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def _nvidia_smi_names() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def _set_library_thread_count(thread_count: int) -> None:
    if thread_count <= 0:
        return
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(thread_count)
    try:
        import torch

        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(max(1, min(4, thread_count)))
    except Exception:
        pass


def _get_process_affinity() -> list[int]:
    if os.name != "nt":
        try:
            return sorted(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except Exception:
            return []
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process, ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        return []
    return [index for index in range((os.cpu_count() or 1)) if process_mask.value & (1 << index)]


def _set_process_affinity(core_ids: list[int]) -> None:
    if not core_ids:
        return
    if os.name != "nt":
        try:
            os.sched_setaffinity(0, set(core_ids))  # type: ignore[attr-defined]
        except Exception:
            pass
        return
    mask = 0
    for core_id in core_ids:
        if core_id >= ctypes.sizeof(ctypes.c_size_t) * 8:
            continue
        mask |= 1 << core_id
    if mask == 0:
        return
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask))
    except Exception:
        pass


def _windows_cpu_cores() -> list[CpuCore]:
    if os.name != "nt":
        return []
    relation_processor_core = 0
    buffer_size = ctypes.c_ulong(0)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(buffer_size))
    if buffer_size.value <= 0:
        return []
    buffer = ctypes.create_string_buffer(buffer_size.value)
    if not kernel32.GetLogicalProcessorInformationEx(
        relation_processor_core, buffer, ctypes.byref(buffer_size)
    ):
        return []
    cores: list[CpuCore] = []
    offset = 0
    logical_index = 0
    while offset + 24 <= buffer_size.value:
        relation = int.from_bytes(buffer.raw[offset : offset + 4], "little")
        size = int.from_bytes(buffer.raw[offset + 4 : offset + 8], "little")
        if size <= 0:
            break
        if relation == relation_processor_core:
            efficiency_class = buffer.raw[offset + 9]
            group_count_offset = offset + 30
            group_count = int.from_bytes(buffer.raw[group_count_offset : group_count_offset + 2], "little")
            group_offset = offset + 32
            for group_index in range(group_count):
                mask_offset = group_offset + group_index * 16
                mask = int.from_bytes(buffer.raw[mask_offset : mask_offset + 8], "little")
                for bit in range(64):
                    if mask & (1 << bit):
                        cores.append(CpuCore(logical_index, efficiency_class))
                        logical_index += 1
        offset += size
    classes = {core.efficiency_class for core in cores if core.efficiency_class is not None}
    if len(classes) <= 1:
        return [CpuCore(core.index) for core in cores]
    lowest_class = min(classes)
    return [
        CpuCore(core.index, 0 if core.efficiency_class == lowest_class else core.efficiency_class)
        for core in cores
    ]
