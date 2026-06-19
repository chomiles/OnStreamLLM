from __future__ import annotations

from live_translate.gpu_driver import (
    NvidiaDriverInfo,
    check_cuda_driver_compatibility,
    driver_meets_cuda_requirement,
    driver_upgrade_message,
    format_version,
    minimum_driver_for_cuda,
    parse_version,
)


def test_parse_version_handles_driver_and_cuda_strings() -> None:
    assert parse_version("570.65") == (570, 65)
    assert parse_version("12.8") == (12, 8)
    assert parse_version("546.12.08") == (546, 12, 8)


def test_driver_meets_cuda_requirement_accepts_new_enough_driver() -> None:
    info = NvidiaDriverInfo(
        driver_version=(576, 52),
        driver_version_raw="576.52",
        cuda_version=(12, 8),
        cuda_version_raw="12.8",
        gpu_names=("NVIDIA GeForce RTX 4090",),
    )
    assert driver_meets_cuda_requirement(info)


def test_driver_meets_cuda_requirement_rejects_old_driver() -> None:
    info = NvidiaDriverInfo(
        driver_version=(546, 12),
        driver_version_raw="546.12",
        cuda_version=(12, 4),
        cuda_version_raw="12.4",
        gpu_names=("NVIDIA GeForce RTX 3060",),
    )
    assert not driver_meets_cuda_requirement(info)


def test_driver_upgrade_message_includes_minimum_driver_text() -> None:
    info = NvidiaDriverInfo(
        driver_version=(546, 12),
        driver_version_raw="546.12",
        cuda_version=(12, 4),
        cuda_version_raw="12.4",
        gpu_names=("NVIDIA GeForce RTX 3060",),
    )
    message = driver_upgrade_message(info)
    assert format_version(minimum_driver_for_cuda()) in message
    assert "546.12" in message
    assert "GeForce RTX 3060" in message
    assert "업데이트" in message


def test_check_cuda_driver_compatibility_skips_non_cuda_dependencies() -> None:
    for name in ("PaddleOCR",):
        compatible, message = check_cuda_driver_compatibility(name)
        assert compatible
        assert message == ""


def test_qwen_asr_is_cuda_runtime_dependency() -> None:
    from live_translate.gpu_driver import CUDA_RUNTIME_DEPENDENCIES

    assert "Qwen ASR" in CUDA_RUNTIME_DEPENDENCIES


def test_check_cuda_driver_compatibility_reports_low_driver(monkeypatch) -> None:
    info = NvidiaDriverInfo(
        driver_version=(520, 0),
        driver_version_raw="520.00",
        cuda_version=(11, 8),
        cuda_version_raw="11.8",
        gpu_names=("NVIDIA GeForce RTX 2060",),
    )
    monkeypatch.setattr("live_translate.gpu_driver.query_nvidia_driver", lambda: info)
    compatible, message = check_cuda_driver_compatibility("Torch")
    assert not compatible
    assert "520.00" in message
    assert "570.65" in message