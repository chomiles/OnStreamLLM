from live_translate.compute import (
    CpuCore,
    effective_cpu_thread_count,
    format_cpu_core_ids,
    normalize_device,
    parse_cpu_core_ids,
    validate_device,
)


def test_cpu_device_is_always_valid() -> None:
    assert normalize_device("cpu") == "cpu"
    assert validate_device("cpu") == (True, "")


def test_cpu_core_ids_are_deduplicated_and_valid(monkeypatch) -> None:
    monkeypatch.setattr(
        "live_translate.compute.list_cpu_cores",
        lambda: [CpuCore(0), CpuCore(1, 0), CpuCore(2, 1)],
    )

    assert parse_cpu_core_ids("1, 2, 2, 99, nope") == [1, 2]
    assert format_cpu_core_ids("1,2") == "1E, 2"


def test_custom_cpu_threads_use_selected_core_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "live_translate.compute.list_cpu_cores",
        lambda: [CpuCore(0), CpuCore(1), CpuCore(2)],
    )

    assert effective_cpu_thread_count(-1, "0,2") == 2
    assert effective_cpu_thread_count(4, "0,2") == 4
    assert effective_cpu_thread_count(0, "", default=3) == 3
