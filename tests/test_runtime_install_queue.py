from __future__ import annotations

import tempfile
from pathlib import Path

from live_translate.runtime_install_queue import (
    clear_runtime_install_queue,
    load_runtime_install_queue,
    save_runtime_install_queue,
)


def test_runtime_install_queue_round_trip(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        queue_path = Path(temp_dir) / "runtime_install_queue.json"
        monkeypatch.setattr(
            "live_translate.runtime_install_queue.runtime_install_queue_path",
            lambda: queue_path,
        )
        save_runtime_install_queue(
            [
                ("Qwen ASR", False),
                ("llama", True),
            ]
        )
        assert load_runtime_install_queue() == [
            ("Qwen ASR", False),
            ("llama", True),
        ]
        clear_runtime_install_queue()
        assert load_runtime_install_queue() == []