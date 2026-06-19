from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import load_settings
from .runtime_bootstrap import configure_runtime_paths


def configure_logging() -> None:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "live_translate.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[handler],
    )


def main() -> None:
    configure_runtime_paths()
    configure_logging()
    from .ui import run_app

    raise SystemExit(run_app(load_settings()))


if __name__ == "__main__":
    main()
