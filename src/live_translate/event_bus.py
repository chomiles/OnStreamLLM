from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from .models import Caption


class CaptionBus:
    def __init__(self) -> None:
        self._listeners: list[Callable[[Caption], None]] = []
        self._lock = Lock()

    def subscribe(self, listener: Callable[[Caption], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def publish(self, caption: Caption) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(caption)

