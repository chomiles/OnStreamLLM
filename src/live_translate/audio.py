from __future__ import annotations

import queue
import threading
import warnings
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .models import SourceKind


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioDevice:
    id: str
    name: str
    is_loopback: bool


def list_microphones(include_loopback: bool = False) -> list[AudioDevice]:
    try:
        import soundcard as sc

        regular_ids = {str(mic.id) for mic in sc.all_microphones(include_loopback=False)}
        microphones = sc.all_microphones(include_loopback=include_loopback)
        if include_loopback:
            microphones = [mic for mic in microphones if str(mic.id) not in regular_ids]
        return [
            AudioDevice(str(mic.id), mic.name, include_loopback)
            for mic in microphones
        ]
    except Exception:
        return []


class AudioCapture:
    """Continuously records chunks and sends speech-sized windows to the pipeline."""

    def __init__(
        self,
        device_id: str,
        source: SourceKind,
        callback: Callable[[SourceKind, np.ndarray, int], None],
        status_callback: Callable[[SourceKind, str], None] | None = None,
        sample_rate: int = 16_000,
        window_seconds: float = 12.0,
        min_speech_seconds: float = 0.5,
        chunk_seconds: float = 0.25,
        end_silence_seconds: float = 0.8,
        preroll_seconds: float = 0.25,
    ) -> None:
        self.device_id = device_id
        self.source = source
        self.callback = callback
        self.status_callback = status_callback or (lambda _source, _status: None)
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.min_speech_seconds = min_speech_seconds
        self.chunk_seconds = chunk_seconds
        self.end_silence_seconds = end_silence_seconds
        self.preroll_seconds = preroll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: queue.Queue[str] = queue.Queue()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            import soundcard as sc
            from soundcard.mediafoundation import SoundcardRuntimeWarning

            mic = next(
                item
                for item in sc.all_microphones(include_loopback=True)
                if str(item.id) == self.device_id
            )
            frames = max(1, int(self.sample_rate * self.chunk_seconds))
            segmenter = SpeechSegmenter(
                self.sample_rate,
                min_speech_seconds=self.min_speech_seconds,
                end_silence_seconds=self.end_silence_seconds,
                preroll_seconds=self.preroll_seconds,
                max_segment_seconds=self.window_seconds,
            )
            with mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                self.status_callback(self.source, "작동중 - 오디오 장치 연결됨")
                while not self._stop.is_set():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="data discontinuity in recording",
                            category=SoundcardRuntimeWarning,
                        )
                        audio = recorder.record(numframes=frames).reshape(-1).astype(np.float32)
                    for segment in segmenter.accept(audio):
                        LOGGER.info(
                            "Audio segment captured: source=%s duration=%.2fs speech=%.2fs",
                            self.source.value,
                            len(segment) / self.sample_rate,
                            speech_duration(segment, self.sample_rate),
                        )
                        self.status_callback(self.source, "작동중 - 음성 감지됨")
                        self.callback(self.source, segment, self.sample_rate)
                for segment in segmenter.flush():
                    LOGGER.info(
                        "Audio segment flushed: source=%s duration=%.2fs speech=%.2fs",
                        self.source.value,
                        len(segment) / self.sample_rate,
                        speech_duration(segment, self.sample_rate),
                    )
                    self.callback(self.source, segment, self.sample_rate)
        except Exception as exc:
            self.errors.put(str(exc))
            self.status_callback(self.source, f"오류 - {exc}")


class SpeechSegmenter:
    def __init__(
        self,
        sample_rate: int,
        min_speech_seconds: float = 0.5,
        end_silence_seconds: float = 0.8,
        preroll_seconds: float = 0.25,
        max_segment_seconds: float = 12.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.min_speech_seconds = min_speech_seconds
        self.end_silence_seconds = end_silence_seconds
        self.max_segment_seconds = max_segment_seconds
        self._preroll_frames = max(0, int(sample_rate * preroll_seconds))
        self._preroll: list[np.ndarray] = []
        self._parts: list[np.ndarray] = []
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0

    def accept(self, chunk: np.ndarray) -> list[np.ndarray]:
        chunk = chunk.reshape(-1).astype(np.float32, copy=False)
        chunk_seconds = len(chunk) / self.sample_rate
        active_seconds = speech_duration(chunk, self.sample_rate)
        has_speech = active_seconds >= min(0.06, max(0.02, chunk_seconds * 0.4))
        emitted: list[np.ndarray] = []
        if has_speech:
            if not self._parts and self._preroll:
                self._parts.extend(self._preroll)
            self._parts.append(chunk.copy())
            self._speech_seconds += active_seconds
            self._silence_seconds = 0.0
        elif self._parts:
            self._parts.append(chunk.copy())
            self._silence_seconds += chunk_seconds
            if self._silence_seconds >= self.end_silence_seconds:
                emitted.extend(self.flush())
        else:
            self._remember_preroll(chunk)
        if self._parts and self._segment_seconds() >= self.max_segment_seconds:
            emitted.extend(self.flush(force=self._speech_seconds >= self.min_speech_seconds))
        return emitted

    def flush(self, force: bool = False) -> list[np.ndarray]:
        if not self._parts:
            return []
        segment = np.concatenate(self._parts).astype(np.float32, copy=False)
        should_emit = force or self._speech_seconds >= self.min_speech_seconds
        self._parts = []
        self._speech_seconds = 0.0
        self._silence_seconds = 0.0
        self._preroll = []
        return [segment] if should_emit else []

    def _segment_seconds(self) -> float:
        return sum(len(part) for part in self._parts) / self.sample_rate

    def _remember_preroll(self, chunk: np.ndarray) -> None:
        if self._preroll_frames <= 0:
            return
        self._preroll.append(chunk.copy())
        while sum(len(part) for part in self._preroll) > self._preroll_frames:
            self._preroll.pop(0)


def speech_duration(
    audio: np.ndarray,
    sample_rate: int,
    frame_seconds: float = 0.02,
    rms_threshold: float = 0.008,
) -> float:
    """Estimate active speech duration from short RMS frames."""
    frame_size = max(1, int(sample_rate * frame_seconds))
    usable = len(audio) - (len(audio) % frame_size)
    if usable <= 0:
        return 0.0
    frames = audio[:usable].reshape(-1, frame_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return float(np.count_nonzero(rms >= rms_threshold) * frame_seconds)
