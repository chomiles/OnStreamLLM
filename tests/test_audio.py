import numpy as np

from live_translate.audio import SpeechSegmenter, speech_duration


def test_short_speech_is_below_minimum_duration() -> None:
    sample_rate = 16_000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[: int(sample_rate * 0.4)] = 0.1
    assert speech_duration(audio, sample_rate) < 0.5


def test_longer_speech_meets_minimum_duration() -> None:
    sample_rate = 16_000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[: int(sample_rate * 0.6)] = 0.1
    assert speech_duration(audio, sample_rate) >= 0.5


def test_speech_segmenter_keeps_tail_across_chunks() -> None:
    sample_rate = 16_000
    chunk_size = int(sample_rate * 0.25)
    segmenter = SpeechSegmenter(
        sample_rate,
        min_speech_seconds=0.5,
        end_silence_seconds=0.75,
        preroll_seconds=0.25,
    )
    silence = np.zeros(chunk_size, dtype=np.float32)
    speech = np.full(chunk_size, 0.1, dtype=np.float32)

    emitted: list[np.ndarray] = []
    emitted.extend(segmenter.accept(silence))
    emitted.extend(segmenter.accept(speech))
    emitted.extend(segmenter.accept(speech))
    emitted.extend(segmenter.accept(speech))
    assert emitted == []

    emitted.extend(segmenter.accept(silence))
    emitted.extend(segmenter.accept(silence))
    emitted.extend(segmenter.accept(silence))

    assert len(emitted) == 1
    assert len(emitted[0]) >= chunk_size * 7
