import sys

from live_translate.engines import (
    _clean_translation,
    _disable_unused_nagisa_forced_alignment,
)


def test_unused_nagisa_forced_alignment_is_stubbed() -> None:
    sys.modules.pop("nagisa", None)
    _disable_unused_nagisa_forced_alignment()
    assert "nagisa" in sys.modules


def test_translation_cleanup_removes_thinking_and_extra_lines() -> None:
    value = _clean_translation("<think>reasoning</think>\n번역문\n반복 문장")
    assert value == "번역문"


def test_translation_cleanup_keeps_multiline_screen_translation() -> None:
    value = _clean_translation("아마도\n아마도\n그 글들은 몇 시간 만에 수천 개가 올라왔어", multiline=True)
    assert value == "아마도\n그 글들은 몇 시간 만에 수천 개가 올라왔어"
