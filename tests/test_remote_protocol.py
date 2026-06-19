import base64
import asyncio

import numpy as np

from live_translate.event_bus import CaptionBus
from live_translate.models import AppSettings, SourceKind
from live_translate.server import (
    HOST_BIND_ADDRESS,
    MAX_AUDIO_BYTES,
    CaptionServer,
    RemoteClient,
    client_ip_allowed,
    parse_ip_list,
    valid_password,
)


def test_float32_audio_round_trip() -> None:
    audio = np.asarray([0.1, -0.25, 0.5], dtype=np.float32)
    encoded = base64.b64encode(audio.tobytes()).decode("ascii")
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
    np.testing.assert_array_equal(decoded, audio)


def test_source_kind_serializes_for_remote_protocol() -> None:
    assert SourceKind.INPUT.value == "input"
    assert SourceKind.OUTPUT.value == "output"


class _FakeWebSocket:
    pass


def test_remote_audio_includes_channel_languages() -> None:
    calls = []
    server = CaptionServer(
        AppSettings(),
        CaptionBus(),
        lambda *args: calls.append(args),
        lambda *_args: None,
    )
    audio = np.asarray([0.1, -0.1], dtype=np.float32)
    asyncio.run(
        server._handle_compute_message(
            _FakeWebSocket(),
            {
                "type": "audio",
                "source": "input",
                "sample_rate": 16000,
                "source_language": "Korean",
                "target_language": "English",
                "audio": base64.b64encode(audio.tobytes()).decode("ascii"),
            },
        )
    )
    assert calls[0][3:] == ("Korean", "English")


def test_remote_text_includes_languages() -> None:
    calls = []
    server = CaptionServer(
        AppSettings(),
        CaptionBus(),
        lambda *_args: None,
        lambda *args: calls.append(args),
    )
    asyncio.run(
        server._handle_compute_message(
            _FakeWebSocket(),
            {
                "type": "text",
                "source": "screen",
                "text": "hello",
                "source_language": "English",
                "target_language": "Korean",
            },
        )
    )
    assert calls[0] == (SourceKind.SCREEN, "hello", "English", "Korean")


def test_remote_client_receives_host_model_info() -> None:
    class FakeSocket:
        async def __aiter__(self):
            yield (
                '{"type":"host_info","asr_model":"asr-model",'
                '"translation_model":"llm-model","demo_mode":false}'
            )

    info = []
    client = RemoteClient("ws://127.0.0.1:8765/ws/client", "test", CaptionBus(), info_callback=info.append)
    asyncio.run(client._receive(FakeSocket()))
    assert info[0]["connected"] is True
    assert info[0]["translation_model"] == "llm-model"


def test_remote_password_is_sent_as_header_not_url() -> None:
    client = RemoteClient("ws://127.0.0.1:8765/ws/client", "secret-value", CaptionBus())
    assert "secret-value" not in client.url
    assert client.authorization == "Bearer secret-value"
    assert valid_password("secret-value", "secret-value")
    assert not valid_password("wrong", "secret-value")


def test_client_ip_whitelist_blocks_unlisted_clients() -> None:
    settings = AppSettings(
        client_ip_whitelist_enabled=True,
        client_ip_whitelist_allowed="192.168.0.10",
    )
    assert client_ip_allowed(settings, "192.168.0.10")
    assert not client_ip_allowed(settings, "192.168.0.11")
    assert not client_ip_allowed(settings, None)


def test_client_ip_whitelist_disabled_allows_any_client() -> None:
    settings = AppSettings(client_ip_whitelist_enabled=False)
    assert client_ip_allowed(settings, "203.0.113.5")


def test_parse_ip_list_and_host_bind_address() -> None:
    assert parse_ip_list(" 1.1.1.1, 2.2.2.2 ,1.1.1.1") == ["1.1.1.1", "2.2.2.2"]
    assert HOST_BIND_ADDRESS == "0.0.0.0"


def test_remote_rejects_oversized_audio() -> None:
    server = CaptionServer(AppSettings(), CaptionBus(), lambda *_args: None, lambda *_args: None)
    message = {
        "type": "audio",
        "source": "input",
        "sample_rate": 16000,
        "audio": "A" * (MAX_AUDIO_BYTES * 2 + 1),
    }
    try:
        asyncio.run(server._handle_compute_message(_FakeWebSocket(), message))
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("Oversized audio payload was accepted")
