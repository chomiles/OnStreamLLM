from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import ipaddress
import json
import socket
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import AppSettings

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .event_bus import CaptionBus
from .models import AppSettings, Caption, SourceKind


OVERLAY_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body { width:100%; height:100%; margin:0; background:transparent; overflow:hidden; }
#captions { position:absolute; left:4%; right:4%; bottom:7%; display:flex;
  flex-direction:column; align-items:center; gap:10px; }
.line { display:none; text-align:center; font-weight:700; white-space:pre-wrap;
  paint-order:stroke fill; }
</style></head><body><div id="captions">
<div id="screen" class="line"></div><div id="input" class="line"></div>
<div id="output" class="line"></div></div>
<script>
let styles = {};
let history = { input: [], output: [], screen: [] };
function applyStyle(el, s) {
  el.style.fontFamily = s.font_family; el.style.fontSize = s.font_size + "px";
  el.style.color = s.color; el.style.webkitTextStroke =
    s.outline_width + "px " + s.outline_color;
}
function show(c) {
  const el = document.getElementById(c.source);
  const lines = history[c.source];
  if (lines[lines.length - 1] !== c.translated) lines.push(c.translated);
  history[c.source] = lines.slice(-3);
  el.textContent = history[c.source].join("\\n");
  applyStyle(el, styles[c.source + "_style"]); el.style.display = "block";
  clearTimeout(el.timer); el.timer = setTimeout(() => el.style.display = "none", 8000);
}
async function connect() {
  const cfg = await (await fetch("/api/settings")).json(); styles = cfg;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/ws/overlay`);
  ws.onmessage = e => show(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1500);
}
connect();
</script></body></html>"""

MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_TEXT_LENGTH = 20_000
HOST_BIND_ADDRESS = "0.0.0.0"


def is_loopback_client(ws: WebSocket) -> bool:
    try:
        return ipaddress.ip_address(ws.client.host).is_loopback if ws.client else False
    except ValueError:
        return False


def is_loopback_host(host: str | None) -> bool:
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def valid_password(provided: str | None, expected: str) -> bool:
    return bool(provided) and hmac.compare_digest(provided, expected)


def parse_ip_list(value: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in value.split(","):
        ip = part.strip()
        if ip and ip not in seen:
            seen.add(ip)
            result.append(ip)
    return result


def join_ip_list(values: list[str]) -> str:
    return ",".join(values)


def list_local_ipv4_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip:
                addresses.append(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in addresses:
                addresses.append(ip)
    except OSError:
        pass
    return sorted(addresses)


def client_ip_allowed(settings: AppSettings, client_host: str | None) -> bool:
    if not settings.client_ip_whitelist_enabled:
        return True
    allowed = parse_ip_list(settings.client_ip_whitelist_allowed)
    if not allowed:
        return False
    if not client_host:
        return False
    return client_host in allowed


class CaptionServer:
    """Runs remote ASR/translation jobs and also serves the optional OBS overlay."""

    def __init__(
        self,
        settings: AppSettings,
        bus: CaptionBus,
        audio_callback: Callable[[SourceKind, np.ndarray, int, str, str], None],
        text_callback: Callable[[SourceKind, str, str, str], None],
        translate_callback: Callable[[str, str, str, bool], tuple[str, str]] | None = None,
    ) -> None:
        self.settings = settings
        self.bus = bus
        self.audio_callback = audio_callback
        self.text_callback = text_callback
        self.translate_callback = translate_callback
        self.app = FastAPI(title="OnStreamLLM Compute Host")
        self._overlay_clients: set[WebSocket] = set()
        self._compute_clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._started = threading.Event()
        self._start_error: Exception | None = None
        self._setup_routes()
        self.bus.subscribe(self.broadcast)

    def _setup_routes(self) -> None:
        @self.app.get("/overlay", response_class=HTMLResponse)
        async def overlay(request: Request) -> str:
            if not is_loopback_host(request.client.host if request.client else None):
                return HTMLResponse("Local access only", status_code=403)
            return OVERLAY_HTML

        @self.app.get("/api/settings")
        async def settings(request: Request) -> dict[str, Any]:
            if not is_loopback_host(request.client.host if request.client else None):
                return HTMLResponse("Local access only", status_code=403)
            return self.settings.public_dict()

        @self.app.websocket("/ws/overlay")
        async def overlay_socket(ws: WebSocket) -> None:
            if not is_loopback_client(ws):
                await ws.close(code=1008, reason="Local access only")
                return
            await ws.accept()
            self._overlay_clients.add(ws)
            try:
                while True:
                    await ws.receive()
            except WebSocketDisconnect:
                pass
            finally:
                self._overlay_clients.discard(ws)

        @self.app.websocket("/ws/client")
        async def client_socket(ws: WebSocket) -> None:
            client_host = ws.client.host if ws.client else None
            if not client_ip_allowed(self.settings, client_host):
                await ws.close(code=1008, reason="IP not allowed")
                return
            authorization = ws.headers.get("authorization", "")
            provided = authorization.removeprefix("Bearer ").strip()
            if not valid_password(provided, self.settings.password):
                await ws.close(code=1008, reason="Invalid password")
                return
            await ws.accept()
            self._compute_clients.add(ws)
            try:
                await ws.send_json(
                    {
                        "type": "host_info",
                        "asr_model": self.settings.asr_model,
                        "translation_model": self.settings.translation_model,
                        "demo_mode": self.settings.demo_mode,
                    }
                )
                while True:
                    await self._handle_compute_message(ws, await ws.receive_json())
            except (WebSocketDisconnect, ValueError, KeyError, binascii.Error):
                pass
            finally:
                self._compute_clients.discard(ws)

    async def _handle_compute_message(self, ws: WebSocket, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "translate":
            await self._handle_translate_message(ws, message)
            return
        source = SourceKind(message["source"])
        if kind == "audio":
            encoded = str(message["audio"])
            if len(encoded) > MAX_AUDIO_BYTES * 2:
                raise ValueError("Audio payload too large")
            raw = base64.b64decode(encoded, validate=True)
            if len(raw) > MAX_AUDIO_BYTES or len(raw) % 4:
                raise ValueError("Invalid audio payload")
            audio = np.frombuffer(raw, dtype=np.float32).copy()
            sample_rate = int(message["sample_rate"])
            if not 8_000 <= sample_rate <= 192_000:
                raise ValueError("Invalid sample rate")
            self.audio_callback(
                source,
                audio,
                sample_rate,
                str(message.get("source_language", "auto")),
                str(message.get("target_language", "Korean")),
            )
        elif kind == "text":
            text = str(message["text"])
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError("Text payload too large")
            self.text_callback(
                source,
                text,
                str(message.get("source_language", "auto")),
                str(message.get("target_language", "Korean")),
            )

    async def _handle_translate_message(self, ws: WebSocket, message: dict[str, Any]) -> None:
        if self.translate_callback is None:
            await ws.send_json(
                {
                    "type": "translate_result",
                    "request_id": message.get("request_id"),
                    "translated": "",
                    "verified": "",
                    "error": "Translation is not available on this host.",
                }
            )
            return
        text = str(message.get("text", ""))
        if not text.strip():
            await ws.send_json(
                {
                    "type": "translate_result",
                    "request_id": message.get("request_id"),
                    "translated": "",
                    "verified": "",
                    "error": "Empty text.",
                }
            )
            return
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError("Text payload too large")
        request_id = message.get("request_id")
        source_language = str(message.get("source_language", "auto"))
        target_language = str(message.get("target_language", "Korean"))
        cross_check = bool(message.get("cross_check", False))
        loop = self._loop
        if loop is None:
            return

        def worker() -> dict[str, Any]:
            try:
                translated, verified = self.translate_callback(
                    text,
                    source_language,
                    target_language,
                    cross_check,
                )
                return {
                    "type": "translate_result",
                    "request_id": request_id,
                    "translated": translated,
                    "verified": verified,
                    "error": "",
                }
            except Exception as exc:
                return {
                    "type": "translate_result",
                    "request_id": request_id,
                    "translated": "",
                    "verified": "",
                    "error": str(exc),
                }

        payload = await asyncio.to_thread(worker)
        await ws.send_json(payload)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._started.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("서버 시작 확인 시간이 초과되었습니다. 포트와 방화벽을 확인하세요.")
        if self._start_error:
            raise RuntimeError(f"서버 시작 실패: {self._start_error}")

    def _run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            config = uvicorn.Config(
                self.app,
                host=HOST_BIND_ADDRESS,
                port=self.settings.port,
                log_level="warning",
                log_config=None,
                access_log=False,
                lifespan="off",
                ws_max_size=MAX_AUDIO_BYTES * 2,
            )
            self._server = uvicorn.Server(config)

            async def serve() -> None:
                task = asyncio.create_task(self._server.serve())
                for _attempt in range(50):
                    if self._server.started or task.done():
                        break
                    await asyncio.sleep(0.1)
                if not self._server.started:
                    if task.done() and task.exception():
                        raise task.exception()
                    raise RuntimeError("포트를 열지 못했습니다.")
                self._started.set()
                await task

            self._loop.run_until_complete(serve())
        except Exception as exc:
            self._start_error = exc
            self._started.set()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                if self._server:
                    self._server.force_exit = True
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=2)

    def broadcast(self, caption: Caption) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(caption), self._loop)

    def broadcast_host_info(self) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._send_many(
                    self._compute_clients,
                    {
                        "type": "host_info",
                        "asr_model": self.settings.asr_model,
                        "translation_model": self.settings.translation_model,
                        "demo_mode": self.settings.demo_mode,
                    },
                ),
                self._loop,
            )

    async def _broadcast(self, caption: Caption) -> None:
        await self._send_many(self._overlay_clients, caption.to_dict())
        await self._send_many(
            self._compute_clients,
            {"type": "caption", "caption": caption.to_dict()},
        )

    async def _send_many(self, clients: set[WebSocket], payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in tuple(clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


class LocalOverlayServer:
    """Serves the OBS browser overlay only on this computer."""

    def __init__(self, settings: AppSettings, bus: CaptionBus) -> None:
        self.settings = settings
        self.bus = bus
        self.app = FastAPI(title="OnStreamLLM Local Overlay")
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None
        self._setup_routes()
        self.bus.subscribe(self.broadcast)

    def _setup_routes(self) -> None:
        @self.app.get("/overlay", response_class=HTMLResponse)
        async def overlay() -> str:
            return OVERLAY_HTML

        @self.app.get("/api/settings")
        async def settings() -> dict[str, Any]:
            return self.settings.public_dict()

        @self.app.websocket("/ws/overlay")
        async def overlay_socket(ws: WebSocket) -> None:
            await ws.accept()
            self._clients.add(ws)
            try:
                while True:
                    await ws.receive()
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(ws)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._started.clear()
        self._start_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise RuntimeError("OBS 로컬 서버 시작 확인 시간이 초과되었습니다.")
        if self._start_error:
            raise RuntimeError(f"OBS 로컬 서버 시작 실패: {self._start_error}")

    def _run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._server = uvicorn.Server(
                uvicorn.Config(
                    self.app,
                    host="127.0.0.1",
                    port=self.settings.overlay_port,
                    log_level="warning",
                    log_config=None,
                    access_log=False,
                    lifespan="off",
                )
            )

            async def serve() -> None:
                task = asyncio.create_task(self._server.serve())
                for _attempt in range(50):
                    if self._server.started or task.done():
                        break
                    await asyncio.sleep(0.1)
                if not self._server.started:
                    if task.done() and task.exception():
                        raise task.exception()
                    raise RuntimeError("OBS 로컬 포트를 열지 못했습니다.")
                self._started.set()
                await task

            self._loop.run_until_complete(serve())
        except BaseException as exc:
            self._start_error = exc
            self._started.set()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                if self._server:
                    self._server.force_exit = True
                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=2)

    def broadcast(self, caption: Caption) -> None:
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send(caption.to_dict()), self._loop)

    async def _send(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in tuple(self._clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


class RemoteClient:
    """Sends captured audio/text to a compute host and receives translated captions."""

    def __init__(
        self,
        url: str,
        password: str,
        bus: CaptionBus,
        status_callback: Callable[[str], None] | None = None,
        info_callback: Callable[[dict[str, Any]], None] | None = None,
        translate_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.url = url
        self.authorization = f"Bearer {password}"
        self.bus = bus
        self.status_callback = status_callback or (lambda _message: None)
        self.info_callback = info_callback or (lambda _info: None)
        self.translate_callback = translate_callback or (lambda _result: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._outgoing: asyncio.Queue[dict[str, Any]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread:
            self._thread.join(timeout=2)

    def send_audio(
        self,
        source: SourceKind,
        audio: np.ndarray,
        sample_rate: int,
        source_language: str,
        target_language: str,
    ) -> None:
        self._queue(
            {
                "type": "audio",
                "source": source.value,
                "sample_rate": sample_rate,
                "source_language": source_language,
                "target_language": target_language,
                "audio": base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii"),
            }
        )

    def send_text(
        self,
        source: SourceKind,
        text: str,
        source_language: str,
        target_language: str,
    ) -> None:
        self._queue(
            {
                "type": "text",
                "source": source.value,
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            }
        )

    def send_translate(
        self,
        request_id: int,
        text: str,
        source_language: str,
        target_language: str,
        cross_check: bool,
    ) -> None:
        self._queue(
            {
                "type": "translate",
                "request_id": request_id,
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "cross_check": cross_check,
            }
        )

    def _queue(self, message: dict[str, Any]) -> None:
        if self._loop and self._outgoing:
            self._loop.call_soon_threadsafe(self._outgoing.put_nowait, message)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._outgoing = asyncio.Queue(maxsize=20)
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    additional_headers={"Authorization": self.authorization},
                    max_size=MAX_AUDIO_BYTES,
                ) as ws:
                    self.status_callback("연산 호스트에 연결되었습니다.")
                    receiver = asyncio.create_task(self._receive(ws))
                    sender = asyncio.create_task(self._send(ws))
                    done, pending = await asyncio.wait(
                        (receiver, sender), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        task.result()
                    self.info_callback({"connected": False})
            except Exception as exc:
                self.status_callback(f"연산 호스트 연결 재시도 중: {exc}")
                self.info_callback({"connected": False})
                await asyncio.sleep(2)

    async def _receive(self, ws: Any) -> None:
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "caption":
                self.bus.publish(Caption.from_dict(message["caption"]))
            elif message.get("type") == "host_info":
                self.info_callback({"connected": True, **message})
            elif message.get("type") == "translate_result":
                self.translate_callback(message)

    async def _send(self, ws: Any) -> None:
        assert self._outgoing is not None
        while True:
            await ws.send(json.dumps(await self._outgoing.get(), ensure_ascii=False))
