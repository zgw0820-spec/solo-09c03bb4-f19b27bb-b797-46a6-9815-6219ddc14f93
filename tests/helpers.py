"""Raw ASGI helpers for exercising client-disconnect behaviour.

``httpx.ASGITransport`` always waits for the application to finish, so a
disconnect cannot be expressed through it.  These helpers call the ASGI
application directly and inject ``http.disconnect`` at a chosen moment.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any


class RawASGIResponse:
    def __init__(self) -> None:
        self.status_code: int | None = None
        self.headers: dict[str, str] = {}
        self.body = bytearray()
        self.finished = asyncio.Event()

    @property
    def json(self) -> Any:
        return json.loads(bytes(self.body) or b"null")


class RawASGIRequest:
    """Send one JSON POST and optionally drop the connection after a delay."""

    def __init__(
        self,
        app: Any,
        path: str,
        payload: dict[str, Any],
        *,
        disconnect_after: float | None = None,
    ) -> None:
        self.app = app
        self.path = path
        self.payload = payload
        self.disconnect_after = disconnect_after
        self.response = RawASGIResponse()
        self._body_sent = False
        self._started_at = 0.0

    async def run(self) -> RawASGIResponse:
        body = json.dumps(self.payload).encode()
        self._started_at = time.monotonic()

        async def receive() -> dict[str, Any]:
            if not self._body_sent:
                self._body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            if self.disconnect_after is None:
                # Behave like an idle keep-alive connection with no further
                # messages pending.
                await asyncio.sleep(0.01)
                return {"type": "http.request", "body": b"", "more_body": False}
            elapsed = time.monotonic() - self._started_at
            if elapsed < self.disconnect_after:
                await asyncio.sleep(min(0.005, self.disconnect_after - elapsed))
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                self.response.status_code = message["status"]
                self.response.headers = {
                    key.decode(): value.decode() for key, value in message.get("headers", [])
                }
            elif message["type"] == "http.response.body":
                self.response.body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    self.response.finished.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": self.path,
            "raw_path": self.path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
            "server": ("test", 80),
            "client": ("127.0.0.1", 45678),
        }
        try:
            await self.app(scope, receive, send)
        finally:
            self.response.finished.set()
        return self.response
