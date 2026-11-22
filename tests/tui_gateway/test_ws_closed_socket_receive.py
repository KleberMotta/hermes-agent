"""A socket that is already closed must not surface as an ERROR from ``handle_ws``.

The server closes a socket itself (send deadline, fanout overflow) or finds the client gone on a send;
the next ``receive_text`` then raises ``WebSocketDisconnect`` or ``RuntimeError('... Need to call "accept"
first.')``. That is routine and logs the cause at INFO in the closing line; only a real failure is an ERROR.
"""

from __future__ import annotations

import asyncio
import logging

from tui_gateway import server
from tui_gateway import ws as ws_mod

_NOT_ACCEPTED = RuntimeError('WebSocket is not connected. Need to call "accept" first.')


def _drive(monkeypatch, caplog, ws_factory):
    """Run handle_ws over a fake socket; returns (errors, closing line, created transports)."""
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)
    monkeypatch.setattr(server, "_close_sessions_for_transport", lambda transport, end_reason: (0, 0))
    created = []
    real_transport = ws_mod.WSTransport
    monkeypatch.setattr(
        ws_mod, "WSTransport",
        lambda ws, loop, **kw: created.append(real_transport(ws, loop, **kw)) or created[-1],
    )
    ws = ws_factory(created)
    with caplog.at_level(logging.DEBUG, logger=ws_mod._log.name):
        asyncio.run(ws_mod.handle_ws(ws))
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    closing = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ws closed")]
    assert len(closing) == 1
    return errors, closing[0], created


class _Socket:
    def __init__(self, created, *, on_receive=None, send_error=None, ready_error=None):
        self.created, self.on_receive = created, on_receive
        self.send_error, self.ready_error, self.sent = send_error, ready_error, 0

    async def accept(self):
        pass

    async def send_text(self, line):
        self.sent += 1
        if self.ready_error is not None and self.sent == 1:
            raise self.ready_error
        if self.send_error is not None and self.sent > 1:
            raise self.send_error

    async def receive_text(self):
        await self.on_receive(self.created[0])
        raise _NOT_ACCEPTED

    async def close(self, code=1000):
        pass


def test_server_abort_then_receive_on_closed_socket_reports_the_server_reason(monkeypatch, caplog):
    async def overflow(transport):
        transport.abort()
        await asyncio.sleep(0)

    errors, closing, _ = _drive(monkeypatch, caplog, lambda created: _Socket(created, on_receive=overflow))

    assert errors == []
    assert "reason=fanout_overflow" in closing


def test_peer_gone_on_a_send_is_a_client_disconnect_not_an_error(monkeypatch, caplog):
    async def send_to_dead_peer(transport):
        assert await transport.write_async({"method": "event", "params": {"type": "tool.progress"}}) is False

    errors, closing, created = _drive(
        monkeypatch, caplog,
        lambda created: _Socket(created, on_receive=send_to_dead_peer, send_error=ws_mod._WebSocketDisconnect()),
    )

    assert errors == []
    assert "reason=client_disconnect(" in closing
    assert created[0].peer_disconnected


def test_peer_gone_before_ready_is_not_an_error(monkeypatch, caplog):
    errors, closing, _ = _drive(
        monkeypatch, caplog, lambda created: _Socket(created, ready_error=ws_mod._WebSocketDisconnect()),
    )

    assert errors == []
    assert "reason=client_disconnect_before_ready" in closing
    assert "send_failures=0" in closing


def test_ready_send_timeout_is_still_an_error(monkeypatch, caplog):
    errors, closing, _ = _drive(
        monkeypatch, caplog, lambda created: _Socket(created, ready_error=TimeoutError("stalled")),
    )

    assert [r.getMessage() for r in errors] == [f"ws ready frame send failed peer={_peer(errors)}"]
    assert "send_failures=1" in closing


def _peer(errors):
    return errors[0].getMessage().rsplit("peer=", 1)[1]
