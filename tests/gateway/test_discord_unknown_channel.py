"""A deleted Discord channel (404, code 10003) is a permanent refusal, logged once and never retried by a restart.

The home-channel "gateway online" notice used to log an ERROR with a traceback at every restart and kept
``.restart_pending.json`` forever; the notice now abandons a target the platform refused for good, without
counting it as delivered (profiles sharing the chat still get theirs).
"""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.discord import adapter as discord_adapter
from tests.gateway.test_discord_send import DiscordAdapter

UNKNOWN_CHANNEL = "404 Not Found (error code: 10003): Unknown Channel"


class _NotFound(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"404 Not Found (error code: {code}): Unknown Channel")
        self.code = code


@pytest.fixture
def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(discord_adapter.discord, "NotFound", _NotFound, raising=False)
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _warnings_about_the_channel(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "10003" in r.getMessage()]


@pytest.mark.asyncio
async def test_deleted_channel_warns_once_without_traceback_and_recovers(adapter, caplog):
    delivered = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=7)))
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=None),
        fetch_channel=AsyncMock(side_effect=[_NotFound(10003), _NotFound(10003), delivered, _NotFound(10003)]),
    )

    with caplog.at_level(logging.DEBUG):
        first = await adapter.send("555", "online")
        second = await adapter.send("555", "online")
        third = await adapter.send("555", "online")
        fourth = await adapter.send("555", "online")

    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert not first.success and not first.retryable and not second.success
    assert third.success, "no negative cache: the channel is tried again and delivers once it is back"
    assert not fourth.success
    warnings = _warnings_about_the_channel(caplog)
    assert len(warnings) == 2, "once per outage: the 2nd failure is DEBUG, a success re-arms the warning"
    assert all(r.exc_info is None for r in warnings)


@pytest.mark.asyncio
async def test_other_not_found_codes_still_log_the_error(adapter, caplog):
    adapter._client = SimpleNamespace(
        get_channel=MagicMock(return_value=None), fetch_channel=AsyncMock(side_effect=_NotFound(10008)),
    )

    with caplog.at_level(logging.DEBUG):
        result = await adapter.send("555", "online")

    assert not result.success
    assert [r for r in caplog.records if r.levelno >= logging.ERROR], "only 10003 is demoted"


def _home_config(chat_id: str) -> GatewayConfig:
    return GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(
        enabled=True, gateway_restart_notification=True,
        home_channel=HomeChannel(platform=Platform.DISCORD, chat_id=chat_id, name=chat_id),
    )})


def _notice_adapter(result: SendResult):
    return SimpleNamespace(send_path_degraded=False, send=AsyncMock(return_value=result))


@pytest.fixture
def two_profiles_one_chat(tmp_path, monkeypatch):
    """The launch profile and ``coder`` share one Discord home channel; ``coder``'s bot is not up yet."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = _home_config("shared-home")
    runner.config.sessions_dir = tmp_path / "sessions"
    runner.adapters = {}
    runner._profile_configs = {"coder": _home_config("shared-home")}
    runner._profile_adapters = {"coder": {}}
    runner._planned_restart_notice_lock = None
    marker = tmp_path / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    return runner, marker


@pytest.mark.asyncio
async def test_deleted_home_is_abandoned_and_a_later_profile_on_the_chat_is_still_notified(two_profiles_one_chat):
    runner, marker = two_profiles_one_chat
    launch = _notice_adapter(SendResult(success=False, error=UNKNOWN_CHANNEL))
    runner.adapters[Platform.DISCORD] = launch

    await runner._replay_pending_planned_restart_notification()

    launch.send.assert_awaited_once()
    assert marker.exists(), "coder's notice is still owed"
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["abandoned_targets"] and not data.get("delivered_targets"), "refused for good is not delivered"

    coder = _notice_adapter(SendResult(success=True, message_id="m"))
    runner._profile_adapters["coder"][Platform.DISCORD] = coder
    await runner._replay_pending_planned_restart_notification()

    coder.send.assert_awaited_once()
    assert launch.send.await_count == 1, "an abandoned target is not retried"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_degraded_send_path_keeps_the_marker_for_the_reconnect(two_profiles_one_chat):
    runner, marker = two_profiles_one_chat
    runner._profile_configs = {}
    runner._profile_adapters = {}
    launch = _notice_adapter(SendResult(success=False, error="send_path_degraded", retryable=True))
    runner.adapters[Platform.DISCORD] = launch

    await runner._replay_pending_planned_restart_notification()

    assert marker.exists()
    assert not json.loads(marker.read_text(encoding="utf-8")).get("abandoned_targets")
