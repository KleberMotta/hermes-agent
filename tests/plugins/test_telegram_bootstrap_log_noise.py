"""Regression tests for PTB's duplicate ERROR log during polling bootstrap delete_webhook.

``Updater.start_polling()`` defaults to ``bootstrap_retries=0``, so PTB's
``network_retry_loop`` logs ``ERROR telegram.ext: Network Retry Loop (Bootstrap
delete Webhook): ... Failed run number 0 of 0. Aborting.`` on the first
transient ``TimedOut``/``NetworkError`` before raising back to
``TelegramAdapter``. The adapter already owns retry/escalation policy (WARNING
per transient reconnect attempt, ERROR on cold-start failure or ladder
exhaustion), so PTB's duplicate ERROR flooded ``errors.log`` during self-healing
network blips and tripped alerting.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _TimedOut(Exception):
    pass


_TimedOut.__name__ = "TimedOut"


class _NetworkError(Exception):
    pass


_NetworkError.__name__ = "NetworkError"


class _BadRequest(Exception):
    pass


_BadRequest.__name__ = "BadRequest"


class _InvalidToken(Exception):
    pass


_InvalidToken.__name__ = "InvalidToken"


class _RetryAfter(Exception):
    pass


_RetryAfter.__name__ = "RetryAfter"


class _Conflict(Exception):
    pass


_Conflict.__name__ = "Conflict"


def _bare_adapter() -> TelegramAdapter:
    a = TelegramAdapter.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_network_error_count = 0
    a._polling_conflict_count = 0
    a._polling_conflict_quiet_since = None
    a._polling_error_callback_ref = None
    a._background_tasks = set()
    a._send_path_degraded = False
    a._polling_teardown_started = False
    a._polling_generation = 0
    a._polling_progress_accepting = True
    a._polling_progress_event = None
    a._polling_progress_verifier_task = None
    a._polling_error_task = None
    a._ptb_proxy_url = None
    return a


def _emit_ptb_bootstrap_log(exc: Exception, *, prefix: str = "Network Retry Loop (Bootstrap delete Webhook):") -> None:
    ptb_logger = logging.getLogger("telegram.ext")
    try:
        raise exc
    except Exception:
        ptb_logger.exception("%s %s Failed run number 0 of 0. Aborting.", prefix, exc)


def test_filter_attachment_is_idempotent():
    ptb_logger = logging.getLogger("telegram.ext")
    tg_adapter._ensure_polling_bootstrap_log_filter()
    tg_adapter._ensure_polling_bootstrap_log_filter()
    attached = [f for f in ptb_logger.filters if isinstance(f, tg_adapter._DowngradePollingBootstrapNetworkError)]
    assert len(attached) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [_TimedOut("Timed out"), _NetworkError("Bad Gateway"), OSError("network unreachable")])
async def test_start_polling_once_demotes_transient_bootstrap_network_error(caplog, exc):
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()

    async def failing_start_polling(**_kwargs):
        _emit_ptb_bootstrap_log(exc)
        raise exc

    app.updater.start_polling = failing_start_polling

    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        with pytest.raises(type(exc)):
            await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

    ptb_records = [r for r in caplog.records if r.name == "telegram.ext"]
    assert len(ptb_records) == 1
    assert ptb_records[0].levelno == logging.DEBUG
    assert ptb_records[0].levelname == "DEBUG"
    assert not [r for r in ptb_records if r.levelno >= logging.WARNING]
    # ContextVar must be reset after _start_polling_once exits.
    assert tg_adapter._SUPPRESS_POLLING_BOOTSTRAP_ERROR.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        _InvalidToken("Invalid token"),
        _BadRequest("Bad Request: invalid parameter"),
        _RetryAfter("Flood control exceeded"),
        _Conflict("Terminated by other getUpdates request"),
        RuntimeError("unexpected failure"),
    ],
)
async def test_start_polling_once_keeps_non_network_bootstrap_errors_at_error(caplog, exc):
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()

    async def failing_start_polling(**_kwargs):
        _emit_ptb_bootstrap_log(exc)
        raise exc

    app.updater.start_polling = failing_start_polling

    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        with pytest.raises(type(exc)):
            await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

    ptb_errors = [r for r in caplog.records if r.name == "telegram.ext" and r.levelno == logging.ERROR]
    assert len(ptb_errors) == 1


@pytest.mark.asyncio
async def test_bootstrap_error_outside_start_polling_once_stays_error(caplog):
    tg_adapter._ensure_polling_bootstrap_log_filter()
    assert tg_adapter._SUPPRESS_POLLING_BOOTSTRAP_ERROR.get() is None

    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        _emit_ptb_bootstrap_log(_TimedOut("Timed out"))

    ptb_errors = [r for r in caplog.records if r.name == "telegram.ext" and r.levelno == logging.ERROR]
    assert len(ptb_errors) == 1


@pytest.mark.asyncio
async def test_non_bootstrap_ptb_network_error_inside_start_polling_once_stays_error(caplog):
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()

    async def failing_start_polling(**_kwargs):
        _emit_ptb_bootstrap_log(_TimedOut("Timed out"), prefix="Network Retry Loop (Polling Updates):")
        raise _TimedOut("Timed out")

    app.updater.start_polling = failing_start_polling

    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        with pytest.raises(_TimedOut):
            await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

    ptb_errors = [r for r in caplog.records if r.name == "telegram.ext" and r.levelno == logging.ERROR]
    assert len(ptb_errors) == 1


@pytest.mark.asyncio
async def test_warning_level_bootstrap_log_not_demoted_to_debug(caplog):
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()

    async def warning_start_polling(**_kwargs):
        ptb_logger = logging.getLogger("telegram.ext")
        try:
            raise _TimedOut("Timed out")
        except Exception:
            ptb_logger.warning(
                "Network Retry Loop (Bootstrap delete Webhook): Timed out",
                exc_info=True,
            )

    app.updater.start_polling = warning_start_polling

    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

    ptb_warnings = [r for r in caplog.records if r.name == "telegram.ext" and r.levelno == logging.WARNING]
    assert len(ptb_warnings) == 1


@pytest.mark.asyncio
async def test_bootstrap_error_without_exc_info_stays_error(caplog):
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()

    async def no_exc_start_polling(**_kwargs):
        logging.getLogger("telegram.ext").error(
            "Network Retry Loop (Bootstrap delete Webhook): Timed out Failed run number 0 of 0. Aborting."
        )

    app.updater.start_polling = no_exc_start_polling

    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

    ptb_errors = [r for r in caplog.records if r.name == "telegram.ext" and r.levelno == logging.ERROR]
    assert len(ptb_errors) == 1


@pytest.mark.asyncio
async def test_generation_error_callback_clears_bootstrap_suppression_context():
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()
    seen_inside_callback: list[object] = []

    async def fake_start_polling(**kwargs):
        # PTB spawns its polling task while _start_polling_once holds the context;
        # invoking the wrapped error_callback must NOT inherit suppression.
        scope = tg_adapter._SUPPRESS_POLLING_BOOTSTRAP_ERROR.get()
        assert scope is not None and scope[0] is True
        kwargs["error_callback"](_TimedOut("polling blip"))

    app.updater.start_polling = fake_start_polling

    await a._start_polling_once(
        app,
        drop_pending_updates=False,
        error_callback=lambda _err: seen_inside_callback.append(tg_adapter._SUPPRESS_POLLING_BOOTSTRAP_ERROR.get()),
        schedule_verifier=False,
    )
    assert seen_inside_callback == [None]
    assert tg_adapter._SUPPRESS_POLLING_BOOTSTRAP_ERROR.get() is None


@pytest.mark.asyncio
async def test_background_task_spawned_inside_start_polling_does_not_keep_suppression_after_exit(caplog):
    """Tasks created via asyncio.create_task inside start_polling() copy the ContextVar;
    flipping the shared mutable scope cell in finally must disable suppression for them too."""
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()
    go = asyncio.Event()
    bg_tasks: list[asyncio.Task] = []

    async def bg_worker():
        await go.wait()
        _emit_ptb_bootstrap_log(_TimedOut("late error from spawned task"))

    async def fake_start_polling(**_kwargs):
        bg_tasks.append(asyncio.create_task(bg_worker()))

    app.updater.start_polling = fake_start_polling

    await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)
    with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
        go.set()
        await asyncio.wait_for(bg_tasks[0], timeout=5)

    ptb_errors = [r for r in caplog.records if r.name == "telegram.ext" and r.levelno == logging.ERROR]
    assert len(ptb_errors) == 1


@pytest.mark.asyncio
async def test_reconnect_ladder_logs_warning_not_error_on_transient_bootstrap_timeout(caplog):
    """Self-healing reconnect (fail attempt 1 with bootstrap TimedOut, succeed attempt 2):
    0 ERRORs emitted; adapter logs WARNING for attempt 1 and INFO on recovery."""
    a = _bare_adapter()
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = False
    attempts = 0

    async def flaky_start_polling(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            _emit_ptb_bootstrap_log(_TimedOut("Timed out"))
            raise _TimedOut("Timed out")
        a._record_polling_progress(a._polling_generation)

    app.updater.start_polling = flaky_start_polling
    a._app = app

    with caplog.at_level(logging.WARNING), \
         patch.object(tg_adapter.asyncio, "sleep", new=AsyncMock()), \
         patch.object(a, "_drain_polling_connections", new=AsyncMock()), \
         patch.object(a, "_schedule_polling_progress_verifier"):
        await a._handle_polling_network_error(_NetworkError("Bad Gateway"))
        chained = a._polling_error_task
        assert chained is not None
        await asyncio.wait_for(chained, timeout=5)

    assert attempts == 2
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "reconnect failed" in r.getMessage()]
    assert errors == []
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_reconnect_ladder_still_logs_error_when_retries_exhausted(caplog):
    """Demoting per-attempt PTB bootstrap errors must not silence fatal escalation after MAX_NETWORK_RETRIES."""
    a = _bare_adapter()
    a._polling_network_error_count = 10  # next attempt is 11 > MAX_NETWORK_RETRIES (10)
    a._notify_fatal_error = AsyncMock()

    with caplog.at_level(logging.WARNING):
        await a._handle_polling_network_error(_TimedOut("Timed out"))

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and "could not reconnect" in r.getMessage()]
    assert len(errors) == 1
    assert a.has_fatal_error is True


@pytest.mark.asyncio
async def test_warning_handler_receives_no_record_for_demoted_bootstrap_timeout():
    """Simulate errors.log (WARNING+ handler): demoted DEBUG records must not reach it."""
    captured: list[logging.LogRecord] = []

    class _WarningListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _WarningListHandler(level=logging.WARNING)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        a = _bare_adapter()
        app = MagicMock()
        app.updater = MagicMock()

        async def failing_start_polling(**_kwargs):
            _emit_ptb_bootstrap_log(_TimedOut("Timed out"))
            raise _TimedOut("Timed out")

        app.updater.start_polling = failing_start_polling
        with pytest.raises(_TimedOut):
            await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

        assert [r for r in captured if r.name == "telegram.ext"] == []
    finally:
        root_logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_real_ptb_updater_bootstrap_timeout_demoted_and_non_network_preserved(caplog):
    """End-to-end with real python-telegram-bot Updater.start_polling -> _bootstrap -> network_retry_loop."""
    tg_error = pytest.importorskip("telegram.error")
    tg_ext = pytest.importorskip("telegram.ext")
    if (
        not isinstance(tg_error.TimedOut, type)
        or not issubclass(tg_error.TimedOut, BaseException)
        or not isinstance(tg_ext.Updater, type)
        or not hasattr(tg_ext.Updater, "_bootstrap")
    ):
        pytest.skip("real python-telegram-bot not loaded (stubbed in sys.modules)")

    a = _bare_adapter()

    # Case 1: real PTB TimedOut and NetworkError("Bad Gateway") on delete_webhook -> demoted to DEBUG (no ERROR).
    for transient_exc in (tg_error.TimedOut("Timed out"), tg_error.NetworkError("Bad Gateway")):
        caplog.clear()
        bot = MagicMock()
        bot.delete_webhook = AsyncMock(side_effect=transient_exc)
        bot.initialize = AsyncMock()
        updater = tg_ext.Updater(bot=bot, update_queue=asyncio.Queue())
        updater._initialized = True
        app = SimpleNamespace(updater=updater)

        with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
            with pytest.raises(type(transient_exc)):
                await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

        records = [r for r in caplog.records if r.name == "telegram.ext" and "Bootstrap delete Webhook" in r.getMessage()]
        assert any(r.levelno == logging.DEBUG and "Aborting" in r.getMessage() for r in records)
        assert not any(r.levelno >= logging.ERROR for r in records)

    # Case 2: real PTB InvalidToken and BadRequest on delete_webhook -> remain ERROR.
    for non_network_exc in (tg_error.InvalidToken("Invalid token"), tg_error.BadRequest("Bad Request: bad param")):
        caplog.clear()
        bot = MagicMock()
        bot.delete_webhook = AsyncMock(side_effect=non_network_exc)
        bot.initialize = AsyncMock()
        updater = tg_ext.Updater(bot=bot, update_queue=asyncio.Queue())
        updater._initialized = True
        app = SimpleNamespace(updater=updater)

        with caplog.at_level(logging.DEBUG, logger="telegram.ext"):
            with pytest.raises(type(non_network_exc)):
                await a._start_polling_once(app, drop_pending_updates=False, error_callback=None, schedule_verifier=False)

        errors = [
            r for r in caplog.records
            if r.name == "telegram.ext" and "Bootstrap delete Webhook" in r.getMessage() and r.levelno == logging.ERROR
        ]
        assert len(errors) == 1
