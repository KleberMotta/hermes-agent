"""Demote PTB's duplicate ``Bootstrap delete Webhook`` ERROR logs inside polling start.

``Updater.start_polling`` defaults to ``bootstrap_retries=0``, so PTB's
``network_retry_loop`` logs ``_LOGGER.exception("... Failed run number 0 of 0. Aborting.")``
on ``telegram.ext`` on the very first transient ``TimedOut``/``NetworkError`` before
raising back to ``TelegramAdapter``. The adapter owns the retry/escalation policy
(WARNING per transient attempt, ERROR only on cold-start failure or after exhausting the
reconnect ladder); keeping PTB's duplicate ERROR floods ``errors.log`` during self-healing
network blips. Non-network errors (``InvalidToken``, ``Forbidden``, ``BadRequest``,
``RetryAfter``, ``Conflict``) and calls outside ``_start_polling_once`` (e.g.
``start_webhook``) are left untouched.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Callable, Optional

# Holds a 1-element mutable flag [active] while ``_start_polling_once`` is inside
# ``start_polling()``. Using a mutable cell (flipped to False in ``finally``) guarantees
# that background tasks spawned via ``asyncio.create_task`` by PTB inside
# ``start_polling()`` never inherit an active suppression flag after
# ``_start_polling_once`` returns.
_SUPPRESS_POLLING_BOOTSTRAP_ERROR: ContextVar[Optional[list[bool]]] = ContextVar(
    "telegram_suppress_polling_bootstrap_error", default=None
)
_BOOTSTRAP_DELETE_WEBHOOK_LOG_PREFIX = "Network Retry Loop (Bootstrap delete Webhook):"


class _DowngradePollingBootstrapNetworkError(logging.Filter):
    """Demote transient network ERRORs from PTB's ``Bootstrap delete Webhook`` to DEBUG."""

    def __init__(self, is_network_error: Callable[[Exception], bool]) -> None:
        super().__init__()
        self._is_network_error = is_network_error

    def filter(self, record: logging.LogRecord) -> bool:
        scope = _SUPPRESS_POLLING_BOOTSTRAP_ERROR.get()
        if not scope or not scope[0]:
            return True
        if record.levelno < logging.ERROR:
            return True
        try:
            message = record.getMessage()
        except (TypeError, ValueError, KeyError):
            return True
        if not message.startswith(_BOOTSTRAP_DELETE_WEBHOOK_LOG_PREFIX):
            return True
        exc_info = record.exc_info
        exc = exc_info[1] if isinstance(exc_info, tuple) and len(exc_info) >= 2 else None
        if not isinstance(exc, Exception) or not self._is_network_error(exc):
            return True
        record.levelno = logging.DEBUG
        record.levelname = logging.getLevelName(logging.DEBUG)
        return True


def _ensure_polling_bootstrap_log_filter(is_network_error: Callable[[Exception], bool]) -> None:
    """Idempotently attach ``_DowngradePollingBootstrapNetworkError`` to ``telegram.ext``."""
    ptb_ext_logger = logging.getLogger("telegram.ext")
    if not any(isinstance(f, _DowngradePollingBootstrapNetworkError) for f in ptb_ext_logger.filters):
        ptb_ext_logger.addFilter(_DowngradePollingBootstrapNetworkError(is_network_error))
