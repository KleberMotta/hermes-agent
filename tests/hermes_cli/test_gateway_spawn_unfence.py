"""A gateway started from a delegate/worker shell must not inherit that shell's Kanban write fence.

Regression: a kanban worker ran ``hermes gateway start`` from its terminal, which carries
``HERMES_DELEGATED_CHILD_CONTEXT`` (from ``scrub_kanban_env``). The daemon was born fenced, so its
embedded dispatcher and cron could never write to the board. The fence stays for everything else.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
    kanban_path_is_fenced,
)
from hermes_cli import gateway_windows


@pytest.fixture
def fenced_env(monkeypatch, tmp_path):
    """A worker's shell env; the board DB is a temp file, never the real board."""
    db = tmp_path / "kanban.db"
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    return db


@pytest.mark.platforms("windows")
def test_spawn_env_drops_delegate_marker(monkeypatch, tmp_path, fenced_env):
    calls = []
    monkeypatch.setattr(
        gateway_windows,
        "_build_gateway_argv",
        lambda home=None: (["python.exe", "-m", "hermes_cli.main", "gateway", "run"], str(tmp_path), {}),
    )
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path / "home")
    monkeypatch.setattr(
        gateway_windows.subprocess, "Popen", lambda argv, **kw: calls.append(kw) or SimpleNamespace(pid=4242)
    )

    assert gateway_windows._spawn_detached() == 4242

    env = calls[0]["env"]
    assert DELEGATED_CHILD_ENV_MARKER not in env
    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_DB" not in env
    assert "HERMES_HOME" in env


def test_gateway_boot_pops_inherited_marker(monkeypatch, tmp_path, caplog, fenced_env):
    from hermes_cli import gateway

    caplog.set_level(logging.WARNING)
    gateway._drop_inherited_delegate_marker()

    assert DELEGATED_CHILD_ENV_MARKER not in os.environ
    assert any(
        r.levelno == logging.WARNING and str(tmp_path) in r.getMessage() for r in caplog.records
    )


def test_fence_still_holds_for_everything_else(monkeypatch, fenced_env):
    from tools.environments.local import build_subprocess_env

    assert build_subprocess_env()[DELEGATED_CHILD_ENV_MARKER] == os.environ[DELEGATED_CHILD_ENV_MARKER]
    assert kanban_path_is_fenced(fenced_env) is True

    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER)
    with delegated_child_context():
        assert kanban_path_is_fenced(fenced_env) is True
