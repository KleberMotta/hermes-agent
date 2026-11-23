"""Embedded dispatcher: live concurrency caps + fair board order.

Incident: the gateway booted while ``kanban.max_in_progress`` was 2 (lowered by
a RAM guard) and kept 2 for hours after config went back to 32; with that host
cap and ``default`` always ticked first, every freed slot went to ``default``
and a second board's ready cards waited over an hour.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _settings(max_in_progress=None, per_profile=None):
    return kwd._DispatcherSettings(60.0, None, max_in_progress, 2, 0, True, None, per_profile)


def _dispatcher(max_in_progress=None, per_profile=None, kb_mod=None):
    kb_mod = kb_mod or SimpleNamespace(DEFAULT_BOARD="default")
    return kwd._KanbanDispatcher(kb_mod, _settings(max_in_progress, per_profile))


# ---------------------------------------------------------------------------
# Live caps
# ---------------------------------------------------------------------------


def test_refresh_caps_applies_raised_and_lowered_cap(caplog):
    d = _dispatcher(max_in_progress=2)
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        d.refresh_caps(lambda: {"kanban": {"max_in_progress": 32, "max_in_progress_per_profile": 3}})
        assert d.settings.max_in_progress == 32
        assert d.settings.max_in_progress_per_profile == 3
        d.refresh_caps(lambda: {"kanban": {"max_in_progress": 32, "max_in_progress_per_profile": 3}})
        d.refresh_caps(lambda: {"kanban": {"max_in_progress": "4"}})
    assert d.settings.max_in_progress == 4
    assert d.settings.max_in_progress_per_profile is None
    changes = [r.getMessage() for r in caplog.records if "applied live" in r.getMessage()]
    # 2 -> 32 and None -> 3, then 32 -> 4 and 3 -> None; the unchanged re-read logs nothing.
    assert len(changes) == 4, changes
    assert not [r for r in caplog.records if r.getMessage().startswith("kanban dispatcher: max_in_progress=")]


def test_refresh_caps_unset_or_invalid_uses_boot_resolution(monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda sample=None: 5)
    d = _dispatcher(max_in_progress=2)
    d.refresh_caps(lambda: {"kanban": {}})
    assert d.settings.max_in_progress == 5
    d.refresh_caps(lambda: {"kanban": {"max_in_progress": 0}})
    assert d.settings.max_in_progress == 5
    d.refresh_caps(lambda: {})
    assert d.settings.max_in_progress == 5


def test_refresh_caps_config_error_keeps_current():
    d = _dispatcher(max_in_progress=7, per_profile=2)

    def boom():
        raise OSError("config.yaml locked")

    d.refresh_caps(boom)
    assert (d.settings.max_in_progress, d.settings.max_in_progress_per_profile) == (7, 2)


def test_tick_once_passes_refreshed_cap_to_dispatch_once(monkeypatch):
    seen = []
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_mod: ["default"])
    d = _dispatcher(max_in_progress=2)
    monkeypatch.setattr(d, "board_db_fingerprint", lambda slug: (slug, None, None))
    monkeypatch.setattr(kbc, "connect", lambda board=None: SimpleNamespace(close=lambda: None))
    from hermes_cli import kanban_db_dispatch as kbd
    monkeypatch.setattr(kbd, "dispatch_once", lambda conn, **kw: seen.append(kw["max_in_progress"]))
    d.tick_once()
    d.refresh_caps(lambda: {"kanban": {"max_in_progress": 32}})
    d.tick_once()
    assert seen == [2, 32]


# ---------------------------------------------------------------------------
# Fair board order
# ---------------------------------------------------------------------------


def test_tick_order_rotates_one_board_per_tick(monkeypatch):
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_mod: ["default", "a", "b"])
    d = _dispatcher()
    orders = [d._tick_order() for _ in range(4)]
    assert orders == [
        ["default", "a", "b"],
        ["a", "b", "default"],
        ["b", "default", "a"],
        ["default", "a", "b"],
    ]


def test_single_board_order_unchanged(monkeypatch):
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_mod: ["default"])
    d = _dispatcher()
    assert [d._tick_order() for _ in range(3)] == [["default"]] * 3


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def test_second_board_gets_a_slot_under_tight_host_cap(kanban_home, monkeypatch, all_assignees_spawnable):
    """Host cap 1, ``default`` always has ready work: the other board must still get a slot."""
    from hermes_cli import kanban_db_dispatch as kbd

    kb.create_board("telas")
    with kbc.connect() as conn:
        for i in range(4):
            kb.create_task(conn, title=f"default-{i}", assignee="alice")
    with kbc.connect(board="telas") as conn:
        telas_tid = kb.create_task(conn, title="telas-0", assignee="alice")

    spawned: list[tuple[str, str]] = []

    def fake_spawn(task, workspace, board=None):
        spawned.append((board or "default", task.id))
        return None

    monkeypatch.setattr(kbd, "_default_spawn", fake_spawn)
    d = _dispatcher(max_in_progress=1, kb_mod=kb)

    def finish_running():
        # The worker of the only slot finishes: free the slot for the next tick.
        for slug in ("default", "telas"):
            with kbc.connect(board=slug) as conn:
                conn.execute("UPDATE tasks SET status='done', claim_lock=NULL WHERE status='running'")
                conn.commit()

    for _ in range(3):
        d.tick_once()
        finish_running()

    boards = [b for b, _ in spawned]
    assert len(spawned) == 3, spawned
    assert "telas" in boards, f"board 'telas' starved: {spawned}"
    assert ("telas", telas_tid) in spawned


# ---------------------------------------------------------------------------
# The real loop: caps change between ticks without a restart
# ---------------------------------------------------------------------------


def test_watcher_loop_applies_cap_raised_between_ticks(kanban_home, monkeypatch, all_assignees_spawnable):
    """Boot with ``max_in_progress=2``, config goes to 32 before tick 2: tick 2 dispatches with 32.

    Drives ``GatewayRunner._kanban_dispatcher_watcher`` itself (the incident was the
    loop never re-reading the cap), with both boards holding ready work: tick 1 fills
    the 2 host slots from ``default``; tick 2 must run with 32 and give ``telas`` its slots.
    """
    import asyncio

    import hermes_cli.config as cfg_mod
    from gateway.run import GatewayRunner
    from hermes_cli import kanban_db_dispatch as kbd

    kb.create_board("telas")
    with kbc.connect() as conn:
        for i in range(5):
            kb.create_task(conn, title=f"default-{i}", assignee="alice")
    with kbc.connect(board="telas") as conn:
        for i in range(2):
            kb.create_task(conn, title=f"telas-{i}", assignee="alice")

    live_cfg = {"kanban": {"dispatch_in_gateway": True, "dispatch_interval_seconds": 1,
                           "auto_decompose": False, "max_in_progress": 2}}
    monkeypatch.setattr(cfg_mod, "load_config", lambda: live_cfg)

    ticks: list[list[tuple[str, int | None]]] = [[]]
    spawned: list[tuple[int, str]] = []
    real_dispatch_once = kbd.dispatch_once

    def recording_dispatch_once(conn, **kw):
        ticks[-1].append((kw.get("board"), kw.get("max_in_progress")))
        return real_dispatch_once(conn, **kw)

    def fake_spawn(task, workspace, board=None):
        spawned.append((len(ticks), board or "default"))
        return None

    monkeypatch.setattr(kbd, "dispatch_once", recording_dispatch_once)
    monkeypatch.setattr(kbd, "_default_spawn", fake_spawn)
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])

    runner = object.__new__(GatewayRunner)
    runner._running = True

    async def run_inline(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if getattr(args[0] if args else None, "__name__", "") == "ready_nonempty":
            # End of a tick: the RAM guard restores the cap; stop after tick 2.
            live_cfg["kanban"]["max_in_progress"] = 32
            if len(ticks) == 2:
                runner._running = False
            else:
                ticks.append([])
        return result

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.run.asyncio.to_thread", run_inline)
    monkeypatch.setattr("gateway.run.asyncio.sleep", no_sleep)

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10.0))

    assert len(ticks) == 2, ticks
    caps = [[cap for _board, cap in tick] for tick in ticks]
    assert caps == [[2, 2], [32, 32]], ticks
    tick1 = [b for t, b in spawned if t == 1]
    tick2 = [b for t, b in spawned if t == 2]
    assert tick1 == ["default", "default"], spawned
    assert tick2.count("telas") == 2, f"board 'telas' got no slot after the cap went to 32: {spawned}"
    assert tick2.count("default") == 3, spawned
