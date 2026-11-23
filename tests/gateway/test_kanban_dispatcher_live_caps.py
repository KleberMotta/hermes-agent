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
