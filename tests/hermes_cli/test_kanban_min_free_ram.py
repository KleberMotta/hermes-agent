"""``kanban.min_free_ram_mb``: the dispatcher refuses new spawns below an
operator-set free-RAM floor, defers (never drops) the ready tasks, and fails
open when free RAM is unknown or the key is unset/invalid."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _floor(monkeypatch, value):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda *a, **k: {"kanban": {"min_free_ram_mb": value}},
    )


def _free(monkeypatch, mb):
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._free_ram_mb", lambda: mb)


def _spawner():
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    return spawns, fake_spawn


@pytest.mark.parametrize("raw,expected", [
    (6144, 6144), ("2048", 2048), (None, None), (0, None), (-1, None), (True, None), ("x", None),
])
def test_configured_min_free_ram_mb_parsing(monkeypatch, raw, expected):
    _floor(monkeypatch, raw)
    assert kbd.configured_min_free_ram_mb() == expected


def test_below_floor_spawns_nothing_and_defers(kanban_home, all_assignees_spawnable, monkeypatch):
    _floor(monkeypatch, 6144)
    free = {"mb": 2048}
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._free_ram_mb", lambda: free["mb"])
    spawns, fake_spawn = _spawner()
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert not spawns and not res.spawned
        assert res.memory_pressure == "critical"
        row = kb.get_task(conn, task)
        assert row is not None and row.status == "ready"

        free["mb"] = 12288
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
    assert spawns == [task]
    assert res.memory_pressure is None


def test_above_floor_spawns_normally(kanban_home, all_assignees_spawnable, monkeypatch):
    _floor(monkeypatch, 6144)
    _free(monkeypatch, 6144)  # equal to the floor is enough
    spawns, fake_spawn = _spawner()
    with kbc.connect() as conn:
        for title in ("a", "b"):
            kb.create_task(conn, title=title, assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
    assert len(spawns) == 2


def test_unknown_free_ram_fails_open(kanban_home, all_assignees_spawnable, monkeypatch):
    _floor(monkeypatch, 6144)
    _free(monkeypatch, None)
    spawns, fake_spawn = _spawner()
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
    assert len(spawns) == 1


def test_unset_floor_never_reads_memory(kanban_home, all_assignees_spawnable, monkeypatch):
    _floor(monkeypatch, None)

    def boom():
        raise AssertionError("free RAM must not be read when the floor is off")

    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._free_ram_mb", boom)
    spawns, fake_spawn = _spawner()
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=fake_spawn)
    assert len(spawns) == 1


def test_free_ram_prefers_linux_sample(monkeypatch):
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {"mem_available_kib": 3 * 1024 * 1024})
    monkeypatch.setattr(kbd, "_windows_free_ram_mb", lambda: pytest.fail("must not be called"))
    assert kbd._free_ram_mb() == 3072


def test_free_ram_windows_fallback(monkeypatch):
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    monkeypatch.setattr(kbd.sys, "platform", "win32")
    monkeypatch.setattr(kbd, "_windows_free_ram_mb", lambda: 1234)
    assert kbd._free_ram_mb() == 1234


def test_free_ram_unknown_elsewhere(monkeypatch):
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    monkeypatch.setattr(kbd.sys, "platform", "darwin")
    assert kbd._free_ram_mb() is None
