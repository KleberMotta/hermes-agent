"""A dispatch pass that spawns nothing must say why.

Incident (23/11): ``hermes kanban --board X dispatch --dry-run`` listed two
ready cards while ``dispatch --max 2`` printed a bare ``Spawned: 0`` with no
skip and no card event. ``--max`` is a live per-board concurrency cap (the
board already had 4 running), so the tick returned before any claim; the dry
run had no ``--max``. A claim CAS that loses after every gate was equally
silent.
"""

from __future__ import annotations

import argparse
import json
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


def _spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return None
    return fake_spawn


def _board_with_running(conn, running: int, ready: int) -> list[str]:
    for i in range(running):
        tid = kb.create_task(conn, title=f"busy-{i}", assignee="alice")
        assert kb.claim_task(conn, tid) is not None
    return [kb.create_task(conn, title=f"ready-{i}", assignee="alice") for i in range(ready)]


def test_max_spawn_cap_names_itself(kanban_home, all_assignees_spawnable):
    """The incident shape: 4 running, --max 2, 2 ready -> 0 spawned, reason set."""
    spawns: list = []
    with kbc.connect() as conn:
        _board_with_running(conn, running=4, ready=2)
        dry = kbd.dispatch_once(conn, dry_run=True)
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_factory(spawns), max_spawn=2)
    assert len(dry.spawned) == 2
    assert spawns == [] and res.spawned == []
    assert res.spawn_capped == {"cap": "max_spawn", "limit": 2, "running": 4}
    assert "max_spawn=1" in kbd.describe_suppression([res])


def test_max_in_progress_cap_names_itself(kanban_home, all_assignees_spawnable):
    with kbc.connect() as conn:
        _board_with_running(conn, running=1, ready=1)
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_factory([]), max_in_progress=1)
    assert res.spawned == []
    assert res.spawn_capped == {"cap": "max_in_progress", "limit": 1, "running": 1}


def test_uncapped_tick_leaves_spawn_capped_unset(kanban_home, all_assignees_spawnable):
    spawns: list = []
    with kbc.connect() as conn:
        ready = _board_with_running(conn, running=1, ready=1)
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_factory(spawns), max_spawn=3)
    assert spawns == ready
    assert res.spawn_capped is None and res.claim_lost == []


def test_lost_claim_is_recorded_on_result_and_card(kanban_home, all_assignees_spawnable, monkeypatch):
    """Another dispatcher wins the CAS between the gates and our claim."""
    real_claim = kb.claim_task

    def raced_claim(conn, task_id, **kwargs):
        assert real_claim(conn, task_id, claimer="other-host:1") is not None
        return real_claim(conn, task_id, **kwargs)

    monkeypatch.setattr("hermes_cli.kanban_db.claim_task", raced_claim)
    spawns: list = []
    with kbc.connect() as conn:
        (tid,) = _board_with_running(conn, running=0, ready=1)
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_factory(spawns))
        events = [
            (r["kind"], json.loads(r["payload"]) if r["payload"] else None)
            for r in conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,))
        ]
    assert spawns == [] and res.spawned == []
    assert res.claim_lost == [(tid, "claimed_by_other")]
    assert events[-1][0] == "claim_lost"
    assert events[-1][1]["reason"] == "claimed_by_other"


def test_parents_rejection_is_not_double_logged(kanban_home, all_assignees_spawnable, monkeypatch):
    """claim_task already writes ``claim_rejected``; no extra ``claim_lost`` row."""
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(conn, title="child", assignee="alice", parents=[parent])
        # Force the inconsistent 'ready with an open parent' state claim_task guards.
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (parent,))
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_factory([]))
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (child,))]
    assert (child, "parents_not_done") in res.claim_lost
    assert kinds.count("claim_rejected") == 1
    assert "claim_lost" not in kinds


def test_cli_prints_the_cap(kanban_home, all_assignees_spawnable, monkeypatch, capsys):
    from hermes_cli import kanban_ops

    with kbc.connect() as conn:
        _board_with_running(conn, running=4, ready=2)
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._default_spawn", _spawn_factory([]))
    args = argparse.Namespace(dry_run=False, max=2, json=False, failure_limit=kbd.DEFAULT_FAILURE_LIMIT)
    assert kanban_ops._cmd_dispatch(args) == 0
    out = capsys.readouterr().out
    assert "Spawned:      0" in out
    assert "Capped: --max/kanban.max_spawn=2" in out and "(4 running)" in out

    args.json = True
    assert kanban_ops._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spawn_capped"] == {"cap": "max_spawn", "limit": 2, "running": 4}
    assert payload["claim_lost"] == []
