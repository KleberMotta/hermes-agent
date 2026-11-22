"""Triage sweep skips tasks stuck in a block loop.

Regression: a worker kept refusing a task ("nothing changed, a human must archive it");
block_loop_detected routed it to triage, the gateway auto-decompose re-specified and
promoted it, the worker refused again — 577 rounds, one paid session each. Past
BLOCK_LOOP_SKIP_COUNT loops in 24 h the sweep must leave the task in triage for a human.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _triage_with_loops(conn, title: str, ages_s: list[int], now: int) -> str:
    """A triage task with one block_loop_detected event per age (seconds before ``now``)."""
    tid = kb.create_task(conn, title=title, triage=True)
    for age in ages_s:
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'block_loop_detected', '{}', ?)", (tid, now - age),
        )
    conn.commit()
    return tid


def test_three_loops_in_24h_are_skipped_others_kept(kanban_home):
    now = int(time.time())
    with kbc.connect() as conn:
        looping = _triage_with_loops(conn, "looping", [60, 3600, 80000], now)
        two = _triage_with_loops(conn, "two loops", [60, 120], now)
        old = _triage_with_loops(conn, "old loops", [60, 120, 24 * 3600 + 5], now)
        fresh = _triage_with_loops(conn, "fresh idea", [], now)
    ids = decomp.list_triage_ids(now=now)
    assert looping not in ids
    assert set(ids) == {two, old, fresh}


def test_include_block_loops_returns_everything(kanban_home):
    now = int(time.time())
    with kbc.connect() as conn:
        looping = _triage_with_loops(conn, "looping", [10, 20, 30], now)
    assert decomp.list_triage_ids(now=now, include_block_loops=True) == [looping]
    assert decomp.list_triage_ids(now=now) == []


def test_loop_events_from_real_block_routing_are_counted(kanban_home):
    """Drive the real block_task router so the event kind stays in sync with the producer."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="designer keeps refusing")
        assert kb.get_task(conn, tid).status == "ready"
        loops = 0
        for _ in range(6):
            assert kb.block_task(conn, tid, reason="RECUSADO, nada mudou", kind="needs_input")
            if kb.get_task(conn, tid).status == "triage":
                loops += 1
                ids = decomp.list_triage_ids()
                assert (tid in ids) == (loops < decomp.BLOCK_LOOP_SKIP_COUNT)
                # what the auto-decomposer does on a single (no-fanout) spec: back to the pool
                conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
                conn.commit()
            else:
                assert kb.unblock_task(conn, tid)
        assert loops >= decomp.BLOCK_LOOP_SKIP_COUNT


def test_auto_decompose_tick_never_hands_a_looping_task_to_the_decomposer(kanban_home, monkeypatch):
    from gateway import kanban_watchers_dispatcher as kwd
    from gateway.kanban_watchers_common import _to_thread_process_service

    now = int(time.time())
    with kbc.connect() as conn:
        looping = _triage_with_loops(conn, "looping", [10, 20, 30], now)
        fresh = _triage_with_loops(conn, "fresh idea", [], now)

    seen = []

    def fake_decompose(task_id, author=None):
        seen.append(task_id)
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    monkeypatch.setattr(decomp, "decompose_task", fake_decompose)
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb_: ["default"])
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    dispatcher = kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)

    # per_tick=1: the looping task must not eat the only slot either
    assert asyncio.run(_to_thread_process_service(dispatcher.auto_decompose_tick, 1)) == 1
    assert seen == [fresh]
    assert looping not in seen
