"""A live update owns the install: the dispatcher neither spends a worker on it nor charges the card.

Regression for 27/11 09:36-11:17 (and 23/11): a source-update completion held the update fence
for 100 minutes. Every worker the dispatcher spawned asked the gateway, read ``draining:
update_paused``, waited ``OWNER_WAIT_S`` and exited 1 with "Kanban owner silent for ~182s:
Gateway draining: update_paused". 22 workers were lost, and a card that never ran got
``gave_up`` after two of them (the dispatcher's failure limit).

  U1  a worker that exits before its owner admitted the run (no ``owner_admitted`` and no
      ``worker_bound`` on the run) with ``Gateway draining: update_paused`` in its output is
      an infrastructure loss: the card goes back to ``ready``, uncharged, never ``gave_up``;
  U2  the same output AFTER the owner admitted the run is the card's own output and is
      charged as before (a card cannot talk its way out of the breaker);
  U3  while the update fence is up (the predicate ``ensure`` answers ``update_paused`` from),
      a dispatcher tick claims and spawns nothing, says so on the result, changes no card,
      and the first tick after the fence drops spawns the card.
"""

from __future__ import annotations

import argparse
import json
import os
import time

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import update_lock
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
from tests.hermes_cli.test_kanban_dispatch_silent_reason import kanban_home  # noqa: F401  (fixture)
from tests.hermes_cli.test_kanban_managed_worker_loss import board  # noqa: F401  (fixture)

# The worker's own log in the incident (t_87ab7165, run 2798).
PAUSED_OUTPUT = (
    "hermes: source-update completion failed: an update is still running; wait for it to exit, "
    "then relaunch Hermes; running with the previous dependencies - run `hermes update` to finish it\n"
    "Kanban owner refused: Kanban owner silent for 182s: Gateway draining: update_paused\n"
)


def _paused_worker(board, tid, *, owner, prior_failures=0):
    """A ``running`` card whose worker died on the update fence, ``prior_failures`` already charged."""
    run_id = board.running(tid, claim_expires_in=900, heartbeat_age=30, owner=owner)
    board.conn.execute("UPDATE tasks SET consecutive_failures = ? WHERE id = ?", (prior_failures, tid))
    board.conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"{PAUSED_OUTPUT}{KANBAN_WORKER_EXIT_TRAILER}1\n", encoding="utf-8")
    return run_id


def _kinds(board, tid):
    return [r["kind"] for r in board.conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]


def test_update_paused_before_hello_never_charges_the_card(board):
    # One failure short of the limit: on the base this loss was the card's gave_up.
    limit = kbd.DEFAULT_FAILURE_LIMIT
    run_id = _paused_worker(board, "t_paused", owner=False, prior_failures=limit - 1)
    board.tick()
    row, run = board.state("t_paused"), board.run(run_id)
    assert row["status"] == "ready"                                      # U1
    assert row["consecutive_failures"] == limit - 1                      # not charged
    assert "gave_up" not in _kinds(board, "t_paused")
    assert run["outcome"] == "crashed"
    assert json.loads(run["metadata"]).get("infrastructure") is True


def test_update_paused_after_the_owner_admitted_the_run_is_charged(board):
    run_id = _paused_worker(board, "t_admitted", owner=True)
    board.tick()
    assert board.state("t_admitted")["consecutive_failures"] == 1       # U2
    assert json.loads(board.run(run_id)["metadata"]).get("infrastructure") is not True


# --- U3: the dispatcher does not spawn into a live update ---------------------------------

def _live_update_marker():
    """A live update claim on the install root, the shape ``hermes update`` publishes."""
    marker = update_lock.update_marker_path()
    marker.write_text(f"{os.getpid()}\n{int(time.time())}\nct:{update_lock.process_create_time():.3f}\n",
                      encoding="utf-8")
    return marker


def test_a_tick_under_a_live_update_spawns_nothing_and_says_why(kanban_home, all_assignees_spawnable):
    spawns: list = []

    def spawn(task, workspace, board=None):
        spawns.append(task.id)

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="waits for the update", assignee="alice")
        marker = _live_update_marker()
        held = kbd.dispatch_once(conn, spawn_fn=spawn)
        row = conn.execute("SELECT status, claim_lock, current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()
        kinds = [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))]
        assert spawns == [] and held.spawned == []
        assert (row["status"], row["claim_lock"], row["current_run_id"]) == ("ready", None, None)
        assert "claimed" not in kinds
        assert held.spawn_capped == {"cap": "update_paused"}
        assert "update_paused=1" in kbd.describe_suppression([held])

        marker.unlink()
        resumed = kbd.dispatch_once(conn, spawn_fn=spawn)
    assert spawns == [tid]                                               # the next tick spawns it
    assert resumed.spawn_capped is None


def test_cli_dispatch_names_the_update_hold(kanban_home, all_assignees_spawnable, monkeypatch, capsys):
    from hermes_cli import kanban_ops

    with kbc.connect() as conn:
        kb.create_task(conn, title="waits for the update", assignee="alice")
    _live_update_marker()
    monkeypatch.setattr("hermes_cli.kanban_db_dispatch._default_spawn", lambda *a, **k: None)
    args = argparse.Namespace(dry_run=False, max=None, json=False, failure_limit=kbd.DEFAULT_FAILURE_LIMIT)
    assert kanban_ops._cmd_dispatch(args) == 0
    out = capsys.readouterr().out
    assert "Spawned:      0" in out
    assert "Held: an update owns the install (update_paused); no claim attempted" in out
