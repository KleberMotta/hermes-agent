"""A managed worker that never said hello never ran the card: its loss is the host's, not the card's.

Regression for a burst of six cold workers of one profile that all hit
``managed_worker_hello_timeout`` in the same tick: ``_account_crashes`` read six identical
fingerprints as a systemic failure and blocked every card sticky on its FIRST attempt
(``failure_limit=1``), so a transient host stall cost six cards an operator unblock each.

  B1  a pre-bootstrap loss (``worker_lost`` reason ``managed_worker_hello_timeout``, not bound)
      never charges ``consecutive_failures``, never trips the breaker and is not sticky, even
      when three or more of them land in one tick;
  B2  the run is booked with ``infrastructure: true`` and the respawn guard spaces the retry
      with the same cooldown as an infrastructure spawn refusal;
  B3  a burst of losses AFTER bootstrap (the card's turn ran) still trips the systemic rule.
"""

from __future__ import annotations

import json

from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli.test_kanban_managed_worker_loss import board

BURST = 6


def _burst(board, reason):
    ids = [f"t_burst{i}" for i in range(BURST)]
    runs = [board.running(tid, claim_expires_in=900, heartbeat_age=30, worker_lost=reason) for tid in ids]
    board.tick(failure_limit=2)
    return ids, runs


def _kinds(board, tid):
    return [r["kind"] for r in board.conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]


def test_a_burst_of_hello_timeouts_keeps_every_card_retryable(board, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    ids, runs = _burst(board, "managed_worker_hello_timeout")
    for tid, run_id in zip(ids, runs):
        row, run = board.state(tid), board.run(run_id)
        assert row["status"] == "ready", (tid, row["status"])            # B1: not blocked
        assert row["consecutive_failures"] == 0                          # B1: not charged
        assert "gave_up" not in _kinds(board, tid)                       # B1: never tripped
        assert json.loads(run["metadata"]).get("infrastructure") is True  # B2
        assert kbd.check_respawn_guard(board.conn, tid) == "infrastructure_cooldown"  # B2


def test_a_burst_of_post_bootstrap_losses_still_trips_the_systemic_rule(board):
    ids, _ = _burst(board, "managed_worker_failed")
    for tid in ids:
        assert board.state(tid)["status"] == "blocked"                   # B3
        assert "gave_up" in _kinds(board, tid)
