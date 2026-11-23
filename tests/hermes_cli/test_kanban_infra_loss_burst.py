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


# --- B4: a card the host keeps losing before hello is bounded, and a real run resets the count ---

CARD = "t_flaky"
HELLO = "managed_worker_hello_timeout"


def _lose_once(board, k):
    """Another pre-hello loss of CARD: seed a fresh run on a scratch card, move it onto CARD."""
    tmp = f"t_scratch{k}"
    run_id = board.running(tmp, claim_expires_in=900, heartbeat_age=30, worker_lost=HELLO)
    scratch = board.state(tmp)
    board.conn.execute("UPDATE task_runs SET task_id = ? WHERE id = ?", (CARD, run_id))
    board.conn.execute("UPDATE task_events SET task_id = ? WHERE task_id = ?", (CARD, tmp))
    board.conn.execute(
        "UPDATE tasks SET status = 'running', claim_lock = ?, claim_expires = ?, worker_pid = ?, "
        "worker_started_at = ?, last_heartbeat_at = ?, current_run_id = ?, started_at = ? WHERE id = ?",
        (scratch["claim_lock"], scratch["claim_expires"], scratch["worker_pid"],
         scratch["worker_started_at"], scratch["last_heartbeat_at"], run_id, scratch["started_at"], CARD))
    board.conn.execute("DELETE FROM tasks WHERE id = ?", (tmp,))
    board.conn.commit()
    board.tick(failure_limit=2)


def _lose_first(board):
    board.running(CARD, claim_expires_in=900, heartbeat_age=30, worker_lost=HELLO)
    board.tick(failure_limit=2)


def _complete_run(board):
    board.conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
        "VALUES (?, 'fan', 'done', 'completed', 1, 2)", (CARD,))
    board.conn.commit()


def test_the_fifth_pre_hello_loss_in_a_row_blocks_the_card_without_gave_up(board):
    _lose_first(board)
    for k in range(1, kbd._INFRA_LOSS_BLOCK_AFTER - 1):
        _lose_once(board, k)
        assert board.state(CARD)["status"] == "ready", k                 # B4: losses 1-4 retry
    _lose_once(board, 99)
    row = board.state(CARD)
    assert row["status"] == "blocked"                                    # B4: the 5th blocks
    assert row["consecutive_failures"] == 0
    assert "gave_up" not in _kinds(board, CARD)
    blocked = board.conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked'", (CARD,)).fetchone()
    assert json.loads(blocked["payload"])["reason"].startswith("infra: 5 perdas antes do hello seguidas")


def test_a_run_that_started_resets_the_pre_hello_loss_count(board):
    _lose_first(board)
    for k in range(1, 3):
        _lose_once(board, k)
    _complete_run(board)                                                 # the card ran: count restarts
    for k in range(10, 12):
        _lose_once(board, k)
    assert board.state(CARD)["status"] == "ready"                      
    _lose_once(board, 12)
    _lose_once(board, 13)
    assert board.state(CARD)["status"] == "ready"                        # 4 since the run
    _lose_once(board, 14)
    assert board.state(CARD)["status"] == "blocked"                      # 5 since the run
