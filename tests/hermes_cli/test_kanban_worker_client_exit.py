"""The worker client reports the attempt's own booked exit; without one an unknown admission is refused."""
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_client
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.kanban_db_connect import connect_closing


def _attempt(tmp_path, monkeypatch, *, book=None, last_output=""):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title="client", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="private-host:owner")
        if book is not None:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "worker_result",
                                 {"pid": os.getpid(), "claim_lock": task.claim_lock, "exit_code": book,
                                  "last_output": last_output}, run_id=task.current_run_id)
    params = {"task_id": task_id, "run_id": task.current_run_id, "claim_lock": task.claim_lock}
    return path, params


def test_unknown_receipt_with_booked_exit_returns_that_exit(tmp_path, monkeypatch):
    path, params = _attempt(tmp_path, monkeypatch, book=75, last_output="quota wall")

    assert kanban_worker_client._exit_code({"status": "unknown"}, params, path) == 75


def test_unknown_receipt_without_booked_exit_still_refuses(tmp_path, monkeypatch):
    path, params = _attempt(tmp_path, monkeypatch)

    with pytest.raises(GatewayClientError, match="execution unknown"):
        kanban_worker_client._exit_code({"status": "unknown"}, params, path)


def test_unknown_receipt_with_another_claims_exit_still_refuses(tmp_path, monkeypatch):
    path, params = _attempt(tmp_path, monkeypatch, book=75)

    with pytest.raises(GatewayClientError, match="execution unknown"):
        kanban_worker_client._exit_code({"status": "unknown"}, dict(params, claim_lock="foreign"), path)


def test_terminal_receipt_without_output_says_why_on_stderr(tmp_path, monkeypatch, capsys):
    path, params = _attempt(tmp_path, monkeypatch, book=130)

    assert kanban_worker_client._exit_code({"status": "terminal", "outcome": "interrupted"}, params, path) == 130

    captured = capsys.readouterr()
    assert "interrupted" in captured.err and captured.out == ""


def test_dispatcher_log_extractor_drops_the_clients_session_line(monkeypatch):
    log = "Session: local-ea855e4f92e1\nKanban worker interrupted without output\n"
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *a, **k: log)

    assert kbd._worker_final_output("t_x") == "Kanban worker interrupted without output"
