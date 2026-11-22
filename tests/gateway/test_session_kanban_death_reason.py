"""A kanban worker's death reaches the card: the booked ``worker_result`` carries why, redacted."""
import json
import os

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli.auth import AuthError, is_rate_limited_auth_error
from hermes_cli.kanban_db_connect import connect_closing

SECRET = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4


def _bound(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr("agent.skill_commands.build_preloaded_skills_prompt", lambda *a, **k: ("", [], []))
    kb.init_db()
    path = kb.kanban_db_path()
    with connect_closing(path) as conn:
        task_id = kb.create_task(conn, title="death", assignee="default")
        task = kb.claim_task(conn, task_id, claimer="private-host:owner")
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "worker_bound", {"pid": os.getpid(), "claim_lock": task.claim_lock},
                             run_id=task.current_run_id)
    context = {"db": str(path), "task_id": task_id, "run_id": task.current_run_id,
               "claim_lock": task.claim_lock, "skills": [], "context": "", "goal_mode": False}
    return path, context, {"text": "work", "policy": {"kanban_json": json.dumps(context)}}


class _Agent:
    session_id = "s1"

    def __init__(self, outcome):
        self.outcome = outcome

    def run_conversation(self, message, conversation_history=None, turn_author=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_exception_with_secret_is_booked_by_name_redacted_and_capped(tmp_path, monkeypatch):
    path, context, frame = _bound(tmp_path, monkeypatch)
    boom = RuntimeError(f"upstream said no, key {SECRET} " + "x" * 2000)

    with pytest.raises(RuntimeError):
        session_kanban.run_worker_turns(_Agent(boom), frame, [])

    result = session_kanban.worker_result(path, context)
    assert result["exit_code"] == 1
    assert result["last_output"].startswith("RuntimeError: ")
    assert SECRET not in result["last_output"] and len(result["last_output"]) <= 500


def test_interrupted_turn_without_reply_books_130_and_a_reason(tmp_path, monkeypatch):
    path, context, frame = _bound(tmp_path, monkeypatch)

    session_kanban.run_worker_turns(_Agent({"interrupted": True, "final_response": ""}), frame, [])

    result = session_kanban.worker_result(path, context)
    assert result["exit_code"] == 130 and result["last_output"]


def test_clean_turn_output_is_unchanged(tmp_path, monkeypatch):
    path, context, frame = _bound(tmp_path, monkeypatch)

    session_kanban.run_worker_turns(_Agent({"completed": True, "final_response": "all done"}), frame, [])

    result = session_kanban.worker_result(path, context)
    assert (result["exit_code"], result["last_output"]) == (0, "all done")


def test_start_failure_books_name_and_redacted_message(tmp_path, monkeypatch):
    path, context, frame = _bound(tmp_path, monkeypatch)

    session_kanban.record_start_failure(frame, ValueError(f"no credentials {SECRET}"))

    result = session_kanban.worker_result(path, context)
    assert result["last_output"].startswith("ValueError: no credentials")
    assert SECRET not in result["last_output"]


def test_model_rate_limited_auth_error_is_a_quota_wall_with_exit_75(tmp_path, monkeypatch):
    path, context, frame = _bound(tmp_path, monkeypatch)
    wall = AuthError("Anthropic credentials are rate-limited", provider="anthropic", code="model_rate_limited")

    assert is_rate_limited_auth_error(wall)
    with pytest.raises(AuthError):
        session_kanban.run_worker_turns(_Agent(wall), frame, [])

    assert session_kanban.worker_exit_code(path, context) == kb.KANBAN_RATE_LIMIT_EXIT_CODE == 75
