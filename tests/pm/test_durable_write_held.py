"""``durable_write_bytes`` rides out a transient Windows hold on the destination (config.yaml, WinError 5/32)."""

import errno
import os

import pytest

from pm import filesystem


def _held() -> PermissionError:
    return PermissionError(errno.EACCES, "The process cannot access the file because it is being used")


def _patch_replace(monkeypatch, failures: list[OSError]) -> list[int]:
    real_replace, calls = os.replace, []

    def replace(source, destination):
        calls.append(1)
        if failures:
            raise failures.pop(0)
        return real_replace(source, destination)

    monkeypatch.setattr(filesystem.os, "replace", replace)
    monkeypatch.setattr(filesystem.time, "sleep", lambda seconds: None)
    return calls


def _leftovers(directory):
    return [p.name for p in directory.iterdir() if p.name.startswith(".publish-")]


def test_publish_succeeds_after_transient_holds(tmp_path, monkeypatch):
    target = tmp_path / "config.yaml"
    target.write_bytes(b"old")
    calls = _patch_replace(monkeypatch, [_held(), _held()])

    filesystem.durable_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"
    assert len(calls) == 3
    assert _leftovers(tmp_path) == []


def test_permanent_hold_raises_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "config.yaml"
    target.write_bytes(b"old")
    _patch_replace(monkeypatch, [_held() for _ in range(20)])

    with pytest.raises(PermissionError):
        filesystem.durable_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert _leftovers(tmp_path) == []


def test_missing_source_raises_on_first_attempt(tmp_path, monkeypatch):
    calls = _patch_replace(monkeypatch, [FileNotFoundError(errno.ENOENT, "gone")])

    with pytest.raises(FileNotFoundError):
        filesystem.durable_write_bytes(tmp_path / "config.yaml", b"new")

    assert len(calls) == 1
    assert _leftovers(tmp_path) == []
