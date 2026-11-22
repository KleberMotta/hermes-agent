"""The gateway loop answers every worker of every profile: a worker.persist must never park it.

Claim (psutil), verification (SQLite) and the write run off the loop; a stuck writer lock delays only that
worker's own reply. Real SQLite, real processes, a temporary HERMES_HOME.
"""
import asyncio
import contextvars
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle
from tests.waiting import async_wait_until

LOCK_HELD_S = 3.0
MAX_LOOP_STALL_S = 0.5


def _child():
    return subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop(process):
    process.stdin.close()
    process.wait(timeout=5)


def _hello(process):
    from hermes_cli.process_identity import worker_birth
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)), 'ancestors': [os.getpid()]}


def _adopted_worker(home, process, *, profile_id=None):
    """A reserved and adopted worker on *home*'s store: ``(connection, persist params, store)``."""
    from gateway.session_worker_reservation import reserve_admission_worker
    db = storage_handle(home)
    seed_session(db, 'owned', 'cli')
    epoch = db.runtime.begin_epoch(instance_id='owner')
    profile = profile_id or str(home)
    woken = []
    authority = SimpleNamespace(epoch=epoch, profile_id=profile, _require_admission_open=lambda: None,
                                wake_after_worker=lambda session_id, owner=None: woken.append((session_id, owner)),
                                logical_owner=lambda session_id: session_id, persistence=db)
    admitted, _ = db.admissions.admit(epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
    db.admissions.claim(epoch=epoch, session_id='owned')
    scope = reserve_admission_worker(authority, admission_id=admitted.admission_id, process=process,
                                     principal_id='human', hello=_hello(process))
    connection = SimpleNamespace(authority=authority, actor=SimpleNamespace(
        subject='human', profile_id=profile, capabilities={'worker:adopt'}))
    return connection, scope, woken


def _last_sequence(connection):
    return raw_holder(connection.authority.persistence)._read_one('SELECT last_sequence FROM worker_executions')[0]


def _persist(scope, sequence=1, operation='session.prompt', payload=None):
    return scope | {'sequence': sequence, 'operation': operation,
                    'payload': {'system_prompt': 'p'} if payload is None else payload}


@pytest.mark.asyncio
async def test_persist_does_not_stall_the_loop_while_the_writer_lock_is_held(tmp_path, monkeypatch):
    from gateway.session_worker import worker_request
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    process = _child()
    try:
        connection, scope, _ = _adopted_worker(tmp_path, process)
        await worker_request(connection, None, {k: v for k, v in scope.items() if k != 'epoch'}, operation='adopt')
        holder = raw_holder(connection.authority.persistence)
        # The read pool at capacity: reads fall back to the writer connection under the write lock
        # (persistence/sqlite/_holder.py::_read_ctx), the path a busy multi-profile gateway lands on.
        monkeypatch.setattr(holder, '_checkout_read_conn', lambda: None)
        held, release = threading.Event(), threading.Event()

        def hold_writer_lock():
            with holder._lock:
                held.set()
                release.wait(LOCK_HELD_S)

        blocker = threading.Thread(target=hold_writer_lock)
        blocker.start()
        assert held.wait(5)
        ticks, stop = [], False

        async def tick():
            last = time.monotonic()
            while not stop:
                await asyncio.sleep(0)  # one trip round the loop: the gap is how long the loop was unavailable
                now = time.monotonic()
                ticks.append(now - last)
                last = now

        ticker = asyncio.create_task(tick())
        await async_wait_until(lambda: ticks)
        started = time.monotonic()
        persisting = asyncio.create_task(worker_request(connection, None, _persist(scope), operation='persist'))
        try:
            result = await asyncio.wait_for(persisting, 30)
        finally:
            stop = True
            await ticker
            release.set()
            blocker.join()
        elapsed = time.monotonic() - started
        assert result is not None and _last_sequence(connection) == 1
        assert elapsed >= LOCK_HELD_S * 0.8, 'the persist cannot finish before the writer lock is released'
        assert max(ticks) < MAX_LOOP_STALL_S, f'the event loop stalled {max(ticks):.2f}s'
    finally:
        _stop(process)


@pytest.mark.asyncio
async def test_finish_wakes_the_fifo_with_the_owner_read_off_the_loop(tmp_path, monkeypatch):
    from gateway.session_worker import worker_request
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    process = _child()
    try:
        connection, scope, woken = _adopted_worker(tmp_path, process)
        loop_thread = threading.get_ident()
        read_on = []

        def logical_owner(session_id):
            read_on.append(threading.get_ident())
            return 'root'
        connection.authority.logical_owner = logical_owner
        await worker_request(connection, None, {k: v for k, v in scope.items() if k != 'epoch'}, operation='adopt')
        result = await worker_request(connection, None, _persist(scope, operation='execution.finish', payload={}),
                                      operation='persist')
        assert result['status'] == 'terminal'
        assert woken == [('owned', 'root')]
        assert read_on and loop_thread not in read_on
    finally:
        _stop(process)


@pytest.mark.asyncio
async def test_off_loop_calls_keep_each_callers_profile_scope(tmp_path):
    """Two profiles' calls in flight at once each see their own contextvar, and nothing leaks back."""
    from gateway.session_persistence import run_off_loop
    scope_var = contextvars.ContextVar('profile_scope', default='launch')
    authorities = [SimpleNamespace(name='a'), SimpleNamespace(name='b')]
    inside = threading.Barrier(2, timeout=10)  # both calls are on threads at the same moment

    async def call(authority, profile):
        scope_var.set(profile)

        def blocking():
            inside.wait()
            seen = scope_var.get()
            scope_var.set('leaked-by-thread')
            return seen
        return await run_off_loop(authority, blocking), scope_var.get()

    first, second = await asyncio.gather(*(asyncio.create_task(call(a, p)) for a, p in zip(authorities, ('profile-a', 'profile-b'))))
    assert first == ('profile-a', 'profile-a')
    assert second == ('profile-b', 'profile-b')
    assert scope_var.get() == 'launch'


@pytest.mark.asyncio
async def test_a_profile_with_stuck_calls_cannot_take_every_thread(tmp_path):
    """One profile's stuck writers occupy at most its own slots; another profile's call still runs."""
    from gateway.session_persistence import _OFF_LOOP_SLOTS, run_off_loop
    stuck, other = SimpleNamespace(), SimpleNamespace()
    release, entered = threading.Event(), []

    def hold():
        entered.append(1)
        release.wait(10)

    blocked = [asyncio.create_task(run_off_loop(stuck, hold)) for _ in range(_OFF_LOOP_SLOTS + 3)]
    try:
        await async_wait_until(lambda: len(entered) >= _OFF_LOOP_SLOTS)
        assert await asyncio.wait_for(run_off_loop(other, lambda: 'ran'), 5) == 'ran'
        assert len(entered) == _OFF_LOOP_SLOTS
    finally:
        release.set()
        await asyncio.gather(*blocked)


@pytest.mark.platforms('windows')
@pytest.mark.asyncio
async def test_windows_persist_never_reads_psutil_status_and_still_refuses_dead_or_recycled_pids(tmp_path, monkeypatch):
    """psutil's Windows ``status()`` does not release the GIL (#142085): it must not be called at all."""
    from gateway.session_worker import worker_request
    from persistence.core.errors import AuthorityRefusal
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    def forbidden(self):
        raise AssertionError('psutil.Process.status() called on Windows')
    monkeypatch.setattr(psutil.Process, 'status', forbidden)
    process, dead = _child(), _child()
    try:
        connection, scope, _ = _adopted_worker(tmp_path, process)
        await worker_request(connection, None, {k: v for k, v in scope.items() if k != 'epoch'}, operation='adopt')
        await worker_request(connection, None, _persist(scope), operation='persist')
        assert _last_sequence(connection) == 1
        # A recycled PID (the right pid, another birth) and a dead one are still refused.
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            await worker_request(connection, None, _persist(scope | {'birth': scope['birth'] + 1000}, sequence=2), operation='persist')
        dead_scope = scope | {'pid': dead.pid, 'birth': _hello(dead)['birth']}
        _stop(dead)
        with pytest.raises(AuthorityRefusal, match='worker_not_live|permission_denied'):
            await worker_request(connection, None, _persist(dead_scope, sequence=2), operation='persist')
    finally:
        _stop(process)
        if dead.poll() is None:
            _stop(dead)
