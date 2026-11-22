"""A resent worker save against the REAL owner store (SQLite): once, in order, and refused when it differs.

The unit suite (``test_worker_receipt_recovery.py``) proves the store resends; this one proves the resend is
safe: the owner's receipt table (execution, sequence, digest) turns a resend of a save whose reply was lost
into the stored result, never a second row.
"""
import itertools
import random
import threading

import pytest

from agent.runtime_session_store import (
    RuntimeSessionStore, WorkerPersistenceError, WorkerReceiptTimeout)
from persistence import AuthorityEpoch, ExecutionGeneration, SessionId
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal
from persistence.workers.executions import WorkerExecutionId, WorkerKind
from tests.contracts._scopes import profile_scope
from tests.contracts._sqlite_backend import _open_sqlite_backend

SECRET = 'adoption-secret-0123456789abcdef'
STAMP = 1_790_000_000.0
PROFILE = profile_scope('profile-a')


class _OwnerRPC:
    """``WorkerRPC``'s call shape over the real ``worker_executions.persist_operation``.

    ``lose_reply(n)`` says whether the n-th call's reply is lost AFTER the owner committed it: the caller
    gets the receipt timeout the wire would raise, the row stays.
    """

    def __init__(self, backend, lose_reply=lambda n: False):
        self.executions = backend.open(PROFILE).worker_executions
        self.lose_reply, self.calls, self.lost = lose_reply, itertools.count(1), 0

    def __call__(self, method, **params):
        assert method == 'worker.persist'
        call = next(self.calls)
        try:
            result = self.executions.persist_operation(
                epoch=AuthorityEpoch(params['epoch']), execution_id=WorkerExecutionId(params['execution_id']),
                session_id=SessionId(params['session_id']), generation=ExecutionGeneration(params['generation']),
                sequence=params['sequence'], operation=params['operation'], payload=params['payload'])
        except AuthorityRefusal as exc:  # the wire's 4001: WorkerRPC raises the refusal's message
            raise WorkerPersistenceError(exc.reason) from exc
        if self.lose_reply(call):
            self.lost += 1
            raise WorkerReceiptTimeout('worker_receipt_timeout')
        return result


class _Cluster:
    """One live epoch and ``n`` sessions, each with its own registered execution, on one profile store."""

    def __init__(self, backend, n):
        self.backend = backend
        store = backend.open(PROFILE)
        self.epoch = backend.begin_epoch(PROFILE, 'instance-idempotence')
        self.scopes = []
        for index in range(n):
            session = SessionId(f'sess-{index}')
            store.sessions.create(NewSession(session_id=session, source='contract'))
            store.worker_executions.register(
                epoch=AuthorityEpoch(self.epoch), execution_id=WorkerExecutionId(f'exec-{index}'),
                session_id=session, generation=ExecutionGeneration(0), kind=WorkerKind.COMPUTE,
                adoption_secret=SECRET)
            self.scopes.append({'session_id': session, 'execution_id': f'exec-{index}',
                                'epoch': self.epoch, 'generation': 0})

    def rows(self, index):
        return [row.content for row in self.backend.open(PROFILE).messages.read_transcript(
            SessionId(f'sess-{index}'))]


@pytest.fixture
def cluster_of():
    with _open_sqlite_backend((PROFILE,)) as backend:
        yield lambda n: _Cluster(backend, n)


def _store(rpc, scope, tmp_path, name='private', **kwargs):
    return RuntimeSessionStore(rpc, scope, tmp_path / name, sleep=lambda s: None, **kwargs)


def _msg(text):
    return {'role': 'user', 'content': text, 'timestamp': STAMP}


def test_a_save_whose_receipt_was_lost_is_stored_once_and_resent_to_the_same_result(cluster_of, tmp_path):
    cluster = cluster_of(1)
    owner = _OwnerRPC(cluster.backend, lose_reply=lambda n: n == 1)
    store = _store(owner, cluster.scopes[0], tmp_path)
    try:
        assert store.append_messages_batch('sess-0', [_msg('hello')]) == 1
        assert owner.lost == 1 and store.failure is None and store.journal['pending'] == []
        assert cluster.rows(0) == ['hello']  # one message, though the owner saw the save twice
    finally:
        store.close()


def test_a_resend_returns_the_stored_result_and_does_not_write_again(cluster_of):
    cluster = cluster_of(1)
    owner = _OwnerRPC(cluster.backend)
    params = dict(cluster.scopes[0], sequence=1, operation='transcript.append',
                  payload={'messages': [_msg('once')], 'turn_lease_holder': None})
    first = owner('worker.persist', **params)
    assert owner('worker.persist', **params) == first
    assert cluster.rows(0) == ['once']


def test_the_same_sequence_with_another_payload_is_refused(cluster_of):
    cluster = cluster_of(1)
    owner = _OwnerRPC(cluster.backend)
    params = dict(cluster.scopes[0], sequence=1, operation='transcript.append')
    owner('worker.persist', **params, payload={'messages': [_msg('one')], 'turn_lease_holder': None})
    with pytest.raises(WorkerPersistenceError, match='admission_conflict'):
        owner('worker.persist', **params, payload={'messages': [_msg('other')], 'turn_lease_holder': None})
    assert cluster.rows(0) == ['one']


def test_a_gateway_restart_between_the_resends_leaves_the_journal_for_adoption(cluster_of, tmp_path):
    """The owner commits seq 1, its reply is lost, the gateway restarts (new epoch): stale_epoch is final, the
    entry stays in the journal, and the adopted successor replays it into the stored receipt, not a new row."""
    cluster = cluster_of(1)
    scope = cluster.scopes[0]
    owner = _OwnerRPC(cluster.backend, lose_reply=lambda n: n == 1)
    store = _store(owner, scope, tmp_path)
    store.append_messages_batch('sess-0', [_msg('before')])  # seq 1, resent once, one row
    new_epoch = cluster.backend.begin_epoch(PROFILE, 'instance-after-restart')
    with pytest.raises(WorkerPersistenceError, match='stale_epoch'):
        store.append_messages_batch('sess-0', [_msg('after')])  # seq 2 under the old epoch
    assert owner.lost == 1 and len(store.journal['pending']) == 1
    assert store.failure == 'stale_epoch'
    store._outbox_owner.close()  # the worker dies (releases the outbox lock); its journal stays on disk
    cluster.backend.open(PROFILE).worker_executions.adopt(
        epoch=AuthorityEpoch(new_epoch), execution_id=WorkerExecutionId('exec-0'),
        session_id=SessionId('sess-0'), generation=ExecutionGeneration(0), adoption_secret=SECRET)
    successor_scope = dict(scope, epoch=new_epoch)
    adopted = {'execution_id': 'exec-0', 'generation': 0, 'session_id': 'sess-0',
               'owner_epoch': new_epoch, 'status': 'running'}
    successor = RuntimeSessionStore(_OwnerRPC(cluster.backend), successor_scope, tmp_path / 'private',
                                    adopted, sleep=lambda s: None)
    try:
        assert successor.journal['pending'] == [] and successor.failure is None
        assert cluster.rows(0) == ['before', 'after']
    finally:
        successor.close()


def test_eight_workers_with_lost_receipts_keep_every_message_once_and_in_order(cluster_of, tmp_path):
    workers, appends = 8, 50
    cluster = cluster_of(workers)
    owners, stores, errors = [], [], []

    def run(index):
        rng = random.Random(index)
        owner = _OwnerRPC(cluster.backend, lose_reply=lambda n: rng.random() < 0.10)
        owners.append(owner)
        store = _store(owner, cluster.scopes[index], tmp_path, name=f'private-{index}')
        stores.append(store)
        try:
            for step in range(appends):
                assert store.append_messages_batch(f'sess-{index}', [_msg(f'w{index}-{step}')]) == 1
        except BaseException as exc:  # surfaced below, with the worker's index
            errors.append((index, repr(exc)))

    threads = [threading.Thread(target=run, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sum(owner.lost for owner in owners) > 0  # the injection really fired
    for index in range(workers):
        assert cluster.rows(index) == [f'w{index}-{step}' for step in range(appends)]
    assert [store.failure for store in stores] == [None] * workers
    assert [store.journal['pending'] for store in stores] == [[]] * workers
    for store in stores:
        store.close()
