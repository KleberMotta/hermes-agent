"""What the store resends, how long it pauses, and what ends the resending.

Resent: a receipt timeout, a connection closed after the send, the owner's 5001 ``storage_unavailable``.
Final at once: every 4001 refusal. The pause is full-jitter and the two deadlines come from
``timeouts.worker.*``.
"""
import json
import random
import threading
from types import SimpleNamespace

import pytest

from agent.runtime_session_store import (
    RuntimeSessionStore, WorkerPersistenceError, WorkerReceiptTimeout, WorkerRPC)

SCOPE = {'session_id': 's', 'execution_id': 'w', 'epoch': 1}


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _save(store):
    return store.append_messages_batch('s', [{'role': 'user', 'content': 'hi'}])


def test_resend_pauses_are_inside_the_jitter_interval_and_not_all_equal(tmp_path):
    clock, pauses, calls = _Clock(), [], []

    def rpc(method, **params):
        calls.append(params)
        clock.now += 1.0
        if len(calls) <= 12:
            raise WorkerReceiptTimeout('worker_receipt_timeout')
        return {'count': 1, 'annotations': [{}]}

    def sleep(seconds):
        pauses.append((seconds, 30.0 - (clock() - 1000.0)))  # the pause and the window left when it began
        clock.now += seconds

    store = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private', receipt_recovery_s=30.0, clock=clock,
                                sleep=sleep, uniform=random.Random(7).uniform)
    try:
        assert _save(store) == 1
    finally:
        store.close()
    assert len(pauses) == 12
    for attempt, (pause, remaining) in enumerate(pauses):
        cap = min(2.0, 0.25 * 2 ** attempt, remaining)
        assert min(0.25, cap) <= pause <= cap, (attempt, pause, cap)
    assert len({pause for pause, _ in pauses}) > 1


def test_no_pause_runs_past_the_recovery_window(tmp_path):
    clock, slept = _Clock(), []

    def rpc(method, **params):
        clock.now += 1.0
        raise WorkerReceiptTimeout('worker_receipt_timeout')

    def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    store = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private', receipt_recovery_s=5.0, clock=clock, sleep=sleep)
    try:
        with pytest.raises(WorkerReceiptTimeout):
            _save(store)
        assert clock() - 1000.0 <= 5.0 + 1.0  # the last send may start just inside the window
        assert store.failure == 'worker_receipt_timeout' and len(store.journal['pending']) == 1
    finally:
        with pytest.raises(WorkerPersistenceError):
            store.close()


def test_abort_resend_ends_the_loop_with_the_last_error(tmp_path):
    calls = []
    holder = {}

    def rpc(method, **params):
        calls.append(params)
        holder['store'].abort_resend()  # e.g. the gateway's Stop arrives while the owner is silent
        raise WorkerReceiptTimeout('worker_receipt_timeout')

    store = holder['store'] = RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private')
    try:
        with pytest.raises(WorkerReceiptTimeout):
            _save(store)
        assert len(calls) == 1
    finally:
        with pytest.raises(WorkerPersistenceError):
            store.close()


def test_the_deadlines_come_from_timeouts_worker_config(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / 'config.yaml').write_text(
        'timeouts:\n  worker:\n    receipt_wait: 7.5\n    receipt_recovery: 42\n', encoding='utf-8')
    store = RuntimeSessionStore(lambda *a, **k: {'count': 1, 'annotations': [{}]}, SCOPE, tmp_path / 'private')
    try:
        assert store.receipt_recovery_s == 42.0
        assert WorkerRPC(tmp_path).receipt_wait_s == 7.5
    finally:
        store.close()


def test_the_documented_defaults_apply_without_config(tmp_path):
    store = RuntimeSessionStore(lambda *a, **k: {'count': 1, 'annotations': [{}]}, SCOPE, tmp_path / 'private')
    try:
        assert (store.receipt_recovery_s, WorkerRPC(tmp_path).receipt_wait_s) == (180.0, 20.0)
    finally:
        store.close()


@pytest.fixture
def owner(monkeypatch):
    """A loopback owner behind the real ``WorkerRPC``; ``script`` decides each request's fate:
    ``'close'`` drops the link without replying, ``'5001'`` / ``'4001'`` answer an error, else commit once."""
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    state = SimpleNamespace(script=[], requests=0, committed=[], receipts={})

    def handler(ws):
        for raw in ws:
            request = json.loads(raw)
            state.requests += 1
            step = state.script.pop(0) if state.script else 'ok'
            if step == 'close':
                ws.close()
                return
            if step in ('5001', '4001'):
                message = 'storage_unavailable' if step == '5001' else 'stale_epoch'
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                                    'error': {'code': int(step), 'message': message}}))
                continue
            sequence = request['params']['sequence']
            if sequence not in state.receipts:
                state.committed.append(sequence)
                state.receipts[sequence] = {'count': 1, 'annotations': [{}], 'value': True}
            ws.send(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': state.receipts[sequence]}))

    with serve(handler, '127.0.0.1', 0, subprotocols=['hermes-gateway-v1']) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = SimpleNamespace(api_origin=f'http://127.0.0.1:{server.socket.getsockname()[1]}', control_home=None)
        monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint',
                            lambda home, timeout: SimpleNamespace(state='ready', endpoint=endpoint))
        monkeypatch.setattr(discovery, 'query_identify', lambda home, timeout: {'pid': -1})
        monkeypatch.setattr(gateway_client, '_session_ticket', lambda home, endpoint, purpose: 'ticket')
        try:
            yield state
        finally:
            server.shutdown()


def _real_store(tmp_path):
    rpc = WorkerRPC(tmp_path)
    return rpc, RuntimeSessionStore(rpc, SCOPE, tmp_path / 'private', sleep=lambda s: None)


def test_a_link_closed_after_the_send_is_resent_and_commits_once(owner, tmp_path):
    owner.script = ['close']
    rpc, store = _real_store(tmp_path)
    try:
        assert _save(store) == 1
        assert owner.requests == 2 and owner.committed == [1]
    finally:
        store.close()
        rpc.close()


def test_storage_unavailable_from_the_owner_is_resent(owner, tmp_path):
    owner.script = ['5001', '5001']
    rpc, store = _real_store(tmp_path)
    try:
        assert _save(store) == 1
        assert owner.requests == 3 and owner.committed == [1]
    finally:
        store.close()
        rpc.close()


def test_an_owner_refusal_4001_is_final_and_never_resent(owner, tmp_path):
    owner.script = ['4001']
    rpc, store = _real_store(tmp_path)
    try:
        with pytest.raises(WorkerPersistenceError, match='stale_epoch'):
            _save(store)
        assert owner.requests == 1 and owner.committed == []
        assert store.failure == 'stale_epoch' and len(store.journal['pending']) == 1
    finally:
        with pytest.raises(WorkerPersistenceError):
            store.close()
        rpc.close()


def test_a_refusal_after_a_lost_receipt_is_final_too(owner, tmp_path):
    owner.script = ['close', '4001']
    rpc, store = _real_store(tmp_path)
    try:
        with pytest.raises(WorkerPersistenceError, match='stale_epoch'):
            _save(store)
        assert owner.requests == 2
    finally:
        with pytest.raises(WorkerPersistenceError):
            store.close()
        rpc.close()
