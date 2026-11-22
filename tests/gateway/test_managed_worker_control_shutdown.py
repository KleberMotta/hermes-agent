"""The managed worker's control reader never strands a frame and never aborts interpreter shutdown.

Regression for #147839: a daemon thread parked in ``sys.stdin.buffer.readline()`` holds the
BufferedReader lock, so a worker that failed with the owner's pipe still open died with
``Fatal Python error: _enter_buffered_busy`` instead of exiting 1 (reported on Windows, Python 3.14).
Adapted from the control-shutdown test in PR #149137.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.waiting import wait_until

ROOT = Path(__file__).resolve().parents[2]

# Real ``main()`` and ``execute()``; only the owner-facing edges (RPC, store, agent factory) are stubbed.
# Construction fails AFTER the control thread is parked inside read_frame, the shape that aborted.
CHILD = r'''
import contextlib, sys, threading, types
import agent.managed_worker as mw
import agent.runtime_session_store as rss

class RPC:
    def __init__(self, home): pass
    def __call__(self, method, **params): return {'owner_epoch': 1}
    def close(self): pass
class Store:
    def __init__(self, *args): pass
    def close(self): pass
class Policy:
    skills_prompt = ''
    model, provider, base_url = 'test', None, None
    toolsets, reasoning_config = (), None
    max_turns, source, platform = 1, 'cli', 'cli'
    ignore_rules, pass_session_id = False, False
    def config(self): return {}

def parked():
    for thread in threading.enumerate():
        if thread.name != 'managed-controls':
            continue
        frame = sys._current_frames().get(thread.ident)
        while frame is not None:
            if frame.f_code.co_name == 'read_frame':
                return True
            frame = frame.f_back
    return False

def construct(*args, **kwargs):
    for _ in range(500):
        if parked():
            raise RuntimeError('construction_failed')
        threading.Event().wait(0.01)
    raise AssertionError('control_reader_not_parked')

mw.hello = lambda: {}
mw.validate_bootstrap = lambda frame: frame
mw.bind_worker_policy = lambda frame: None
mw.outbox_dir = lambda *args: 'unused'
mw.discover_profile_mcp = lambda policy: None
mw._construct_agent = construct
rss.WorkerRPC, rss.RuntimeSessionStore = RPC, Store
import gateway.session_kanban as sk
sk.bind_worker_context = lambda frame: None
import gateway.session_policy as sp
sp.restore_policy = lambda frame: Policy()
sp.register_worker_hooks = lambda policy: None
sp.policy_scope = lambda policy, **kwargs: contextlib.nullcontext()
import tools.process_registry as pr
pr.process_registry.recover_from_checkpoint = lambda: None
import tools.approval as approval
approval.register_gateway_notify = lambda *args: None
approval.unregister_gateway_notify = lambda *args: None
import tools.approval_context as approval_context
approval_context.set_current_session_key = lambda *args: None
import gateway.session_worker_construct as wc
wc.checkpoint_agent_kwargs = lambda *args: {}
wc.construct_kwargs = lambda *args: {}
wc.routing_agent_kwargs = lambda *args: {}
run_agent = types.ModuleType('run_agent')
run_agent.AIAgent = object
sys.modules['run_agent'] = run_agent
sys.exit(mw.main())
'''

BOOTSTRAP = {'home': 'unused', 'route': 'route', 'api_key': None, 'user_id': 'user', 'chat_id': 'chat',
             'text': 'text', 'policy': {},
             'scope': {'profile_id': 'unused', 'session_id': 'session', 'execution_id': 'execution',
                       'generation': 1, 'pid': 1, 'birth': 'birth', 'secret': 'secret', 'epoch': 1}}


@pytest.mark.platforms('windows', 'posix')
def test_worker_failure_with_the_owner_pipe_open_exits_one_without_a_fatal_abort(tmp_path):
    proc = subprocess.Popen([sys.executable, '-c', CHILD], cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        proc.stdin.write(json.dumps(BOOTSTRAP).encode() + b'\n')  # the owner keeps the pipe open afterwards
        proc.stdin.flush()
        error = proc.stderr.read()  # EOF when the worker exits; a hang here is the bug
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    text = error.decode(errors='replace')
    assert 'Fatal Python error' not in text and '_enter_buffered_busy' not in text, text
    assert 'RuntimeError: construction_failed' in text, text
    assert proc.returncode == 1, text


@pytest.mark.platforms('windows', 'posix')
def test_stop_and_finish_in_the_same_write_as_the_bootstrap_are_both_seen():
    from agent import managed_worker as mw
    read_fd, write_fd = os.pipe()
    controls = None
    try:
        # One write: whatever the reader reads ahead must still reach the control thread.
        os.write(write_fd, b'{"type":"bootstrap"}\n{"type":"stop"}\n{"type":"finish"}\n')
        reader = mw._ControlPipeReader(read_fd)
        assert reader.read_frame() == {'type': 'bootstrap'}
        controls = mw.WorkerControls(channel=None, route='route', control_reader=reader)
        wait_until(controls.finish.is_set, timeout=10, what='finish frame seen')
        assert controls.stopped.is_set()
        controls.reader.join(timeout=5)
        assert not controls.reader.is_alive()
    finally:
        if controls is not None:
            controls.close()
        os.close(write_fd)
        os.close(read_fd)
