"""Read-pool ceilings sized for a multiplexed gateway, and an honest exhaustion warning.

A gateway serving N profiles opens one state.db per profile in ONE process. The old process ceiling (24, fixed at
import) was spent by three profiles' pools, after which every read of the other profiles queued behind the single
writer's lock on the gateway's event loop. The ceiling is now sized at runtime (24 on a descriptor-poor host, 192
otherwise), both ceilings are configurable from ``database.read_pool_max`` / ``database.read_pool_process_max``, the
warning names the ceiling that refused, and the claim-origin lookup has a partial index.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager

import pytest

from persistence.sqlite import _backend
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._state import readpool
from persistence.sqlite.workers._compression import worker_claim_origin

PROFILES = 24
READERS_PER_PROFILE = 3


@pytest.fixture()
def fresh_ceilings(monkeypatch):
    """A process ceiling not yet sized, on a host whose descriptor limit cannot be measured (Windows-like), and no
    per-file overrides or rate-limited warnings left over from other tests."""
    monkeypatch.setattr(readpool, "_fd_soft_limit", lambda: None)
    monkeypatch.setattr(readpool, "_process_read_permits", readpool._PermitCounter())
    monkeypatch.setattr(readpool, "_per_file_limits", {})
    monkeypatch.setattr(readpool, "_pool_exhaustion_warned_at", {})
    monkeypatch.setattr(readpool, "_pool_exhaustion_suppressed", {})


@pytest.fixture()
def profile_dbs(tmp_path):
    holders = [StoreHolder(tmp_path / f"p{i}" / "state.db") for i in range(PROFILES)]
    yield holders
    for holder in holders:
        holder.close()


@contextmanager
def _writers_hold_every_db(holders):
    """One in-flight write per database: the writer connection's lock is taken, as during a long write."""
    for holder in holders:
        holder._lock.acquire()
    try:
        yield
    finally:
        for holder in holders:
            holder._lock.release()


def _concurrent_checkouts(holders):
    """Every reader thread checks a read connection out and holds it until all have tried; returns how many got one."""
    got, lock = [], threading.Lock()
    all_tried = threading.Event()
    pending = [len(holders) * READERS_PER_PROFILE]

    def reader(holder):
        conn = holder._checkout_read_conn()
        with lock:
            got.append(conn is not None)
            pending[0] -= 1
            if pending[0] == 0:
                all_tried.set()
        assert all_tried.wait(30)
        if conn is not None:
            holder._close_read_conn(conn)

    threads = [threading.Thread(target=reader, args=(h,)) for h in holders for _ in range(READERS_PER_PROFILE)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return sum(got)


def _at_capacity(caplog):
    return [r.getMessage() for r in caplog.records if "read pool at capacity" in r.getMessage()]


@pytest.mark.requires_wal
def test_default_ceiling_serves_every_profile_while_writers_hold_their_dbs(fresh_ceilings, profile_dbs, caplog):
    """24 profile databases x 3 concurrent readers: no read falls onto a writer's lock."""
    served, lock = [], threading.Lock()

    def read(holder):
        with holder._read_ctx() as conn:
            conn.execute("SELECT 1").fetchone()
        with lock:
            served.append(holder.db_path)

    with caplog.at_level(logging.WARNING, logger="hermes_state"), _writers_hold_every_db(profile_dbs):
        threads = [threading.Thread(target=read, args=(h,)) for h in profile_dbs for _ in range(READERS_PER_PROFILE)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)  # a read that fell onto the writer's lock cannot finish while the lock is held
        stuck = [t for t in threads if t.is_alive()]
    for t in threads:
        t.join(30)
    assert not stuck, f"{len(stuck)} reads queued behind a writer's lock"
    assert len(served) == PROFILES * READERS_PER_PROFILE
    assert not _at_capacity(caplog)
    assert readpool.process_read_limit() == 192


@pytest.mark.requires_wal
def test_configured_process_ceiling_denies_and_the_warning_says_process(fresh_ceilings, profile_dbs, caplog):
    readpool.configure_read_pool(profile_dbs[0].db_path, process=24)
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        granted = _concurrent_checkouts(profile_dbs)
    assert granted == 24
    warnings = _at_capacity(caplog)
    assert warnings and all("(process 24)" in w for w in warnings), warnings


@pytest.mark.requires_wal
def test_per_file_denial_says_per_db_8(fresh_ceilings, tmp_path, caplog):
    holder = StoreHolder(tmp_path / "state.db")
    try:
        held = [holder._checkout_read_conn() for _ in range(8)]
        assert all(c is not None for c in held)
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            assert holder._checkout_read_conn() is None
        warnings = _at_capacity(caplog)
        assert len(warnings) == 1 and "(per-db 8)" in warnings[0], warnings
        for conn in held:
            holder._close_read_conn(conn)
    finally:
        holder.close()


@pytest.mark.requires_wal
def test_configured_per_file_ceiling_is_the_one_named(fresh_ceilings, tmp_path, caplog):
    readpool.configure_read_pool(tmp_path / "state.db", per_file=3)
    holder = StoreHolder(tmp_path / "state.db")
    try:
        held = [holder._checkout_read_conn() for _ in range(3)]
        assert all(c is not None for c in held)
        assert holder._read_pool.maxsize == 3
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            assert holder._checkout_read_conn() is None
        assert "(per-db 3)" in _at_capacity(caplog)[0]
        for conn in held:
            holder._close_read_conn(conn)
    finally:
        holder.close()


@pytest.mark.requires_wal
def test_descriptor_headroom_denial_says_fd_headroom(fresh_ceilings, tmp_path, monkeypatch, caplog):
    holder = StoreHolder(tmp_path / "state.db")
    try:
        monkeypatch.setattr(readpool, "_fd_soft_limit", lambda: 256)
        monkeypatch.setattr(readpool, "_open_fd_count", lambda: 250)
        monkeypatch.setattr(readpool, "_fd_usage_cache", (0.0, None))
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            assert holder._checkout_read_conn() is None
        warnings = _at_capacity(caplog)
        assert len(warnings) == 1 and "fd-headroom 6 free" in warnings[0], warnings
    finally:
        holder.close()


@pytest.mark.parametrize("soft, expected", [(256, 24), (1023, 24), (1024, 192), (65536, 192), (None, 192)])
def test_process_ceiling_is_sized_from_the_descriptor_limit_at_first_use(monkeypatch, soft, expected):
    monkeypatch.setattr(readpool, "_fd_soft_limit", lambda: soft)
    monkeypatch.setattr(readpool, "_process_read_permits", readpool._PermitCounter())
    assert readpool.process_read_limit() == expected


def test_release_never_raises_when_the_ceiling_changed_under_live_permits():
    permits = readpool._PermitCounter(4)
    assert all(permits.acquire(blocking=False) for _ in range(4))
    permits.set_limit(1)
    assert not permits.acquire(blocking=False)  # over the new ceiling until permits come back
    for _ in range(6):  # more releases than acquires, too
        permits.release()
    assert permits.acquire(blocking=False)
    assert not permits.acquire(blocking=False)


def _write_config(home, **database):
    home.mkdir(parents=True, exist_ok=True)
    lines = ["database:"] + [f"  {key}: {value}" for key, value in database.items()]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_launch_home_config_sets_both_ceilings_at_store_open(fresh_ceilings, tmp_path, monkeypatch):
    home = tmp_path / "launch"
    _write_config(home, read_pool_max=5, read_pool_process_max=30)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _backend._configure_read_pool(home / "state.db")
    assert readpool.process_read_limit() == 30
    assert readpool._read_budget_for(home / "state.db").limit == 5


def test_secondary_profile_config_cannot_move_the_process_ceiling(fresh_ceilings, tmp_path, monkeypatch):
    launch, other = tmp_path / "launch", tmp_path / "launch" / "profiles" / "other"
    _write_config(launch, read_pool_process_max=40)
    _write_config(other, read_pool_max=4, read_pool_process_max=9)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    _backend._configure_read_pool(launch / "state.db")
    with _backend._home_bound(other):
        _backend._configure_read_pool(other / "state.db")
    assert readpool.process_read_limit() == 40
    assert readpool._read_budget_for(other / "state.db").limit == 4  # its own file ceiling still applies


def test_invalid_ceilings_are_ignored_with_a_warning(fresh_ceilings, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        readpool.configure_read_pool(tmp_path / "state.db", per_file=0, process="many")
    assert readpool._read_budget_for(tmp_path / "state.db").limit == 8
    assert readpool.process_read_limit() == 192
    assert len(caplog.records) == 2


# ── claim-origin index ──

def _seed_receipts(db_path, rows):
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO worker_receipts(execution_id, sequence, payload_digest, result_json) VALUES (?,?,?,?)",
            [(e, s, f"d{s}", json.dumps({"worker_assignment": a})) for e, s, a in rows],
        )
        conn.commit()
    finally:
        conn.close()


def _index_names(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        conn.close()


@pytest.mark.requires_wal
def test_claim_origin_lookup_uses_the_partial_index_and_keeps_its_answers(tmp_path):
    holder = StoreHolder(tmp_path / "state.db")
    try:
        rows = []
        for seq in range(1, 60):  # receipts without a parent, then the one that names it, then a later one
            rows.append(("e1", seq, None))
        rows += [("e1", 60, {"parent": "origin-A"}), ("e1", 61, {"parent": "origin-B"})]
        rows += [("e2", 1, {"other": 1}), ("e2", 2, None)]  # never names a parent
        rows += [("e3", 1, {"parent": "origin-C"})]
        _seed_receipts(holder.db_path, rows)

        statements = []
        original = holder._read_ctx

        @contextmanager
        def spying_read_ctx(*args, **kwargs):
            with original(*args, **kwargs) as conn:
                conn.set_trace_callback(statements.append)
                try:
                    yield conn
                finally:
                    conn.set_trace_callback(None)

        holder._read_ctx = spying_read_ctx
        try:
            answers = {e: worker_claim_origin(holder, {"execution_id": e, "session_id": f"fallback-{e}"})
                       for e in ("e1", "e2", "e3", "missing")}
        finally:
            del holder._read_ctx

        assert answers == {"e1": "origin-A", "e2": "fallback-e2", "e3": "origin-C", "missing": "fallback-missing"}

        lookup = next(s for s in statements if "worker_receipts" in s and "ORDER BY sequence" in s)
        conn = sqlite3.connect(holder.db_path)
        try:
            plan = " ".join(row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + lookup))
        finally:
            conn.close()
        assert "idx_worker_receipts_claim_origin" in plan, plan
    finally:
        holder.close()


@pytest.mark.requires_wal
def test_claim_origin_index_is_created_on_an_existing_database_and_is_idempotent(tmp_path):
    path = tmp_path / "state.db"
    StoreHolder(path).close()
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP INDEX idx_worker_receipts_claim_origin")  # a database from before the index existed
        conn.commit()
    finally:
        conn.close()
    assert "idx_worker_receipts_claim_origin" not in _index_names(path)
    for _ in range(2):  # the second open finds it already there
        StoreHolder(path).close()
        assert "idx_worker_receipts_claim_origin" in _index_names(path)
