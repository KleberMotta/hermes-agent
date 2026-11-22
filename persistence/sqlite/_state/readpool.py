"""Read-connection budgeting for the store holder's WAL read path: per-file and
process-wide permit ceilings plus a descriptor-headroom gate, so N handles on
one state.db cannot walk the process into EMFILE. Idle pooled connections keep
their permit and are reclaimable across peers on the same path."""

import errno
import logging
import os
import threading
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:  # pragma: no cover
    from persistence.sqlite._holder import StoreHolder

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

# Ceiling on read-only connections ALIVE at once against one database FILE (idle
# pooled + checked out, over every holder on that file). One constant for both
# the pool maxsize and the permit count: a LifoQueue only caps how many are
# *returned*, and EMFILE is a peak-instant condition, so a connection holds a
# permit for its whole lifetime; once permits are gone reads degrade to the
# locked writer connection — slower, but not a wedge the supervisor can't see.
# Transient SQLITE_IOERR retry budget for READ-ONLY opens (#100436). A WAL database being actively written
# (checkpoint, WAL reset/truncate, frame flush) can surface "disk I/O error" to a concurrent ``mode=ro``
# reader in a millisecond-wide transition window: the read-only connection cannot perform the WAL recovery a
# read through a stale or mid-update -shm file needs, because recovery requires writing the -shm index,
# which mode=ro refuses. The window closes on its own (the writer finishes the transition), so a bounded
# number of short retries makes the open succeed instead of 500-ing the whole /api/sessions poll (or any
# other read-only opener). Deliberately NOT attempted on writable opens: a writer owns the transition, so an
# IOERR there means a real storage/fd problem.
_READ_POOL_MAX = 8

# Ceiling ALIVE in this PROCESS across every state.db (a multiplexed gateway opens one per profile).
# A per-file ceiling alone lets the descriptor cost grow with the profile count (#98573), so the process
# has its own; past it, readers degrade to their writer connection instead of opening descriptors: a slow
# read path is recoverable, a process-wide EMFILE is not.
# Sized at RUNTIME (never at import): a process that can hold few descriptors keeps 24, anything else gets
# room for every profile of a large gateway at the per-file ceiling (23 x 8 = 184 <= 192). Both ceilings are
# overridable by ``database.read_pool_max`` / ``database.read_pool_process_max``.
_PROCESS_MAX_LOW_FD = 24
_PROCESS_MAX_DEFAULT = 192
_LOW_FD_SOFT_LIMIT = 1024

# Warn past this many store holders on one file in one process (diagnostic:
# writer connections cannot be rationed the way read connections can).
_HANDLES_PER_PATH_WARN = 4

# Descriptors kept in reserve for everything that is NOT this module (httpx
# sockets, terminal pipes, log files): the EMFILE SQLite pushes over surfaces elsewhere.
# The ceilings above bound Hermes's SQLite descriptors, which is only ever part of the fd table. The #98573
# report is exactly that case: ~20 state.db descriptors were not the whole 256, they were the share that
# pushed httpx and terminal pipes over, and the EMFILE surfaced in tools/terminal_tool.py rather than here.
# So the read pool also yields when the PROCESS is close to its limit, whatever is consuming it.
_FD_HEADROOM_RESERVE = 64

# The fd count is a directory listing; cache it briefly so a read burst isn't a
# syscall per query (staleness lets through at most the ceiling's worth of opens).
_FD_USAGE_CACHE_SECONDS = 0.25


class _PermitCounter:
    """Non-blocking permit counter whose ceiling may change while permits are out. ``release()`` never
    raises: a ceiling lowered under live permits just refuses new ones until enough come back."""

    def __init__(self, limit: Optional[int] = None) -> None:
        self._lock = threading.Lock()
        self._limit = limit
        self._held = 0

    @property
    def limit(self) -> int:
        with self._lock:
            return self._resolve_limit()

    def _resolve_limit(self) -> int:
        if self._limit is None:
            self._limit = _default_process_limit()
        return self._limit

    def set_limit(self, limit: int) -> None:
        with self._lock:
            self._limit = limit

    def acquire(self, blocking: bool = False) -> bool:
        with self._lock:
            if self._held >= self._resolve_limit():
                return False
            self._held += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._held > 0:
                self._held -= 1


def _default_process_limit() -> int:
    soft = _fd_soft_limit()
    return _PROCESS_MAX_LOW_FD if soft is not None and soft < _LOW_FD_SOFT_LIMIT else _PROCESS_MAX_DEFAULT


# Unsized until first use (or until the launch home's config sets it).
_process_read_permits = _PermitCounter()

# Per-file ceilings the opening config asked for, by canonical path; read when that file's budget is created.
_per_file_limits: dict[str, int] = {}
_per_file_limits_lock = threading.Lock()

# Why the last refused acquire() on this thread was refused: (ceiling, number), read by note_read_pool_exhausted.
_denial = threading.local()


def process_read_limit() -> int:
    """The process-wide read-connection ceiling in force right now."""
    return _process_read_permits.limit


def _positive_int(raw, name: str) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        value = 0
    if value < 1:
        logger.warning("ignoring database.%s=%r: expected a positive integer", name, raw)
        return None
    return value


def configure_read_pool(db_path, *, per_file=None, process=None) -> None:
    """Apply ``database.read_pool_max`` (this file) and ``database.read_pool_process_max`` (the process) from the
    config read at store open. ``process`` is passed only for the launch home, so a secondary profile's config
    cannot move the ceiling every profile shares. A budget already alive keeps its size."""
    per_file = _positive_int(per_file, "read_pool_max")
    process = _positive_int(process, "read_pool_process_max")
    if per_file is not None:
        with _per_file_limits_lock:
            _per_file_limits[_read_budget_key(db_path)] = per_file
    if process is not None:
        _process_read_permits.set_limit(process)

# Read opens refused for low descriptor headroom — the only visible signal the
# guard fires. Guarded by _read_budgets_lock.
_read_open_denied_fd_headroom = 0

_fd_usage_lock = threading.Lock()
_fd_usage_cache: tuple[float, Optional[int]] = (0.0, None)


def _proc_fd_targets(pid: int) -> Iterator[tuple[str, str]]:
    """Yield ``(readlink target, fd path)`` for every entry in /proc/<pid>/fd (unreadable
    links skipped). Raises OSError when the fd directory itself cannot be listed."""
    fd_dir = f"/proc/{pid}/fd"
    for fd in os.listdir(fd_dir):
        fd_path = f"{fd_dir}/{fd}"
        try:
            yield os.readlink(fd_path), fd_path
        except OSError:
            continue


def _open_fd_count() -> Optional[int]:
    """Open descriptors in THIS process; None when unmeasurable (Windows: no fd
    dir, correctly inert); -1 when the probe itself hit EMFILE/ENFILE (no headroom)."""
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except OSError as exc:
            if exc.errno in (errno.EMFILE, errno.ENFILE):
                return -1
    return None


def _fd_soft_limit() -> Optional[int]:
    """The process's soft RLIMIT_NOFILE, or None when there is no usable one."""
    try:
        import resource
    except ImportError:
        return None
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (AttributeError, OSError, ValueError):
        return None
    if soft in (resource.RLIM_INFINITY, -1):
        return None
    return int(soft)


def _fd_headroom_ok() -> bool:
    """Can the process spare a descriptor for a new read connection? Fails OPEN
    when unmeasurable (refusing every read would be a self-inflicted convoy);
    fails CLOSED only on evidence (measured shortfall or a starved probe)."""
    free = _fd_free()
    return free is None or free > _FD_HEADROOM_RESERVE


def _fd_free() -> Optional[int]:
    """Descriptors the process can still open; None when unmeasurable, 0 when the probe was starved."""
    soft = _fd_soft_limit()
    if soft is None:
        return None
    global _fd_usage_cache
    now = time.monotonic()
    with _fd_usage_lock:
        stamp, cached = _fd_usage_cache
        fresh = cached is not None and (now - stamp) < _FD_USAGE_CACHE_SECONDS
    if not fresh:
        cached = _open_fd_count()
        with _fd_usage_lock:
            _fd_usage_cache = (now, cached)
    if cached is None:
        return None
    return max(soft - cached, 0) if cached >= 0 else 0


def _reclaim_idle_read_conn_anywhere() -> bool:
    """Close one idle read connection on ANY path: the process ceiling is shared
    across files, so a quiet profile must not hold descriptors a busy one needs."""
    with _read_budgets_lock:
        budgets = list(_read_budgets.values())
    return any(budget.reclaim_idle() for budget in budgets)


class _PathReadBudget:
    """Read-connection permits for ONE database file, shared process-wide
    (per-instance semaphores let N holders peak at N x (1 + MAX)). An idle
    pooled connection keeps its permit, so a permit miss first reclaims an IDLE
    connection from a peer on the same path.

    The permits belong to the *file*, not to each holder: the descriptors are spent on the file, so a
    per-holder allowance would let N holders on one state.db peak at ``N x (1 + _READ_POOL_MAX)``. A
    long-lived gateway holds at least two per profile path and the count grows with the profile count
    (#98573: EMFILE in a healthy process).
    """

    def __init__(self, per_file_limit: int = _READ_POOL_MAX) -> None:
        self.limit = per_file_limit
        self.permits = _PermitCounter(per_file_limit)
        self._lock = threading.Lock()
        # Weak: a holder dropped without close() must not pin peers' budget.
        self._members: weakref.WeakSet[StoreHolder] = weakref.WeakSet()
        self._duplicate_handles_warned = False

    def register(self, db: "StoreHolder") -> None:
        with self._lock:
            self._members.add(db)
            # Only writable handles carry the cost the warning names (writer connection, write
            # lock, close-time checkpoint). Read-only attaches (dashboard routers, status/lookup
            # one-shots) open per request by design and must not trip it.
            handles = sum(
                1 for member in self._members
                if not member.read_only and member._conn is not None
            )
            warn = (handles > _HANDLES_PER_PATH_WARN and not self._duplicate_handles_warned)
            if warn:
                self._duplicate_handles_warned = True
                # Same population as `handles`: the read-only attaches the count skips
                # must not reappear in the list, or the list no longer explains the number.
                creation_sites = ", ".join(sorted(
                    member._creation_site for member in self._members if not member.read_only
                ))
        if warn:
            # Writer connections cannot be capped; the only bound is not opening
            # redundant handles, so make the duplicate visible before it's an incident.
            logger.warning(
                # The only real bound on writers is not opening redundant handles in the first place (which
                # is what GatewayRunner borrowing SessionStore's handle does, #98573), so the next duplicate
                # should be visible before it becomes an incident rather than inferred from an lsof after
                # one.
                "%d live SessionDB handles on %s in this process; each holds "
                "its own writer connection (read connections are capped at %d "
                "for the file). A long-lived process should share one handle per path. "
                "Created at: %s",
                handles, db.db_path, self.limit, creation_sites,
            )

    def unregister(self, db: "StoreHolder") -> None:
        """Remove a closed writer from duplicate-handle diagnostics immediately."""
        with self._lock:
            self._members.discard(db)

    def acquire(self, requester: "StoreHolder") -> bool:
        """Take a permit for a new read connection, or refuse (caller degrades to the
        locked writer connection). Gates, broadest first: fd headroom, process
        ceiling, this file's ceiling."""
        _denial.reason = None
        free = _fd_free()
        if free is not None and free <= _FD_HEADROOM_RESERVE:
            global _read_open_denied_fd_headroom
            with _read_budgets_lock:
                _read_open_denied_fd_headroom += 1
            _denial.reason = ("fd-headroom", free)
            return False
        if not self._acquire_process_permit():
            _denial.reason = ("process", _process_read_permits.limit)
            return False
        if self._acquire_path_permit(requester):
            return True
        _process_read_permits.release()
        _denial.reason = ("per-db", self.limit)
        return False

    @staticmethod
    def take_denial() -> Optional[tuple[str, int]]:
        """The (ceiling, number) that refused this thread's last acquire(), once; None if unknown."""
        reason = getattr(_denial, "reason", None)
        _denial.reason = None
        return reason

    def release(self) -> None:
        """Return one connection's permits. Pairs with a successful acquire()."""
        self.permits.release()
        _process_read_permits.release()

    def _acquire_process_permit(self) -> bool:
        # Another thread may take a freed permit first: legitimate loss, no looping.
        return _process_read_permits.acquire(blocking=False) or (
            _reclaim_idle_read_conn_anywhere() and _process_read_permits.acquire(blocking=False)
        )

    def _acquire_path_permit(self, requester: "StoreHolder") -> bool:
        return self.permits.acquire(blocking=False) or (
            self.reclaim_idle(exclude=requester) and self.permits.acquire(blocking=False)
        )

    def reclaim_idle(self, exclude: "Optional[StoreHolder]" = None) -> bool:
        """Close one idle pooled connection held by a member; True if one went.
        Its release() returns both permits, so both ceilings reclaim through here."""
        with self._lock:
            members = [db for db in self._members if db is not exclude]
        return any(member._evict_one_idle_read_conn() for member in members)


# canonical db path -> permits for that file. Weak values: the budget lives only
# while some holder on the path holds it, so tmp_path churn can't grow this.
_read_budgets: weakref.WeakValueDictionary[str, _PathReadBudget] = (weakref.WeakValueDictionary())
_read_budgets_lock = threading.Lock()


def _read_budget_key(db_path) -> str:
    """Canonicalise a db path so two spellings share one budget."""
    try:
        return str(Path(db_path).resolve())
    except OSError:
        return str(db_path)


def _read_budget_for(db_path) -> _PathReadBudget:
    key = _read_budget_key(db_path)
    with _read_budgets_lock:
        budget = _read_budgets.get(key)
        if budget is None:
            with _per_file_limits_lock:
                limit = _per_file_limits.get(key, _READ_POOL_MAX)
            budget = _PathReadBudget(limit)
            _read_budgets[key] = budget
        return budget


# A read that finds every permit out is served on the writer connection under SessionDB._lock, so it queues
# behind every writer, including one parked on a slow -shm lock (network filesystems). It used to be DEBUG,
# invisible where it mattered; now one WARNING per database per window, counting the degraded reads between.
_POOL_EXHAUSTION_WARN_INTERVAL_S = 60.0
_pool_exhaustion_lock = threading.Lock()
_pool_exhaustion_warned_at: dict[str, float] = {}
_pool_exhaustion_suppressed: dict[str, int] = {}


def _describe_denial(db_path, denial: Optional[tuple[str, int]]) -> str:
    if denial is None:
        budget = _read_budgets.get(_read_budget_key(db_path))
        denial = ("per-db", budget.limit if budget is not None else _READ_POOL_MAX)
    kind, number = denial
    if kind == "fd-headroom":
        return f"fd-headroom {number} free, needs more than {_FD_HEADROOM_RESERVE}"
    return f"{kind} {number}"


def note_read_pool_exhausted(db_path, denial: Optional[tuple[str, int]] = None) -> None:
    """Log (rate-limited) that a read degraded to the locked writer connection for ``db_path``. ``denial`` is
    ``_PathReadBudget.take_denial()``: which ceiling refused (per-db, process or fd-headroom) and its number."""
    key = str(db_path)
    now = time.monotonic()
    with _pool_exhaustion_lock:
        last = _pool_exhaustion_warned_at.get(key)
        if last is not None and now - last < _POOL_EXHAUSTION_WARN_INTERVAL_S:
            _pool_exhaustion_suppressed[key] = _pool_exhaustion_suppressed.get(key, 0) + 1
            return
        _pool_exhaustion_warned_at[key] = now
        suppressed = _pool_exhaustion_suppressed.pop(key, 0)
    logger.warning(
        "read pool at capacity (%s) for %s; serving this read from the locked writer connection, so it waits "
        "behind every in-flight write%s. Sustained, this means long writes or slow storage (a network "
        "filesystem) are holding readers too. Logged at most once per %.0fs per database.",
        _describe_denial(db_path, denial), db_path,
        f" ({suppressed} more reads since the last warning)" if last is not None else "",
        _POOL_EXHAUSTION_WARN_INTERVAL_S,
    )
