"""Running the slow things in the background, and letting the browser watch.

A backtest with the ablation matrix takes minutes. An HTTP request cannot hold that open,
so every command becomes a job: start it, get an id, poll for state and log lines.

Two decisions worth explaining.

**One worker, not a pool.** ``max_workers=1``. These jobs each build a full pandas panel
and fit models over it; running three at once would triple peak memory and make all three
slower on a laptop. Queueing is the honest behaviour, and the queue position is reported.

**The log comes from the pipeline's own logging, not from new instrumentation.** A
:class:`logging.Handler` is attached for the duration of a job and captures records from
the ``swingbot`` logger tree. Every ``log.info`` already in the codebase — fold summaries,
calibration coverage, PBO path selections, capacity truncations, the loud Kelly fallback —
streams to the browser for free, and stays in sync with what the CLI prints, because it is
the same source. Adding a parallel progress channel would have been a second thing to keep
correct.

Jobs live in memory and die with the process. That is deliberate: the durable record of
what happened is the run directory, which already exists and is already the thing the rest
of the system refers to. The UI says so rather than implying job history is kept.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "failed"]

#: Log lines kept per job. Enough to follow a full ablation run, bounded so a runaway
#: loop cannot exhaust memory.
MAX_LOG_LINES = 2000

#: Finished jobs kept for inspection. The run directory is the permanent record.
MAX_JOBS = 40


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    market: str
    options: dict[str, Any] = field(default_factory=dict)
    state: JobState = "queued"
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    #: Set when the job produced a run directory, which is how the UI links to results.
    run_id: str = ""
    #: A short human summary of what came out, shown next to the job.
    summary: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    log: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))

    def to_dict(self, *, with_log: bool = True, log_from: int = 0) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "market": self.market,
            "options": self.options,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "run_id": self.run_id,
            "summary": self.summary,
            "error": self.error,
            "n_log": len(self.log),
        }
        if with_log:
            # Sliced by index so the browser can ask for "everything after what I have"
            # and append, rather than re-rendering the whole log every poll.
            payload["log"] = list(self.log)[log_from:]
        return payload


class _JobLogHandler(logging.Handler):
    """Routes ``swingbot`` log records into whichever job is running on this thread.

    Thread-local rather than global: the worker thread owns exactly one job at a time, so
    a record is attributed by the thread that emitted it. A handler that appended to "the
    current job" would misfile anything logged by a request handler on another thread.
    """

    def __init__(self, local: threading.local) -> None:
        super().__init__(level=logging.INFO)
        self._local = local

    def emit(self, record: logging.LogRecord) -> None:
        job: Job | None = getattr(self._local, "job", None)
        if job is None:
            return
        # Never let a bad format string in some far-off module take down the job it is
        # reporting on. A dropped log line is a nuisance; a crashed backtest is not.
        with contextlib.suppress(Exception):
            name = record.name.removeprefix("swingbot.")
            job.log.append(f"{record.levelname:<7} {name} — {record.getMessage()}")


class JobRunner:
    """Starts jobs, keeps their state, and captures their logs."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swingbot-job")
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._local = threading.local()

        self._handler = _JobLogHandler(self._local)
        self._root = logging.getLogger("swingbot")
        self._root.addHandler(self._handler)
        # The pipeline's own log level governs what a job can capture, so make sure INFO
        # records are not filtered out before the handler ever sees them.
        if self._root.level > logging.INFO or self._root.level == logging.NOTSET:
            self._root.setLevel(logging.INFO)

    # ------------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self._root.removeHandler(self._handler)
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------------------- submit

    def submit(
        self,
        kind: str,
        market: str,
        work: Callable[[], tuple[str, dict[str, Any]]],
        *,
        options: dict[str, Any] | None = None,
    ) -> Job:
        """Queue ``work``, which returns ``(run_id, summary)`` when it succeeds."""
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            market=market,
            options=dict(options or {}),
            created_at=_now(),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()

        self._pool.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[], tuple[str, dict[str, Any]]]) -> None:
        self._local.job = job
        job.state = "running"
        job.started_at = _now()
        job.log.append(f"INFO    {job.kind} — started for {job.market}")
        try:
            run_id, summary = work()
            job.run_id = run_id or ""
            job.summary = summary or {}
            job.state = "done"
            job.log.append(f"INFO    {job.kind} — finished")
        except Exception as exc:
            job.state = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(f"ERROR   {job.kind} — {job.error}")
            log.exception("job %s (%s) failed", job.id, job.kind)
        finally:
            job.finished_at = _now()
            self._local.job = None

    # ----------------------------------------------------------------------------- reads

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            ids = list(self._order)[-limit:]
            return [self._jobs[i] for i in reversed(ids) if i in self._jobs]

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(j.state in ("queued", "running") for j in self._jobs.values())

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs. Never evicts one still queued or running."""
        while len(self._order) > MAX_JOBS:
            for index, job_id in enumerate(self._order):
                job = self._jobs.get(job_id)
                if job is None or job.state in ("done", "failed"):
                    del self._order[index]
                    self._jobs.pop(job_id, None)
                    break
            else:
                return


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
