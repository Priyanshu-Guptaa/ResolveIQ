"""Asynchronous LLM answer enhancement (Chat Assistant Phase 37).

DETERMINISTIC ANSWER FIRST -> OPTIONAL LLM ENHANCEMENT -> SAFE
VALIDATION -> ASYNC DELIVERY. Every phase since Phase 1 has already
established that the deterministic answer (``ChatOrchestrator.
_compose_answer``) is fully computable BEFORE the LLM is ever touched
-- retrieval, ranking, and ``StructuredResolution`` construction
(``RecommendationEngine.generate()``) are 100% deterministic and
already complete by the time ``_generate_answer`` would otherwise block
on ``OllamaProvider.generate()``. This module is what lets
``ChatOrchestrator`` return that already-complete deterministic answer
immediately and run LLM generation in the background instead, without
ever making the user wait for qwen2.5:3b (Phase 35: 87.5% success but
still multi-second-to-tens-of-seconds latency, and qwen3:4b-class
models can run for minutes).

Deliberately the smallest possible abstraction, not a new job-queue
framework: a single, small, bounded ``ThreadPoolExecutor`` (the exact
same stdlib primitive already used, synchronously, by
``app.engines.external_knowledge.service.ExternalKnowledgeService.
gather()`` for its own TFS/Wiki fan-out -- reused here as an
already-established, zero-new-dependency pattern in this codebase, not
introduced fresh). No Redis, no Celery, no message broker: this is a
single-process FastAPI application with no existing distributed-task
infrastructure (a repository-wide search for BackgroundTasks/Celery/
WebSocket/job_id/task_queue/redis/StreamingResponse/asyncio.create_task
found none), so a process-local, in-memory job registry is the correct,
minimal answer for this application's actual current architecture --
never persisted to ``data/resolveiq.db`` or Chroma, and never required
to be (Absolute Rule: STOP before introducing DB persistence for job
state -- this module deliberately never does).

``ChatEnhancementService`` is a PROCESS-LIFETIME SINGLETON (constructed
once via ``app.api.dependencies``' ``@lru_cache`` pattern, the same
idiom already used for ``_llm_provider()``), never per-request state --
``get_chat_orchestrator()`` constructs a fresh ``ChatOrchestrator`` on
every call, so a job submitted while handling one request must remain
pollable from a later request's own, different ``ChatOrchestrator``
instance. The service, not the orchestrator, owns the thread pool and
the job registry.

Concurrency is bounded conservatively: Phase 35's own ``ollama ps``
measurement found qwen2.5:3b running at 100% CPU with no GPU/iGPU
acceleration active on this hardware -- a second concurrent generation
would only contend for the same CPU cycles Ollama is already using,
never add real throughput. ``max_concurrent`` therefore defaults to 1
(configurable via ``Settings.llm_max_concurrent_jobs``), and a small,
separately-bounded queue depth (``max_queued``, default 1, configurable
via ``Settings.llm_enhancement_max_queue``) determines how many further
jobs may wait before a new submission is rejected outright -- REJECTED,
never a blocked request and never an unbounded queue.

Job lifecycle: PENDING (registered, not yet started) -> RUNNING
(worker thread executing) -> exactly one of COMPLETED (validated
answer available), FAILED (a real ``LLMProviderError`` or unexpected
worker exception), REJECTED (either the queue was full at submission
time, or the LLM's raw output failed the existing safety validators --
see ``ChatOrchestrator._attempt_llm_answer``, the single source of that
decision, never reimplemented here), or TIMED_OUT (specifically an
Ollama request that timed out, distinguished from other failures by
inspecting the real, existing ``LLMProviderError`` message text
``OllamaProvider`` already raises verbatim -- "Ollama request to
{url} timed out." -- not a new timeout mechanism of its own; the
underlying HTTP call already bounds worker-thread lifetime via
``Settings.ollama_timeout_seconds``, so no additional cancellation
plumbing is needed to keep a job from running forever). Finished jobs
(COMPLETED/FAILED/REJECTED/TIMED_OUT) are purged from the in-memory
registry after ``_JOB_TTL_SECONDS`` (15 minutes -- long enough to poll
a real answer, short enough that an abandoned session's job record
does not accumulate forever in a long-running process); PENDING/RUNNING
jobs are never purged early.

Known, honest limitation: Python's ``ThreadPoolExecutor`` has no
cooperative cancellation -- a job already RUNNING when its caller stops
polling will still run to completion (or its own bounded timeout) in
the background; its result is simply never read. This is a bounded,
documented trade-off (the job cannot run longer than
``ollama_timeout_seconds`` already allows), not an unbounded resource
leak, and was not solved with additional complexity this phase's own
"smallest safe abstraction" principle does not ask for.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from app.domain.chat import EnhancementStatus

logger = logging.getLogger(__name__)

_JOB_TTL_SECONDS = 900.0
"""How long a finished (COMPLETED/FAILED/REJECTED/TIMED_OUT) job stays
pollable before being purged -- see module docstring."""


@dataclass
class ChatEnhancementJob:
    """One LLM enhancement attempt's current state. Never contains raw,
    unvalidated LLM output -- ``answer_text`` is populated only when
    ``status == COMPLETED``, meaning it already passed every existing
    safety validator (see ``ChatOrchestrator._attempt_llm_answer``)."""

    id: str
    session_id: str
    status: EnhancementStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    answer_text: str | None = None
    error: str | None = None


class ChatEnhancementService:
    """See module docstring. Owns one small, bounded thread pool and an
    in-memory job registry; never touches the database or Chroma."""

    def __init__(self, *, max_concurrent: int = 1, max_queued: int = 1) -> None:
        self._max_concurrent = max(1, max_concurrent)
        self._max_queued = max(0, max_queued)
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent, thread_name_prefix="llm-enhancement")
        self._lock = threading.Lock()
        self._jobs: dict[str, ChatEnhancementJob] = {}
        self._in_flight: set[str] = set()
        """Job ids currently PENDING or RUNNING -- the live queue-depth
        signal ``submit()`` checks against, kept as an explicit set
        rather than inferred from ``ThreadPoolExecutor`` internals."""

    def submit(self, session_id: str, work: Callable[[], str | None]) -> ChatEnhancementJob:
        """``work`` is a zero-arg callable returning the validated
        enhanced answer text, or ``None`` when the LLM's output failed
        an existing safety validator or generation itself failed to
        produce anything to validate. Never blocks the caller -- either
        a job is registered and handed to the thread pool, or the queue
        is already full and a REJECTED job is returned immediately, in
        both cases before this method returns."""
        self._purge_expired()
        with self._lock:
            if len(self._in_flight) >= self._max_concurrent + self._max_queued:
                job = ChatEnhancementJob(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    status=EnhancementStatus.REJECTED,
                    created_at=time.time(),
                    finished_at=time.time(),
                    error="Enhancement queue is full; the deterministic answer stands.",
                )
                self._jobs[job.id] = job
                return job
            job = ChatEnhancementJob(
                id=str(uuid.uuid4()), session_id=session_id, status=EnhancementStatus.PENDING, created_at=time.time()
            )
            self._jobs[job.id] = job
            self._in_flight.add(job.id)
        self._executor.submit(self._run, job.id, work)
        return job

    def _run(self, job_id: str, work: Callable[[], str | None]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:  # pragma: no cover -- defensive only, job is always registered before submit()
                return
            job.status = EnhancementStatus.RUNNING
            job.started_at = time.time()
        try:
            result = work()
        except Exception as exc:  # noqa: BLE001 -- a worker-thread exception must never crash the pool or reach the user; it can only ever downgrade this one job. The deterministic answer, already returned to the caller before this job was even submitted, is entirely unaffected.
            logger.warning("LLM enhancement job %s raised: %s", job_id, exc)
            status = EnhancementStatus.TIMED_OUT if "timed out" in str(exc).lower() else EnhancementStatus.FAILED
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job.status = status
                    job.error = str(exc)
                    job.finished_at = time.time()
                self._in_flight.discard(job_id)
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                if result is None:
                    job.status = EnhancementStatus.REJECTED
                    job.error = "LLM output failed safety validation, or generation produced nothing to validate; the deterministic answer stands."
                else:
                    job.status = EnhancementStatus.COMPLETED
                    job.answer_text = result
                job.finished_at = time.time()
            self._in_flight.discard(job_id)

    def get(self, job_id: str) -> ChatEnhancementJob | None:
        self._purge_expired()
        with self._lock:
            return self._jobs.get(job_id)

    def _purge_expired(self) -> None:
        cutoff = time.time() - _JOB_TTL_SECONDS
        with self._lock:
            expired = [jid for jid, j in self._jobs.items() if j.finished_at is not None and j.finished_at < cutoff]
            for jid in expired:
                del self._jobs[jid]
