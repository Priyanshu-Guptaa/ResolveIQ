"""Tests for app.engines.chat.enhancement (Chat Assistant Phase 37).

Pure unit tests against the real ChatEnhancementService -- no database,
no HTTP, no real Ollama. Exercises the job lifecycle
(PENDING/RUNNING/COMPLETED/FAILED/REJECTED/TIMED_OUT), concurrency/
queue bounds, and cleanup directly, with controllable work callables
(threading.Event-gated) rather than real sleeps, for fast, deterministic
tests.
"""

from __future__ import annotations

import threading
import time

from app.domain.chat import EnhancementStatus
from app.engines.chat import enhancement as enhancement_module
from app.engines.llm.provider import LLMProviderError
from app.engines.chat.enhancement import ChatEnhancementService


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_successful_job_completes_with_answer_text():
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    job = service.submit("session-1", lambda: "a validated enhanced answer")
    assert _wait_until(lambda: service.get(job.id).status == EnhancementStatus.COMPLETED)
    finished = service.get(job.id)
    assert finished.answer_text == "a validated enhanced answer"
    assert finished.error is None
    assert finished.session_id == "session-1"


def test_job_returning_none_is_rejected():
    """None means "the LLM's output failed an existing safety
    validator, or generation produced nothing to validate" -- see
    ChatOrchestrator._finalize_llm_answer's contract."""
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    job = service.submit("session-1", lambda: None)
    assert _wait_until(lambda: service.get(job.id).status == EnhancementStatus.REJECTED)
    finished = service.get(job.id)
    assert finished.answer_text is None
    assert finished.error is not None


def test_job_raising_generic_exception_is_failed():
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)

    def boom():
        raise RuntimeError("worker exploded")

    job = service.submit("session-1", boom)
    assert _wait_until(lambda: service.get(job.id).status == EnhancementStatus.FAILED)
    finished = service.get(job.id)
    assert "worker exploded" in finished.error


def test_job_raising_timeout_error_is_timed_out():
    """Classified via the exact, real LLMProviderError message
    OllamaProvider itself raises for a timeout -- see
    ollama_provider.py: 'Ollama request to {url} timed out.'"""
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)

    def timeout_work():
        raise LLMProviderError("Ollama request to http://localhost:11434 timed out.")

    job = service.submit("session-1", timeout_work)
    assert _wait_until(lambda: service.get(job.id).status == EnhancementStatus.TIMED_OUT)


def test_non_timeout_provider_error_is_failed_not_timed_out():
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)

    def connect_error_work():
        raise LLMProviderError("Could not reach Ollama at http://localhost:11434 -- is it running?")

    job = service.submit("session-1", connect_error_work)
    assert _wait_until(lambda: service.get(job.id).status == EnhancementStatus.FAILED)


def test_unknown_job_id_returns_none():
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    assert service.get("does-not-exist") is None


def test_worker_exception_never_crashes_the_service():
    """A worker-thread exception must only ever downgrade its own job
    -- the service, and every other job, remains fully usable."""
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)

    def boom():
        raise RuntimeError("boom")

    job1 = service.submit("s1", boom)
    assert _wait_until(lambda: service.get(job1.id).status == EnhancementStatus.FAILED)
    job2 = service.submit("s2", lambda: "still works")
    assert _wait_until(lambda: service.get(job2.id).status == EnhancementStatus.COMPLETED)


def test_concurrency_and_queue_limits_enforced():
    """max_concurrent=1, max_queued=1 -- a 3rd simultaneous submission
    must be REJECTED immediately (never blocking the caller), while the
    2nd (queued, not yet running) must NOT be rejected."""
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    started = threading.Event()
    release = threading.Event()

    def slow_work():
        started.set()
        release.wait(timeout=5)
        return "first done"

    job1 = service.submit("s1", slow_work)
    assert started.wait(timeout=2), "job1 never started running"
    assert job1.status in (EnhancementStatus.PENDING, EnhancementStatus.RUNNING)

    job2 = service.submit("s2", lambda: "second done")  # fills the queue slot
    assert job2.status in (EnhancementStatus.PENDING, EnhancementStatus.RUNNING)

    job3 = service.submit("s3", lambda: "third done")  # queue is now full
    assert job3.status == EnhancementStatus.REJECTED
    assert "queue is full" in job3.error.lower()

    release.set()
    assert _wait_until(lambda: service.get(job1.id).status == EnhancementStatus.COMPLETED)
    assert _wait_until(lambda: service.get(job2.id).status == EnhancementStatus.COMPLETED)


def test_submit_never_blocks_the_caller():
    """The defining property of this whole module: submitting a job,
    however slow the work will be, returns immediately."""
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    release = threading.Event()

    def slow_work():
        release.wait(timeout=5)
        return "eventually"

    t0 = time.time()
    job = service.submit("s1", slow_work)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"submit() blocked for {elapsed}s"
    assert job.status in (EnhancementStatus.PENDING, EnhancementStatus.RUNNING)
    release.set()
    assert _wait_until(lambda: service.get(job.id).status == EnhancementStatus.COMPLETED)


def test_finished_jobs_are_purged_after_ttl(monkeypatch):
    monkeypatch.setattr(enhancement_module, "_JOB_TTL_SECONDS", 0.05)
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    job = service.submit("s1", lambda: "done")
    assert _wait_until(lambda: service.get(job.id) is not None and service.get(job.id).status == EnhancementStatus.COMPLETED)
    time.sleep(0.1)
    assert service.get(job.id) is None


def test_pending_or_running_jobs_are_never_purged_early(monkeypatch):
    monkeypatch.setattr(enhancement_module, "_JOB_TTL_SECONDS", 0.01)
    service = ChatEnhancementService(max_concurrent=1, max_queued=1)
    release = threading.Event()
    started = threading.Event()

    def slow_work():
        started.set()
        release.wait(timeout=5)
        return "done"

    job = service.submit("s1", slow_work)
    assert started.wait(timeout=2)
    time.sleep(0.05)  # well past the (shrunk) TTL, but the job is still RUNNING
    assert service.get(job.id) is not None
    release.set()
