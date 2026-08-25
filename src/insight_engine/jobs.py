"""In-process background job runner.

The old prototype ran audio transcription + LLM parsing synchronously inside
the HTTP request handler — for a real ~47-minute call that's minutes of CPU
work, so the request either timed out or just looked hung with no feedback.
Here every upload starts a background thread immediately, the browser gets a
job id back straight away, and polls `/jobs/{id}` for step-by-step progress —
exactly the 5-step pipeline animation from the mockup, except the steps and
their timing are real.

A thread pool (not asyncio) is deliberate: faster-whisper and Ollama calls
here are blocking C-extension / network calls, not cooperatively async.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


@dataclass
class JobStep:
    name: str
    detail: str


@dataclass
class Job:
    id: str
    status: str = STATUS_PENDING
    steps: list[JobStep] = field(default_factory=list)
    error: str | None = None
    result: Any = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=str(uuid.uuid4()))
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job_id: str, fn: Callable[[Callable[[str, str], None]], Any]) -> None:
        """Run `fn(on_step)` in a background thread, updating job state."""

        def on_step(name: str, detail: str) -> None:
            with self._lock:
                self._jobs[job_id].steps.append(JobStep(name=name, detail=detail))

        def target() -> None:
            with self._lock:
                self._jobs[job_id].status = STATUS_RUNNING
            try:
                result = fn(on_step)
                with self._lock:
                    self._jobs[job_id].result = result
                    self._jobs[job_id].status = STATUS_DONE
            except Exception as exc:  # noqa: BLE001 — turned into a job-visible error
                with self._lock:
                    self._jobs[job_id].error = str(exc)
                    self._jobs[job_id].status = STATUS_ERROR
                traceback.print_exc()

        threading.Thread(target=target, daemon=True).start()


job_manager = JobManager()
