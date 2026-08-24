"""Thread-safe run-wide budgets, deadlines, and usage accounting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from rlm.config import RLMConfig
from rlm.errors import BackendProtocolError, LimitExceededError
from rlm.types import TokenUsage


class Ledger:
    """State shared by the root RLM run and all of its recursive branches.

    Budget reservations are deliberately separate:

    * ``reserve_subcalls(n)`` counts requested leaf or recursive operations.
    * ``reserve_model_call()`` counts every actual upstream backend attempt,
      including controller calls and calls that later fail.
    * ``model_slot()`` bounds concurrent backend I/O but consumes no budget.

    Keeping these operations separate lets the engine reserve a batch of
    subcalls atomically, while counting each resulting backend request exactly
    once.  All counters and token totals are protected by one lock.  A single
    instance must be passed to every recursive child.
    """

    def __init__(
        self,
        config: RLMConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self.started_at = clock()
        self.deadline_at = self.started_at + config.limits.deadline_seconds

        self._lock = threading.Lock()
        self._model_slots = threading.BoundedSemaphore(config.limits.max_parallel_model_calls)
        self._model_calls = 0
        self._subcalls = 0
        self._active_model_calls = 0
        self._peak_parallel_model_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._unreported_calls = 0

    def check(self) -> None:
        """Raise if the shared wall-clock deadline has been reached."""
        now = self._clock()
        if now >= self.deadline_at:
            self._raise_deadline(now)

    def reserve_model_call(self) -> int:
        """Atomically reserve one upstream call and return its 1-based number."""
        with self._lock:
            now = self._clock()
            self._check_deadline_locked(now)
            if self._model_calls >= self.config.limits.max_model_calls:
                raise LimitExceededError(
                    f"model call limit reached ({self.config.limits.max_model_calls})",
                    code="model_call_limit",
                )
            self._model_calls += 1
            return self._model_calls

    def reserve_subcalls(self, count: int = 1) -> int:
        """Atomically reserve ``count`` subcalls and return the new total.

        The reservation is all-or-nothing, which lets a batch fail before any
        member starts.  Zero is accepted as a convenient no-op for empty
        batches; negative and boolean values are rejected.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("subcall count must be a non-negative integer")
        with self._lock:
            now = self._clock()
            self._check_deadline_locked(now)
            requested_total = self._subcalls + count
            if requested_total > self.config.limits.max_subcalls:
                raise LimitExceededError(
                    (
                        f"subcall limit would be exceeded: requested {count}, "
                        f"used {self._subcalls}, limit {self.config.limits.max_subcalls}"
                    ),
                    code="subcall_limit",
                )
            self._subcalls = requested_total
            return self._subcalls

    @contextmanager
    def model_slot(self) -> Iterator[None]:
        """Hold one run-wide backend-call slot until the context exits.

        Waiting is bounded by the run deadline.  Hold the slot only around one
        actual ``backend.complete`` call, never around a whole recursive child
        branch.  Applying this context to every backend call gives the complete
        recursive run one concurrency ceiling without nested-slot deadlocks.
        """
        self.check()
        remaining = self.remaining_seconds
        acquired = self._model_slots.acquire(timeout=remaining)
        if not acquired:
            self._raise_deadline(self._clock())

        entered = False
        try:
            with self._lock:
                now = self._clock()
                self._check_deadline_locked(now)
                self._active_model_calls += 1
                self._peak_parallel_model_calls = max(
                    self._peak_parallel_model_calls,
                    self._active_model_calls,
                )
                entered = True
            yield
        finally:
            if entered:
                with self._lock:
                    self._active_model_calls -= 1
            self._model_slots.release()

    @contextmanager
    def subcall_slot(self) -> Iterator[None]:
        """Backward-compatible alias for :meth:`model_slot`."""
        with self.model_slot():
            yield

    def record_response(self, response: Mapping[str, Any]) -> TokenUsage:
        """Record provider usage from an OpenAI Responses object.

        When an aggregate token budget is configured, every response must
        include valid input and output token counts so enforcement cannot hide
        behind missing provider accounting.
        """
        usage = usage_from_response(response)
        with self._lock:
            if self.config.limits.max_total_tokens is not None and usage.unreported_calls:
                raise BackendProtocolError(
                    "token budget requires usage from every backend response"
                )
            self._input_tokens += usage.input_tokens
            self._output_tokens += usage.output_tokens
            self._unreported_calls += usage.unreported_calls
            if (
                self.config.limits.max_total_tokens is not None
                and self._input_tokens + self._output_tokens > self.config.limits.max_total_tokens
            ):
                raise LimitExceededError(
                    "token limit exceeded "
                    f"({self._input_tokens + self._output_tokens}/"
                    f"{self.config.limits.max_total_tokens})",
                    code="token_limit",
                )
        return usage

    def snapshot(self) -> TokenUsage:
        """Return a detached aggregate usage snapshot."""
        with self._lock:
            return self._usage_locked()

    def stats(self) -> dict[str, Any]:
        """Return a consistent, JSON-serializable operational snapshot."""
        now = self._clock()
        with self._lock:
            usage = self._usage_locked()
            return {
                "model_calls": self._model_calls,
                "subcalls": self._subcalls,
                "active_model_calls": self._active_model_calls,
                "peak_parallel_model_calls": self._peak_parallel_model_calls,
                "max_model_calls": self.config.limits.max_model_calls,
                "max_subcalls": self.config.limits.max_subcalls,
                "max_parallel_model_calls": self.config.limits.max_parallel_model_calls,
                "max_total_tokens": self.config.limits.max_total_tokens,
                "started_at": self.started_at,
                "deadline_at": self.deadline_at,
                "deadline_seconds": self.config.limits.deadline_seconds,
                "elapsed_seconds": max(0.0, now - self.started_at),
                "remaining_seconds": max(0.0, self.deadline_at - now),
                "usage": usage.to_dict(),
            }

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - self._clock())

    def timeout(self, cap: float | None = None) -> float:
        """Return a positive blocking timeout bounded by the run deadline."""
        if cap is not None and cap <= 0:
            raise ValueError("timeout cap must be positive")
        now = self._clock()
        remaining = self.deadline_at - now
        if remaining <= 0:
            self._raise_deadline(now)
        return remaining if cap is None else min(remaining, cap)

    def _usage_locked(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            calls=self._model_calls,
            unreported_calls=self._unreported_calls,
        )

    def _check_deadline_locked(self, now: float) -> None:
        if now >= self.deadline_at:
            self._raise_deadline(now)

    def _raise_deadline(self, now: float) -> None:
        elapsed = max(0.0, now - self.started_at)
        raise LimitExceededError(
            (
                f"run deadline exceeded after {elapsed:.3f} seconds "
                f"(limit {self.config.limits.deadline_seconds:.3f} seconds)"
            ),
            code="deadline_exceeded",
        )


# A descriptive alias for callers that prefer to make the sharing scope
# explicit.  ``Ledger`` remains the small public API name.
RunLedger = Ledger


def usage_from_response(response: Mapping[str, Any]) -> TokenUsage:
    """Extract token counts from a Responses response."""
    raw = response.get("usage")
    if raw is None or "usage" not in response:
        return TokenUsage(calls=1, unreported_calls=1)
    if not isinstance(raw, Mapping):
        raise BackendProtocolError("backend response usage must be an object or null")
    input_tokens = _token_count(raw.get("input_tokens"), name="input_tokens")
    output_tokens = _token_count(raw.get("output_tokens"), name="output_tokens")
    total_tokens = _token_count(raw.get("total_tokens"), name="total_tokens")
    if total_tokens != input_tokens + output_tokens:
        raise BackendProtocolError(
            "backend response usage total_tokens must equal input_tokens + output_tokens"
        )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        calls=1,
    )


def _token_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendProtocolError(f"backend response usage {name} must be a non-negative integer")
    return value
