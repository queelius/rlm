"""Provider-neutral OpenAI-compatible HTTP transport."""

from __future__ import annotations

import asyncio
import copy
import math
import os
import threading
from collections.abc import Coroutine, Mapping
from contextlib import suppress
from typing import Any, Protocol, runtime_checkable

import httpx

from rlm.errors import InvalidRequestError, UpstreamError
from rlm.json import json_compatibility_error, strict_json_loads
from rlm.types import ModelCallContext


@runtime_checkable
class ModelBackend(Protocol):
    def complete(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
        context: ModelCallContext,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ModelCatalogBackend(Protocol):
    def list_models(self) -> dict[str, Any]: ...


@runtime_checkable
class CloseableBackend(Protocol):
    def close(self) -> None: ...


class OpenAIEndpoint:
    """A thin raw-JSON client for an OpenAI-compatible base URL."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 300.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = _positive_timeout(timeout, name="timeout")
        self.headers = dict(headers or {})
        self.client = client

    def complete(
        self,
        request: Mapping[str, Any],
        *,
        timeout: float,
        context: ModelCallContext,
    ) -> dict[str, Any]:
        body = copy.deepcopy(dict(request))
        json_error = json_compatibility_error(body)
        if json_error is not None:
            raise InvalidRequestError(
                f"request must be strict JSON: {json_error}",
                code="invalid_json_value",
                param="request",
            )
        if "stream" in body and body["stream"] is not False:
            raise InvalidRequestError(
                "RLM supports only non-streaming requests; set stream to false or omit it",
                code="streaming_not_supported",
                param="stream",
            )
        endpoint = f"{self.base_url}/responses"
        call_timeout = min(self.timeout, _positive_timeout(timeout, name="timeout"))
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return self._run_async(
            self._request_json(
                method="POST",
                endpoint=endpoint,
                headers=headers,
                body=body,
                timeout=call_timeout,
            ),
            endpoint=endpoint,
        )

    def list_models(self) -> dict[str, Any]:
        """Return the configured upstream's OpenAI-compatible model catalog."""

        endpoint = f"{self.base_url}/models"
        headers = {"Accept": "application/json", **self.headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return self._run_async(
            self._request_json(
                method="GET",
                endpoint=endpoint,
                headers=headers,
                body=None,
                timeout=self.timeout,
            ),
            endpoint=endpoint,
        )

    def close(self) -> None:
        """Close is a no-op: owned async transports are closed per request."""

    def __enter__(self) -> OpenAIEndpoint:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run_async(
        self,
        operation: Coroutine[Any, Any, dict[str, Any]],
        *,
        endpoint: str,
    ) -> dict[str, Any]:
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(operation)
            return _run_in_helper_thread(operation)
        except asyncio.TimeoutError as exc:
            raise UpstreamError(
                502,
                {"message": "upstream request exceeded the absolute timeout"},
                endpoint=endpoint,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(502, {"message": str(exc)}, endpoint=endpoint) from exc

    async def _request_json(
        self,
        *,
        method: str,
        endpoint: str,
        headers: dict[str, str],
        body: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        if self.client is None:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await _request_json(
                    client,
                    method=method,
                    endpoint=endpoint,
                    headers=headers,
                    body=body,
                    timeout=timeout,
                )
        return await _request_json(
            self.client,
            method=method,
            endpoint=endpoint,
            headers=headers,
            body=body,
            timeout=timeout,
        )


def _response_json_object(response: httpx.Response, *, endpoint: str) -> dict[str, Any]:
    """Validate one JSON object response and preserve upstream errors."""

    try:
        payload: Any = strict_json_loads(response.content)
    except (TypeError, ValueError) as exc:
        if response.is_success:
            raise UpstreamError(
                502,
                {"message": "upstream returned a non-JSON success body", "body": response.text},
                endpoint=endpoint,
            ) from exc
        payload = {
            "message": response.text,
            "content_type": response.headers.get("content-type"),
        }
    if not response.is_success:
        raise UpstreamError(response.status_code, payload, endpoint=endpoint)
    if not isinstance(payload, dict):
        raise UpstreamError(
            502,
            {"message": f"upstream returned {type(payload).__name__}, expected a JSON object"},
            endpoint=endpoint,
        )
    json_error = json_compatibility_error(payload)
    if json_error is not None:
        raise UpstreamError(
            502,
            {"message": f"upstream returned a non-JSON response object: {json_error}"},
            endpoint=endpoint,
        )
    return payload


async def _request_json(
    client: httpx.AsyncClient,
    *,
    method: str,
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    """Read one response under an absolute cancellation deadline."""

    async def read() -> dict[str, Any]:
        async with client.stream(
            method,
            endpoint,
            json=body,
            headers=headers,
            timeout=timeout,
        ) as response:
            await response.aread()
            return _response_json_object(response, endpoint=endpoint)

    return await asyncio.wait_for(read(), timeout=timeout)


class _HelperTaskCancellation:
    """Thread-safe cancellation handoff for one helper-loop task."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[dict[str, Any]] | None = None
        self._cancel_requested = False

    def bind(self, *, loop: asyncio.AbstractEventLoop, task: asyncio.Task[dict[str, Any]]) -> None:
        with self._lock:
            self._loop = loop
            self._task = task
            cancel_requested = self._cancel_requested
        if cancel_requested:
            self._schedule_cancel(loop, task)

    def request_cancel(self) -> None:
        with self._lock:
            self._cancel_requested = True
            loop = self._loop
            task = self._task
        if loop is not None and task is not None:
            self._schedule_cancel(loop, task)

    @staticmethod
    def _schedule_cancel(
        loop: asyncio.AbstractEventLoop,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        # A closed loop has already completed the helper task and cleanup.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(task.cancel)


async def _run_bound_helper_operation(
    operation: Coroutine[Any, Any, dict[str, Any]],
    cancellation: _HelperTaskCancellation,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(operation)
    cancellation.bind(loop=loop, task=task)
    return await task


def _run_in_helper_thread(operation: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    """Synchronously join a fresh event loop when the caller loop is active."""

    outcome: dict[str, Any] = {}
    cancellation = _HelperTaskCancellation()
    completed = threading.Event()

    def run() -> None:
        try:
            outcome["result"] = asyncio.run(_run_bound_helper_operation(operation, cancellation))
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            completed.set()

    thread = threading.Thread(target=run, name="rlm-http-bridge", daemon=False)
    try:
        thread.start()
    except BaseException:
        operation.close()
        raise
    interruption: BaseException | None = None
    while True:
        try:
            if completed.is_set():
                break
            thread.join()
            if completed.is_set():
                break
            completed.wait()
        except BaseException as exc:
            if interruption is None:
                interruption = exc
            cancellation.request_cancel()
    reaped = False
    while not reaped:
        try:
            thread.join()
            reaped = True
        except BaseException as exc:
            if interruption is None:
                interruption = exc
            cancellation.request_cancel()
    if interruption is not None:
        raise interruption
    if "error" in outcome:
        raise outcome["error"]
    result = outcome.get("result")
    if not isinstance(result, dict):  # pragma: no cover - _request_json contract
        raise RuntimeError("async HTTP bridge returned an invalid result")
    return result


def _positive_timeout(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return timeout
