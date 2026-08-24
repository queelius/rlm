from __future__ import annotations

import asyncio
import gc
import json
import os
import signal
import threading
import time
import warnings

import httpx
import pytest

from rlm.backend import (
    OpenAIEndpoint,
    _HelperTaskCancellation,
    _run_bound_helper_operation,
    _run_in_helper_thread,
)
from rlm.errors import InvalidRequestError, UpstreamError
from rlm.types import ModelCallContext, ModelRole
from tests.fakes import request, responses_text

CALL_CONTEXT = ModelCallContext(
    run_id="run-test",
    branch_id="branch-root",
    call_id="call-1",
    role=ModelRole.PUBLIC,
    depth=0,
)


def test_endpoint_posts_the_exact_request_to_responses() -> None:
    request_body = {"model": "model-a", "input": "hello", "provider_field": {"keep": True}}
    response_body = {"object": "response", "output": [], "usage": {}}

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://provider.test/v1/responses"
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["x-provider"] == "keep"
        assert json.loads(request.content) == request_body
        return httpx.Response(200, json=response_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        endpoint = OpenAIEndpoint(
            base_url="https://provider.test/v1/",
            api_key="secret",
            headers={"X-Provider": "keep"},
            client=client,
        )
        assert endpoint.complete(request_body, timeout=5, context=CALL_CONTEXT) == response_body
    finally:
        asyncio.run(client.aclose())


def test_endpoint_rejects_streaming_before_network_io() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("unexpected upstream request"))
    )
    try:
        endpoint = OpenAIEndpoint(base_url="https://provider.test/v1", client=client)
        with pytest.raises(InvalidRequestError) as caught:
            endpoint.complete(
                {"model": "model-a", "input": "hello", "stream": True},
                timeout=5,
                context=CALL_CONTEXT,
            )
    finally:
        asyncio.run(client.aclose())

    assert caught.value.code == "streaming_not_supported"


def test_endpoint_preserves_upstream_error_status_and_body() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(429, json={"error": {"message": "rate limited"}})
        )
    )
    try:
        endpoint = OpenAIEndpoint(base_url="https://provider.test/v1", client=client)
        with pytest.raises(UpstreamError) as caught:
            endpoint.complete(
                {"model": "model-a", "input": "hello"}, timeout=5, context=CALL_CONTEXT
            )
    finally:
        asyncio.run(client.aclose())

    assert caught.value.status_code == 429
    assert caught.value.to_body() == {"error": {"message": "rate limited"}}


def test_successful_non_json_response_is_an_upstream_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    transport = httpx.MockTransport(handler)
    endpoint = OpenAIEndpoint(
        base_url="https://provider.test/v1", client=httpx.AsyncClient(transport=transport)
    )

    with pytest.raises(UpstreamError, match="non-JSON success body"):
        endpoint.complete(request(), timeout=5, context=CALL_CONTEXT)


def test_successful_non_json_models_body_is_an_upstream_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    transport = httpx.MockTransport(handler)
    endpoint = OpenAIEndpoint(
        base_url="https://provider.test/v1",
        client=httpx.AsyncClient(transport=transport),
    )

    with pytest.raises(UpstreamError, match="non-JSON success body"):
        endpoint.list_models()


def test_endpoint_uses_smaller_call_timeout() -> None:
    seen: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(float(request.extensions["timeout"]["read"]))
        return httpx.Response(200, json=responses_text("done"))

    endpoint = OpenAIEndpoint(
        base_url="https://provider.test/v1",
        timeout=300,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    endpoint.complete(request(), timeout=0.25, context=CALL_CONTEXT)
    assert seen == [pytest.approx(0.25)]


def test_endpoint_cancels_a_trickling_response_at_the_absolute_call_deadline() -> None:
    class TrickleStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            for _ in range(10):
                await asyncio.sleep(0.02)
                yield b"x"

        async def aclose(self) -> None:
            self.closed = True

    stream = TrickleStream()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    endpoint = OpenAIEndpoint(base_url="https://provider.test/v1", client=client)
    started = time.monotonic()
    try:
        with pytest.raises(UpstreamError, match="absolute timeout"):
            endpoint.complete(request(), timeout=0.08, context=CALL_CONTEXT)
    finally:
        asyncio.run(client.aclose())

    assert time.monotonic() - started < 0.16
    assert stream.closed


def test_sync_endpoint_calls_inside_an_active_event_loop_preserve_results_and_errors() -> None:
    async def success(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses_text("done"))

    async def failure(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    async def exercise() -> None:
        success_client = httpx.AsyncClient(transport=httpx.MockTransport(success))
        failure_client = httpx.AsyncClient(transport=httpx.MockTransport(failure))
        try:
            endpoint = OpenAIEndpoint(base_url="https://provider.test/v1", client=success_client)
            result = endpoint.complete(request(), timeout=5, context=CALL_CONTEXT)
            assert result["output"][0]["content"][0]["text"] == "done"
            failing_endpoint = OpenAIEndpoint(
                base_url="https://provider.test/v1", client=failure_client
            )
            with pytest.raises(UpstreamError, match="unavailable") as caught:
                failing_endpoint.complete(request(), timeout=5, context=CALL_CONTEXT)
            assert caught.value.status_code == 503
        finally:
            await success_client.aclose()
            await failure_client.aclose()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(exercise())
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_active_loop_bridge_joins_its_helper_before_reraising_an_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_started = threading.Event()
    cancellation_observed = threading.Event()
    cleanup_complete = threading.Event()
    completed_normally = threading.Event()
    original_join = threading.Thread.join
    interrupts = 0

    async def operation() -> dict[str, str]:
        operation_started.set()
        try:
            await asyncio.sleep(1)
            completed_normally.set()
            return {"status": "finished"}
        except asyncio.CancelledError:
            cancellation_observed.set()
            raise
        finally:
            cleanup_complete.set()

    def interrupt_once(thread: threading.Thread, *args: object, **kwargs: object) -> None:
        nonlocal interrupts
        if thread.name == "rlm-http-bridge" and interrupts == 0:
            interrupts += 1
            assert operation_started.wait(timeout=1)
            raise KeyboardInterrupt
        original_join(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "join", interrupt_once)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run_in_helper_thread(operation())

    assert time.monotonic() - started < 0.25
    assert interrupts == 1
    assert cancellation_observed.is_set()
    assert cleanup_complete.is_set()
    assert not completed_normally.is_set()
    assert not any(
        thread.name == "rlm-http-bridge" and thread.is_alive() for thread in threading.enumerate()
    )


def test_active_loop_bridge_waits_for_request_cleanup_after_real_sigint() -> None:
    operation_started = threading.Event()
    cancellation_observed = threading.Event()
    cleanup_complete = threading.Event()
    completed_normally = threading.Event()

    async def operation() -> dict[str, str]:
        operation_started.set()
        try:
            await asyncio.sleep(1)
            completed_normally.set()
            return {"status": "finished"}
        except asyncio.CancelledError:
            cancellation_observed.set()
            raise
        finally:
            await asyncio.sleep(0.05)
            cleanup_complete.set()

    def send_sigint() -> None:
        assert operation_started.wait(timeout=1)
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=send_sigint, name="rlm-test-interrupter", daemon=False)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            _run_in_helper_thread(operation())
    finally:
        interrupter.join()

    assert cancellation_observed.is_set()
    assert cleanup_complete.is_set()
    assert not completed_normally.is_set()
    assert not any(
        thread.name == "rlm-http-bridge" and thread.is_alive() for thread in threading.enumerate()
    )


def test_helper_task_cancellation_remembers_an_interrupt_before_task_binding() -> None:
    cancellation_observed = threading.Event()
    cleanup_complete = threading.Event()
    cancellation = _HelperTaskCancellation()
    cancellation.request_cancel()

    async def operation() -> dict[str, str]:
        try:
            await asyncio.sleep(1)
            return {"status": "finished"}
        except asyncio.CancelledError:
            cancellation_observed.set()
            raise
        finally:
            cleanup_complete.set()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_run_bound_helper_operation(operation(), cancellation))

    assert cancellation_observed.is_set()
    assert cleanup_complete.is_set()
