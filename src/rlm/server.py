"""FastAPI adapter for the non-streaming OpenAI-compatible RLM surface."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from rlm.engine import RLM
from rlm.errors import InvalidRequestError, RLMError
from rlm.types import API, RunResult


def create_app(
    rlm: RLM,
    *,
    close_on_shutdown: bool = True,
) -> FastAPI:
    """Create an HTTP app around one shared :class:`~rlm.engine.RLM`.

    Routes intentionally use raw request dictionaries instead of Pydantic
    models.  This preserves unknown provider extensions and keeps validation
    responsibility at the wrapped endpoint, as required by the model-decorator
    contract.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if close_on_shutdown:
                await run_in_threadpool(rlm.close)

    app = FastAPI(
        title="RLM",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.rlm = rlm

    @app.exception_handler(RLMError)
    async def handle_rlm_error(_: Request, error: RLMError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content=error.to_body())

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await _run_request(rlm, "chat.completions", request)

    @app.post("/v1/responses")
    async def responses(request: Request) -> JSONResponse:
        return await _run_request(rlm, "responses", request)

    return app


async def _run_request(rlm: RLM, api: API, request: Request) -> JSONResponse:
    body = await _read_json_object(request)
    result = await run_in_threadpool(rlm.run, api, body)
    return JSONResponse(content=result.response, headers=_run_headers(result))


async def _read_json_object(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidRequestError(
            "request body must be a valid JSON object",
            code="invalid_json",
        ) from exc
    if not isinstance(value, Mapping):
        raise InvalidRequestError(
            f"request body must be a JSON object, got {type(value).__name__}",
            param="request",
        )
    # ``dict`` retains every JSON field and value while satisfying the engine's
    # concrete request type.  The engine makes its own defensive deep copy.
    return dict(value)


def _run_headers(result: RunResult) -> dict[str, str]:
    return {
        "X-RLM-Run-ID": result.run_id,
        "X-RLM-Stop-Reason": result.stop_reason,
        "X-RLM-Turns": str(result.turns),
        "X-RLM-Duration-Seconds": f"{result.duration_seconds:.6f}",
        "X-RLM-Model-Calls": str(result.usage.calls),
        "X-RLM-Input-Tokens": str(result.usage.input_tokens),
        "X-RLM-Output-Tokens": str(result.usage.output_tokens),
    }
