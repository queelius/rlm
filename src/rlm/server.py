"""FastAPI adapter for the non-streaming OpenAI-compatible RLM surface."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from rlm.attestation import RunAttestation
from rlm.backend import ModelCatalogBackend
from rlm.engine import RLM
from rlm.errors import InvalidRequestError, RLMError
from rlm.json import StrictJSONError, strict_json_loads
from rlm.types import RunResult


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

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        if not isinstance(rlm.backend, ModelCatalogBackend):
            raise RLMError(
                "configured public backend does not support model listing",
                code="models_not_supported",
                status_code=501,
            )
        payload = await run_in_threadpool(rlm.backend.list_models)
        return JSONResponse(content=payload)

    @app.post("/v1/responses")
    async def responses(request: Request) -> JSONResponse:
        body = await _read_json_object(request)
        result = await run_in_threadpool(rlm.run, body)
        return JSONResponse(content=result.response, headers=_run_headers(result))

    return app


async def _read_json_object(request: Request) -> dict[str, Any]:
    try:
        value = strict_json_loads(await request.body())
    except (StrictJSONError, UnicodeDecodeError) as exc:
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
    return RunAttestation.from_result(result).to_headers()
