from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from rlm.backend import OpenAIEndpoint
from rlm.config import ControllerConfig, RLMConfig, RunLimits
from rlm.engine import RLM
from rlm.json import strict_json_sha256
from rlm.prompts import default_harness_spec
from rlm.server import create_app
from rlm.types import ControllerRunIdentity, RunResult, TokenUsage
from tests.fakes import ScriptedBackend, controller_code, responses_text


def test_models_route_passes_through_the_upstream_catalog() -> None:
    catalog = {
        "object": "list",
        "data": [{"id": "model-a", "object": "model", "owned_by": "provider"}],
    }

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://provider.test/v1/models"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json=catalog)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    backend = OpenAIEndpoint(
        base_url="https://provider.test/v1/",
        api_key="secret",
        client=http_client,
    )
    try:
        with TestClient(create_app(RLM(backend), close_on_shutdown=False)) as client:
            response = client.get("/v1/models")
    finally:
        asyncio.run(http_client.aclose())

    assert response.status_code == 200
    assert response.json() == catalog


def test_models_route_reports_an_unsupported_custom_backend() -> None:
    with TestClient(create_app(RLM(ScriptedBackend()), close_on_shutdown=False)) as client:
        response = client.get("/v1/models")

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "models_not_supported"


def test_success_headers_attest_effective_controller_identity_and_usage() -> None:
    options = {"seed": 42, "temperature": 0}
    config = RLMConfig(controller=ControllerConfig(model="qwen", options=options))
    app = create_app(
        RLM(ScriptedBackend([controller_code('FINAL_TEXT("done")')]), config=config),
        close_on_shutdown=False,
    )

    with TestClient(app) as client:
        response = client.post("/v1/responses", json={"model": "qwen", "input": "task"})

    assert response.status_code == 200
    assert response.headers["x-rlm-controller-model"] == "qwen"
    assert response.headers["x-rlm-controller-options-sha256"] == strict_json_sha256(options)
    assert response.headers["x-rlm-harness-fingerprint"] == default_harness_spec().fingerprint()
    assert int(response.headers["x-rlm-input-tokens"]) >= 0
    assert int(response.headers["x-rlm-output-tokens"]) >= 0
    assert int(response.headers["x-rlm-usage-unreported-calls"]) == 0


def test_run_result_requires_immutable_complete_attestation() -> None:
    response = responses_text("done")
    usage = TokenUsage(input_tokens=1)
    result = RunResult(
        response=response,
        run_id="run",
        stop_reason="done",
        usage=usage,
        turns=1,
        duration_seconds=1.0,
        controller_identity=ControllerRunIdentity("qwen", "a" * 64),
        harness_fingerprint="b" * 64,
    )
    response["model"] = "mutated"
    usage.input_tokens = 99
    assert result.response["model"] != "mutated"
    assert result.usage.input_tokens == 1
    with pytest.raises(AttributeError):
        result.harness_fingerprint = "c" * 64  # type: ignore[misc]


def test_responses_route_preserves_the_request() -> None:
    request = {
        "model": "public-model",
        "instructions": "Preserve me.",
        "input": [{"role": "user", "content": "Hello"}],
        "provider_extension": {"keep": [1, 2, 3]},
    }
    public_response = responses_text("public answer", model="public-model")
    public = ScriptedBackend([public_response], name="public")
    controller = ScriptedBackend(
        [controller_code("response = model_complete()\nFINAL_RESPONSE(response)")],
        name="controller",
    )
    rlm = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(controller=ControllerConfig(model="controller")),
    )

    with TestClient(create_app(rlm, close_on_shutdown=False)) as client:
        response = client.post("/v1/responses", json=request)

    assert response.status_code == 200
    assert response.json() == public_response
    assert response.headers["x-rlm-stop-reason"] == "final_response"
    assert response.headers["x-rlm-turns"] == "1"
    assert response.headers["x-rlm-model-calls"] == "2"
    assert public.calls[0].request == request


def test_chat_completions_route_does_not_exist() -> None:
    with TestClient(create_app(RLM(ScriptedBackend()), close_on_shutdown=False)) as client:
        response = client.post("/v1/chat/completions", json={})

    assert response.status_code == 404


def test_unrepaired_controller_failure_is_an_error_not_a_fallback_response() -> None:
    rlm = RLM(
        ScriptedBackend(name="public"),
        controller_backend=ScriptedBackend([responses_text("not Python")]),
        config=RLMConfig(
            controller=ControllerConfig(model="controller"),
            limits=RunLimits(max_turns=1),
        ),
    )

    with TestClient(create_app(rlm, close_on_shutdown=False)) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "public-model", "input": "hello"},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "max_turns_exceeded"


def test_streaming_is_rejected() -> None:
    with TestClient(create_app(RLM(ScriptedBackend()), close_on_shutdown=False)) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "public-model", "input": "hello", "stream": True},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "streaming_not_supported"


def test_route_rejects_invalid_or_non_object_json() -> None:
    with TestClient(create_app(RLM(ScriptedBackend()), close_on_shutdown=False)) as client:
        invalid = client.post(
            "/v1/responses",
            content="{",
            headers={"content-type": "application/json"},
        )
        non_object = client.post("/v1/responses", json=[])

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_json"
    assert non_object.status_code == 400
    assert non_object.json()["error"]["param"] == "request"


@pytest.mark.parametrize(
    "body",
    [b'{"model":"m","input":"a","input":"b"}', b'{"model":"m","temperature":NaN}'],
)
def test_public_request_rejects_non_strict_json(body: bytes) -> None:
    app = create_app(RLM(ScriptedBackend()), close_on_shutdown=False)
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"
