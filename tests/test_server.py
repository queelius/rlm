from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from rlm.config import RLMConfig
from rlm.engine import RLM
from rlm.server import create_app
from rlm.types import API
from tests.fakes import ScriptedBackend, chat_text, responses_text


@pytest.mark.parametrize(
    ("api", "route", "public_request", "public_response"),
    [
        (
            "chat.completions",
            "/v1/chat/completions",
            {
                "model": "public-model",
                "messages": [
                    {"role": "system", "content": "Preserve me."},
                    {"role": "user", "content": "Use a tool if needed."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "seed": 9,
                "provider_extension": {"keep": [1, 2, 3]},
            },
            chat_text("public chat response", model="public-model"),
        ),
        (
            "responses",
            "/v1/responses",
            {
                "model": "public-model",
                "instructions": "Preserve me.",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "description": "Look up data",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
                "previous_response_id": "resp_previous",
                "provider_extension": {"keep": [1, 2, 3]},
            },
            responses_text("public Responses response", model="public-model"),
        ),
    ],
)
def test_rest_route_preserves_request_and_routes_api_family(
    api: API,
    route: str,
    public_request: dict[str, Any],
    public_response: dict[str, Any],
) -> None:
    public = ScriptedBackend([public_response], name="public")
    controller = ScriptedBackend([chat_text("planning prose without an action")], name="controller")
    rlm = RLM(
        public,
        controller_backend=controller,
        config=RLMConfig(
            controller_model="controller",
            max_consecutive_errors=1,
        ),
    )

    with TestClient(create_app(rlm, close_on_shutdown=False)) as client:
        response = client.post(route, json=public_request)

    assert response.status_code == 200
    assert response.json() == public_response
    assert response.headers["x-rlm-stop-reason"] == ("consecutive_protocol_errors_direct_fallback")
    assert response.headers["x-rlm-turns"] == "1"
    assert response.headers["x-rlm-model-calls"] == "2"
    assert response.headers["x-rlm-run-id"]
    assert len(public.calls) == 1
    assert public.calls[0].api == api
    assert public.calls[0].request == public_request
    public.assert_exhausted()
    controller.assert_exhausted()


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
def test_rest_route_returns_openai_error_for_streaming(route: str) -> None:
    backend = ScriptedBackend(name="public")
    rlm = RLM(backend, config=RLMConfig(controller_model="controller"))

    with TestClient(create_app(rlm, close_on_shutdown=False)) as client:
        response = client.post(route, json={"model": "public-model", "stream": True})

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "RLM supports only non-streaming requests; set stream to false or omit it",
            "type": "invalid_request_error",
            "param": "stream",
            "code": "streaming_not_supported",
        }
    }
    assert backend.calls == []


@pytest.mark.parametrize(
    ("body", "expected_code", "expected_param"),
    [
        ("{", "invalid_json", None),
        ("[]", "invalid_request", "request"),
    ],
)
def test_rest_route_rejects_invalid_or_non_object_json(
    body: str,
    expected_code: str,
    expected_param: str | None,
) -> None:
    backend = ScriptedBackend(name="public")
    rlm = RLM(backend, config=RLMConfig(controller_model="controller"))

    with TestClient(create_app(rlm, close_on_shutdown=False)) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["param"] == expected_param
    assert backend.calls == []
