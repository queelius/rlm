"""Provider-neutral OpenAI-compatible HTTP transport."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from rlm.errors import InvalidRequestError, UpstreamError
from rlm.response import json_compatibility_error
from rlm.types import API


@runtime_checkable
class ModelBackend(Protocol):
    def complete(self, api: API, request: Mapping[str, Any]) -> dict[str, Any]: ...


class OpenAIEndpoint:
    """A thin raw-JSON client for an OpenAI-compatible base URL."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 300.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.headers = dict(headers or {})
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout)

    def complete(self, api: API, request: Mapping[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(dict(request))
        json_error = json_compatibility_error(body)
        if json_error is not None:
            raise InvalidRequestError(
                f"request must be strict JSON: {json_error}",
                code="invalid_json_value",
                param="request",
            )
        if body.get("stream") is True:
            raise InvalidRequestError(
                "RLM supports only non-streaming requests; set stream to false or omit it",
                code="streaming_not_supported",
                param="stream",
            )
        path = "/chat/completions" if api == "chat.completions" else "/responses"
        endpoint = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        try:
            response = self.client.post(endpoint, json=body, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise UpstreamError(502, {"message": str(exc)}, endpoint=endpoint) from exc
        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"message": response.text}
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

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> OpenAIEndpoint:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
