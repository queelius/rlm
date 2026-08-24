from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

import pytest

from rlm import RLM, RLMConfig
from rlm.abi import (
    ENVIRONMENT_ABI,
    EnvironmentABI,
    HostOperation,
    environment_function,
    validate_environment_bindings,
)
from rlm.errors import HarnessContractError, HostProtocolError
from rlm.executor import ActionHandler, IPythonExecutor
from rlm.json import json_compatibility_error
from rlm.prompts import default_harness_spec
from rlm.protocol import extract_text
from rlm.types import ExecutionResult, TextSubmission
from tests.fakes import ScriptedBackend, controller_code, request


def test_environment_abi_has_unique_names_and_operations() -> None:
    assert ENVIRONMENT_ABI.version == "2"
    assert len(ENVIRONMENT_ABI.functions) == len({item.name for item in ENVIRONMENT_ABI.functions})
    host_operations = [
        item.metadata.operation
        for item in ENVIRONMENT_ABI.functions
        if item.metadata.operation is not None
    ]
    assert len(host_operations) == len(set(host_operations))
    assert HostOperation.MODEL_COMPLETE in host_operations
    assert IPythonExecutor.abi_version == ENVIRONMENT_ABI.version
    assert all("self" not in item.render() for item in ENVIRONMENT_ABI.functions)
    assert json_compatibility_error(ENVIRONMENT_ABI.to_dict()) is None


def test_final_text_abi_requires_exact_text() -> None:
    final_text = next(item for item in ENVIRONMENT_ABI.functions if item.name == "FINAL_TEXT")

    assert str(final_text.signature) == "(text: str) -> NoneType"


def test_harness_fingerprint_includes_exact_environment_abi() -> None:
    spec = default_harness_spec()
    first = ENVIRONMENT_ABI.functions[0]
    changed_first = replace(
        first,
        metadata=replace(first.metadata, description=first.metadata.description + " changed"),
    )
    changed = replace(
        ENVIRONMENT_ABI,
        functions=(changed_first, *ENVIRONMENT_ABI.functions[1:]),
    )

    assert spec.abi_digest == ENVIRONMENT_ABI.digest()
    assert changed.digest() != ENVIRONMENT_ABI.digest()
    assert replace(spec, abi_digest=changed.digest()).fingerprint() != spec.fingerprint()


def test_run_rejects_stale_abi_digest_before_a_model_call() -> None:
    backend = ScriptedBackend([controller_code('FINAL_TEXT("unused")')])
    harness = replace(default_harness_spec(), abi_digest="0" * 64)

    with pytest.raises(HarnessContractError, match="ABI digest"):
        RLM(backend, config=RLMConfig(harness=harness)).run(request())

    assert backend.calls == []


def test_worker_binding_rejects_signature_drift_and_extra_callable() -> None:
    class TinyNamespace(Protocol):
        @staticmethod
        @environment_function(description="Return text.")
        def ask(prompt: str) -> str: ...

    def correct(prompt: str) -> str:
        return prompt

    def wrong() -> str:
        return "wrong"

    abi = EnvironmentABI.from_protocol(TinyNamespace, version="test", request_name="request")
    validate_environment_bindings(abi, {"ask": correct})

    with pytest.raises(HostProtocolError, match="signature"):
        validate_environment_bindings(abi, {"ask": wrong})
    with pytest.raises(HostProtocolError, match="undocumented"):
        validate_environment_bindings(abi, {"ask": correct, "extra": correct})


def test_host_operation_rejects_the_wrong_payload_shape() -> None:
    with pytest.raises(HostProtocolError, match="model_complete"):
        HostOperation.MODEL_COMPLETE.decode_payload({"requests": []})


def test_explicit_falsey_execution_factory_is_used() -> None:
    class FinalEnvironment:
        abi_version = ENVIRONMENT_ABI.version

        def __enter__(self) -> FinalEnvironment:
            return self

        def __exit__(self, *exc_info: object) -> None:
            return None

        def execute(
            self,
            code: str,
            *,
            action_handler: ActionHandler,
            timeout: float,
        ) -> ExecutionResult:
            return ExecutionResult(submission=TextSubmission("done"))

    class FalseyFactory:
        def __init__(self) -> None:
            self.called = False

        def __bool__(self) -> bool:
            return False

        def __call__(self, **kwargs: Any) -> FinalEnvironment:
            self.called = True
            return FinalEnvironment()

    factory = FalseyFactory()
    controller = ScriptedBackend([controller_code("FINAL_TEXT('ignored by fake')")])
    result = RLM(controller, execution_factory=factory).run(request())

    assert factory.called
    assert extract_text(result.response) == "done"
