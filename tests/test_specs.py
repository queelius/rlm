from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rlm import ExecutionConfig, RLMConfig
from rlm.abi import ENVIRONMENT_ABI
from rlm.json import json_compatibility_error
from rlm.prompts import default_harness_spec
from rlm.specs import (
    HarnessSpec,
    ObservationSpec,
    PromptDigests,
    PromptSpec,
    harness_spec_from_dict,
)


def test_harness_fingerprint_is_stable_and_content_addressed() -> None:
    original = default_harness_spec()
    same = replace(original)
    changed = replace(
        original,
        prompt=replace(original.prompt, policy=original.prompt.policy + "\nVerify once."),
    )

    assert same.fingerprint() == original.fingerprint()
    assert changed.fingerprint() != original.fingerprint()
    assert len(original.fingerprint()) == 64


def test_harness_spec_is_strict_json() -> None:
    spec = default_harness_spec()

    assert spec.to_dict()["schema_version"] == "1"
    assert spec.to_dict()["context"]["preserve_reasoning"] is True


def test_new_harness_specs_default_to_the_installed_abi_identity() -> None:
    spec = HarnessSpec(
        prompt=PromptSpec(name="test", version="1", protocol="protocol", policy="policy"),
        rendered_prompts=PromptDigests("0" * 64, "0" * 64),
    )

    assert spec.abi_version == ENVIRONMENT_ABI.version
    assert spec.abi_digest == ENVIRONMENT_ABI.digest()


def test_bootstrap_contract_round_trips_and_changes_harness_identity() -> None:
    original = default_harness_spec()
    bootstrap = getattr(original, "bootstrap", None)

    assert bootstrap is not None, "HarnessSpec must carry its controller bootstrap contract"
    changed = replace(original, bootstrap=replace(bootstrap, schema_version="3"))
    reconstructed = harness_spec_from_dict(original.to_dict())

    assert reconstructed == original
    assert reconstructed.to_dict()["bootstrap"] == {
        "type": "rlm.controller_bootstrap",
        "schema_version": "2",
        "content_in_message": False,
    }
    assert changed.fingerprint() != original.fingerprint()


def test_observation_contract_round_trips_and_changes_harness_identity() -> None:
    original = default_harness_spec()
    observation = getattr(original, "observation", None)

    assert observation is not None, "HarnessSpec must carry its controller observation contract"
    changed = replace(original, observation=replace(observation, schema_version="3"))
    reconstructed = harness_spec_from_dict(original.to_dict())

    assert reconstructed == original
    assert reconstructed.to_dict()["observation"] == {
        "type": "rlm.controller_observation",
        "schema_version": "2",
    }
    assert changed.fingerprint() != original.fingerprint()


def test_observation_contract_rejects_untyped_discriminators() -> None:
    with pytest.raises(TypeError, match="ObservationType"):
        ObservationSpec(type="rlm.controller_observation")  # type: ignore[arg-type]


def test_harness_prompt_digests_are_lowercase_sha256_values() -> None:
    digests = default_harness_spec().rendered_prompts

    assert len(digests.nonrecursive) == len(digests.recursive) == 64
    assert PromptDigests(digests.nonrecursive, digests.recursive) == digests


def test_harness_prompt_digests_reject_noncanonical_values() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        PromptDigests("A" * 64, "0" * 64)


def test_runtime_config_serializes_without_credentials_or_python_objects() -> None:
    config = RLMConfig(execution=ExecutionConfig(working_directory=Path("work")))
    value = config.to_dict()

    assert value["execution"]["working_directory"] == "work"
    assert json_compatibility_error(value) is None
