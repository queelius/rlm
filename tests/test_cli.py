from __future__ import annotations

import pytest

from rlm.backend import OpenAIEndpoint
from rlm.cli import _build_model, build_parser
from rlm.codex_backend import CodexAgentBackend


def parse_serve(*arguments: str):
    return build_parser().parse_args(("serve", *arguments))


def test_serve_parser_defaults_to_openai_compatible_controller() -> None:
    args = parse_serve()

    assert args.controller_backend == "openai"
    assert args.codex_reasoning_effort is None
    assert args.action_mode == "tool"


def test_build_model_selects_codex_only_for_private_controller() -> None:
    args = parse_serve(
        "--upstream-base-url",
        "http://192.168.0.204:11434/v1",
        "--upstream-api-key",
        "ollama",
        "--controller-backend",
        "codex",
        "--controller-model",
        "gpt-5.6-sol",
        "--codex-reasoning-effort",
        "max",
        "--action-mode",
        "code",
        "--worker-model",
        "qwen3.5:latest",
    )

    model = _build_model(args)
    try:
        assert isinstance(model.backend, OpenAIEndpoint)
        assert model.worker_backend is model.backend
        assert isinstance(model.controller_backend, CodexAgentBackend)
        assert model.controller_backend is not model.backend
        assert model.controller_backend.model == "gpt-5.6-sol"
        assert model.controller_backend.reasoning_effort == "max"
        assert model.config.worker_model == "qwen3.5:latest"
        assert model.config.action_mode == "code"
    finally:
        model.close()


def test_build_model_defaults_all_roles_to_openai_compatible_upstream() -> None:
    model = _build_model(parse_serve("--controller-model", "qwen3.5:latest"))
    try:
        assert model.controller_backend is model.backend
        assert model.worker_backend is model.backend
    finally:
        model.close()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--controller-backend", "codex", "--action-mode", "code"), "--controller-model"),
        (
            ("--controller-backend", "codex", "--controller-model", "gpt-5.6-sol"),
            "--action-mode code",
        ),
        (
            ("--codex-reasoning-effort", "max"),
            "--controller-backend codex",
        ),
        (
            (
                "--controller-backend",
                "codex",
                "--controller-model",
                "gpt-5.6-sol",
                "--action-mode",
                "code",
                "--controller-option",
                "temperature=0",
            ),
            "--controller-option",
        ),
    ],
)
def test_build_model_rejects_invalid_controller_combinations(
    arguments: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_model(parse_serve(*arguments))
