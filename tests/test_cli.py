from __future__ import annotations

import argparse

import pytest

from rlm import ControllerConfig
from rlm.cli import _build_model, _parse_options, _positive_float, build_parser


def parse_serve(*arguments: str):
    return build_parser().parse_args(("serve", *arguments))


def test_serve_parser_defaults_to_the_openai_compatible_upstream() -> None:
    args = parse_serve()

    assert args.controller_model is None
    assert not hasattr(args, "worker_model")
    assert not hasattr(args, "action_mode")
    assert args.max_parallel_model_calls == 8
    assert args.max_total_tokens is None
    assert args.trace_dir is None
    assert args.trace_markdown is False
    assert not hasattr(args, "max_parallel_subcalls")
    assert not hasattr(args, "debug_dir")


def test_build_model_uses_one_backend_by_default() -> None:
    model = _build_model(parse_serve("--controller-model", "qwen3.5:latest"))
    try:
        assert model.controller_backend is model.backend
    finally:
        model.close()


def test_controller_option_requires_valid_json_value() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        _parse_options(["reasoning_effort=high"])

    assert _parse_options(['reasoning_effort="high"', "temperature=0", "think=false"]) == {
        "reasoning_effort": "high",
        "temperature": 0,
        "think": False,
    }

    with pytest.raises(ValueError, match="valid JSON"):
        _parse_options(["temperature=NaN"])


def test_background_controller_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="background"):
        ControllerConfig(options={"background": True})


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_positive_float_rejects_non_finite_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float(value)
