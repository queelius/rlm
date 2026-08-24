"""Command-line entry point for the RLM HTTP proxy."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rlm.backend import OpenAIEndpoint
from rlm.config import ControllerConfig, ExecutionConfig, RLMConfig, RunLimits, TraceConfig
from rlm.engine import RLM
from rlm.json import StrictJSONError, strict_json_loads
from rlm.server import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm",
        description="Run a small request-preserving Recursive Language Model runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="serve an OpenAI-compatible Responses endpoint")
    serve.add_argument("--host", default="127.0.0.1", help="address to bind (default: %(default)s)")
    serve.add_argument("--port", type=int, default=8000, help="port to bind (default: %(default)s)")
    serve.add_argument(
        "--upstream-base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible upstream base URL",
    )
    serve.add_argument(
        "--upstream-api-key",
        default=None,
        help="upstream API key (default: OPENAI_API_KEY)",
    )
    serve.add_argument(
        "--upstream-timeout",
        type=_positive_float,
        default=300.0,
        metavar="SECONDS",
    )
    serve.add_argument("--controller-model", default=None)
    serve.add_argument(
        "--controller-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable controller option; VALUE must be strict JSON (for example think=false)",
    )
    serve.add_argument("--max-turns", type=_positive_int, default=12)
    serve.add_argument("--max-model-calls", type=_positive_int, default=64)
    serve.add_argument("--max-subcalls", type=_nonnegative_int, default=48)
    serve.add_argument("--max-parallel-model-calls", type=_positive_int, default=8)
    serve.add_argument("--max-depth", type=_nonnegative_int, default=0)
    serve.add_argument("--max-total-tokens", type=_positive_int, default=None)
    serve.add_argument("--deadline-seconds", type=_positive_float, default=600.0)
    serve.add_argument("--execution-timeout", type=_positive_float, default=120.0)
    serve.add_argument("--max-execution-output-chars", type=_positive_int, default=1_000_000)
    serve.add_argument("--max-observation-chars", type=_positive_int, default=6_000)
    serve.add_argument("--working-directory", type=Path, default=None)
    serve.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help="enable canonical traces in DIRECTORY",
    )
    serve.add_argument("--trace-markdown", action="store_true", help="also write a Markdown trace")
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--no-access-log", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        try:
            return _serve(args)
        except ValueError as exc:
            parser.error(str(exc))
    parser.error(f"unknown command: {args.command!r}")


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError("the serve command requires uvicorn; install rlm[server]") from exc

    model = _build_model(args)
    app = create_app(model, close_on_shutdown=True)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=not args.no_access_log,
    )
    return 0


def _build_model(args: argparse.Namespace) -> RLM:
    controller_options = _parse_options(args.controller_option)
    tracing = TraceConfig(
        enabled=args.trace_dir is not None,
        directory=args.trace_dir or "runs",
        markdown=args.trace_markdown,
    )
    config = RLMConfig(
        controller=ControllerConfig(model=args.controller_model, options=controller_options),
        limits=RunLimits(
            max_turns=args.max_turns,
            max_model_calls=args.max_model_calls,
            max_subcalls=args.max_subcalls,
            max_parallel_model_calls=args.max_parallel_model_calls,
            max_depth=args.max_depth,
            max_total_tokens=args.max_total_tokens,
            deadline_seconds=args.deadline_seconds,
            execution_timeout_seconds=args.execution_timeout,
            max_execution_output_chars=args.max_execution_output_chars,
            max_observation_chars=args.max_observation_chars,
        ),
        execution=ExecutionConfig(working_directory=args.working_directory),
        tracing=tracing,
    )
    upstream = OpenAIEndpoint(
        base_url=args.upstream_base_url,
        api_key=args.upstream_api_key,
        timeout=args.upstream_timeout,
    )
    try:
        return RLM(upstream, config=config)
    except BaseException:
        upstream.close()
        raise


def _parse_options(values: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for value in values:
        key, separator, raw = value.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"option must have the form KEY=VALUE: {value!r}")
        try:
            parsed: Any = strict_json_loads(raw)
        except StrictJSONError as exc:
            raise ValueError(f"option VALUE must be valid JSON: {value!r}") from exc
        options[key] = parsed
    return options


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
