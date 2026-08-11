# Repository Guidelines

## Project Structure & Module Organization

Runtime code lives in `src/rlm/`. Keep public interfaces in `__init__.py`, entry points in `cli.py` and `server.py`, and orchestration in focused modules such as `engine.py`, `executor.py`, and `backend.py`. Tests in `tests/` generally mirror source modules; shared test doubles belong in `tests/fakes.py`. Architectural contracts and research boundaries are documented in `DESIGN.md` and `RESEARCH.md`. Generated debug output belongs under the ignored `runs/` directory.

## Build, Test, and Development Commands

- `uv sync --extra dev` creates or updates the environment with test, lint, server, and OpenAI dependencies.
- `uv run pytest` runs the full test suite using the quiet settings in `pyproject.toml`.
- `uv run pytest tests/test_engine.py -k final_text` runs a focused subset while iterating.
- `uv run ruff check .` checks imports, Python errors, upgrades, and configured style rules.
- `uv run ruff format --check .` verifies formatting; omit `--check` to apply it.
- `uv build` creates source and wheel distributions through Hatchling.
- `uv run rlm serve --help` lists options for running the local proxy.

## Coding Style & Naming Conventions

Use four-space indentation, Python 3.10-compatible syntax, and type annotations for public APIs and nontrivial internals. Ruff enforces a 100-character line limit and the `E`, `F`, `I`, `UP`, `B`, and `SIM` rule sets. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and leading underscores for private implementation details. Preserve complete OpenAI-compatible request objects rather than silently normalizing unknown fields.

## Testing Guidelines

Pytest discovers `tests/test_*.py`. Name tests after observable behavior, for example `test_streaming_request_is_rejected`. Prefer deterministic fake backends over live network calls, and cover both `chat.completions` and `responses` when behavior is API-family dependent. Add regression tests with every bug fix. No numeric coverage threshold is configured; prioritize protocol, limit, error, and cleanup paths.

## Commit & Pull Request Guidelines

This repository currently has no commit history from which to infer a convention. Use short, imperative subjects such as `Reject invalid response payloads`, and keep each commit scoped to one logical change. Pull requests should explain behavior and design impact, list verification commands, link relevant issues, and update `README.md` or `DESIGN.md` when contracts change. Include trace excerpts only when useful, and redact requests, credentials, and model output.

## Security Notes

The IPython executor is process-isolated but not sandboxed. Never run untrusted generated code, commit API keys, or check data-sensitive debug traces into source control.
