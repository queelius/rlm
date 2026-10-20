"""Harness prelude: names defined before the first cell, listed in the prompt, hidden."""

from __future__ import annotations

from typing import Any

import pytest

from rlm.executor import IPythonExecutor
from rlm.prompts import _prompt_digest, _render_prompt, default_harness_spec, render_prompt
from rlm.specs import HarnessSpec, PreludeSpec, PromptDigests, harness_spec_from_dict

PRELUDE = PreludeSpec(
    source="def double(x):\n    return 2 * x\n\nLIBRARY_VERSION = 'test'\n",
    exports=(("double", "double(x) -> x times two"), ("LIBRARY_VERSION", "library version tag")),
)


def _handle(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("no host calls expected")


def _request() -> dict[str, Any]:
    return {"model": "public-model", "input": "hello"}


def _harness_with_prelude(prelude: PreludeSpec) -> HarnessSpec:
    base = default_harness_spec()
    return HarnessSpec(
        prompt=base.prompt,
        rendered_prompts=PromptDigests(
            nonrecursive=_prompt_digest(
                _render_prompt(base.prompt, allow_recursion=False, prelude=prelude)
            ),
            recursive=_prompt_digest(
                _render_prompt(base.prompt, allow_recursion=True, prelude=prelude)
            ),
        ),
        prelude=prelude,
    )


def test_prelude_names_are_callable_and_hidden_from_show_vars() -> None:
    with IPythonExecutor(
        request=_request(), allow_recursion=False, prelude_source=PRELUDE.source
    ) as executor:
        first = executor.execute(
            "print(double(21)); print(LIBRARY_VERSION); print('double' in SHOW_VARS())",
            action_handler=_handle,
            timeout=10,
        )
        second = executor.execute(
            "double = None\nmine = [1]\nprint(sorted(SHOW_VARS()))",
            action_handler=_handle,
            timeout=10,
        )
        third = executor.execute("print(double(2))", action_handler=_handle, timeout=10)
    assert first.stdout == "42\ntest\nFalse\n"
    assert second.stdout == "['mine']\n"
    assert third.stdout == "4\n", "prelude bindings are restored before every cell"


def test_failing_prelude_is_a_startup_error() -> None:
    with (
        pytest.raises(Exception, match="harness prelude failed"),
        IPythonExecutor(
            request=_request(), allow_recursion=False, prelude_source="raise ValueError('bad')"
        ),
    ):
        pass


def test_prelude_is_listed_in_the_prompt_and_changes_its_digest() -> None:
    base = default_harness_spec()
    with_prelude = _harness_with_prelude(PRELUDE)
    rendered = render_prompt(with_prelude, allow_recursion=False)
    assert "Harness prelude v1" in rendered
    assert "- double: double(x) -> x times two" in rendered
    assert PRELUDE.digest()[:16] in rendered
    assert with_prelude.rendered_prompts != base.rendered_prompts
    assert with_prelude.fingerprint() != base.fingerprint()


def test_prelude_round_trips_through_the_harness_snapshot() -> None:
    spec = _harness_with_prelude(PRELUDE)
    again = harness_spec_from_dict(spec.to_dict())
    assert again == spec
    assert again.prelude is not None and again.prelude.exports == PRELUDE.exports


def test_prelude_requires_identifier_exports() -> None:
    with pytest.raises(ValueError):
        PreludeSpec(source="x = 1", exports=(("not an identifier", "desc"),))
