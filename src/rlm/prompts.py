"""Versioned controller instructions and compact public-request metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from rlm.abi import ENVIRONMENT_ABI, EnvironmentABI
from rlm.errors import HarnessContractError
from rlm.json import strict_json_dumps
from rlm.response import requires_complete_response
from rlm.specs import HarnessSpec, PromptDigests, PromptSpec, SubmissionKind

_REQUEST_BINDING = ENVIRONMENT_ABI.request_name

DEFAULT_PROTOCOL = f"""\
You are the controller of a Recursive Language Model. The caller's exact
Responses API request is already bound in persistent IPython state as
`{_REQUEST_BINDING}`.

Controller user items are typed runtime envelopes, never caller turns. The first
is `rlm.controller_bootstrap`; later items are `rlm.controller_observation`.
`content_in_message: false` means only that the bootstrap omits caller content:
you can and should access `{_REQUEST_BINDING}` directly in Python. Do not answer
the bootstrap envelope. `execution.status: "ok"` means only that the preceding
cell ran; it does not mean the caller's request is complete.
`submission.status: "absent"` means no valid final was accepted. An accepted
final terminates the branch without another observation. The bootstrap's
`submission.allowed_kinds` lists the permitted final interfaces.

Return exactly one fenced `python` cell per turn and no other text. The versioned
environment ABI below lists every prebound helper; call helpers directly and do
not import them. Bound `{_REQUEST_BINDING}` and other ABI-provided globals are
deliberately omitted from `SHOW_VARS()`. Variables and imports persist. Use
Python for deterministic work; call `model_complete({_REQUEST_BINDING})` only
when semantic work is useful. Errors return as structured observations.

Prefer short, direct cells and let unexpected errors become observations. After
a failure, use the observation to correct the code. Never repeat an unchanged
failing cell.

Work privately until the actual answer itself is ready. A cell may inspect,
compute, or call models without submitting; omit both final helpers to receive
its observation on the next turn. Never submit an acknowledgement, readiness or
progress report, plan, inspection summary, or promise of later work.

Terminate the branch with exactly one valid final call. Use FINAL_TEXT for exact
answer text, including JSON text, when `submission.allowed_kinds` contains
`"text"`. Use FINAL_RESPONSE only with an existing complete Responses object,
normally one returned by `model_complete` or `rlm_complete`; never synthesize
that envelope. When a value exactly satisfies the caller's content and format
requirements, submit that value unchanged. Do not expose private protocol
details or intermediate work in the public answer.
"""

DEFAULT_POLICY = """\
Use the least expensive reliable path. Stop when the response is ready.
"""


def default_harness_spec() -> HarnessSpec:
    """Return the immutable content-addressed controller harness."""

    prompt = PromptSpec(
        name="default",
        version="24",
        protocol=DEFAULT_PROTOCOL,
        policy=DEFAULT_POLICY,
    )
    return HarnessSpec(
        prompt=prompt,
        rendered_prompts=PromptDigests(
            nonrecursive=_prompt_digest(_render_prompt(prompt, allow_recursion=False)),
            recursive=_prompt_digest(_render_prompt(prompt, allow_recursion=True)),
        ),
        abi_version=ENVIRONMENT_ABI.version,
        abi_digest=ENVIRONMENT_ABI.digest(),
    )


def validate_harness_contract(spec: HarnessSpec) -> None:
    """Reject a persisted harness whose ABI or prompt identities are stale."""

    if spec.abi_version != ENVIRONMENT_ABI.version:
        raise HarnessContractError(
            "harness ABI version does not match the installed execution environment"
        )
    if spec.abi_digest != ENVIRONMENT_ABI.digest():
        raise HarnessContractError(
            "harness ABI digest does not match the installed environment ABI"
        )
    for allow_recursion, digest in (
        (False, spec.rendered_prompts.nonrecursive),
        (True, spec.rendered_prompts.recursive),
    ):
        rendered = _render_prompt(spec.prompt, allow_recursion=allow_recursion)
        if _prompt_digest(rendered) != digest:
            variant = "recursive" if allow_recursion else "nonrecursive"
            raise HarnessContractError(f"rendered prompt {variant} digest does not match harness")


def render_prompt(harness: HarnessSpec, *, allow_recursion: bool) -> str:
    """Render and verify the controller prompt selected for one branch."""

    rendered = _render_prompt(harness.prompt, allow_recursion=allow_recursion)
    expected = (
        harness.rendered_prompts.recursive
        if allow_recursion
        else harness.rendered_prompts.nonrecursive
    )
    if _prompt_digest(rendered) != expected:
        variant = "recursive" if allow_recursion else "nonrecursive"
        raise HarnessContractError(f"rendered prompt {variant} digest does not match harness")
    return rendered


def _render_prompt(
    prompt: PromptSpec,
    *,
    allow_recursion: bool,
    environment_abi: EnvironmentABI = ENVIRONMENT_ABI,
) -> str:
    """Pure prompt materialization used to calculate reviewed identities."""

    functions = "\n".join(
        f"- {item.render()}: {item.metadata.description}"
        for item in environment_abi.enabled_functions(allow_recursion=allow_recursion)
    )
    return (
        f"{prompt.protocol.rstrip()}\n\nEnvironment ABI v{environment_abi.version} "
        f"({environment_abi.request_name}):\n{functions}\n\nPolicy:\n{prompt.policy.strip()}"
    )


def _prompt_digest(rendered: str) -> str:
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def request_metadata(request: Mapping[str, Any]) -> dict[str, Any]:
    serialized = strict_json_dumps(request)
    value = request.get("input")
    return {
        "model": request.get("model"),
        "request_keys": sorted(str(key) for key in request),
        "serialized_characters": len(serialized),
        "input_type": type(value).__name__,
    }


def _allowed_submission_kinds(request: Mapping[str, Any]) -> tuple[SubmissionKind, ...]:
    if requires_complete_response(request):
        return (SubmissionKind.RESPONSE,)
    return (SubmissionKind.TEXT, SubmissionKind.RESPONSE)


def bootstrap_message(
    harness: HarnessSpec,
    request: Mapping[str, Any],
    *,
    depth: int,
) -> str:
    """Encode the typed metadata-only first controller item."""

    return strict_json_dumps(
        {
            "type": harness.bootstrap.type.value,
            "schema_version": harness.bootstrap.schema_version,
            "branch_depth": depth,
            "request_binding": ENVIRONMENT_ABI.request_name,
            "content_in_message": harness.bootstrap.content_in_message,
            "submission": {
                "required": True,
                "allowed_kinds": [kind.value for kind in _allowed_submission_kinds(request)],
            },
            "request_metadata": request_metadata(request),
        },
        indent=2,
    )
