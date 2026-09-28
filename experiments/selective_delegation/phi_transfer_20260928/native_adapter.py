"""Exact, fail-closed native Phi stop-token binding; original native actions unchanged."""

import hashlib
import importlib.util
import inspect
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import acquire
import qualify

LIBRARY = Path(__file__).resolve().parent.parent
BINDER = acquire.ROOT / "source-textcraft-recipe-binder-003"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rewrite(function, namespace, substitutions):
    """Transform only counted, pinned local-source expressions; never model/remote code."""
    original = textwrap.dedent(inspect.getsource(function))
    changed = original
    for old, new in substitutions:
        if changed.count(old) != 1:
            raise ValueError("native source seam changed: " + old)
        changed = changed.replace(old, new)
    compiled = {}
    exec(compile(changed, __file__, "exec"), namespace, compiled)
    return compiled[function.__name__], dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        transformed_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        substitutions=substitutions,
    )


@lru_cache(maxsize=2)
def implementation(assistance):
    if assistance not in ("raw", "binder"):
        raise ValueError("unknown observed-recipe assistance")
    source = LIBRARY if assistance == "raw" else BINDER
    collector = load(source / "eval_textcraft.py", "phi_native_collector_" + assistance)
    auditor = load(source / "analyze_textcraft.py", "phi_native_auditor_" + assistance)
    collector.BASE = acquire.MODEL
    auditor.collector, auditor.bridge = collector, collector.bridge
    collector.PHI_STOP_IDS = auditor.PHI_STOP_IDS = qualify.STOP_IDS
    call, call_binding = rewrite(
        collector.NativeClient.call,
        vars(collector),
        [
            (
                '"context_limit": 8192,',
                '"context_limit": 8192,\n        "generation_stop_ids": PHI_STOP_IDS,',
            ),
            ("eos_token_id=self.tokenizer.eos_token_id,", "eos_token_id=PHI_STOP_IDS,"),
            ("emitted[-1] == self.tokenizer.eos_token_id", "emitted[-1] in PHI_STOP_IDS"),
        ],
    )
    collector.NativeClient.call = call
    original_audit, audit_binding = rewrite(
        auditor.audit_call,
        vars(auditor),
        [
            ("emitted[-1] == tokenizer.eos_token_id", "emitted[-1] in PHI_STOP_IDS"),
        ],
    )

    def audit_call(call, *args):
        if call["request"].get("generation_stop_ids") != qualify.STOP_IDS:
            raise ValueError("native Phi stop-token request differs")
        return original_audit(call, *args)

    auditor.audit_call = audit_call
    binding = dict(
        schema="phi-native-client-stop-binding-20260928-v1",
        assistance=assistance,
        stop_ids=qualify.STOP_IDS,
        native_client=call_binding,
        saved_call_auditor=audit_binding,
        source_sha256={
            str(p.resolve()): acquire.sha(p)
            for p in (
                Path(__file__),
                Path(collector.__file__),
                Path(auditor.__file__),
                Path(collector.bridge.__file__),
            )
        },
        semantics="Only model path and native stop IDs change. Phi stop IDs are in every "
        "saved request, used by GenerationConfig and checked by saved-response audit. "
        "Native action parsing/execution/scoring, request seeds, all budgets and error accounting "
        "remain unchanged. This transforms inspected local source, not remote model code.",
    )
    return collector, auditor, binding
