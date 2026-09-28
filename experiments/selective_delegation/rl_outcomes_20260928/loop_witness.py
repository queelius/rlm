"""Small saved-call witnesses separating sampled payload and executed behavior."""

import json
from pathlib import Path

import analyze as a


def observed(directory, index):
    audit_path = directory / "NATIVE-AUDIT.json"
    audit = a.read(audit_path)
    call_path = directory / "calls" / f"t04-r0-flat-c{index:03d}.json"
    node_path = directory / "nodes/t04-r0-flat-n0.json"
    call = a.read(call_path, audit["receipt_sha256"][str(call_path)])
    node = a.read(node_path, audit["receipt_sha256"][str(node_path)])
    return (
        call,
        node["public_history"][index],
        {
            str(audit_path): a.sha(audit_path),
            str(call_path): a.sha(call_path),
            str(node_path): a.sha(node_path),
        },
    )


def witness():
    root = a.ROOT / "textcraft-rl-assist-20260928-001"
    paths = [root / "raw/readout-warm", root / "binder/readout-warm", root / "binder/readout-0001"]
    raw, raw_event, pins = observed(paths[0], 9)
    warm, warm_event, more = observed(paths[1], 9)
    pins.update(more)
    trained, trained_event, more = observed(paths[2], 9)
    pins.update(more)
    assert raw["request"]["input_token_ids"] == warm["request"]["input_token_ids"]
    assert raw["text"] == warm["text"]
    assert warm_event == trained_event
    early = dict(
        index=9,
        raw_requested=raw["text"],
        raw_executed=raw_event,
        warm_binder_executed=warm_event,
        raw_vs_binder_same_input_ids_and_sampled_text=True,
        trained_binder_requested=trained["text"],
        warm_vs_trained_binder_same_executed_action_and_feedback=True,
        warm_vs_trained_binder_same_input_ids=(
            warm["request"]["input_token_ids"] == trained["request"]["input_token_ids"]
        ),
    )
    warm, warm_event, more = observed(paths[1], 27)
    pins.update(more)
    trained, trained_event, more = observed(paths[2], 27)
    pins.update(more)
    states = [
        json.loads(c["request"]["prompt"][len(a.bridge.INSTRUCTION) :]) for c in (warm, trained)
    ]
    differing = [key for key in states[0] if states[0][key] != states[1][key]]
    assert differing == ["global_output_tokens_remaining"]
    later = dict(
        index=27,
        warm_executed=warm_event,
        trained_executed=trained_event,
        input_ids_equal=False,
        differing_public_state_fields=differing,
        warm_output_tokens_remaining=states[0][differing[0]],
        trained_output_tokens_remaining=states[1][differing[0]],
        history_identical=True,
    )
    return dict(
        task_id="textcraft_synth.train.429",
        repeat=0,
        early_binding_witness=early,
        later_executed_choice_witness=later,
        inputs_sha256=pins,
        source_sha256=a.sha(Path(__file__)),
        interpretation="The binder changes a stock/history trajectory at the same sampled action. "
        "Early trained-vs-warm payload changes need not change execution, but still change token "
        "budgets. The later useful target change is not an exact-input causal contrast. "
        "No parse relaxation, model call, native continuation or causal-mechanism proof.",
    )


if __name__ == "__main__":
    result = witness()
    a.save(a.OUTPUT / "LOOP-WITNESS.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "inputs_sha256"}, indent=2))
