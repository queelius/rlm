"""Fixed terminal065/extra-SFT endpoints on the unchanged queue007 fresh16 interface."""

import argparse
import copy
import importlib.metadata
import json
import sys
from pathlib import Path

import eval_textcraft as c
import psutil
import train_textcraft_matched_sft as control

TEMPLATE = c.ROOT / "textcraft-fresh-public-001/PLAN.json"
TEMPLATE_SHA = "7296007858d36d651e3ce6a7c5836f10351c3e25bff4c2b28896a96253f784b9"
ORDER = c.ROOT / "TEXTCRAFT-EXTRA-SFT-ORDER-001.json"
ORDER_SHA = "f4488f75ce7c316b3e429590d1b5490cc0472ca6c4895df615caa201281fb58b"


def template_inputs():
    if c.inputs.sha(TEMPLATE) != TEMPLATE_SHA:
        raise ValueError("accepted queue007 fresh-public PLAN changed")
    plan = control.read(TEMPLATE)
    prepared = Path(plan["prepared"])
    for filename, key in [("MANIFEST.json", "manifest_sha256"), ("tasks.jsonl", "tasks_sha256")]:
        if c.inputs.sha(prepared / filename) != plan[key]:
            raise ValueError("frozen fresh16 input changed")
    for module in (c, c.bridge, c.inputs, c.probe, c.probe.runtime, c.probe.campaign):
        path = Path(module.__file__).resolve()
        digests = {
            digest for name, digest in plan["source_sha256"].items() if Path(name).name == path.name
        }
        if c.inputs.sha(path) not in digests:
            raise ValueError("collector dependency differs from accepted007: " + str(path))
    tasks = list(map(json.loads, (prepared / "tasks.jsonl").read_text().splitlines()))
    if len(tasks) != 16 or len(plan["jobs"]) != 32:
        raise ValueError("fixed16parents32slots required")
    return plan, tasks


def validate_endpoint(kind, terminal, state, commit, steps):
    if (
        kind not in ("rl", "matched_sft")
        or not 1 <= steps <= 2
        or terminal.get("complete") is not True
        or terminal.get("failure")
        or terminal.get("stopped")
        or terminal.get("endpoint_usable") is not True
        or terminal.get("actual_optimizer_steps") != steps
        or state.get("step") != steps
        or commit.get("step") != steps
    ):
        raise ValueError("only exact normal-completion final positive-step endpoint accepted")
    if kind == "rl" and terminal.get("committed_optimizer_steps") != steps:
        raise ValueError("RL actual/committed dose differs")


def released_terminal(output):
    owners = list(output.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("single completed training owner required")
    owner = control.read(owners[0])
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("training owner still active")
    except psutil.NoSuchProcess:
        pass
    path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    return control.read(path), {str(p): c.inputs.sha(p) for p in (owners[0], path)}


def endpoint(kind, training_output, *, stopped_amendment=None):
    training_output = training_output.resolve()
    training_path = training_output / "PLAN.json"
    training = control.read(training_path)
    rl_output = training_output if kind == "rl" else Path(training["rl_output"]).resolve()
    if stopped_amendment:
        import textcraft_stopped_amendment as amendment

        rl_plan, budgets, rl_pins = amendment.qualify(stopped_amendment, rl_output)
    else:
        rl_plan, budgets, rl_pins = control.rl_dose(rl_output)
    if not budgets:
        raise ValueError("zero RL updates; no duplicated warm-policy readout")
    steps = len(budgets)
    terminal, pins = released_terminal(training_output)
    adapter = Path(terminal["endpoint"]).resolve()
    state, commit = control.read(adapter / "STATE.json"), control.read(adapter / "COMMIT.json")
    if stopped_amendment and kind == "rl":
        amendment.validate_stopped(
            terminal, control.read(rl_output / "SUMMARY.json"), state, commit
        )
    else:
        validate_endpoint(kind, terminal, state, commit, steps)
    if (
        stopped_amendment
        and kind == "matched_sft"
        and training.get("stopped_run_amendment")
        != dict(path=str(stopped_amendment.resolve()), sha256=c.inputs.sha(stopped_amendment))
    ):
        raise ValueError("control not bound to the exact stopped-run amendment")
    if kind == "rl":
        expected = (
            training_output
            / "boundaries"
            / f"sample-{terminal['committed_sampled_batches']:04d}"
            / f"checkpoint-{steps:04d}"
        )
    elif kind == "matched_sft":
        if (
            training.get("schema") != "textcraft-token-update-matched-extra-sft-v1"
            or training.get("warm") != rl_plan["warm"]
            or training.get("rl_receipt_sha256") != rl_pins
            or training.get("frozen_sha256") != ORDER_SHA
            or c.inputs.sha(ORDER) != ORDER_SHA
            or training.get("planned_updates") != steps
        ):
            raise ValueError("matched control ancestry/dose differs")
        frozen, examples = control.teacher_inputs()
        schedule = control.row_schedule(examples, frozen["order"], budgets)
        if training["schedule"] != schedule or state.get("planned_updates") != steps:
            raise ValueError("matched control schedule differs from actual RL credit")
        update_paths = sorted((training_output / "updates").glob("*.json"))
        if len(update_paths) != steps:
            raise ValueError("all actual control optimizer receipts required")
        for index, path in enumerate(update_paths, 1):
            row = control.read(path)
            if row["step"] != index or any(row.get(k) != v for k, v in schedule[index - 1].items()):
                raise ValueError("control actual update dose differs")
            pins[str(path)] = c.inputs.sha(path)
        expected = training_output / f"checkpoint-{steps:04d}"
    else:
        raise ValueError("unsupported endpoint kind")
    if adapter != expected.resolve() or state["plan_sha256"] != c.inputs.sha(training_path):
        raise ValueError("selected or foreign checkpoint rejected")
    required = {
        "STATE.json",
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "rng.pt",
    }
    if not required <= set(commit["files"]):
        raise ValueError("full committed checkpoint required")
    # Small metadata + final adapter once; no model weights/optimizer ancestry rehash.
    for name in ("STATE.json", "adapter_config.json", "adapter_model.safetensors"):
        if c.inputs.sha(adapter / name) != commit["files"][name]:
            raise ValueError("final committed adapter content changed")
    config = control.read(adapter / "adapter_config.json")
    if (
        config["r"],
        config["lora_alpha"],
        config["lora_dropout"],
        config["base_model_name_or_path"],
    ) != (8, 16, 0, str(c.BASE)):
        raise ValueError("endpoint base/LoRA configuration differs")
    for path in (training_path, adapter / "COMMIT.json", adapter / "STATE.json"):
        pins[str(path)] = c.inputs.sha(path)
    result = dict(
        path=str(adapter),
        sha256=commit["files"]["adapter_model.safetensors"],
        commit_sha256=c.inputs.sha(adapter / "COMMIT.json"),
        state=state,
        training_plan_sha256=c.inputs.sha(training_path),
        endpoint_kind=kind,
        actual_optimizer_steps=steps,
        credited_rl_token_budgets=budgets,
        training_receipt_sha256=pins,
        rl_receipt_sha256=rl_pins,
        public056_warm_sha256=control.WARM_SHA,
    )
    if stopped_amendment:
        result["stopped_run_amendment"] = dict(
            path=str(stopped_amendment.resolve()), sha256=c.inputs.sha(stopped_amendment)
        )
    return result


def bound_plan(template, kind, binding, training_output):
    plan = copy.deepcopy(template)
    condition = "fresh_" + kind + "_terminal"
    plan.pop("teacher", None)
    plan.update(
        schema="textcraft-fresh16-final-training-endpoint-v1",
        endpoint_kind=kind,
        training_output=str(training_output),
        adapter=binding,
        fixed_adapter=binding["path"],
        training_plan_sha256=binding["training_plan_sha256"],
        jobs=[dict(job, condition=condition) for job in template["jobs"]],
        conditions=[condition],
        reference007_plan_sha256=TEMPLATE_SHA,
        endpoint_selection="Normal terminal endpoint only; no best checkpoint selection",
        prompt_difference="Exact queue007 original flat prompt/interface/sampling/budgets; "
        "only fixed final trained adapter changes.",
        caveat=template["caveat"] + " Reused prospective fresh panel for a later fixed "
        "training comparison, not a new unseen panel. No outcome-driven selection.",
    )
    plan["environment"] = dict(
        python=sys.version,
        executable=sys.executable,
        **{n: importlib.metadata.version(n) for n in ("torch", "transformers")},
    )
    for module in (
        c,
        c.bridge,
        c.inputs,
        c.probe,
        control,
        control.recipe,
        control.checkpoint,
        control.credit_math,
    ):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = c.inputs.sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = c.inputs.sha(Path(__file__))
    return plan


def prepare(args):
    template, tasks = template_inputs()
    if args.validate_inputs_only:
        return template, tasks, None
    amendment = getattr(args, "stopped_amendment", None)
    binding = endpoint(args.kind, args.training_output, stopped_amendment=amendment)
    plan = bound_plan(template, args.kind, binding, args.training_output.resolve())
    if amendment:
        import textcraft_stopped_amendment

        plan["stopped_run_amendment"] = binding["stopped_run_amendment"]
        plan["endpoint_selection"] = (
            "Explicit stopped-run one-step amendment; only committed nonzero checkpoint, "
            "not outcome-selected; failed RL002 remains unusable as a normal endpoint."
        )
        condition = "fresh_" + args.kind + "_stopped_step1"
        plan["conditions"] = [condition]
        plan["jobs"] = [dict(job, condition=condition) for job in plan["jobs"]]
        module = Path(textcraft_stopped_amendment.__file__)
        plan["source_sha256"][str(module)] = c.inputs.sha(module)
    path = args.output / "PLAN.json"
    if path.exists():
        if control.read(path) != plan:
            raise ValueError("immutable final endpoint readout PLAN changed")
    else:
        c.save(path, plan)
    return plan, tasks, binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("rl", "matched_sft"), required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-inputs-only", action="store_true")
    parser.add_argument("--stopped-amendment", type=Path)
    args = parser.parse_args()
    plan, tasks, binding = prepare(args)
    if args.validate_inputs_only:
        print(
            json.dumps(
                dict(
                    endpoint_pending=True,
                    GPU_loaded=False,
                    planned_episodes=32,
                    parent_tasks=16,
                    template_sha256=TEMPLATE_SHA,
                )
            )
        )
    else:
        c.run(args, prepared_run=(plan, tasks), adapter=binding)
