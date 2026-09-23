"""Four matched public-observation memory modes at three fixed actor endpoints."""

import argparse
import copy
import json
from functools import cache
from itertools import combinations
from pathlib import Path

import analyze_textcraft as audit
import analyze_textcraft_endpoint_fresh as comparison
import eval_textcraft_endpoint_fresh as fixed
import eval_textcraft_fp16_endpoint as fp16
import eval_textcraft_fresh as fresh
import textcraft_recipe_memory as memory
import textcraft_teacher_seed as seed2

c, read, sha = fixed.c, audit.read, fixed.c.inputs.sha
POLICIES = ("public1", "public2291", "rl_cp2")


def output(policy, mode):
    return c.ROOT / f"textcraft-memory-{policy}-{mode}-001"


def report_path(policy, mode):
    return c.ROOT / f"analysis-textcraft-memory-{policy}-{mode}-001.json"


@cache
def endpoint(policy):
    if policy == "public1":
        return fresh.endpoint("public")
    if policy == "public2291":
        return seed2.endpoint("public")
    if policy == "rl_cp2":
        return fp16.endpoint()
    raise ValueError("unknown fixed policy")


def build_plan(template, policy, mode, binding):
    if policy not in POLICIES or mode not in memory.MODES:
        raise ValueError("unknown fixed memory/policy condition")
    plan = fixed.bound_plan(
        template, f"memory_{policy}_{mode}", binding, Path(binding["path"]).parent
    )
    plan.update(
        schema="textcraft-public-recipe-memory-v1",
        memory_mode=mode,
        memory_policy=policy,
        budget_seconds=7200,
        memory_contract="All modes add identical memory_description. Notebook copies only static "
        "facts from actual prior get_info responses, never stale inventory or hidden recipes. "
        "recent4 keeps last four complete history entries; current inventory/goal stay visible.",
        prompt_difference="Only public notebook inclusion and full versus recent4 history; "
        "native actions, rewards, initial state and per-episode budgets unchanged.",
        initial_token_audit=None,
        caveat="Exposed fixed16 roots in one shared recipe world. Four new matched2hour arms "
        "per actor; historical1hour baseline supplemental only. Missing outcomes remain unknown. "
        "Ledger changes tokens/context and recent4 removes observations; not free computation.",
    )
    for module in (
        c,
        c.bridge,
        c.inputs,
        c.probe,
        c.probe.runtime,
        c.probe.campaign,
        fixed,
        fresh,
        fp16,
        seed2,
        memory,
        audit,
        comparison,
    ):
        path = Path(module.__file__).resolve()
        plan["source_sha256"][str(path)] = sha(path)
    plan["source_sha256"][str(Path(__file__).resolve())] = sha(Path(__file__))
    return plan


@cache
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(c.BASE, local_files_only=True, trust_remote_code=False)


def prepare(policy, mode):
    template, tasks = fixed.template_inputs()
    binding = endpoint(policy)
    plan = build_plan(template, policy, mode, binding)
    lengths = []
    world = c.bridge.load_world()
    with memory.installed(mode):
        for task in tasks:
            frame = c.bridge.Frame(
                world,
                task["misc"]["initial_inventory"],
                task["misc"]["target_items"],
                c.bridge.Budget(),
                0,
            )
            prompt = c.render_prompt(frame, [], goal=task["goal"], profile="original")
            ids = tokenizer().apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            lengths.append(len(ids))
    if max(lengths) + 256 > 8192:
        raise ValueError("initial prompt exceeds unchanged context cap")
    plan["initial_token_audit"] = dict(
        prompt_tokens=lengths, max_prompt_plus_cap=max(lengths) + 256
    )
    path = output(policy, mode) / "PLAN.json"
    if path.exists():
        if read(path) != plan:
            raise ValueError("immutable memory PLAN changed")
    else:
        c.save(path, plan)
    return plan, tasks, binding


def collect(policy, mode, prepare_only):
    plan, tasks, binding = prepare(policy, mode)
    with memory.installed(mode):
        c.run(
            argparse.Namespace(output=output(policy, mode), prepare_only=prepare_only),
            prepared_run=(plan, tasks),
            adapter=binding,
        )


def audit_arm(policy, mode):
    directory, report = output(policy, mode), report_path(policy, mode)
    if report.exists() or report.with_suffix(".md").exists():
        raise FileExistsError("immutable audit report exists")
    plan, _, _ = prepare(policy, mode)
    with memory.installed(mode):
        result = audit.analyze(directory, tokenizer(), expected_task_count=16)
    result.pop("paired", None)
    result.pop("depth_strata", None)
    result.update(
        memory_mode=mode,
        memory_policy=policy,
        caveat=plan["caveat"],
        memory_source_sha256=sha(Path(memory.__file__)),
        wrapper_sha256=sha(Path(__file__)),
    )
    c.save(report, result)
    with report.with_suffix(".md").open("x") as stream:
        stream.write(
            f"# Public recipe memory: {policy}, {mode}\n\n"
            + plan["caveat"]
            + "\n\n```json\n"
            + json.dumps(
                dict(
                    groups=result["groups"],
                    physical_cost=result["physical_cost"],
                    owner_wall_seconds=result["owner_wall_seconds"],
                ),
                indent=2,
            )
            + "\n```\n"
        )


def compare_policy(policy):
    modes = memory.MODES
    plans = [read(output(policy, mode) / "PLAN.json") for mode in modes]
    comparison.match_slots(plans)
    if any(
        p["adapter"] != endpoint(policy)
        or p["memory_policy"] != policy
        or p["memory_mode"] != mode
        or p["budget_seconds"] != 7200
        for p, mode in zip(plans, modes, strict=True)
    ):
        raise ValueError("fixed actor/memory/caps differ")
    reports, rows, pins = {}, {}, {}
    for mode in modes:
        path = report_path(policy, mode)
        reports[mode] = read(path)
        pins[str(path)] = sha(path)
        plan_path = output(policy, mode) / "PLAN.json"
        if reports[mode]["sha256"].get(str(plan_path)) != sha(plan_path):
            raise ValueError("audit PLAN differs")
        rows[mode] = {}
        for path in (output(policy, mode) / "episodes").glob("*.json"):
            if reports[mode]["sha256"].get(str(path)) != sha(path):
                raise ValueError("episode differs from native audit")
            row = read(path)
            rows[mode][row["task_id"], row["repeat"]] = row
    jobs = plans[0]["jobs"]
    pairs = {
        f"{b}_minus_{a}": comparison.profiles.compare(jobs, rows[a], rows[b])
        for a, b in combinations(modes, 2)
    }
    # Derived per-slot ledger benefits retain unknowns and paired task clustering.
    benefits = []
    for no_ledger, ledger in (("history_full", "ledger_full"), ("recent4", "ledger_recent4")):
        effect = comparison.profiles.compare(jobs, rows[no_ledger], rows[ledger])
        benefits.append(
            {
                (r["task_id"], r["repeat"]): dict(observed=r["known"], native_score=r["delta"])
                for r in effect["rows"]
            }
        )
    interaction = comparison.profiles.compare(jobs, *benefits)
    result = dict(
        policy=policy,
        planned_per_arm=32,
        parent_count=16,
        repeats=2,
        paired_comparisons=pairs,
        ledger_benefit_recent_minus_full=interaction,
        arms={
            mode: dict(
                groups=reports[mode]["groups"],
                cost=reports[mode]["physical_cost"],
                wall_seconds=reports[mode]["owner_wall_seconds"],
            )
            for mode in modes
        },
        source_sha256=pins,
        wrapper_sha256=sha(Path(__file__)),
        caveat=plans[0]["caveat"],
    )
    path = c.ROOT / f"analysis-textcraft-memory-{policy}-factorial-001.json"
    c.save(path, result)
    concise = copy.deepcopy(result)
    for effect in list(concise["paired_comparisons"].values()) + [
        concise["ledger_benefit_recent_minus_full"]
    ]:
        effect.pop("rows", None)
    with path.with_suffix(".md").open("x") as stream:
        stream.write(
            f"# Public recipe-memory factorial: {policy}\n\n"
            + result["caveat"]
            + "\n\n```json\n"
            + json.dumps(concise, indent=2)
            + "\n```\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--mode", choices=memory.MODES)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare_policy(args.policy)
    elif not args.mode:
        parser.error("--mode required for collection/audit")
    elif args.audit:
        audit_arm(args.policy, args.mode)
    else:
        collect(args.policy, args.mode, args.prepare_only)
