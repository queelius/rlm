"""Independent native-receipt and trusted-action replay of the fixed TextCraft screen."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

import eval_textcraft as collector

bridge = collector.bridge
SEED = 2026092206


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def paired(jobs, rows, draws=20000):
    parents = sorted({j["task_id"] for j in jobs})
    differences, wins, losses, ties, unknown = {}, 0, 0, 0, 0
    for parent in parents:
        values = []
        selected = [j for j in jobs if j["task_id"] == parent]
        for repeat in sorted({j["repeat"] for j in selected}):
            pair = {
                j["policy"]: rows.get(j["episode_id"]) for j in selected if j["repeat"] == repeat
            }
            if set(pair) != {"flat", "recursive"} or not all(
                row and row["observed"] for row in pair.values()
            ):
                unknown += 1
                continue
            delta = pair["recursive"]["native_score"] - pair["flat"]["native_score"]
            values.append(delta)
            wins += delta > 0
            losses += delta < 0
            ties += delta == 0
        if len(values) == len({j["repeat"] for j in selected}):
            differences[parent] = mean(values)
    estimate = interval = None
    if len(differences) == len(parents) and parents:
        values = list(differences.values())
        estimate = mean(values)
        rng = random.Random(SEED)
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
        interval = [samples[int((draws - 1) * q)] for q in (0.025, 0.975)]
    return dict(
        parents=len(parents),
        complete_parents=len(differences),
        paired_attempts=wins + losses + ties,
        unknown_pairs=unknown,
        wins=wins,
        losses=losses,
        ties=ties,
        difference=estimate,
        ci95=interval,
        parent_differences=differences,
    )


def audit_call(call, spec, plan, plan_sha, tokenizer):
    req = call["request"]
    require(1 <= spec["cap"] <= 256, "native output cap mismatch")
    require(all(req.get(k) == v for k, v in spec.items()), "native public request mismatch")
    require(
        call["request_digest"] == collector.probe.runtime.digest(req)
        and call["plan_sha256"] == plan_sha,
        "request/PLAN digest mismatch",
    )
    require(
        req["model"] == plan["model"]
        and req["model_manifest_sha256"] == plan["model_manifest_sha256"]
        and req["adapter_enabled"] == bool(plan.get("adapter"))
        and req["adapter_sha256"] == (plan["adapter"]["sha256"] if plan.get("adapter") else None),
        "frozen base model identity mismatch",
    )
    if plan.get("adapter"):
        require(
            req.get("adapter_path") == plan["adapter"]["path"]
            and req.get("adapter_commit_sha256") == plan["adapter"]["commit_sha256"],
            "adapter checkpoint receipt mismatch",
        )
    require(
        req["context_limit"] == 8192
        and req["truncation"] is False
        and req["sampling"]
        == dict(
            temperature=0.5,
            top_p=1.0,
            top_k=0,
            max_new_tokens=spec["cap"],
            do_sample=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            max_time=90.0,
        ),
        "sampling/context mismatch",
    )
    require(
        all(
            call[k] == spec[k]
            for k in ("call_id", "role", "node_id", "parent_node_id", "depth", "episode_id")
        )
        and call["condition"] == spec.get("condition", spec["policy"]),
        "call role/tree identity mismatch",
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": req["prompt"]}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    require(
        ids == req["input_token_ids"] == call["input_token_ids"]
        and call["usage"]["prompt_tokens"] == len(ids),
        "native input token mismatch",
    )
    require(len(ids) + spec["cap"] <= 8192, "native context exceeded")
    if call["available"]:
        emitted = call["output_token_ids"]
        require(
            1 <= len(emitted) <= spec["cap"] and call["usage"]["completion_tokens"] == len(emitted),
            "output token mismatch",
        )
        require(
            tokenizer.decode(emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            == call["text"],
            "native decode mismatch",
        )
        require(
            call["finish_reason"]
            == ("eos" if emitted[-1] == tokenizer.eos_token_id else "length_or_time"),
            "native EOS mismatch",
        )


def audit_episode(task, job, row, calls, nodes, plan, plan_sha, tokenizer, world):
    profile = job.get("prompt_profile", plan.get("profile", "original"))

    def public_prompt(frame, history, context, goal):
        prompt = bridge.public_prompt(frame, history, context=context, goal=goal)
        if profile == "instruction_control":
            require(
                plan["instruction_reminder"] == collector.INSTRUCTION_REMINDER,
                "unqualified instruction reminder",
            )
            prompt += plan["instruction_reminder"]
        elif profile == "procedure_control":
            require(
                plan["procedural_instruction"] == collector.PROCEDURAL_INSTRUCTION,
                "unqualified procedure instruction",
            )
            prompt += plan["procedural_instruction"]
        else:
            require(profile == "original", "unknown prompt profile")
        return prompt

    require(all(row[k] == v for k, v in job.items()), "episode planned identity mismatch")
    require(
        set(nodes) == set(row["node_ids"]) and len(nodes) == row["node_count"],
        "saved node inventory mismatch",
    )
    require(
        len(row["call_ids"]) == len(set(row["call_ids"])) == row["global_calls"],
        "duplicated/global call count mismatch",
    )
    require(
        row["global_calls"] <= 96 and row["global_output_tokens"] <= 8192, "global budget exceeded"
    )
    budget = bridge.Budget()
    root = bridge.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        budget,
        max_depth=0 if job["policy"] == "flat" else 2,
    )
    errors, actions = Counter(), Counter()
    visited, used, child_scores = [], [], []
    quantity_met_calls = 0

    def visit(frame, parent, context, goal):
        nonlocal quantity_met_calls
        node_id = f"n{len(visited)}"
        visited.append(node_id)
        saved = nodes[node_id]
        require(
            saved["parent_node_id"] == parent
            and saved["depth"] == frame.depth
            and saved["targets"] == frame.targets
            and saved["initial_inventory"] == frame.initial_inventory,
            "node parent/depth/initial inventory mismatch",
        )
        history = []
        for cid in saved["call_ids"]:
            require(not frame.finished, "model call after node finish")
            index, cap = budget.calls, budget.reserve()
            require(cid == f"{job['episode_id']}-c{index:03d}", "global call order mismatch")
            prompt = public_prompt(frame, history, context, goal)
            spec = dict(
                call_id=cid,
                prompt=prompt,
                policy=job["policy"],
                role="root" if frame.depth == 0 else "child",
                depth=frame.depth,
                node_id=node_id,
                parent_node_id=parent,
                episode_id=job["episode_id"],
                task_id=task["id"],
                global_call_index=index,
                cap=cap,
                seed=job["seed"]
                + int(hashlib.sha256(task["id"].encode()).hexdigest()[:8], 16)
                + 1000 * index,
            )
            if "condition" in job:
                spec.update(condition=job["condition"], prompt_profile=profile)
            call = calls[cid]
            audit_call(call, spec, plan, plan_sha, tokenizer)
            used.append(cid)
            if not call["available"]:
                raise ValueError("observed episode includes unavailable call")
            budget.charge(len(call["output_token_ids"]))
            if frame.depth == 0 and all(
                frame.inventory.get(k, 0) - frame.initial_inventory.get(k, 0) >= v
                for k, v in frame.targets.items()
            ):
                quantity_met_calls += 1
            try:
                action = bridge.parse_action(call["text"])
            except ValueError as exc:
                errors["invalid_schema"] += 1
                history.append(
                    dict(
                        response=call["text"],
                        feedback=f"Rejected action: {exc}. Inventory unchanged.",
                    )
                )
                continue
            actions[action["action"]] += 1
            try:
                response = frame.apply(action)
            except ValueError as exc:
                errors["rejected_action"] += 1
                history.append(
                    dict(action=action, feedback=f"Rejected action: {exc}. Inventory unchanged.")
                )
                continue
            if action["action"] == "delegate":
                child = visit(response, node_id, action["context"], None)
                feedback = dict(
                    child_status=child["status"],
                    child_message=child["finish_message"],
                    inventory_after_return=dict(frame.inventory),
                )
            else:
                feedback = response
                if isinstance(response, str) and response.startswith("Error:"):
                    errors["native_action_error"] += 1
            history.append(dict(action=action, feedback=feedback))
        score, details = frame.score()
        require(
            saved["final_inventory"] == frame.inventory
            and saved["public_history"] == history
            and saved["native_score"] == score
            and saved["native_details"] == details
            and saved["finish_message"] == frame.message,
            "native state/history/checker mismatch",
        )
        if frame.finished:
            status = "finished"
        elif budget.calls >= 96:
            status = "global_call_cap"
        elif budget.output_tokens >= 8192:
            status = "global_token_cap"
        else:
            ids = tokenizer.apply_chat_template(
                [
                    dict(
                        role="user",
                        content=public_prompt(frame, history, context, goal),
                    )
                ],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            require(len(ids) + budget.reserve() > 8192, "unexplained observed node termination")
            status = "context_cap"
        require(saved["status"] == status and saved["error"] is None, "node termination mismatch")
        if frame.depth:
            child_scores.append(score)
        return saved

    if not row["observed"]:
        require(row["native_score"] is None, "unknown episode must not have reward")
        # Native binding still checked; external failure may leave only partial histories.
        for index, cid in enumerate(row["call_ids"]):
            call = calls[cid]
            req = call["request"]
            require(
                req["global_call_index"] == index and req["episode_id"] == job["episode_id"],
                "unknown episode call binding mismatch",
            )
            audit_call(
                call,
                {
                    k: req[k]
                    for k in (
                        "call_id",
                        "prompt",
                        "policy",
                        "role",
                        "depth",
                        "node_id",
                        "parent_node_id",
                        "episode_id",
                        "task_id",
                        "global_call_index",
                        "cap",
                        "seed",
                    )
                },
                plan,
                plan_sha,
                tokenizer,
            )
        return dict(
            replayed=False,
            native_score=None,
            calls=len(row["call_ids"]),
            reason="external failure; native bindings checked, full transition replay unavailable",
        )
    final = visit(root, None, "", task["goal"])
    require(
        used == row["call_ids"] and set(visited) == set(nodes), "tree call/node coverage mismatch"
    )
    require(
        root.initial_inventory == row["root_initial_inventory"]
        and root.inventory == row["final_inventory"]
        and final["native_score"] == row["native_score"]
        and final["status"] == row["status"]
        and dict(errors) == row["errors"]
        and budget.output_tokens == row["global_output_tokens"],
        "episode replay mismatch",
    )
    return dict(
        replayed=True,
        native_score=final["native_score"],
        calls=budget.calls,
        output_tokens=budget.output_tokens,
        errors=dict(errors),
        actions=dict(actions),
        child_nodes=len(child_scores),
        child_successes=sum(child_scores),
        root_calls_after_quantity_met=quantity_met_calls,
        node_statuses=dict(Counter(n["status"] for n in nodes.values())),
    )


def analyze(output, tokenizer, draws=20000, expected_collector_sha256=None):
    import psutil

    plan = read(output / "PLAN.json")
    plan_sha = collector.inputs.sha(output / "PLAN.json")
    owners = list(output.glob("OWNER-*.json"))
    require(len(owners) == 1, "single native owner required")
    terminal_path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    require(terminal_path.exists(), "terminal run required; no partial outcome analysis")
    owner, terminal = read(owners[0]), read(terminal_path)
    try:
        process = psutil.Process(owner["pid"])
        require(
            abs(process.create_time() - owner["create_time"]) >= 0.01
            or process.status() == psutil.STATUS_ZOMBIE,
            "native owner still live",
        )
    except psutil.NoSuchProcess:
        pass
    hashes = {
        str(p): collector.inputs.sha(p)
        for p in (
            output / "PLAN.json",
            owners[0],
            terminal_path,
            Path(__file__),
            Path(collector.__file__),
            Path(bridge.__file__),
        )
    }
    for path, digest in plan["source_sha256"].items():
        require(collector.inputs.sha(Path(path)) == digest, "sealed source changed: " + path)
        hashes[path] = digest
    require(str(Path(owner["source"]).resolve()) in plan["source_sha256"], "owner source unbound")
    require(
        collector.inputs.sha(Path(bridge.__file__)) in plan["source_sha256"].values(),
        "analyzer bridge not identical to collector bridge",
    )
    require(
        (expected_collector_sha256 or collector.inputs.sha(Path(collector.__file__)))
        in {
            digest
            for path, digest in plan["source_sha256"].items()
            if Path(path).name == "eval_textcraft.py"
        },
        "analyzer collector helpers differ from sealed source",
    )
    prepared = Path(plan["prepared"])
    require(
        collector.inputs.sha(prepared / "MANIFEST.json") == plan["manifest_sha256"]
        and collector.inputs.sha(prepared / "tasks.jsonl") == plan["tasks_sha256"],
        "frozen inputs changed",
    )
    require(
        collector.inputs.sha(Path(plan["model"]) / "local-research-manifest.json")
        == plan["model_manifest_sha256"],
        "model manifest changed",
    )
    tasks = {
        v["id"]: v
        for v in (json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines())
    }
    require(
        len(tasks) == 8
        and len(plan["jobs"]) in (16, 32)
        and plan["planned_episodes"] == len(plan["jobs"]),
        "fixed eight-parent profile inventory required",
    )
    rows, calls, node_files, starts = {}, {}, {}, {}
    for folder, target in (
        ("episodes", rows),
        ("calls", calls),
        ("nodes", node_files),
        ("starts", starts),
    ):
        for path in sorted((output / folder).glob("*.json")):
            target[path.stem] = read(path)
            hashes[str(path)] = collector.inputs.sha(path)
    require(set(rows) <= {j["episode_id"] for j in plan["jobs"]}, "unplanned episodes")
    require(set(calls) <= set(starts), "native call missing start receipt")
    jobs_by_id = {j["episode_id"]: j for j in plan["jobs"]}
    for cid, call in calls.items():
        require(
            starts[cid]["request"] == call["request"]
            and starts[cid]["request_digest"] == call["request_digest"],
            "start/call mismatch",
        )
        req = call["request"]
        job = jobs_by_id[req["episode_id"]]
        index = req["global_call_index"]
        require(
            type(index) is int and 0 <= index < 96 and cid == f"{job['episode_id']}-c{index:03d}",
            "unplanned native call",
        )
        spec = {
            k: req[k]
            for k in (
                "call_id",
                "prompt",
                "depth",
                "node_id",
                "parent_node_id",
                "global_call_index",
                "cap",
            )
        }
        spec.update(
            policy=job["policy"],
            role="root" if req["depth"] == 0 else "child",
            episode_id=job["episode_id"],
            task_id=job["task_id"],
            seed=job["seed"]
            + int(hashlib.sha256(job["task_id"].encode()).hexdigest()[:8], 16)
            + 1000 * index,
        )
        if "condition" in job:
            spec.update(condition=job["condition"], prompt_profile=job["prompt_profile"])
        audit_call(call, spec, plan, plan_sha, tokenizer)
    world, audits, used = bridge.load_world(), {}, set()
    for job in plan["jobs"]:
        eid = job["episode_id"]
        if eid not in rows:
            continue
        row = rows[eid]
        require(not (used & set(row["call_ids"])), "calls reused across episodes")
        used.update(row["call_ids"])
        nodes = {nid: node_files[f"{eid}-{nid}"] for nid in row["node_ids"]}
        audits[eid] = audit_episode(
            tasks[job["task_id"]], job, row, calls, nodes, plan, plan_sha, tokenizer, world
        )
    groups = {}
    for policy in sorted({j.get("condition", j["policy"]) for j in plan["jobs"]}):
        selected = [r for r in rows.values() if r.get("condition", r["policy"]) == policy]
        known = [r for r in selected if r["observed"]]
        successes = sum(r["native_score"] for r in known)
        children = [audits[r["episode_id"]] for r in known]
        groups[policy] = dict(
            planned=16,
            recorded=len(selected),
            observed=len(known),
            missing=16 - len(selected),
            unavailable=len(selected) - len(known),
            observed_failures=len(known) - successes,
            won=successes,
            success_rate_bounds=[successes / 16, (successes + 16 - len(known)) / 16],
            statuses=dict(Counter(r["status"] for r in selected)),
            invalids=dict(sum((Counter(r["errors"]) for r in selected), Counter())),
            actions=dict(sum((Counter(a["actions"]) for a in children), Counter())),
            child_nodes=sum(a["child_nodes"] for a in children),
            child_successes=sum(a["child_successes"] for a in children),
            failed_roots_with_successful_child=sum(
                a["child_successes"] > 0 and a["native_score"] == 0 for a in children
            ),
            root_calls_after_quantity_met=sum(a["root_calls_after_quantity_met"] for a in children),
            cost=collector.cost([c for c in calls.values() if c["condition"] == policy]),
            role_cost={
                role: collector.cost(
                    [c for c in calls.values() if c["condition"] == policy and c["role"] == role]
                )
                for role in ("root", "child")
            },
        )
    strata = {}
    for label, depths in (("depth2_3", (2, 3)), ("depth4", (4,))):
        jobs = [j for j in plan["jobs"] if tasks[j["task_id"]]["misc"]["max_depth"] in depths]
        strata[label] = paired(jobs, rows, draws)
        strata[label]["successes"] = {
            p: sum(
                rows.get(j["episode_id"], {}).get("native_score") == 1
                for j in jobs
                if j["policy"] == p
            )
            for p in groups
        }
    return dict(
        method=dict(
            parent_count=8,
            repeats=2,
            bootstrap_draws=draws,
            bootstrap_seed=SEED,
            bootstrap_unit="Task parent; retain both seeds. "
            "Shared recipe world is not atomic independence.",
            unknown_policy="No imputed loss; bounds use full planned denominator; "
            "CI only complete panel.",
            interpretation="Child success and work after sufficient inventory are descriptive; "
            "finish still required. No causal claim that child work is necessary/unnecessary "
            "without trace/counterfactual evidence.",
        ),
        groups=groups,
        paired=paired(plan["jobs"], rows, draws),
        depth_strata=strata,
        physical_cost=collector.cost(list(calls.values())),
        owner_wall_seconds=terminal.get("elapsed_seconds"),
        terminal=terminal,
        unresolved_starts=sorted(set(starts) - set(calls)),
        calls_without_episode=sorted(set(calls) - used),
        audits=audits,
        sha256=hashes,
        input_manifest_sha256=plan["manifest_sha256"],
        model_manifest_sha256=plan["model_manifest_sha256"],
    )


def markdown(report):
    lines = [
        "# TextCraft fixed-panel exploratory readout",
        "",
        "Eight official VAL task parents × two seeds; shared recipe world. "
        "No independent16-parent claim.",
        "",
        "| Policy | Won/planned | Observed failures | Missing / unknown | Calls | Input / output |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p, g in report["groups"].items():
        c = g["cost"]
        lines.append(
            f"| {p} | {g['won']}/16 | {g['observed_failures']} | "
            f"{g['missing']} / {g['unavailable']} | "
            f"{c['calls']} | {c['prompt_tokens']} / {c['completion_tokens']} |"
        )
    lines += [
        "",
        "Recursive minus flat (parent bootstrap): " + json.dumps(report["paired"]),
        "",
        "Depth strata (descriptive): " + json.dumps(report["depth_strata"]),
        "",
        "All available native requests/IDs/decodes verified; observed episodes replayed "
        "through trusted actions "
        "and native checker, including tree budgets, inventories and rejection/child feedback.",
        "",
        report["method"]["interpretation"],
        "",
        f"Bootstrap: {report['method']['bootstrap_draws']} draws, seed {SEED}; "
        "unadjusted exploratory intervals.",
        "Observed context/global-budget failures remain failures; "
        "unavailable/missing remain unknown.",
        "",
        "Physical service seconds: "
        + str(report["physical_cost"]["native_service_seconds"])
        + "; owner wall seconds: "
        + str(report["owner_wall_seconds"])
        + ".",
        "",
    ]
    for policy, g in report["groups"].items():
        lines += [
            f"{policy}: statuses {g['statuses']}; invalids {g['invalids']}. "
            f"Child native success {g['child_successes']}/{g['child_nodes']}; "
            f"failed roots with successful child {g['failed_roots_with_successful_child']}; "
            f"root calls after inventory quantity sufficient {g['root_calls_after_quantity_met']} "
            "(includes required finish).",
            "",
            f"Root/child native costs: {json.dumps(g['role_cost'])}",
            "",
        ]
    return "\n".join(lines)


def main(args):
    from transformers import AutoTokenizer

    require(
        not args.report.exists() and not args.report.with_suffix(".md").exists(),
        "immutable report already exists",
    )
    plan = read(args.output / "PLAN.json")
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    result = analyze(args.output, tokenizer)
    collector.inputs.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
    print(
        json.dumps(dict(report=str(args.report), groups=result["groups"], paired=result["paired"]))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args())
