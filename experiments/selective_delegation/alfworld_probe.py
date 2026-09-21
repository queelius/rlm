"""Frozen base-model, affordance-assisted ALFWorld scaffolding screen (not training)."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import eval_planner as evaluation
import probe

MANIFEST = Path(
    "/project/alex_phd/research-cache/datasets/alfworld-text-0.4.2-20260921/readiness/MANIFEST.json"
)
MANIFEST_SHA = "7c0a86472377cd73af66585ec04dff4010c75de3a2346e9f2988a86840eb76c5"
POLICIES = ("flat", "manager_worker")
SEEDS = (2026092178, 2026092179)
ACTION_LIMIT, TOKEN_LIMIT, CALL_CAP, CONTEXT = 50, 2048, 128, 8192
GOAL_RESERVE = 512


def save(path, value):
    probe.runtime.save(Path(path), value)


def parse_output(text, role, admissible):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    obj = json.loads(text, object_pairs_hook=unique)
    key = "goal" if role == "manager" else "action"
    if not isinstance(obj, dict) or set(obj) != {key} or not isinstance(obj[key], str):
        raise ValueError("strict single-string JSON field required")
    value = obj[key]
    if role == "manager":
        if not value.strip() or len(value) > 240:
            raise ValueError("goal must contain1..240 characters")
    elif value not in admissible:
        raise ValueError("action is not exactly in current admissible commands")
    return value


def prompts(initial, current, history, goal):
    context = {
        "initial_observation": initial,
        "current_feedback": current["feedback"],
        "history": [{"action": h["action"], "feedback": h["feedback"]} for h in history],
        "admissible_commands": list(current["admissible_commands"]),
    }
    shared = "Solve the household task using the supplied public observations. "
    action = (
        "Return ONLY a JSON object with one string field named action. "
        "Its value must exactly equal one current admissible command. "
    )
    instructions = {
        "flat": shared + action + "Choose the next useful action.",
        "manager": shared + "Return ONLY a JSON object with one string field named goal. "
        "State a short immediate goal (at most240 characters) for the next four actions.",
        "worker": shared
        + action
        + "Choose the next useful action toward the current short-term goal.",
    }
    return {
        role: instruction
        + "\n"
        + json.dumps(
            {"public_context": context, **({"current_goal": goal} if role == "worker" else {})},
            ensure_ascii=False,
        )
        for role, instruction in instructions.items()
    }


def bounded_prompts(client, initial, current, history, goal, cap):
    dropped = 0
    original = None
    while True:
        neutral = prompts(initial, current, history[dropped:], "")
        maximum = max(client.token_count(p) for p in neutral.values())
        if original is None:
            original = maximum
        if maximum + GOAL_RESERVE + CALL_CAP <= CONTEXT:
            result = prompts(initial, current, history[dropped:], goal)
            lengths = {r: client.token_count(p) for r, p in result.items()}
            if max(lengths.values()) + cap > CONTEXT:
                raise ValueError("bounded goal exceeded reserved context; no asymmetric trimming")
            return result, {
                "dropped_history": dropped,
                "original_history": len(history),
                "full_neutral_max_tokens": original,
                "kept_neutral_max_tokens": maximum,
                "actual_prompt_tokens_by_role": lengths,
                "goal_reserve_tokens": GOAL_RESERVE,
            }
        if dropped >= len(history):
            raise ValueError("mandatory initial/current context exceeds limit")
        dropped += 1


class Bridge:
    def __init__(self, game, python, data, deadline, stderr_path):
        self.deadline = deadline
        self.stderr = Path(stderr_path).open("x")  # noqa: SIM115 -- bridge owns until close()
        self.process = subprocess.Popen(
            [
                str(python),
                str(Path(__file__).with_name("alfworld_bridge.py")),
                "--game",
                game["game"],
                "--sha256",
                game["game_sha256"],
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            env={**os.environ, "ALFWORLD_DATA": str(data), "CUDA_VISIBLE_DEVICES": ""},
        )

    def receive(self):
        timeout = min(30, self.deadline - time.time())
        if timeout <= 0 or not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError("ALFWorld bridge response deadline")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ALFWorld bridge closed before response; inspect stderr")
        row = json.loads(line)
        if set(row["public"]) != {"feedback", "admissible_commands"}:
            raise ValueError("bridge public projection differs")
        return row

    def reset(self):
        return self.receive()

    def step(self, action):
        self.process.stdin.write(json.dumps({"action": action}) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write('{"op":"close"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=5)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
        finally:
            self.stderr.close()


def play_episode(client, env, job, output, stopping=lambda: False):
    output = Path(output)
    (output / "episode-progress").mkdir(parents=True, exist_ok=True)
    identity, policy = job["episode_id"], job["policy"]
    result = {
        **job,
        "observed": False,
        "won": False,
        "actions": 0,
        "generated_tokens": 0,
        "call_ids": [],
        "invalid_outputs": 0,
        "termination": "starting",
        "started": time.time(),
    }
    records, history = [], []
    invalid, action_attempt, manager_attempt = 0, 0, 0
    goal, goal_at = "", -1
    try:
        event = env.reset()
        save(output / "observations" / (identity + "-000.json"), event)
        initial, current = event["public"]["feedback"], event["public"]
        if event["host"]["won"]:
            raise ValueError("initially won task is outside frozen readiness contract")
        while True:
            if stopping():
                result["termination"] = "owner_stopped_or_capped"
                break
            if result["generated_tokens"] >= TOKEN_LIMIT:
                result.update(observed=True, termination="token_budget")
                break
            if result["actions"] >= ACTION_LIMIT:
                result.update(observed=True, termination="action_budget")
                break
            need_goal = policy == "manager_worker" and (
                not goal or result["actions"] - goal_at >= 4
            )
            role = "manager" if need_goal else "flat" if policy == "flat" else "worker"
            if role == "manager":
                seed = job["seed"] + 100000 + manager_attempt
                manager_attempt += 1
            else:
                seed = job["seed"] + action_attempt
                action_attempt += 1
            cap = min(CALL_CAP, TOKEN_LIMIT - result["generated_tokens"])
            texts, trimming = bounded_prompts(client, initial, current, history, goal, cap)
            cid = identity + f"-call-{len(records):03d}-{role}"
            record = client.call(cid, texts[role], policy, role, seed, cap, trimming)
            records.append(record)
            result["call_ids"].append(cid)
            if not record["available"]:
                raise RuntimeError("native inference failed; no retry or zero reward substitution")
            tokens = record["usage"]["completion_tokens"]
            if not isinstance(tokens, int) or tokens < 1 or tokens > cap:
                raise ValueError("native output token receipt outside episode cap")
            result["generated_tokens"] += tokens
            decision = {"call_id": cid, "role": role, "trimming": trimming}
            try:
                value = parse_output(record["text"], role, current["admissible_commands"])
            except (ValueError, TypeError) as exc:
                invalid += 1
                result["invalid_outputs"] += 1
                decision.update(valid=False, error=str(exc))
            else:
                decision.update(valid=True, parsed=value)
                invalid = 0
                if role == "manager":
                    goal, goal_at = value, result["actions"]
                else:
                    event = env.step(value)
                    result["actions"] += 1
                    save(
                        output / "observations" / (identity + f"-{result['actions']:03d}.json"),
                        event,
                    )
                    current = event["public"]
                    history.append({"action": value, "feedback": current["feedback"]})
                    if event["host"]["done"] or event["host"]["won"]:
                        result.update(
                            observed=True, won=bool(event["host"]["won"]), termination="native_done"
                        )
            save(output / "decisions" / (cid + ".json"), decision)
            probe.campaign.snapshot(output / "episode-progress" / (identity + ".json"), result)
            if result["observed"]:
                break
            if invalid >= 3:
                result.update(observed=True, termination="three_consecutive_invalid")
                break
    except Exception as exc:
        result.update(
            termination="inference_or_environment_error", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        result.update(ended=time.time(), native_cost=evaluation.cost(records))
        save(output / "episodes" / (identity + ".json"), result)
    return result


class BaseClient:
    def __init__(self, model, tokenizer, output, deadline):
        self.model, self.tokenizer, self.output, self.deadline = (
            model,
            tokenizer,
            Path(output),
            deadline,
        )
        self.returned = self.failed = 0

    def ids(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def token_count(self, text):
        return len(self.ids(text))

    def call(self, identity, prompt, policy, role, seed, cap, trimming):
        import torch

        ids = self.ids(prompt)
        request = {
            "prompt": prompt,
            "input_token_ids": ids,
            "condition": policy,
            "role": role,
            "model": str(evaluation.planner.BASE),
            "adapter_enabled": False,
            "adapter_sha256": None,
            "seed": seed,
            "sampling": {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": cap,
                "max_time": 90.0,
                "do_sample": True,
            },
        }
        row = {
            "call_id": identity,
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "condition": policy,
            "role": role,
            "available": False,
            "text": None,
            "input_token_ids": ids,
            "usage": {},
            "trimming": trimming,
            "started": time.time(),
        }
        save(self.output / "starts" / (identity + ".json"), row)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if remaining <= 0 or len(ids) + cap > CONTEXT or not 1 <= cap <= CALL_CAP:
                raise ValueError("generation deadline/context/cap violated")
            torch.manual_seed(seed)
            if str(self.model.device).startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            inputs = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            row["effective_max_time"] = remaining
            row["usage"]["prompt_tokens"] = len(ids)
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=True,
                    temperature=0.5,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=cap,
                    max_time=remaining,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            values = generated[0].detach().cpu().tolist()
            if values[: len(ids)] != ids or len(values) <= len(ids):
                raise ValueError("not a real continuation of saved prompt")
            completion = values[len(ids) :]
            row.update(
                available=True,
                output_token_ids=completion,
                text=self.tokenizer.decode(
                    completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
                usage={"prompt_tokens": len(ids), "completion_tokens": len(completion)},
                finish_reason="eos"
                if completion[-1] == self.tokenizer.eos_token_id
                else "length_or_time",
            )
            self.returned += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.failed += 1
        row["ended"] = time.time()
        save(self.output / "calls" / (identity + ".json"), row)
        probe.campaign.snapshot(
            self.output / "STATUS.json",
            {
                "state": "scientific_calls",
                "returned": self.returned,
                "failed": self.failed,
                "last_call": identity,
                "updated": time.time(),
            },
        )
        if self.returned == 1 and row["ended"] - row["started"] > 90:
            raise TimeoutError("first real scientific response exceeded90seconds")
        return row


def prepare(output, hours):
    if not 0 < hours <= 1:
        raise ValueError("at most one cumulative hour")
    if probe.campaign.sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("readiness manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    selected = manifest["proposed_screen"]
    if selected["seed"] != 2026092177 or len(selected["games"]) != 8:
        raise ValueError("frozen eight-game inventory differs")
    jobs = []
    for index, game in enumerate(selected["games"]):
        if probe.campaign.sha(Path(game["game"])) != game["game_sha256"]:
            raise ValueError("game changed")
        for seed in SEEDS:
            for policy in POLICIES:
                jobs.append(
                    {
                        "episode_id": f"game-{index:02d}-seed-{seed}-{policy}",
                        "game_index": index,
                        "game": game,
                        "seed": seed,
                        "policy": policy,
                    }
                )
    env = Path(manifest["environment"]["path"])
    for name in ("pyproject.toml", "uv.lock"):
        if (
            probe.campaign.sha(env / name)
            != manifest["inspected_source_and_environment_sha256"][str(env / name)]
        ):
            raise ValueError("isolated environment specification changed")
    plan = {
        "schema": "alfworld-frozen-scaffolding-v1",
        "readiness_manifest": str(MANIFEST),
        "readiness_manifest_sha256": MANIFEST_SHA,
        "cases": jobs,
        "planned_episodes": 32,
        "policies": POLICIES,
        "seeds": SEEDS,
        "model": str(evaluation.planner.BASE),
        "model_manifest_sha256": probe.campaign.sha(
            evaluation.planner.BASE / "local-research-manifest.json"
        ),
        "adapters": None,
        "training": False,
        "budget_seconds": hours * 3600,
        "action_limit": ACTION_LIMIT,
        "token_limit": TOKEN_LIMIT,
        "request_cap": CALL_CAP,
        "context_limit": CONTEXT,
        "goal_reserve": GOAL_RESERVE,
        "alfworld_python": str(env / ".venv/bin/python"),
        "data_root": str(MANIFEST.parent.parent),
        "smoke_rule": "first game/first seed/both arms must execute>=1action; no won gate",
        "prompt_templates": prompts(
            "INITIAL", {"feedback": "CURRENT", "admissible_commands": []}, [], "GOAL"
        ),
        "source_sha256": {
            str(p.resolve()): probe.campaign.sha(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("alfworld_bridge.py"),
                Path(evaluation.__file__),
                Path(probe.__file__),
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        },
    }
    # JSON canonicalization makes repeated CPU preparation and launch identical.
    plan = json.loads(json.dumps(plan))
    path = Path(output) / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable plan changed")
    else:
        save(path, plan)
    return plan, jobs


def summarize(output, plan):
    episodes = [json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")]
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    groups = {}
    for policy in POLICIES:
        rows = [e for e in episodes if e["policy"] == policy]
        groups[policy] = {
            "planned": 16,
            "recorded": len(rows),
            "observed": sum(r["observed"] for r in rows),
            "missing_or_unobserved": 16 - sum(r["observed"] for r in rows),
            "won": sum(r["won"] for r in rows),
            "success_lower_bound": sum(r["won"] for r in rows) / 16,
            "terminations": dict(Counter(r["termination"] for r in rows)),
            "actions": sum(r["actions"] for r in rows),
            "cost": evaluation.cost([c for c in calls if c["condition"] == policy]),
        }
    values = {(e["game_index"], e["seed"], e["policy"]): e for e in episodes}
    changes = []
    for index in range(8):
        for seed in SEEDS:
            a, b = [values.get((index, seed, p)) for p in POLICIES]
            if a and b and a["observed"] and b["observed"] and a["won"] != b["won"]:
                changes.append(
                    {
                        "game_index": index,
                        "seed": seed,
                        "flat_won": a["won"],
                        "manager_won": b["won"],
                    }
                )
    return {
        "plan_sha256": probe.campaign.sha(output / "PLAN.json"),
        "groups": groups,
        "paired_observed_changes": changes,
        "physical_cost": evaluation.cost(calls),
        "all_slots_recorded": len(episodes) == plan["planned_episodes"],
        "note": "Missing is unobserved. Affordance-assisted exposed development screen; "
        "no novelty/RL claim.",
    }


def run(args):
    output = args.output.resolve()
    plan, jobs = prepare(output, args.hours)
    if args.prepare_only:
        print(json.dumps({"planned": len(jobs), "model_loaded": False, "output": str(output)}))
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("existing owner: no implicit retry/resume of environment episodes")
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation budget")
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"], local_files_only=True, trust_remote_code=False
    )
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    stopped, failure, model = False, None, None
    try:
        save(
            output / f"OWNER-{invocation}.json",
            {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "started": started,
                "deadline": deadline,
                "allocation_end": lease,
                "source": str(Path(__file__).resolve()),
            },
        )

        def stop(*_):
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly oneGPU")
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        save(
            output / f"LOAD-{invocation}.json",
            {
                "adapters_loaded": False,
                "optimizer_created": False,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = BaseClient(model, tokenizer, output, deadline)
        smoke = []

        def stopping():
            return stopped or time.time() >= deadline - 5 or (output / "STOP").exists()

        for index, job in enumerate(jobs):
            if stopping():
                stopped = True
                break
            env = Bridge(
                job["game"],
                plan["alfworld_python"],
                plan["data_root"],
                deadline,
                output / (job["episode_id"] + "-bridge.stderr"),
            )
            try:
                result = play_episode(client, env, job, output, stopping)
            finally:
                env.close()
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            if index < 2:
                smoke.append(result)
            if index == 1:
                passed = all(r["actions"] > 0 for r in smoke)
                save(
                    output / "SMOKE.json",
                    {
                        "passed": passed,
                        "episode_ids": [r["episode_id"] for r in smoke],
                        "rule": plan["smoke_rule"],
                    },
                )
                if not passed:
                    stopped = True
                    break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
        save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "elapsed_seconds": time.time() - started,
                "ended": time.time(),
                "deadline": deadline,
            },
        )
        probe.campaign.snapshot(
            output / "STATUS.json",
            {
                "state": "failed" if failure else "finished_or_capped",
                "failure": failure,
                "updated": time.time(),
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
