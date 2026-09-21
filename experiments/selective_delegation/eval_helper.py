"""Paired helper adaptation with frozen saved SFT roots and permanently base finals."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import random
import signal
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import eval_planner as evaluation
import probe

CONDITIONS = ("base_helper", "trained_helper", "format_reminder")
FORMAT_REMINDER = "\nReturn ONLY a JSON object with one string field named answer."
ADAPTER_NAME = "helper_sft"
SEED = 2026092112
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def validate_helper_training(training_plan, model):
    if training_plan.get("role") != "helper" or training_plan.get("model") != str(model):
        raise ValueError("helper training role/model mismatch")


def enabled_for(condition, role):
    if condition not in CONDITIONS or role not in ("helper", "final"):
        raise ValueError("unknown helper evaluation condition/role; roots are never generated")
    return condition == "trained_helper" and role == "helper"


def helper_prompt(case, question, condition):
    enabled_for(condition, "helper")
    return evaluation.isolated_helper_prompt(case, question) + (
        FORMAT_REMINDER if condition == "format_reminder" else ""
    )


@contextmanager
def route(model, condition, role):
    enabled = enabled_for(condition, role)
    model.set_adapter(ADAPTER_NAME)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.active_adapters != [ADAPTER_NAME]:
        raise ValueError("unexpected active adapter composition")
    try:
        if enabled:
            yield
        else:
            with model.disable_adapter():
                yield
    finally:
        # PEFT enable_adapter_layers on context exit restores trainability unless frozen again.
        for parameter in model.parameters():
            parameter.requires_grad_(False)


class HelperClient:
    """Sequential native HF inference; adapter routing is explicit, never process monkeypatching."""

    def __init__(self, model, tokenizer, output, deadline, adapter_sha):
        self.model, self.tokenizer = model, tokenizer
        self.output, self.deadline, self.adapter_sha = Path(output), deadline, adapter_sha
        self.returned = self.failed = self.consecutive_failures = 0

    def call(self, identity, prompt, condition, role, seed, *, max_new_tokens):
        import torch
        from transformers import GenerationConfig

        enabled = enabled_for(condition, role)
        cap = max_new_tokens
        if type(cap) is not int or not 1 <= cap <= {"helper": 384, "final": 128}[role]:
            raise ValueError("invalid role token cap")
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        request = {
            "prompt": prompt,
            "input_token_ids": ids,
            "condition": condition,
            "role": role,
            "model": str(evaluation.planner.BASE),
            "adapter_enabled": enabled,
            "adapter_name": ADAPTER_NAME if enabled else None,
            "adapter_sha256": self.adapter_sha if enabled else None,
            "seed": seed,
            "sampling": {
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "max_new_tokens": cap,
                "max_time": 90.0,
            },
        }
        path = self.output / "calls" / (identity + ".json")
        if path.exists():
            saved = read(path)
            if (
                saved["request_digest"] != probe.runtime.digest(request)
                or probe.runtime.digest(saved["request"]) != saved["request_digest"]
            ):
                raise ValueError("cached native request differs")
            return saved
        start_path = self.output / "starts" / (identity + ".json")
        if start_path.exists():
            raise RuntimeError("unresolved native start receipt; no implicit retry")
        record = {
            "call_id": identity,
            "condition": condition,
            "role": role,
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "adapter_enabled": enabled,
            "adapter_name": request["adapter_name"],
            "adapter_sha256": request["adapter_sha256"],
            "input_token_ids": ids,
            "available": False,
            "text": None,
            "usage": {},
            "started": time.time(),
        }
        probe.runtime.save(start_path, record)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if STOP or remaining <= 0:
                raise TimeoutError("helper evaluation deadline")
            if len(ids) + cap > 8192:
                raise ValueError("context limit exceeded; no truncation")
            torch.manual_seed(seed)
            if str(self.model.device).startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            tensor = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            record["usage"] = {"prompt_tokens": len(ids)}
            record["effective_max_time"] = remaining
            config = GenerationConfig(
                do_sample=True,
                temperature=0.5,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=cap,
                max_time=remaining,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            with route(self.model, condition, role), torch.no_grad():
                generated = self.model.generate(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    generation_config=config,
                )
            actual = generated[0].detach().cpu().tolist()
            if actual[: len(ids)] != ids or len(actual) <= len(ids):
                raise ValueError("native generation did not extend saved prompt")
            emitted = actual[len(ids) :]
            record.update(
                output_token_ids=emitted,
                usage={"prompt_tokens": len(ids), "completion_tokens": len(emitted)},
            )
            text = self.tokenizer.decode(
                emitted, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if not text.strip():
                raise ValueError("empty scientific model response")
            if self.returned == 0 and time.time() - record["started"] > 90:
                raise TimeoutError("first scientific response exceeded90 seconds")
            record.update(
                available=True,
                text=text,
                finish_reason="eos"
                if emitted[-1] == self.tokenizer.eos_token_id
                else "length"
                if len(emitted) == cap
                else "max_time_or_other_stop",
            )
            self.returned += 1
            self.consecutive_failures = 0
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            self.failed += 1
            self.consecutive_failures += 1
        record["ended"] = time.time()
        probe.runtime.save(path, record)
        print(
            json.dumps(
                {
                    "call_id": identity,
                    "role": role,
                    "condition": condition,
                    "available": record["available"],
                    "usage": record["usage"],
                    "seconds": record["ended"] - record["started"],
                }
            ),
            flush=True,
        )
        return record


def validate_root(root, case, model, adapter_sha):
    request = root["request"]
    if probe.runtime.digest(request) != root["request_digest"]:
        raise ValueError("source root request digest mismatch")
    if request["prompt"] != evaluation.planner_prompt(case):
        raise ValueError("source root prompt differs from public reconstruction")
    if (
        root["role"] != "root"
        or root["condition"] != "sft"
        or request["model"] != model
        or not request["adapter_enabled"]
        or request["adapter_sha256"] != adapter_sha
        or root["adapter_sha256"] != adapter_sha
        or not root["adapter_enabled"]
    ):
        raise ValueError("source root role/adapter mismatch")
    if not root["available"]:
        return None
    try:
        return evaluation.parse_plan(root["text"])
    except (ValueError, TypeError):
        return None


def prepare_roots(source, cases_path, root_adapter):
    source, root_adapter = Path(source), Path(root_adapter)
    source_plan = read(source / "PLAN.json")
    binding = evaluation.adapter_identity(root_adapter)
    if (
        read(root_adapter / "STATE.json")["step"] != 48
        or source_plan["adapter_files_sha256"] != binding
        or source_plan["cases_sha256"] != probe.campaign.sha(cases_path)
        or source_plan["execution"] != "isolated"
        or "sft" not in source_plan["conditions"]
        or len(source_plan["case_ids"]) != 32
        or source_plan["repeats"] != 2
    ):
        raise ValueError("expected completed32-parent two-repeat SFT48 source panel")
    owners = {p.stem.removeprefix("OWNER-") for p in source.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in source.glob("TERMINAL-*.json")}
    if not owners or owners != terminals:
        raise ValueError("source owner not completed/released")
    hashes = {str(source / "PLAN.json"): probe.campaign.sha(source / "PLAN.json")}
    for owner_path in source.glob("OWNER-*.json"):
        owner = read(owner_path)
        source_file = Path(owner["source"])
        if probe.campaign.sha(source_file) != source_plan["source_sha256"]:
            raise ValueError("source owner code changed")
        hashes[str(source_file)] = source_plan["source_sha256"]
        hashes[str(owner_path)] = probe.campaign.sha(owner_path)
    for p, digest in source_plan.get("dependencies", {}).items():
        if probe.campaign.sha(Path(p)) != digest:
            raise ValueError("source dependency changed")
        hashes[p] = digest
    for name, digest in binding.items():
        hashes[str(root_adapter / name)] = digest
    with Path(cases_path).open() as stream:
        cases = {c["id"]: c for c in map(json.loads, stream)}
    indexed = {}
    for path in (source / "episodes").glob("*.json"):
        row = read(path)
        if row["condition"] == "sft":
            key = row["case_id"], row["repeat"]
            if key in indexed:
                raise ValueError("duplicate source SFT root episode")
            indexed[key] = path, row
    jobs = []
    for cid in source_plan["case_ids"]:
        if cases[cid]["split"] != "validation":
            raise ValueError("helper development requires validation parents")
        for repeat in range(2):
            path, row = indexed[cid, repeat]
            root_path = source / "calls" / (row["episode_id"] + "-root.json")
            root = read(root_path)
            if root["call_id"] != root_path.stem or root["call_id"] not in row["call_ids"]:
                raise ValueError("source root receipt identity mismatch")
            plan = validate_root(
                root, cases[cid], source_plan["model"], binding["adapter_model.safetensors"]
            )
            if bool(plan) != row["plan_valid"] or (plan and plan != row["plan"]):
                raise ValueError("source parsed root differs from saved episode")
            pair_hashes = {str(p): probe.campaign.sha(p) for p in (path, root_path)}
            hashes.update(pair_hashes)
            jobs.append(
                {
                    "case_id": cid,
                    "repeat": repeat,
                    "root": root,
                    "plan": plan,
                    "seed": SEED + int(probe.runtime.digest(cid)[:6], 16) + repeat * 100,
                    "source_episode_id": row["episode_id"],
                    "source_hashes": pair_hashes,
                }
            )
    if len(indexed) != 64:
        raise ValueError("unexpected source SFT episode inventory")
    return source_plan, cases, jobs, hashes


def collect_episode(client, case, job, condition):
    identity = f"{case['id']}-r{job['repeat']}-{condition}"
    path = client.output / "episodes" / (identity + ".json")
    if path.exists():
        return read(path)
    plan, records, trace, answers = job["plan"], [], [], []
    row = {
        "episode_id": identity,
        "case_id": case["id"],
        "repeat": job["repeat"],
        "condition": condition,
        "seed": job["seed"],
        "plan": plan,
        "plan_valid": bool(plan),
        "source_episode_id": job["source_episode_id"],
        "source_hashes": job["source_hashes"],
        "reused_root_call_id": job["root"]["call_id"],
        "root_request_digest": job["root"]["request_digest"],
        "available": False,
        "valid": False,
        "correct": False,
        "f1": 0.0,
        "status": "invalid_plan" if job["root"]["available"] else "source_root_unavailable",
        "started": time.time(),
    }
    try:
        if plan:
            cap = 384 // len(plan["subquestions"])
            for index, question in enumerate(plan["subquestions"]):
                row["status"] = "invalid_dependency"
                resolved = evaluation.bind_question(question, answers)
                row["status"] = "helper_failure"
                helper = client.call(
                    identity + f"-helper-{index + 1}",
                    helper_prompt(case, resolved, condition),
                    condition,
                    "helper",
                    job["seed"] + index + 1,
                    max_new_tokens=cap,
                )
                records.append(helper)
                if not helper["available"]:
                    raise RuntimeError("helper generation unavailable")
                row["status"] = "invalid_helper"
                answer = evaluation.parse_helper_answer(helper["text"])
                answers.append(answer)
                trace.append(
                    {
                        "step": index + 1,
                        "question": question,
                        "resolved_question": resolved,
                        "answer": answer,
                    }
                )
            row["status"] = "final_failure"
            final = client.call(
                identity + "-final",
                evaluation.final_prompt(case, plan, {"execution": "isolated", "steps": trace}),
                condition,
                "final",
                job["seed"] + 2,
                max_new_tokens=128,
            )
            records.append(final)
            if not final["available"]:
                raise RuntimeError("final generation unavailable")
            grade = probe.grade(final["text"], case)
            row.update(
                available=True,
                valid=grade["valid"],
                correct=grade["correct"],
                f1=grade["f1"],
                parsed=grade["parsed"],
                status="scored" if grade["valid"] else "invalid_final",
            )
    except (ValueError, TypeError, RuntimeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row.update(
        call_ids=[r["call_id"] for r in records],
        helper_trace=trace,
        helper_answers=len(trace),
        helper_answer_lengths=[len(a) for a in answers],
        new_physical_cost=evaluation.cost(records),
        deployed_cost=evaluation.cost([job["root"], *records]),
        ended=time.time(),
    )
    probe.runtime.save(path, row)
    return row


def summarize(output, plan, jobs):
    rows = [read(p) for p in (output / "episodes").glob("*.json")]
    calls = [read(p) for p in (output / "calls").glob("*.json")]
    starts = [read(p) for p in (output / "starts").glob("*.json")]
    returned_ids = {r["call_id"] for r in calls}
    unresolved = [r for r in starts if r["call_id"] not in returned_ids]
    denom = len(jobs)
    groups = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        helpers = [r for r in calls if r["condition"] == condition and r["role"] == "helper"]
        valid_helpers = 0
        for helper in helpers:
            try:
                evaluation.parse_helper_answer(helper["text"])
                valid_helpers += bool(helper["available"])
            except (ValueError, TypeError):
                pass
        groups[condition] = {
            "planned": denom,
            "recorded": len(subset),
            "missing": denom - len(subset),
            "correct": sum(r["correct"] for r in subset),
            "em": sum(r["correct"] for r in subset) / denom,
            "f1": sum(r["f1"] for r in subset) / denom,
            "status_counts": dict(Counter(r["status"] for r in subset)),
            "helper_calls": len(helpers),
            "valid_helper_json": valid_helpers,
            "helper_answer_lengths": [n for r in subset for n in r["helper_answer_lengths"]],
        }
    indexed = {(r["case_id"], r["repeat"], r["condition"]): r for r in rows}
    wins, losses, differences, f1_differences = Counter(), Counter(), [], []
    for cid in plan["case_ids"]:
        delta = 0.0
        f1_delta = 0.0
        for repeat in range(plan["repeats"]):
            left = indexed.get((cid, repeat, "base_helper"), {})
            right = indexed.get((cid, repeat, "trained_helper"), {})
            change = int(right.get("correct", False)) - int(left.get("correct", False))
            delta += change / plan["repeats"]
            f1_delta += (right.get("f1", 0.0) - left.get("f1", 0.0)) / plan["repeats"]
            if change:
                category = (
                    "both_scored"
                    if left.get("valid") and right.get("valid")
                    else (
                        "missing_involved"
                        if not left
                        or not right
                        or any(
                            r.get("status", "").endswith("failure")
                            or r.get("status") == "source_root_unavailable"
                            for r in (left, right)
                        )
                        else "protocol_involved"
                    )
                )
                (wins if change > 0 else losses)[category] += 1
        differences.append(delta)
        f1_differences.append(f1_delta)
    rng = random.Random(SEED)
    samples = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(2000)
    )
    rng = random.Random(SEED)
    f1_samples = sorted(
        sum(rng.choice(f1_differences) for _ in f1_differences) / len(f1_differences)
        for _ in range(2000)
    )
    roots = [job["root"] for job in jobs]
    return {
        "conditions": groups,
        "paired_em_trained_minus_base": {
            "estimate": sum(differences) / len(differences),
            "parent_bootstrap_95": [samples[49], samples[1949]],
            "draws": 2000,
            "seed": SEED,
            "wins": dict(wins),
            "losses": dict(losses),
        },
        "paired_f1_trained_minus_base": {
            "estimate": sum(f1_differences) / len(f1_differences),
            "parent_bootstrap_95": [f1_samples[49], f1_samples[1949]],
            "draws": 2000,
            "seed": SEED,
        },
        "new_physical_cost": evaluation.cost(calls + unresolved),
        "physical_start_receipts": len(starts),
        "unresolved_start_receipts": len(unresolved),
        "shared_root_acquisition_cost": evaluation.cost(roots),
        "physical_including_shared_roots_once": evaluation.cost(roots + calls + unresolved),
        "all_planned_complete": len(rows) == denom * len(CONDITIONS),
        "note": "Missing outcomes remain in denominators; incomplete rates are lower bounds. "
        "Roots are historical acquisition, not new calls. Deployment charges one root per arm. "
        "Parent bootstrap does not remove within-split atomic-component dependence. "
        "No annotated-step accuracy is inferred for generated subquestions.",
        "updated": time.time(),
    }


def run(args):
    global STOP
    STOP = False
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source, output = args.source_output.resolve(), args.output.resolve()
    helper_adapter = args.helper_adapter.resolve()
    if source == output or not 0 < args.hours <= 1:
        raise ValueError("separate output and at most one hour required")
    source_plan, cases, jobs, hashes = prepare_roots(
        source, args.cases, args.root_adapter.resolve()
    )
    helper_binding = evaluation.adapter_identity(helper_adapter)
    if read(helper_adapter / "STATE.json")["step"] != 36:
        raise ValueError("primary helper checkpoint must be fixed update36")
    model_path = evaluation.planner.BASE
    if (
        str(model_path) != source_plan["model"]
        or probe.campaign.sha(model_path / "local-research-manifest.json")
        != source_plan["model_manifest_sha256"]
    ):
        raise ValueError("base model identity differs")
    helper_training = read(helper_adapter.parent / "PLAN.json")
    validate_helper_training(helper_training, model_path)
    plan = {
        "schema": "matched-frozen-root-helper-evaluation-v1",
        "source_output": str(source),
        "source_hashes": hashes,
        "cases_sha256": probe.campaign.sha(args.cases),
        "case_ids": source_plan["case_ids"],
        "repeats": 2,
        "conditions": list(CONDITIONS),
        "model": str(model_path),
        "model_manifest_sha256": source_plan["model_manifest_sha256"],
        "root_adapter": str(args.root_adapter.resolve()),
        "root_adapter_binding": source_plan["adapter_files_sha256"],
        "helper_adapter": str(helper_adapter),
        "helper_adapter_binding": helper_binding,
        "helper_training_plan_sha256": probe.campaign.sha(helper_adapter.parent / "PLAN.json"),
        "helper_training_plan": helper_training,
        "seed": SEED,
        "format_reminder": FORMAT_REMINDER,
        "caps": {"helper_total": 384, "final": 128},
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": 0,
        "budget_seconds": args.hours * 3600,
        "architecture": "Same saved SFT48 title-only roots; isolated helper predictions; "
        "full-source base final. Only trained_helper helper calls enable helper_sft.",
        "dependencies": {
            str(p): probe.campaign.sha(p)
            for p in (
                Path(__file__),
                Path(evaluation.__file__),
                Path(evaluation.planner.__file__),
                Path(probe.__file__),
                probe.MUSIQUE / "metrics/answer.py",
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "PLAN.json").exists():
        if read(output / "PLAN.json") != plan:
            raise ValueError("immutable helper evaluation plan differs")
    else:
        probe.runtime.save(output / "PLAN.json", plan)
    owners = {p.stem.removeprefix("OWNER-") for p in output.glob("OWNER-*.json")}
    terminals = {p.stem.removeprefix("TERMINAL-") for p in output.glob("TERMINAL-*.json")}
    if owners != terminals:
        raise ValueError("previous owner unresolved")
    spent = sum(read(p)["elapsed_seconds"] for p in output.glob("TERMINAL-*.json"))
    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + max(0, args.hours * 3600 - spent), lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient remaining cumulative/allocation budget")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    model, failure = None, None
    try:
        probe.runtime.save(
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
            global STOP
            STOP = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("main must assign exactly one GPU")
        torch.set_num_threads(4)
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base,
            helper_adapter,
            adapter_name=ADAPTER_NAME,
            is_trainable=False,
            autocast_adapter_dtype=True,
        )
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        probe.runtime.save(
            output / f"LOAD-{invocation}.json",
            {
                "loaded_adapter_names": list(model.peft_config),
                "root_adapter_loaded": False,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "helper_binding": helper_binding,
                "model_training": model.training,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = HelperClient(
            model, tokenizer, output, deadline, helper_binding["adapter_model.safetensors"]
        )
        for job in jobs:
            order = (
                CONDITIONS
                if (int(probe.runtime.digest(job["case_id"])[:4], 16) + job["repeat"]) % 2 == 0
                else CONDITIONS[::-1]
            )
            for condition in order:
                if STOP or time.time() >= deadline - 5 or (output / "STOP").exists():
                    STOP = True
                    break
                collect_episode(client, cases[job["case_id"]], job, condition)
                probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan, jobs))
                probe.campaign.snapshot(
                    output / "STATUS.json",
                    {
                        "state": "evaluating",
                        "returned": client.returned,
                        "failed": client.failed,
                        "updated": time.time(),
                    },
                )
                if (client.returned == 0 and client.failed) or client.consecutive_failures >= 2:
                    raise RuntimeError("scientific response check failed")
            if STOP:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan, jobs))
        probe.runtime.save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": STOP,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
            },
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--root-adapter", type=Path, required=True)
    parser.add_argument("--helper-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    run(parser.parse_args())
