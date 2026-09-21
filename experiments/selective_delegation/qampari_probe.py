"""Frozen fixed-pool QAMPARI direct reading versus four-block map and exact union."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib.metadata
import importlib.util
import json
import os
import signal
import sys
import time
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path

import prepare_qampari_panel as panel
import probe

CONDITIONS = ("direct200", "map50")
SEEDS = (2026092187, 2026092188)
CONTEXT = 40960
CASES_SHA = "4dd4215394608958bd1e3a2096ae8f0c8a065eb726d78d1373e1557f6731e962"
METRIC = panel.REPO / "models/evaluation/reader_metrics.py"
METRIC_SHA = "d0d1e2beca6280b9e81be78bbdf2324f245a3dcc4233dc13fe92c8dfeac8ebaf"
METRIC_KEYS = ("precision", "recall", "f1", "rec_above", "f1_above")


def save(path, value):
    probe.runtime.save(Path(path), value)


def parse_answers(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    obj = json.loads(text, object_pairs_hook=unique)
    if (
        not isinstance(obj, dict)
        or set(obj) != {"answers"}
        or not isinstance(obj["answers"], list)
        or any(not isinstance(v, str) or not v.strip() for v in obj["answers"])
    ):
        raise ValueError("expected exact answers:list of nonempty strings")
    return obj["answers"]


@lru_cache(maxsize=1)
def official():
    if panel.sha(METRIC) != METRIC_SHA:
        raise ValueError("official grader changed")
    spec = importlib.util.spec_from_file_location("qampari_official_reader", METRIC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grade(answers, gold):
    if not gold or any(not a["aliases"] for a in gold):
        raise ValueError("empty reference entity/alias inventory")
    if not answers:
        return dict.fromkeys(METRIC_KEYS, 0.0)
    result = official().compute_metrics_qampari([{"predictions": answers, "answer_list": gold}])
    return {key: float(result[key]) for key in METRIC_KEYS}


def requests(case, job):
    docs, question = case["public"]["documents"], case["public"]["question"]
    if len(docs) != 200 or job["condition"] not in CONDITIONS:
        raise ValueError("requires fixed200 public passages and recognized condition")
    direct = job["condition"] == "direct200"
    blocks = [docs] if direct else [docs[i : i + 50] for i in range(0, 200, 50)]
    return [
        {
            "call_id": f"{job['episode_id']}-b{index}",
            "condition": job["condition"],
            "role": "direct" if direct else "map",
            "seed": job["seed"] + index,
            "cap": 1024 if direct else 256,
            "prompt": panel.make_prompt({"question": question, "documents": block}),
        }
        for index, block in enumerate(blocks)
    ]


class NativeClient:
    """Own frozen-base native path; no 8k/128-token cap inherited from previous tasks."""

    def __init__(self, model, tokenizer, output, deadline):
        self.model, self.tokenizer = model, tokenizer
        self.output, self.deadline = Path(output), deadline
        self.returned, self.failed = 0, 0
        self.model_manifest_sha = panel.sha(panel.MODEL / "local-research-manifest.json")
        self.plan_sha = (
            panel.sha(self.output / "PLAN.json") if (self.output / "PLAN.json").exists() else None
        )
        if hasattr(model, "peft_config") or any(p.requires_grad for p in model.parameters()):
            raise ValueError("requires frozen base only, no PEFT adapters or trainable parameters")

    def call(self, spec):
        import torch

        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": spec["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        expected = {"direct200": ("direct", 1024), "map50": ("map", 256)}
        if expected.get(spec["condition"]) != (spec["role"], spec["cap"]):
            raise ValueError("condition/role/generation cap mismatch")
        request = {
            "prompt": spec["prompt"],
            "input_token_ids": ids,
            "model": str(panel.MODEL),
            "model_manifest_sha256": self.model_manifest_sha,
            "adapter_enabled": False,
            "adapter_sha256": None,
            "condition": spec["condition"],
            "role": spec["role"],
            "seed": spec["seed"],
            "sampling": {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": spec["cap"],
                "max_time": 90.0,
                "do_sample": True,
            },
            "context_limit": CONTEXT,
            "truncation": False,
        }
        row = {
            "call_id": spec["call_id"],
            "request": request,
            "request_digest": probe.runtime.digest(request),
            "plan_sha256": self.plan_sha,
            "condition": spec["condition"],
            "role": spec["role"],
            "available": False,
            "text": None,
            "input_token_ids": ids,
            "usage": {},
            "started": time.time(),
        }
        save(self.output / "starts" / (spec["call_id"] + ".json"), row)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if remaining <= 0 or len(ids) + spec["cap"] > CONTEXT:
                raise ValueError("generation context/deadline exceeded; no truncation")
            torch.manual_seed(spec["seed"])
            if str(self.model.device).startswith("cuda"):
                torch.cuda.manual_seed_all(spec["seed"])
                torch.cuda.reset_peak_memory_stats()
            inputs = torch.tensor([ids], dtype=torch.long, device=self.model.device)
            row.update(effective_max_time=remaining, usage={"prompt_tokens": len(ids)})
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=True,
                    temperature=0.5,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=spec["cap"],
                    max_time=remaining,
                    use_cache=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id,
                )
            values = generated[0].detach().cpu().tolist()
            if values[: len(ids)] != ids or len(values) <= len(ids):
                raise ValueError("native continuation missing or prefix changed")
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
        if str(self.model.device).startswith("cuda"):
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        save(self.output / "calls" / (spec["call_id"] + ".json"), row)
        probe.campaign.snapshot(
            self.output / "STATUS.json",
            {
                "state": "scientific_calls",
                "returned": self.returned,
                "failed": self.failed,
                "last_call": spec["call_id"],
                "updated": time.time(),
            },
        )
        if self.returned == 1 and row["ended"] - row["started"] > 90:
            raise TimeoutError("first scientific response exceeded90seconds")
        return row


def execute(client, case, job, output, stopping=lambda: False):
    result = {
        **job,
        "question_type": case.get("question_type"),
        "call_ids": [],
        "request_digests": [],
        "blocks": [],
        "answers": None,
        "observed": False,
        "status": "missing",
        "metrics": dict.fromkeys(METRIC_KEYS, 0.0),
    }
    for request in requests(case, job):
        if stopping():
            result["status"] = "capped_unavailable"
            break
        call = client.call(request)
        result["call_ids"].append(call["call_id"])
        result["request_digests"].append(call["request_digest"])
        if not call["available"]:
            result["status"] = "inference_unavailable"
            break
        try:
            answers = parse_answers(call["text"])
            block = {"valid": True, "answers": answers}
        except (ValueError, TypeError) as exc:
            block = {"valid": False, "answers": None, "error": str(exc)}
        result["blocks"].append(block)
        probe.campaign.snapshot(Path(output) / ("PROGRESS-" + job["episode_id"] + ".json"), result)
    else:
        result["observed"] = True
        if all(b["valid"] for b in result["blocks"]):
            result["answers"] = list(
                dict.fromkeys(answer for b in result["blocks"] for answer in b["answers"])
            )
            result["metrics"] = grade(result["answers"], case["answer_list"])
            result["status"] = "scored"
        else:
            result["status"] = "protocol_invalid"
    save(Path(output) / "episodes" / (job["episode_id"] + ".json"), result)
    return result


def prepare(cases_path, output, hours):
    if not 0 < hours <= 1:
        raise ValueError("at most one cumulative hour")
    if panel.sha(cases_path) != CASES_SHA:
        raise ValueError("frozen case file changed")
    manifest_path = cases_path.with_name("MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["cases_sha256"] != CASES_SHA or manifest["selection_seed"] != 2026092186:
        raise ValueError("frozen panel manifest differs")
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if len(cases) != 16 or len({c["id"] for c in cases}) != 16:
        raise ValueError("requires frozen16-parent inventory")
    official()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        panel.MODEL, local_files_only=True, trust_remote_code=False
    )
    budgets = {b["id"]: b for b in manifest["prompt_budgets"]}
    jobs = []
    for index, case in enumerate(cases):
        if case["split"] != "development" or not case["answer_list"]:
            raise ValueError("nondevelopment or unscorable reference")
        grade([], case["answer_list"])
        for repeat, seed in enumerate(SEEDS):
            # Alternate arm order, independent of labels/outcomes.
            conditions = CONDITIONS if (index + repeat) % 2 == 0 else CONDITIONS[::-1]
            for condition in conditions:
                job = {
                    "episode_id": f"{case['id']}-r{repeat}-{condition}",
                    "case_id": case["id"],
                    "repeat": repeat,
                    "seed": seed,
                    "condition": condition,
                }
                for block, request in enumerate(requests(case, job)):
                    count = len(
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": request["prompt"]}],
                            tokenize=True,
                            return_dict=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                    expected = (
                        budgets[case["id"]]["direct_input_tokens"]
                        if (condition == "direct200")
                        else budgets[case["id"]]["map_input_tokens"][block]
                    )
                    if count != expected or count + request["cap"] > CONTEXT:
                        raise ValueError("exact prepared token budget changed or exceeds runtime")
                jobs.append(job)
    sources = [
        Path(__file__),
        Path(panel.__file__),
        Path(probe.__file__),
        Path(probe.campaign.__file__),
        Path(probe.runtime.__file__),
        Path(probe.runtime.__file__).with_name("runner.py"),
        METRIC,
    ]
    plan = {
        "schema": "qampari-native-fixed-pool-v1",
        "cases": str(cases_path),
        "cases_sha256": CASES_SHA,
        "manifest_sha256": panel.sha(manifest_path),
        "model": str(panel.MODEL),
        "model_manifest_sha256": panel.sha(panel.MODEL / "local-research-manifest.json"),
        "adapters": None,
        "training": False,
        "split": "development",
        "parents": 16,
        "repeats": 2,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "jobs": jobs,
        "planned_episodes": 64,
        "planned_calls": 160,
        "planned_per_condition": 32,
        "sampling": {"temperature": 0.5, "top_p": 1.0, "top_k": 0},
        "caps": {"direct200": 1024, "map50": 256},
        "context_limit": CONTEXT,
        "truncation": False,
        "budget_seconds": hours * 3600,
        "prompt_instruction": panel.INSTRUCTION,
        "prompt_budgets": manifest["prompt_budgets"],
        "merge": "first-occurrence exact-string union, applied to bothconditions",
        "metric": "official QAMPARI entity-set metric; empty prediction list explicit zero",
        "invalid_map_policy": "any returned-invalid block makes full arm zero; no partial union",
        "unavailable_policy": "missing block is unobserved; inference failure halts without retry",
        "source_sha256": {str(p.resolve()): panel.sha(p) for p in sources},
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "numpy", "pandas", "regex")
            },
        },
    }
    path = output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        save(path, plan)
    return plan, cases, tokenizer


def measured(calls):
    result = {
        "calls": len(calls),
        "failed_calls": sum(not c["available"] for c in calls),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "unknown_usage_calls": 0,
        "native_call_seconds": sum(c["ended"] - c["started"] for c in calls),
    }
    for call in calls:
        unknown = False
        for key in ("prompt_tokens", "completion_tokens"):
            value = call.get("usage", {}).get(key)
            if type(value) is int and value >= 0:
                result[key] += value
            else:
                unknown = True
        result["unknown_usage_calls"] += unknown
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    result["token_totals_are_lower_bounds"] = bool(result["unknown_usage_calls"])
    return result


def summarize(output, plan):
    episodes = {p.stem: json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")}
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    conditions = {}
    for condition in CONDITIONS:
        jobs = [j for j in plan["jobs"] if j["condition"] == condition]
        rows = [episodes[j["episode_id"]] for j in jobs if j["episode_id"] in episodes]
        conditions[condition] = {
            "planned": len(jobs),
            "recorded": len(rows),
            "observed": sum(r["observed"] for r in rows),
            "missing_or_unavailable": len(jobs) - sum(r["observed"] for r in rows),
            "valid": sum(r["status"] == "scored" for r in rows),
            "returned_protocol_invalid": sum(r["status"] == "protocol_invalid" for r in rows),
            "metrics": {k: sum(r["metrics"][k] for r in rows) / len(jobs) for k in METRIC_KEYS},
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "cost": measured([c for c in calls if c["condition"] == condition]),
        }
    lookup = {(e["case_id"], e["repeat"], e["condition"]): e for e in episodes.values()}
    comparisons = []
    for job in [j for j in plan["jobs"] if j["condition"] == "direct200"]:
        a, b = [lookup.get((job["case_id"], job["repeat"], c)) for c in CONDITIONS]
        if a and b and a["observed"] and b["observed"]:
            comparisons.append(
                {
                    "case_id": job["case_id"],
                    "repeat": job["repeat"],
                    "question_type": a["question_type"],
                    "both_valid": a["status"] == b["status"] == "scored",
                    "map_minus_direct": {k: b["metrics"][k] - a["metrics"][k] for k in METRIC_KEYS},
                }
            )
    returned_ids = {c["call_id"] for c in calls}
    starts = {p.stem for p in (output / "starts").glob("*.json")}
    return {
        "plan_sha256": panel.sha(output / "PLAN.json"),
        "conditions": conditions,
        "physical_cost": measured(calls),
        "planned_calls": plan["planned_calls"],
        "recorded_episodes": len(episodes),
        "planned_episodes": plan["planned_episodes"],
        "unresolved_starts": sorted(starts - returned_ids),
        "paired_observed": comparisons,
        "limits": "Fixed-pool package comparison, not adaptive retrieval or novelPIG. "
        "Missing is unobserved; primary planned metrics are lower bounds if incomplete. "
        "More calls, repeated question tokens, and partitioned generation limits differ.",
    }


def markdown(report):
    lines = [
        "# QAMPARI fixed-pool reading screen",
        "",
        "| Arm | Valid / planned | Precision | Recall | F1 | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in report["conditions"].items():
        m, cost = group["metrics"], group["cost"]
        lines.append(
            f"| {name} | {group['valid']}/{group['planned']} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {cost['calls']} | {cost['total_tokens']} |"
        )
        lines.append("")
    lines += [report["limits"], "", "No bootstrap or confirmatory claim in this native summary."]
    return "\n".join(lines) + "\n"


def run(args):
    output = args.output.resolve()
    plan, cases, tokenizer = prepare(args.cases.resolve(), output, args.hours)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "planned_calls": 160,
                    "planned_episodes": 64,
                    "model_loaded": False,
                    "context_limit": CONTEXT,
                }
            )
        )
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("existing attempt: no implicit retry/resume")
    import psutil
    import torch
    from transformers import AutoModelForCausalLM

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation")
    lock = (probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation, started = uuid.uuid4().hex[:12], time.time()
    model, failure, stopped = None, None, False
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
            str(panel.MODEL),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        save(
            output / f"LOAD-{invocation}.json",
            {
                "adapters_loaded": False,
                "optimizer_created": False,
                "trainable_parameters": 0,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "context_limit": CONTEXT,
            },
        )
        client = NativeClient(model, tokenizer, output, deadline)
        by_id = {c["id"]: c for c in cases}

        def stopping():
            return stopped or time.time() >= deadline - 5 or (output / "STOP").exists()

        for job in plan["jobs"]:
            if stopping():
                stopped = True
                break
            result = execute(client, by_id[job["case_id"]], job, output, stopping)
            probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan))
            if result["status"] == "inference_unavailable":
                raise RuntimeError("native inference failure; halt without retry")
            if result["status"] == "capped_unavailable":
                stopped = True
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        report = summarize(output, plan)
        probe.campaign.snapshot(output / "SUMMARY.json", report)
        save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
                "recorded_episodes": report["recorded_episodes"],
                "calls": report["physical_cost"]["calls"],
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
        save(output / "ANALYSIS.json", report)
        with (output / "ANALYSIS.md").open("x") as stream:
            stream.write(markdown(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
