"""Separate fixed32 QAMPARI direct-only official sampling-package qualification."""

import argparse
import fcntl
import gc
import json
import os
import signal
import time
import uuid
from collections import Counter
from pathlib import Path

import qampari_probe as original

panel, probe, save = original.panel, original.probe, original.save
CONTEXT = original.CONTEXT
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def prepare(cases_path, output, hours, baseline):
    if not 0 < hours <= 0.5:
        raise ValueError("maximum30minute owner")
    if panel.sha(cases_path) != original.CASES_SHA:
        raise ValueError("exact frozen028cases required")
    old = json.loads((baseline / "PLAN.json").read_text())
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if len(cases) != 16 or len({c["id"] for c in cases}) != 16:
        raise ValueError("all16 frozenparents required")
    jobs = [j for j in old["jobs"] if j["condition"] == "direct200"]
    expected = {
        (c["id"], repeat, seed) for c in cases for repeat, seed in enumerate(original.SEEDS)
    }
    if (
        len(jobs) != 32
        or {(j["case_id"], j["repeat"], j["seed"]) for j in jobs} != expected
        or old["model"] != str(panel.MODEL)
        or old["cases_sha256"] != original.CASES_SHA
        or old["context_limit"] != CONTEXT
        or old["caps"]["direct200"] != 1024
        or old["sampling"] != {"temperature": 0.5, "top_p": 1.0, "top_k": 0}
    ):
        raise ValueError("baseline scientific contract differs")
    original.official()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        panel.MODEL, local_files_only=True, trust_remote_code=False
    )
    lookup, hashes = {c["id"]: c for c in cases}, {}
    for job in jobs:
        spec = original.requests(lookup[job["case_id"]], job)[0]
        path = baseline / "calls" / (spec["call_id"] + ".json")
        call = json.loads(path.read_text())
        request = call["request"]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": spec["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if (
            not call["available"]
            or request["prompt"] != spec["prompt"]
            or request["seed"] != spec["seed"]
            or request["input_token_ids"] != ids
            or call["input_token_ids"] != ids
            or request["adapter_enabled"]
            or request["model"] != str(panel.MODEL)
            or request["truncation"]
            or request["context_limit"] != CONTEXT
            or request["sampling"]
            != {**old["sampling"], "max_new_tokens": 1024, "max_time": 90.0, "do_sample": True}
            or call["request_digest"] != probe.runtime.digest(request)
            or len(ids) + 1024 > CONTEXT
        ):
            raise ValueError("original direct native request mismatch")
        hashes[str(path)] = panel.sha(path)
    readme = panel.MODEL / "README.md"
    if "Temperature=0.7" not in readme.read_text() or "TopP=0.8" not in readme.read_text():
        raise ValueError("cached model recommendation changed")
    plan = {
        **old,
        "schema": "qampari-direct-sampling-control-v1",
        "cases": str(cases_path),
        "manifest_sha256": panel.sha(cases_path.with_name("MANIFEST.json")),
        "conditions": ["direct200"],
        "jobs": jobs,
        "planned_calls": 32,
        "planned_episodes": 32,
        "planned_per_condition": 32,
        "sampling": SAMPLING,
        "caps": {"direct200": 1024},
        "budget_seconds": hours * 3600,
        "baseline_output": str(baseline),
        "baseline_plan_sha256": panel.sha(baseline / "PLAN.json"),
        "baseline_direct_calls_sha256": hashes,
        "model_readme_sha256": panel.sha(readme),
        "intervention": "T.7/top_p.8/top_k20 package only; no penalty changes",
        "invalid_map_policy": None,
        "source_sha256": {
            str(p.resolve()): panel.sha(p)
            for p in (
                Path(__file__),
                Path(original.__file__),
                Path(panel.__file__),
                Path(probe.__file__),
                Path(probe.campaign.__file__),
                Path(probe.runtime.__file__),
                original.METRIC,
                readme,
            )
        },
    }
    if (output / "PLAN.json").exists():
        if json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        save(output / "PLAN.json", plan)
    return plan, cases, tokenizer


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
        expected = {"direct200": ("direct", 1024)}
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
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
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
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
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


def summarize(output, plan):
    episodes = {p.stem: json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")}
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    planned = {j["episode_id"] for j in plan["jobs"]}
    if set(episodes) - planned:
        raise ValueError("unplanned episode")
    rows = list(episodes.values())
    cost = original.measured(calls)
    starts = {p.stem for p in (output / "starts").glob("*.json")}
    return {
        "plan_sha256": panel.sha(output / "PLAN.json"),
        "planned_episodes": 32,
        "planned_calls": 32,
        "recorded_episodes": len(rows),
        "observed": sum(r["observed"] for r in rows),
        "missing_or_unavailable": 32 - sum(r["observed"] for r in rows),
        "valid": sum(r["status"] == "scored" for r in rows),
        "returned_protocol_invalid": sum(r["status"] == "protocol_invalid" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "metrics": {k: sum(r["metrics"][k] for r in rows) / 32 for k in original.METRIC_KEYS},
        "physical_cost": cost,
        "finish_reason_counts": dict(Counter(c.get("finish_reason") for c in calls)),
        "output_cap_hits": sum(c.get("usage", {}).get("completion_tokens") == 1024 for c in calls),
        "unresolved_starts": sorted(starts - {c["call_id"] for c in calls}),
        "limits": "Same exposed16parents/twoseeds; sampling-package qualification, not "
        "temperature-only causality or novel architecture. Missing is unobserved; "
        "planned metrics lower bounds if incomplete; no repair.",
    }


def complete(summary):
    return all(
        summary.get(k) == v
        for k, v in {
            "planned_episodes": 32,
            "recorded_episodes": 32,
            "planned_calls": 32,
            "observed": 32,
            "missing_or_unavailable": 0,
        }.items()
    ) and (
        summary.get("physical_cost", {}).get("calls") == 32
        and summary.get("physical_cost", {}).get("failed_calls") == 0
        and not summary.get("unresolved_starts")
    )


def markdown(report):
    return (
        "# QAMPARI direct sampling qualification\n\n"
        + json.dumps({k: v for k, v in report.items() if k != "plan_sha256"}, indent=2)
        + "\n"
    )


def run(args):
    output = args.output.resolve()
    plan, cases, tokenizer = prepare(
        args.cases.resolve(), output, args.hours, args.baseline.resolve()
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "planned_calls": 32,
                    "planned_episodes": 32,
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

        for index, job in enumerate(plan["jobs"]):
            if stopping():
                stopped = True
                break
            result = original.execute(client, by_id[job["case_id"]], job, output, stopping)
            if (index + 1) % 8 == 0:
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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
