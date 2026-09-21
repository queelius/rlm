"""Fixed step32 answer/sufficiency adapters on the unchanged64-variant DEV panel."""

import argparse
import fcntl
import gc
import json
import os
import signal
import time
import uuid
from pathlib import Path

import alfworld_probe as native
import prepare_sufficiency_panel as panel
import sufficiency_probe as baseline

prompt = baseline.prompt
parse_output = baseline.parse_output
summarize = baseline.summarize


def adapter_identity(adapter, arm):
    """Require the fixed committed endpoint, not a selected earlier checkpoint."""
    commit = json.loads((adapter / "COMMIT.json").read_text())
    state = json.loads((adapter / "STATE.json").read_text())
    training = json.loads((adapter.parent / "PLAN.json").read_text())
    if (
        commit["step"] != 32
        or state["step"] != 32
        or state["epoch"] != 1
        or state["cursor"] != 0
        or training.get("comparison_arm") != arm
        or training.get("role") != "sufficiency"
        or training.get("planned_updates") != 32
        or training.get("seed") != 2026092189
    ):
        raise ValueError("requires fixed completed sufficiency step32 for the declared arm")
    files = {
        name: panel.sha256(adapter / name)
        for name in ("STATE.json", "adapter_config.json", "adapter_model.safetensors")
    }
    if any(commit["files"].get(name) != sha for name, sha in files.items()):
        raise ValueError("checkpoint commit differs")
    if training["model_manifest_sha256"] != panel.sha256(
        native.evaluation.planner.BASE / "local-research-manifest.json"
    ):
        raise ValueError("training base differs")
    return {
        "path": str(adapter),
        "files": files,
        "commit_sha256": panel.sha256(adapter / "COMMIT.json"),
        "training_plan_sha256": panel.sha256(adapter.parent / "PLAN.json"),
        "arm": arm,
        "step": 32,
    }


def prepare(cases_path, output, hours, adapter, arm, baseline_output):
    if not 0 < hours <= 1 / 3 + 1e-9:
        raise ValueError("at most20minutes")
    original = json.loads((baseline_output / "PLAN.json").read_text())
    manifest_path = cases_path.with_name("MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    if panel.sha256(cases_path) != original["cases_sha256"] or (
        panel.sha256(cases_path) != manifest["cases_sha256"]
    ):
        raise ValueError("must reuse the exact baseline panel")
    if original["seeds"] != list(baseline.SEEDS) or original["planned_calls"] != 128:
        raise ValueError("baseline schedule differs")
    if original["prompt_instruction"] != baseline.INSTRUCTION:
        raise ValueError("baseline prompt differs")
    for path, sha in manifest["official_metric_sha256"].items():
        if panel.sha256(Path(path)) != sha:
            raise ValueError("official metric changed")
    cases = panel.read_jsonl(cases_path)
    expected = [
        {
            "episode_id": f"{c['id']}-{seed}",
            "case_id": c["id"],
            "parent_id": c["parent_id"],
            "seed": seed,
        }
        for c in cases
        for seed in baseline.SEEDS
    ]
    if original["jobs"] != expected or len(cases) != 64:
        raise ValueError("baseline inventory differs")
    # Check actual saved baseline requests, not only declarations.
    for job in expected:
        call = json.loads((baseline_output / "calls" / (job["episode_id"] + ".json")).read_text())
        case = next(c for c in cases if c["id"] == job["case_id"])
        req = call["request"]
        if (
            not call["available"]
            or req["prompt"] != prompt(case)
            or req["seed"] != job["seed"]
            or req["adapter_enabled"]
            or req["model"] != str(native.evaluation.planner.BASE)
            or req["sampling"]
            != {
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 0,
                "max_new_tokens": 128,
                "max_time": 90.0,
                "do_sample": True,
            }
            or call["request_digest"] != native.probe.runtime.digest(req)
        ):
            raise ValueError("baseline native request unavailable or incompatible")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        native.evaluation.planner.BASE, local_files_only=True, trust_remote_code=False
    )
    lengths = {
        c["id"]: len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(c)}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        for c in cases
    }
    if lengths != original["prompt_token_counts"] or max(lengths.values()) + 128 > 8192:
        raise ValueError("baseline tokenization differs; no truncation")
    identity = adapter_identity(adapter, arm)
    plan = {
        **original,
        "schema": "musique-sufficiency-fixed-adapter-v1",
        "cases": str(cases_path),
        "manifest_sha256": panel.sha256(manifest_path),
        "comparison_arm": arm,
        "adapters": identity,
        "budget_seconds": hours * 3600,
        "baseline_output": str(baseline_output),
        "baseline_plan_sha256": panel.sha256(baseline_output / "PLAN.json"),
        "baseline_calls_sha256": {
            p.name: panel.sha256(p) for p in sorted((baseline_output / "calls").glob("*.json"))
        },
        "source_sha256": {
            str(p.resolve()): panel.sha256(p)
            for p in (
                Path(__file__),
                Path(baseline.__file__),
                Path(panel.__file__),
                Path(native.__file__),
                Path(native.probe.__file__),
                Path(native.evaluation.__file__),
            )
        },
    }
    if (output / "PLAN.json").exists():
        if json.loads((output / "PLAN.json").read_text()) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        native.save(output / "PLAN.json", plan)
    return plan, cases, tokenizer


class AdapterClient(native.BaseClient):
    def __init__(self, model, tokenizer, output, deadline, identity):
        self.model, self.tokenizer, self.output, self.deadline = (
            model,
            tokenizer,
            Path(output),
            deadline,
        )
        self.identity = identity
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
            "model": str(native.evaluation.planner.BASE),
            "adapter_enabled": True,
            "adapter_sha256": self.identity["files"]["adapter_model.safetensors"],
            "adapter": self.identity,
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
            "request_digest": native.probe.runtime.digest(request),
            "condition": policy,
            "role": role,
            "available": False,
            "text": None,
            "input_token_ids": ids,
            "usage": {},
            "trimming": trimming,
            "started": time.time(),
        }
        native.save(self.output / "starts" / (identity + ".json"), row)
        try:
            remaining = min(90.0, self.deadline - time.time() - 1)
            if remaining <= 0 or len(ids) + cap > 8192 or not 1 <= cap <= 128:
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
        native.save(self.output / "calls" / (identity + ".json"), row)
        native.probe.campaign.snapshot(
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


def run(args):
    output = args.output.resolve()
    plan, cases, tokenizer = prepare(
        args.cases.resolve(),
        output,
        args.hours,
        args.adapter.resolve(),
        args.arm,
        args.baseline.resolve(),
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "planned_calls": 128,
                    "max_prompt_tokens": max(plan["prompt_token_counts"].values()),
                    "model_loaded": False,
                }
            )
        )
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("existing owner; no implicit retry or resume")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    lease = int(os.environ["SLURM_JOB_END_TIME"])
    deadline = min(time.time() + args.hours * 3600, lease - 600)
    if deadline < time.time() + 180:
        raise ValueError("insufficient allocation")
    lock = (native.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open(
        "a"
    )
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    invocation = uuid.uuid4().hex[:12]
    started, stopped, failure, model = time.time(), False, None, None
    try:
        native.save(
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
            raise RuntimeError("requires exactly one GPU assigned by main")
        torch.set_num_threads(4)
        model = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(model, args.adapter.resolve(), is_trainable=False)
        model.eval()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        native.save(
            output / f"LOAD-{invocation}.json",
            {
                "adapters_loaded": True,
                "adapter": plan["adapters"],
                "comparison_arm": args.arm,
                "optimizer_created": False,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        client = AdapterClient(model, tokenizer, output, deadline, plan["adapters"])
        lookup = {c["id"]: c for c in cases}
        for index, job in enumerate(plan["jobs"]):
            if stopped or time.time() >= deadline - 5 or (output / "STOP").exists():
                stopped = True
                break
            call = client.call(
                job["episode_id"],
                prompt(lookup[job["case_id"]]),
                args.arm,
                "sufficiency",
                job["seed"],
                128,
                {"truncation": False},
            )
            prediction, error = None, None
            if call["available"]:
                try:
                    prediction = parse_output(call["text"])
                except (ValueError, TypeError) as exc:
                    error = str(exc)
            native.save(
                output / "episodes" / (job["episode_id"] + ".json"),
                {
                    **job,
                    "call_id": call["call_id"],
                    "available": call["available"],
                    "prediction": prediction,
                    "protocol_error": error,
                    "request_digest": call["request_digest"],
                },
            )
            if (index + 1) % 16 == 0:
                native.probe.campaign.snapshot(
                    output / "SUMMARY.json", summarize(output, plan, cases)
                )
            if not call["available"]:
                raise RuntimeError("native inference failed; halt without retry")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        native.probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan, cases))
        native.save(
            output / f"TERMINAL-{invocation}.json",
            {
                "failure": failure,
                "stopped": stopped,
                "ended": time.time(),
                "elapsed_seconds": time.time() - started,
                "deadline": deadline,
            },
        )
        native.probe.campaign.snapshot(
            output / "STATUS.json",
            {"state": "failed" if failure else "finished_or_capped", "updated": time.time()},
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--arm", choices=("joint", "positive_only"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1 / 3)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
