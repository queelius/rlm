"""Full-context frozen-base MuSiQue answer-sufficiency baseline; no helpers or adapters."""

import argparse
import fcntl
import gc
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import alfworld_probe as native
import prepare_sufficiency_panel as panel

SEEDS = (2026092181, 2026092182)
INSTRUCTION = (
    "Answer only if the supplied documents support the entire question. Do not answer from "
    "memory or unsupported inference. Return ONLY an exact JSON object with two fields: "
    '"answerable" (a boolean) and "answer" (a string). If the documents are insufficient, '
    "set answerable to false and answer to an empty string. Otherwise set answerable to true "
    "and provide only the short entity, number, date, or phrase answering the question. "
    "Keep necessary qualifiers. No explanation or additional fields.\n"
)


def prompt(case):
    public = case["public"]
    clean = {
        "question": public["question"],
        "documents": [
            {k: doc[k] for k in ("docid", "title", "text")} for doc in public["documents"]
        ],
    }
    return INSTRUCTION + json.dumps(clean, ensure_ascii=False)


def parse_output(text):
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
        or set(obj) != {"answerable", "answer"}
        or type(obj["answerable"]) is not bool
        or not isinstance(obj["answer"], str)
    ):
        raise ValueError("expected exact answerable:boolean, answer:string JSON")
    return obj


def official_metrics():
    if str(panel.OFFICIAL) not in sys.path:
        sys.path.insert(0, str(panel.OFFICIAL))
    from metrics.answer import AnswerMetric
    from metrics.group_answer_sufficiency import GroupAnswerSufficiencyMetric

    return AnswerMetric, GroupAnswerSufficiencyMetric


def group_score(positive, predictions):
    if any(p is None for p in predictions):
        return {"em": 0.0, "f1": 0.0, "suff": 0.0}
    _, metric_class = official_metrics()
    metric = metric_class()
    gold = [positive["answer"], *positive.get("answer_aliases", [])]
    for label, predicted in zip((True, False), predictions, strict=True):
        metric(
            predicted["answer"],
            gold if label else ["UNUSED"],
            predicted["answerable"],
            label,
            "pair",
        )
    return metric.get_metric(reset=True)


def prepare(cases_path, output, hours):
    if not 0 < hours <= 1 / 3 + 1e-9:
        raise ValueError("at most20minutes")
    manifest_path = cases_path.with_name("MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    if panel.sha256(cases_path) != manifest["cases_sha256"]:
        raise ValueError("panel cases changed")
    for path, sha in manifest["official_metric_sha256"].items():
        if panel.sha256(Path(path)) != sha:
            raise ValueError("official metric changed")
    cases = panel.read_jsonl(cases_path)
    if len(cases) != 64 or len({c["parent_id"] for c in cases}) != 32:
        raise ValueError("expected32complete pairs")
    for parent in {c["parent_id"] for c in cases}:
        pair = [c for c in cases if c["parent_id"] == parent]
        if len(pair) != 2 or {c["answerable"] for c in pair} != {True, False}:
            raise ValueError("malformed panel pair")
    from transformers import AutoTokenizer

    model = native.evaluation.planner.BASE
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
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
    if max(lengths.values()) + 128 > 8192:
        raise ValueError("frozen panel exceeds context; no truncation/reselection")
    jobs = [
        {
            "episode_id": f"{c['id']}-{seed}",
            "case_id": c["id"],
            "parent_id": c["parent_id"],
            "seed": seed,
        }
        for c in cases
        for seed in SEEDS
    ]
    plan = {
        "schema": "musique-sufficiency-native-v1",
        "cases": str(cases_path),
        "cases_sha256": panel.sha256(cases_path),
        "manifest_sha256": panel.sha256(manifest_path),
        "planned_calls": 128,
        "planned_groups": 64,
        "seeds": list(SEEDS),
        "jobs": jobs,
        "model": str(model),
        "model_manifest_sha256": panel.sha256(model / "local-research-manifest.json"),
        "adapters": None,
        "prompt_instruction": INSTRUCTION,
        "temperature": 0.5,
        "top_k": 0,
        "top_p": 1.0,
        "max_new_tokens": 128,
        "context_limit": 8192,
        "truncation": False,
        "prompt_token_counts": lengths,
        "budget_seconds": hours * 3600,
        "source_sha256": {
            str(p.resolve()): panel.sha256(p)
            for p in (
                Path(__file__),
                Path(panel.__file__),
                Path(native.__file__),
                Path(native.probe.__file__),
                Path(native.evaluation.__file__),
            )
        },
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        },
    }
    path = output / "PLAN.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("immutable PLAN differs")
    else:
        native.save(path, plan)
    return plan, cases, tokenizer


def summarize(output, plan, cases):
    calls = [json.loads(p.read_text()) for p in (output / "calls").glob("*.json")]
    episodes = {p.stem: json.loads(p.read_text()) for p in (output / "episodes").glob("*.json")}
    groups, supported, false_abstention, overanswer = [], [], 0, 0
    valid, invalid, unavailable = 0, 0, 0
    for job in plan["jobs"]:
        e = episodes.get(job["episode_id"])
        if e and e["available"]:
            valid += e["prediction"] is not None
            invalid += e["prediction"] is None
        else:
            unavailable += 1
    answer_class, _ = official_metrics()
    for parent in sorted({c["parent_id"] for c in cases}):
        pair = sorted(
            [c for c in cases if c["parent_id"] == parent], key=lambda c: not c["answerable"]
        )
        for seed in SEEDS:
            records = [episodes.get(f"{c['id']}-{seed}") for c in pair]
            predictions = [r.get("prediction") if r else None for r in records]
            score = group_score(pair[0], predictions)
            groups.append(
                {
                    "parent_id": parent,
                    "seed": seed,
                    **score,
                    "both_available": all(r and r["available"] for r in records),
                    "both_valid": all(p is not None for p in predictions),
                }
            )
            metric = answer_class()
            if predictions[0] is not None:
                metric(predictions[0]["answer"], [pair[0]["answer"], *pair[0]["answer_aliases"]])
                false_abstention += not predictions[0]["answerable"]
            supported.append(metric.get_metric(reset=True))
            if predictions[1] is not None:
                overanswer += predictions[1]["answerable"]
    return {
        "plan_sha256": panel.sha256(output / "PLAN.json"),
        "groups": groups,
        "planned_variant_attempts": 128,
        "planned_pair_attempts": 64,
        "returned_valid": valid,
        "returned_protocol_invalid": invalid,
        "missing_or_inference_unavailable": unavailable,
        "group_answer_sufficiency": {
            k: sum(g[k] for g in groups) / 64 for k in ("em", "f1", "suff")
        },
        "supported_answer_em": sum(s[0] for s in supported) / 64,
        "supported_answer_f1": sum(s[1] for s in supported) / 64,
        "false_abstentions": false_abstention,
        "label_false_overanswers": overanswer,
        "each_label_planned_denominator": 64,
        "physical_cost": native.evaluation.cost(calls),
        "native_call_seconds": sum(c["ended"] - c["started"] for c in calls),
        "missing_note": "Missing is unobserved, not scientific failure; planned metrics "
        "are lower bounds when unavailable. Returned malformed JSON receives zero.",
    }


def run(args):
    output = args.output.resolve()
    plan, cases, tokenizer = prepare(args.cases.resolve(), output, args.hours)
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
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        native.save(
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
        client = native.BaseClient(model, tokenizer, output, deadline)
        lookup = {c["id"]: c for c in cases}
        for job in plan["jobs"]:
            if stopped or time.time() >= deadline - 5 or (output / "STOP").exists():
                stopped = True
                break
            call = client.call(
                job["episode_id"],
                prompt(lookup[job["case_id"]]),
                "full_context",
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
            native.probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, plan, cases))
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
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=1 / 3)
    parser.add_argument(
        "--prepare-only", "--validate-only", dest="prepare_only", action="store_true"
    )
    run(parser.parse_args())
