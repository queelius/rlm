"""Independent32-call direct QAMPARI sampling-package audit against source028."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_qampari as prior
import qampari_sampling_control as control

original = control.original


def measured_including_unresolved(records):
    returned = [r for r in records if "ended" in r]
    unresolved = [r for r in records if "ended" not in r]
    cost = original.measured(returned)
    cost["calls"] += len(unresolved)
    cost["unresolved_calls"] = cost["unknown_latency_calls"] = len(unresolved)
    for record in unresolved:
        unknown = False
        for field in ("prompt_tokens", "completion_tokens"):
            value = record.get("usage", {}).get(field)
            if type(value) is int and value >= 0:
                cost[field] += value
            else:
                unknown = True
        cost["unknown_usage_calls"] += unknown
    cost["total_tokens"] = cost["prompt_tokens"] + cost["completion_tokens"]
    cost["token_totals_are_lower_bounds"] = bool(cost["unknown_usage_calls"])
    return cost


def audit(row, spec, plan, plan_sha, ids, tokenizer):
    expected = {
        "prompt": spec["prompt"],
        "input_token_ids": ids,
        "model": plan["model"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        "adapter_enabled": False,
        "adapter_sha256": None,
        "condition": "direct200",
        "role": "direct",
        "seed": spec["seed"],
        "sampling": {
            **control.SAMPLING,
            "max_new_tokens": 1024,
            "max_time": 90.0,
            "do_sample": True,
        },
        "context_limit": 40960,
        "truncation": False,
    }
    if row["request"] != expected or row["request_digest"] != original.probe.runtime.digest(
        expected
    ):
        raise ValueError("native request/sampling/prompt/IDs/seed differs")
    if (
        row["call_id"] != spec["call_id"]
        or row["plan_sha256"] != plan_sha
        or row["input_token_ids"] != ids
        or row["condition"] != "direct200"
        or row["role"] != "direct"
        or row["ended"] < row["started"]
    ):
        raise ValueError("native identity/PLAN/latency differs")
    if row.get("usage", {}).get("prompt_tokens", len(ids)) != len(ids):
        raise ValueError("input usage differs")
    if not row["available"]:
        return {"stop": "inference_unavailable", "output_tokens": None}
    tokens = row["output_token_ids"]
    if (
        not tokens
        or len(tokens) > 1024
        or row["usage"]["completion_tokens"] != len(tokens)
        or tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        != row["text"]
    ):
        raise ValueError("native output usage/decoding differs")
    eos = tokens[-1] == tokenizer.eos_token_id
    if row["finish_reason"] != ("eos" if eos else "length_or_time"):
        raise ValueError("native EOS differs")
    return {
        "stop": "eos" if eos else "at_output_cap" if len(tokens) == 1024 else "non_eos_before_cap",
        "output_tokens": len(tokens),
        "native_seconds": row["ended"] - row["started"],
    }


def change(old, new):
    observed = old["observed"] and new["observed"]
    return {
        "category": "unobserved_involved"
        if not observed
        else "both_valid"
        if old["valid"] and new["valid"]
        else "protocol_involved",
        "f1_delta": new["metrics"]["f1"] - old["metrics"]["f1"] if observed else None,
    }


def analyze(output, cases_path, baseline_report):
    output, cases_path, baseline_report = map(Path.resolve, (output, cases_path, baseline_report))
    hashes, cache = {}, {}

    def read(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in cache:
            raw = path.read_bytes()
            hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            cache[str(path)] = raw
        if expected is not None and hashes[str(path)] != expected:
            raise ValueError("consumed source/receipt changed: " + str(path))
        return json.loads(cache[str(path)])

    import psutil

    owners = list(output.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("one completed owner required")
    owner = read(owners[0])
    terminal = read(owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-")))
    try:
        process = psutil.Process(owner["pid"])
        if (
            abs(process.create_time() - owner["create_time"]) < 0.01
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("owner live; no partial readout")
    except psutil.NoSuchProcess:
        pass
    plan = read(output / "PLAN.json")
    plan_sha = hashes[str(output / "PLAN.json")]
    old = read(baseline_report)
    base_output = Path(plan["baseline_output"])
    base_plan = read(base_output / "PLAN.json", plan["baseline_plan_sha256"])
    if (
        old["source_output"] != str(base_output)
        or old["plan"] != base_plan
        or plan["schema"] != "qampari-direct-sampling-control-v1"
        or plan["sampling"] != control.SAMPLING
        or plan["planned_calls"] != 32
        or plan["caps"] != {"direct200": 1024}
        or plan["jobs"] != [j for j in base_plan["jobs"] if j["condition"] == "direct200"]
    ):
        raise ValueError("fixed sampling-package comparison differs")
    for key in (
        "seeds",
        "model",
        "model_manifest_sha256",
        "context_limit",
        "truncation",
        "split",
        "cases_sha256",
    ):
        if plan[key] != base_plan[key]:
            raise ValueError("baseline scientific contract differs: " + key)
    raw = cases_path.read_bytes()
    hashes[str(cases_path)] = hashlib.sha256(raw).hexdigest()
    if hashes[str(cases_path)] != original.CASES_SHA or plan["cases_sha256"] != original.CASES_SHA:
        raise ValueError("fixed16 cases changed")
    cases = [json.loads(line) for line in raw.splitlines()]
    manifest = read(cases_path.with_name("MANIFEST.json"), plan["manifest_sha256"])
    for path, digest in plan["source_sha256"].items():
        raw_source = Path(path).read_bytes()
        if hashlib.sha256(raw_source).hexdigest() != digest:
            raise ValueError("frozen source changed")
        hashes[path] = digest
    for module in (control, original, original.panel):
        match = [
            v
            for p, v in plan["source_sha256"].items()
            if Path(p).name == Path(module.__file__).name
        ]
        if match != [original.panel.sha(module.__file__)]:
            raise ValueError("analysis reconstruction differs from source036")
    model = Path(plan["model"])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        read(model / name, manifest["source_sha256"][str(model / name)])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    lookup = {c["id"]: c for c in cases}
    new_calls = {p.stem: read(p) for p in (output / "calls").glob("*.json")}
    starts = {p.stem: read(p) for p in (output / "starts").glob("*.json")}
    episodes = {p.stem: read(p) for p in (output / "episodes").glob("*.json")}
    jobs = plan["jobs"]
    expected_cids = {original.requests(lookup[j["case_id"]], j)[0]["call_id"] for j in jobs}
    if (
        set(new_calls) - expected_cids
        or set(starts) - expected_cids
        or set(episodes) - {j["episode_id"] for j in jobs}
    ):
        raise ValueError("unplanned calls/episodes")
    rows, audits, original_calls, linked = [], {}, [], set()
    old_rows = {
        (r["case_id"], r["repeat"]): r for r in old["rows"] if r["condition"] == "direct200"
    }
    for job in jobs:
        case = lookup[job["case_id"]]
        spec = original.requests(case, job)[0]
        cid = spec["call_id"]
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": spec["prompt"]}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        base_path = base_output / "calls" / (cid + ".json")
        base = read(base_path, plan["baseline_direct_calls_sha256"][str(base_path)])
        if old["input_source_receipt_sha256"][str(base_path)] != hashes[str(base_path)]:
            raise ValueError("baseline audit consumed different native call")
        prior.audit_call(base, spec, base_plan, plan["baseline_plan_sha256"], ids, tokenizer)
        original_calls.append(base)
        before = prior.score_native([base], 1, case["answer_list"])
        if any(
            before[k] != old_rows[case["id"], job["repeat"]][k]
            for k in ("metrics", "answers", "valid", "observed")
        ):
            raise ValueError("baseline official native regrade differs")
        call = new_calls.get(cid)
        if cid in starts:
            audit(
                {**starts[cid], "ended": starts[cid]["started"]},
                spec,
                plan,
                plan_sha,
                ids,
                tokenizer,
            )
        if call is not None:
            if cid not in starts or any(
                call[k] != starts[cid][k]
                for k in ("request", "request_digest", "started", "plan_sha256")
            ):
                raise ValueError("native start/return mismatch")
            audits[cid] = audit(call, spec, plan, plan_sha, ids, tokenizer)
        after = prior.score_native([call], 1, case["answer_list"])
        e = episodes.get(job["episode_id"])
        if e is not None:
            if any(e[k] != job[k] for k in job) or e["call_ids"] != ([cid] if call else []):
                raise ValueError("episode identity/reference differs")
            if e["request_digests"] != ([call["request_digest"]] if call else []):
                raise ValueError("episode digest differs")
            if any(e[k] != after[k] for k in ("answers", "metrics", "observed")):
                raise ValueError("collector score disagrees with official native regrade")
            linked.update(e["call_ids"])
        rows.append(
            {
                "case_id": case["id"],
                "repeat": job["repeat"],
                "question": case["public"]["question"],
                "question_type": case["question_type"],
                "baseline": before,
                "official_sampling": after,
                "call_id": cid,
                "episode_present": e is not None,
                **change(before, after),
            }
        )
    groups = {}
    for condition, records in (
        ("baseline", original_calls),
        ("official_sampling", list(new_calls.values())),
    ):
        groups[condition] = {
            "planned": 32,
            "observed": sum(r[condition]["observed"] for r in rows),
            "valid": sum(r[condition]["valid"] for r in rows),
            "protocol_invalid": sum(
                r[condition]["observed"] and not r[condition]["valid"] for r in rows
            ),
            "unknown": sum(not r[condition]["observed"] for r in rows),
            "metrics": {k: mean(r[condition]["metrics"][k] for r in rows) for k in prior.KEYS},
            "physical_cost": original.measured(records),
        }
    intervals = None
    if all(r["official_sampling"]["observed"] and r["baseline"]["observed"] for r in rows):
        intervals = {
            key: prior.parent_interval(
                {
                    c["id"]: [
                        r["official_sampling"]["metrics"][key] - r["baseline"]["metrics"][key]
                        for r in rows
                        if r["case_id"] == c["id"]
                    ]
                    for c in cases
                }
            )
            for key in prior.KEYS
        }
    unresolved = [v for cid, v in starts.items() if cid not in new_calls]
    for module in (prior, control, original):
        hashes[str(Path(module.__file__).resolve())] = original.panel.sha(module.__file__)
    hashes[str(Path(__file__).resolve())] = original.panel.sha(Path(__file__))
    return {
        "output": str(output),
        "baseline_report": str(baseline_report),
        "terminal": terminal,
        "plan": plan,
        "groups": groups,
        "paired_official_minus_baseline": intervals,
        "rows": rows,
        "change_categories": {
            category: {
                "wins": sum(r["category"] == category and (r["f1_delta"] or 0) > 0 for r in rows),
                "losses": sum(r["category"] == category and (r["f1_delta"] or 0) < 0 for r in rows),
            }
            for category in ("both_valid", "protocol_involved", "unobserved_involved")
        },
        "native_call_audits": audits,
        "native_stop_counts": dict(Counter(a["stop"] for a in audits.values())),
        "new_physical_cost_including_unresolved": measured_including_unresolved(
            list(new_calls.values()) + unresolved
        ),
        "unresolved_starts": sorted(set(starts) - set(new_calls)),
        "unlinked_calls": sorted(set(new_calls) - linked),
        "input_source_receipt_sha256": hashes,
        "method": {
            "parent_bootstrap_draws": 20000,
            "seed": prior.SEED,
            "parents": 16,
            "repeats": 2,
        },
        "limitations": [
            "Sampling-package qualification on exposed16 DEV parents; "
            "not independent confirmation.",
            "Temperature, top-p and top-k change together. Prompts/caps/seeds/evidence stay fixed.",
            "Strict JSON and official alias grader; empty lists explicitly zero; no repair.",
            "Missing outcomes remain unknown; planned metrics lower bounds if incomplete.",
            "Parent bootstrap retains both repeats; no atomic independence claim.",
        ],
    }


def markdown(report):
    lines = [
        "# QAMPARI direct sampling-package control",
        "",
        "All 16 fixed parents × two seeds; official sampling package versus original direct.",
        "",
        "| Condition | Valid / 32 | Unknown | Precision | Recall | F1 | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in report["groups"].items():
        m, cost = r["metrics"], r["physical_cost"]
        lines.append(
            f"| {name} | {r['valid']} | {r['unknown']} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {cost['calls']} | {cost['total_tokens']} |"
        )
    lines += [
        "",
        "Paired parent-bootstrap differences: "
        + json.dumps(report["paired_official_minus_baseline"]),
        "",
        "Protocol/content changes: " + json.dumps(report["change_categories"]),
        "",
        "Native stops: " + json.dumps(report["native_stop_counts"]),
        "",
        *["- " + x for x in report["limitations"]],
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "cases", "baseline-report", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    result = analyze(args.output, args.cases, args.baseline_report)
    original.save(args.report, result)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(markdown(result))
