"""Independent native-receipt QAMPARI regrading and paired parent bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import qampari_probe as collector

SEED = 2026092190
KEYS = ("precision", "recall", "f1", "rec_above", "f1_above")


def score_native(calls, expected, gold):
    result = {
        "observed": False,
        "valid": False,
        "status": "missing_or_unavailable",
        "metrics": dict.fromkeys(KEYS, 0.0),
        "answers": None,
        "blocks": [],
    }
    if len(calls) != expected or any(c is None or not c["available"] for c in calls):
        return result
    result["observed"] = True
    for call in calls:
        try:
            answers = collector.parse_answers(call["text"])
            result["blocks"].append({"valid": True, "answers": answers})
        except (ValueError, TypeError):
            result["blocks"].append({"valid": False, "answers": None})
    if not all(b["valid"] for b in result["blocks"]):
        result["status"] = "protocol_invalid"
        return result
    answers = list(dict.fromkeys(a for b in result["blocks"] for a in b["answers"]))
    result.update(valid=True, status="scored", answers=answers)
    # Independently grade the native-text union, not collector episode scores.
    if answers:
        scores = collector.official().compute_metrics_qampari(
            [{"answer_list": gold, "predictions": answers}]
        )
        result["metrics"] = {k: float(scores[k]) for k in KEYS}
    return result


def parent_interval(differences, draws=20000, seed=SEED):
    if draws < 1 or not differences or any(len(v) != 2 for v in differences.values()):
        raise ValueError("requires two repeats for every parent")
    values = np.array([mean(differences[p]) for p in sorted(differences)], dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "parents": len(values),
        "repeats": 2,
    }


def audit_call(row, spec, plan, plan_sha, expected_ids, tokenizer):
    expected = {
        "prompt": spec["prompt"],
        "input_token_ids": expected_ids,
        "model": plan["model"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
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
        "context_limit": 40960,
        "truncation": False,
    }
    if row["request"] != expected or row["request_digest"] != collector.probe.runtime.digest(
        expected
    ):
        raise ValueError("native prompt/input IDs/seed/role/cap/model contract differs")
    if (
        row["call_id"] != spec["call_id"]
        or row["plan_sha256"] != plan_sha
        or row["condition"] != spec["condition"]
        or row["role"] != spec["role"]
        or row["input_token_ids"] != expected_ids
    ):
        raise ValueError("native identity or PLAN binding differs")
    if row["ended"] < row["started"]:
        raise ValueError("negative native latency")
    if "prompt_tokens" in row.get("usage", {}) and row["usage"]["prompt_tokens"] != len(
        expected_ids
    ):
        raise ValueError("input usage differs from native IDs")
    if not row["available"]:
        return {"stop": "inference_unavailable", "output_tokens": None, "protocol_valid": False}
    output = row["output_token_ids"]
    if not output or len(output) > spec["cap"] or row["usage"]["completion_tokens"] != len(output):
        raise ValueError("output IDs missing, over cap, or usage mismatch")
    decoded = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if decoded != row["text"]:
        raise ValueError("native decode differs from returned text")
    eos = output[-1] == tokenizer.eos_token_id
    if row["finish_reason"] != ("eos" if eos else "length_or_time"):
        raise ValueError("finish reason differs from native EOS")
    stop = "eos" if eos else "at_output_cap" if len(output) == spec["cap"] else "non_eos_before_cap"
    try:
        collector.parse_answers(decoded)
        valid = True
    except (ValueError, TypeError):
        valid = False
    return {
        "stop": stop,
        "output_tokens": len(output),
        "protocol_valid": valid,
        "elapsed_seconds": row["ended"] - row["started"],
        "effective_max_time": row.get("effective_max_time"),
        "time_budget_reached": row.get("effective_max_time") is not None
        and row["ended"] - row["started"] >= row["effective_max_time"],
    }


def literal_coverage(cases):
    rows = []
    for case in cases:
        docs = [(d["title"] + " " + d["text"]).casefold() for d in case["public"]["documents"]]
        entities = []
        for gold in case["answer_list"]:
            aliases = [a.casefold() for a in gold["aliases"] if a]
            indices = [i for i, doc in enumerate(docs) if any(alias in doc for alias in aliases)]
            entities.append(
                {
                    "answer_text": gold["answer_text"],
                    "document_ranks": indices,
                    "blocks": sorted({i // 50 for i in indices}),
                }
            )
        covered = sum(bool(e["document_ranks"]) for e in entities)
        rows.append(
            {
                "case_id": case["id"],
                "question_type": case["question_type"],
                "gold_entities": len(entities),
                "literal_covered_entities": covered,
                "literal_coverage_fraction": covered / len(entities),
                "entities": entities,
                "covered_entities_by_block": [
                    sum(b in e["blocks"] for e in entities) for b in range(4)
                ],
            }
        )
    return {
        "method": "Case-insensitive literal alias substring within each title+text; "
        "no cross-document concatenation, relation validation, or inference.",
        "warning": "Host-only diagnostic, NOT proof of supporting evidence, an oracle "
        "achievable ceiling, or an outcome-based selection/filter.",
        "macro_literal_coverage": mean(r["literal_coverage_fraction"] for r in rows),
        "total_gold_entities": sum(r["gold_entities"] for r in rows),
        "total_literal_covered_entities": sum(r["literal_covered_entities"] for r in rows),
        "rows": rows,
    }


def analyze(output, cases_path, draws=20000):
    output, cases_path = Path(output).resolve(), Path(cases_path).resolve()
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = str(Path(path).resolve())
        if path not in cache:
            cache[path] = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(cache[path]).hexdigest()
        if expected is not None and hashes[path] != expected:
            raise ValueError("consumed input/source changed: " + path)
        return cache[path]

    def read(path):
        return json.loads(track(path))

    owners = list(output.glob("OWNER-*.json"))
    if len(owners) != 1:
        raise ValueError("requires one finished native owner")
    owner = read(owners[0])
    terminal_path = owners[0].with_name(owners[0].name.replace("OWNER-", "TERMINAL-"))
    if not terminal_path.exists():
        raise ValueError("owner still unfinished; do not analyze partial outcomes")
    terminal = read(terminal_path)
    import psutil

    try:
        process = psutil.Process(owner["pid"])
        if abs(process.create_time() - owner["create_time"]) < 0.01 and (
            process.status() != psutil.STATUS_ZOMBIE
        ):
            raise ValueError("authenticated owner still live")
    except psutil.NoSuchProcess:
        pass
    plan = read(output / "PLAN.json")
    plan_sha = hashes[str(output / "PLAN.json")]
    expected_fields = {
        "schema": "qampari-native-fixed-pool-v1",
        "parents": 16,
        "repeats": 2,
        "planned_episodes": 64,
        "planned_calls": 160,
        "planned_per_condition": 32,
        "seeds": [2026092187, 2026092188],
        "conditions": ["direct200", "map50"],
        "adapters": None,
        "training": False,
        "context_limit": 40960,
        "truncation": False,
        "split": "development",
        "caps": {"direct200": 1024, "map50": 256},
        "sampling": {"temperature": 0.5, "top_p": 1.0, "top_k": 0},
    }
    if any(plan.get(k) != v for k, v in expected_fields.items()):
        raise ValueError("scientific panel contract changed")
    raw = track(cases_path, collector.CASES_SHA)
    if plan["cases_sha256"] != collector.CASES_SHA:
        raise ValueError("PLAN cases hash differs")
    cases = [json.loads(line) for line in raw.splitlines()]
    by_id = {c["id"]: c for c in cases}
    if len(cases) != 16 or len(by_id) != 16:
        raise ValueError("invalid frozen inventory")
    manifest = read(cases_path.with_name("MANIFEST.json"))
    track(cases_path.with_name("MANIFEST.json"), plan["manifest_sha256"])
    for path, digest in plan["source_sha256"].items():
        track(path, digest)
    for module in (collector, collector.panel):
        matches = [
            h
            for p, h in plan["source_sha256"].items()
            if Path(p).name == Path(module.__file__).name
        ]
        if len(matches) != 1 or collector.panel.sha(module.__file__) != matches[0]:
            raise ValueError("analysis reconstruction source differs from frozen collector")
    model = Path(plan["model"])
    track(model / "local-research-manifest.json", plan["model_manifest_sha256"])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        track(model / name, manifest["source_sha256"][str(model / name)])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    inventory = {
        (c["id"], r, condition)
        for c in cases
        for r in range(2)
        for condition in collector.CONDITIONS
    }
    jobs = {(j["case_id"], j["repeat"], j["condition"]): j for j in plan["jobs"]}
    if set(jobs) != inventory or len(plan["jobs"]) != 64:
        raise ValueError("job inventory differs")
    specs = {}
    for key, job in jobs.items():
        p, repeat, condition = key
        if (
            job["seed"] != collector.SEEDS[repeat]
            or job["episode_id"] != f"{p}-r{repeat}-{condition}"
        ):
            raise ValueError("logical episode identity/seed differs")
        for spec in collector.requests(by_id[p], job):
            specs[spec["call_id"]] = spec
    calls, starts, episodes, audited, token_cache = {}, {}, {}, {}, {}
    for directory, destination in (("calls", calls), ("starts", starts)):
        for path in sorted((output / directory).glob("*.json")):
            row = read(path)
            cid = row["call_id"]
            if cid != path.stem or cid not in specs or cid in destination:
                raise ValueError("unexpected/duplicate physical call")
            spec = specs[cid]
            prompt = spec["prompt"]
            if prompt not in token_cache:
                token_cache[prompt] = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            if directory == "calls":
                audited[cid] = audit_call(row, spec, plan, plan_sha, token_cache[prompt], tokenizer)
            else:
                # A synthetic zero duration audits only the saved start's request/identity;
                # it is never included in measured returned-call cost or outcome totals.
                audit_call(
                    {**row, "ended": row["started"]},
                    spec,
                    plan,
                    plan_sha,
                    token_cache[prompt],
                    tokenizer,
                )
            destination[cid] = row
    for cid, call in calls.items():
        if cid not in starts or any(
            call[k] != starts[cid][k]
            for k in ("request", "request_digest", "started", "plan_sha256")
        ):
            raise ValueError("native call lacks matching start")
    for path in sorted((output / "episodes").glob("*.json")):
        e = read(path)
        key = e["case_id"], e["repeat"], e["condition"]
        if key not in inventory or key in episodes or path.stem != jobs[key]["episode_id"]:
            raise ValueError("unexpected/duplicate episode")
        episodes[key] = e
    rows, linked = [], set()
    for key in sorted(inventory):
        parent, repeat, condition = key
        expected = collector.requests(by_id[parent], jobs[key])
        native = [calls.get(s["call_id"]) for s in expected]
        result = score_native(native, len(expected), by_id[parent]["answer_list"])
        e = episodes.get(key)
        if e:
            if (
                e["call_ids"] != [s["call_id"] for s in expected][: len(e["call_ids"])]
                or len(e["call_ids"]) > len(expected)
                or any(cid not in calls for cid in e["call_ids"])
            ):
                raise ValueError("episode physical references differ")
            linked.update(e["call_ids"])
            if e["request_digests"] != [calls[cid]["request_digest"] for cid in e["call_ids"]]:
                raise ValueError("episode request references differ")
            if e["observed"] != result["observed"] or e["answers"] != result["answers"]:
                raise ValueError("episode observation/union disagrees with native regrade")
            if result["observed"] and e["status"] != result["status"]:
                raise ValueError("episode protocol classification differs")
            if any(abs(e["metrics"][k] - result["metrics"][k]) > 1e-12 for k in KEYS):
                raise ValueError("episode score disagrees with official native regrade")
        rows.append(
            {
                "case_id": parent,
                "repeat": repeat,
                "condition": condition,
                "question_type": by_id[parent]["question_type"],
                "question": by_id[parent]["public"]["question"],
                "episode_receipt_present": e is not None,
                **result,
                "call_ids": [s["call_id"] for s in expected if s["call_id"] in calls],
            }
        )
    groups = {}
    for condition in collector.CONDITIONS:
        current = [r for r in rows if r["condition"] == condition]
        selected_calls = [c for c in calls.values() if c["condition"] == condition]
        groups[condition] = {
            "planned": 32,
            "observed": sum(r["observed"] for r in current),
            "unobserved": sum(not r["observed"] for r in current),
            "protocol_invalid": sum(r["observed"] and not r["valid"] for r in current),
            "valid": sum(r["valid"] for r in current),
            "metrics": {k: mean(r["metrics"][k] for r in current) for k in KEYS},
            "physical_cost": collector.measured(selected_calls),
            "native_stop_counts": dict(
                Counter(audited[c["call_id"]]["stop"] for c in selected_calls)
            ),
            "invalid_native_stop_counts": dict(
                Counter(
                    audited[c["call_id"]]["stop"]
                    for c in selected_calls
                    if not audited[c["call_id"]]["protocol_valid"]
                )
            ),
            "output_token_lengths": [
                audited[c["call_id"]]["output_tokens"] for c in selected_calls
            ],
            "native_time_budget_reached": sum(
                audited[c["call_id"]].get("time_budget_reached", False) for c in selected_calls
            ),
        }
    lookup = {(r["case_id"], r["repeat"], r["condition"]): r for r in rows}
    differences, changes = {}, []
    for metric in KEYS:
        delta = {
            c["id"]: [
                lookup[c["id"], rep, "map50"]["metrics"][metric]
                - lookup[c["id"], rep, "direct200"]["metrics"][metric]
                for rep in range(2)
            ]
            for c in cases
        }
        differences[metric] = parent_interval(delta, draws, SEED)
    for case in cases:
        for repeat in range(2):
            a, b = [lookup[case["id"], repeat, c] for c in collector.CONDITIONS]
            delta = b["metrics"]["f1"] - a["metrics"]["f1"]
            if abs(delta) < 1e-12:
                continue
            category = (
                "unobserved_involved"
                if not (a["observed"] and b["observed"])
                else ("both_valid" if a["valid"] and b["valid"] else "protocol_involved")
            )
            changes.append(
                {
                    "case_id": case["id"],
                    "repeat": repeat,
                    "question_type": case["question_type"],
                    "f1_delta": delta,
                    "category": category,
                }
            )
    types = {}
    for kind in sorted({c["question_type"] for c in cases}):
        types[kind] = {}
        for condition in collector.CONDITIONS:
            subset = [r for r in rows if r["condition"] == condition and r["question_type"] == kind]
            types[kind][condition] = {
                "planned": len(subset),
                "observed": sum(r["observed"] for r in subset),
                "valid": sum(r["valid"] for r in subset),
                "metrics": {k: mean(r["metrics"][k] for r in subset) for k in KEYS},
            }
    for path in (Path(__file__), Path(collector.__file__), Path(collector.panel.__file__)):
        track(path)
    return {
        "method": {
            "parents": 16,
            "repeats": 2,
            "bootstrap_draws": draws,
            "seed": SEED,
            "bootstrap_unit": "parent, retaining both repeats; NOT atomic-component independence",
            "metric": "official QAMPARI entity-set precision/recall/F1; explicit empty-list zero",
        },
        "source_output": str(output),
        "terminal": terminal,
        "plan": plan,
        "groups": groups,
        "paired_map_minus_direct": differences,
        "f1_changes": changes,
        "change_categories": {
            direction: dict(
                Counter(
                    c["category"] for c in changes if (c["f1_delta"] > 0) == (direction == "wins")
                )
            )
            for direction in ("wins", "losses")
        },
        "f1_delta_contributions_planned_denominator": {
            category: sum(c["f1_delta"] for c in changes if c["category"] == category) / 32
            for category in ("both_valid", "protocol_involved", "unobserved_involved")
        },
        "type_breakdown_descriptive": types,
        "rows": rows,
        "literal_alias_coverage": literal_coverage(cases),
        "native_call_audits": audited,
        "physical_cost": collector.measured(list(calls.values())),
        "coverage": {
            "planned_calls": 160,
            "started_calls": len(starts),
            "returned_receipts": len(calls),
            "planned_episodes": 64,
            "episode_receipts": len(episodes),
            "unresolved_starts": sorted(set(starts) - set(calls)),
            "unlinked_native_calls": sorted(set(calls) - linked),
        },
        "incomplete_difference_is_not_effect_estimate": any(not r["observed"] for r in rows),
        "input_source_receipt_sha256": hashes,
        "analysis_environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
        },
        "limitations": [
            "Exploratory 16-parent natural-type packaged DEV panel; strata are descriptive.",
            "Parent bootstrap does not establish independent entities, documents, or atomic facts.",
            "Same fixed 200 retrieved passages; not adaptive retrieval or a new PIG algorithm.",
            "Calls, repeated-question tokens, and output-budget partitioning differ.",
            "Union cannot join facts that are individually insufficient across blocks.",
            "Non-EOS before cap is a possible time stop, not proof of its cause.",
            "Protocol-invalid outputs are observed zeros; missing outcomes remain unobserved.",
        ],
    }


def markdown(report):
    lines = [
        "# QAMPARI: fixed retrieved-pool reading",
        "",
        "16 development parents × two repeats. Parent bootstrap preserves both repeats.",
        "",
        "| Arm | Valid / planned | Precision | Recall | F1 | Calls | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, g in report["groups"].items():
        m, cost = g["metrics"], g["physical_cost"]
        lines.append(
            f"| {name} | {g['valid']}/32 | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {cost['calls']} | {cost['total_tokens']} |"
        )
    lines += ["", "Map minus direct, paired parent-bootstrap95% intervals:", ""]
    for key, row in report["paired_map_minus_direct"].items():
        lines.append(
            f"- {key}: {row['estimate']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]"
        )
    lines += ["", f"F1 win/loss categories: {report['change_categories']}", ""]
    for name, group in report["groups"].items():
        lines.append(
            f"{name}: {group['unobserved']} unobserved; {group['protocol_invalid']} "
            f"protocol-invalid policy attempts; native stops {group['native_stop_counts']}."
        )
    lines += [
        "",
        "Type breakdown (descriptive):",
        "",
        "| Type | Parents | Direct F1 | Map F1 |",
        "|---|---:|---:|---:|",
    ]
    for kind, values in report["type_breakdown_descriptive"].items():
        lines.append(
            f"| {kind} | {values['direct200']['planned'] // 2} | "
            f"{values['direct200']['metrics']['f1']:.4f} | {values['map50']['metrics']['f1']:.4f} |"
        )
    coverage = report["literal_alias_coverage"]
    lines += [
        "",
        f"Host-only literal alias coverage: {coverage['total_literal_covered_entities']}/"
        f"{coverage['total_gold_entities']} entities; "
        f"parent-macro {coverage['macro_literal_coverage']:.4f}.",
        coverage["warning"],
        "",
        *report["limitations"],
    ]
    if report["incomplete_difference_is_not_effect_estimate"]:
        lines += [
            "",
            "INCOMPLETE: planned-denominator metrics are lower bounds, not an effect estimate.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report destination already exists")
    if args.coverage_only:
        if collector.panel.sha(args.cases) != collector.CASES_SHA:
            raise ValueError("frozen cases changed")
        cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
        report = {
            "literal_alias_coverage": literal_coverage(cases),
            "cases_sha256": collector.CASES_SHA,
            "cases": str(args.cases.resolve()),
            "source_sha256": collector.panel.sha(Path(__file__)),
            "model_outcomes_inspected": False,
        }
        md = (
            "# QAMPARI host-only literal alias audit\n\n"
            + report["literal_alias_coverage"]["warning"]
            + "\n"
        )
    else:
        if args.output is None:
            parser.error("--output required for completed native analysis")
        report = analyze(args.output, args.cases, args.draws)
        md = markdown(report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(md)
