"""Official Hotpot types on the completed exposed32 panel; no model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import audit_hotpot_dose
import probe
import score_hotpot

outcome = audit_hotpot_dose.outcome
POLICIES = ("base", "sft", "rl", "direct")
SEED = 2026092122
MIRROR_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"


def join_types(cases, mirror_rows):
    mirror = {r["id"]: r for r in mirror_rows}
    if len(mirror) != len(mirror_rows):
        raise ValueError("duplicate mirror source ID")
    if len({c["id"] for c in cases}) != len(cases) or len(
        {c["metadata"]["source_id"] for c in cases}
    ) != len(cases):
        raise ValueError("duplicate case/source ID")
    result = {}
    for case in cases:
        original = mirror.get(case["metadata"]["source_id"])
        if original is None or original["question"] != case["question"]:
            raise ValueError("source ID missing or exact question mismatch")
        if original["type"] not in ("bridge", "comparison"):
            raise ValueError("unknown official type")
        result[case["id"]] = original["type"]
    return result


def types(cases_path, parquet_path):
    import pyarrow.parquet as pq

    cases = [json.loads(line) for line in Path(cases_path).read_text().splitlines() if line]
    return join_types(
        cases, pq.read_table(parquet_path, columns=["id", "type", "question"]).to_pylist()
    )


def summarize(parents, values, *, draws):
    groups, comparisons = {}, {}
    for policy in POLICIES:
        rows = [values[p, r, policy] for p in parents for r in range(2)]
        groups[policy] = dict(
            planned=len(rows),
            correct=sum(r["em"] for r in rows),
            em=mean(r["em"] for r in rows),
            f1=mean(r["f1"] for r in rows),
            valid=sum(r["valid"] for r in rows),
            unobserved=sum(r["unobserved"] for r in rows),
            status_counts=dict(Counter(r["status"] for r in rows)),
        )
    for right in ("base", "rl", "direct"):
        pair = {"left": "sft", "right": right}
        for metric in ("em", "f1"):
            delta = {
                p: mean(values[p, r, right][metric] - values[p, r, "sft"][metric] for r in range(2))
                for p in parents
            }
            pair[metric] = analyze_helper.clustered_interval(
                delta, [[p] for p in parents], draws, SEED
            )
        for label, sign in (("wins", 1), ("losses", -1)):
            changes = []
            for p in parents:
                for repeat in range(2):
                    a, b = values[p, repeat, "sft"], values[p, repeat, right]
                    if b["em"] - a["em"] != sign:
                        continue
                    category = (
                        "unobserved_involved"
                        if a["unobserved"] or b["unobserved"]
                        else "both_valid"
                        if a["valid"] and b["valid"]
                        else "protocol_involved"
                    )
                    changes.append(
                        dict(
                            case_id=p,
                            repeat=repeat,
                            category=category,
                            sft_status=a["status"],
                            other_status=b["status"],
                        )
                    )
            pair[label] = dict(
                episodes=len(changes),
                parents=sorted({r["case_id"] for r in changes}),
                categories=dict(Counter(r["category"] for r in changes)),
                rows=changes,
            )
        comparisons[right + "_minus_sft"] = pair
    return dict(parents=len(parents), parent_ids=parents, groups=groups, comparisons=comparisons)


def analyze(root, mirror, *, draws=20000):
    hashes, cache = {}, {}

    def track(path, expected=None):
        path = Path(path).resolve()
        if str(path) not in hashes:
            hashes[str(path)] = score_hotpot.sha256(path)
        if expected is not None and hashes[str(path)] != expected:
            raise ValueError("source/receipt hash mismatch: " + str(path))
        return path

    def read(path, expected=None):
        path = track(path, expected)
        if str(path) not in cache:
            cache[str(path)] = json.loads(path.read_text())
        return cache[str(path)]

    acquisition = read(mirror.parent / "ACQUISITION.json")
    if acquisition["source"]["revision"] != MIRROR_REVISION:
        raise ValueError("wrong pinned HF mirror revision")
    track(mirror, acquisition["artifact"]["sha256"])
    cases_path = root / "hotpot-inputs-001/cases.jsonl"
    track(cases_path)
    cases = score_hotpot._load_cases(cases_path)
    labels = types(cases_path, mirror)
    if len(cases) != 32 or draws < 1:
        raise ValueError("fixed32 exposed Hotpot parents and positive draws required")
    reports = [
        read(root / name / "REPORT.json")
        for name in ("analysis-transfer-hotpot-001", "analysis-transfer-hotpot-direct-001")
    ]
    values, identities, first, sampling_seen = {}, {}, None, {}
    for report in reports:
        if report["sources"]["cases_sha256"] != hashes[str(cases_path.resolve())]:
            raise ValueError("official report cases differ")
        track(score_hotpot.EVALUATOR, report["scoring"]["evaluator_sha256"])
        for source in report["sources"]["evaluations"]:
            output = Path(source["evaluation"])
            plan = read(output / "PLAN.json", source["plan_sha256"])
            owners = list(output.glob("OWNER-*.json"))
            if not owners or {p.stem[6:] for p in owners} != {
                p.stem[9:] for p in output.glob("TERMINAL-*.json")
            }:
                raise ValueError("source owner is not completed")
            for owner in owners:
                terminal = read(owner.with_name(owner.name.replace("OWNER-", "TERMINAL-")))
                if terminal["failure"] or terminal["stopped"]:
                    raise ValueError("failed/stopped source needs separate accounting")
            first = first or plan
            for field in (
                "cases_sha256",
                "case_ids",
                "repeats",
                "seed",
                "temperature",
                "top_p",
                "top_k",
                "model",
                "model_manifest_sha256",
                "split",
            ):
                if plan.get(field) != first.get(field):
                    raise ValueError("paired panel differs: " + field)
            if (
                plan["repeats"] != 2
                or set(plan["case_ids"]) != set(cases)
                or plan["caps"]["final"] != 128
            ):
                raise ValueError("wrong fixed32 x2 panel/final cap")
            if plan.get("mode", "planner") not in ("planner", "direct"):
                raise ValueError("unexpected policy mode")
            if score_hotpot.tree_sha256(output / "episodes") != source["episodes_sha256"]:
                raise ValueError("episode inventory changed since official audit")
            indexed = {}
            for path in sorted((output / "episodes").glob("*.json")):
                row = read(path)
                key = row["case_id"], row["repeat"], row["condition"]
                if key in indexed:
                    raise ValueError("duplicate episode")
                indexed[key] = row
            if set(indexed) != {
                (p, r, c) for p in cases for r in range(2) for c in plan["conditions"]
            }:
                raise ValueError("incomplete planned episode inventory")
            for condition in plan["conditions"]:
                policy = "direct" if plan.get("mode") == "direct" else condition
                if policy in identities or policy not in POLICIES:
                    raise ValueError("duplicate or unknown policy")
                identities[policy] = dict(output=str(output), plan=plan, official_audit=source)
                for p in plan["case_ids"]:
                    for repeat in range(2):
                        episode = indexed[p, repeat, condition]
                        cid = score_hotpot._final_call_id(episode)
                        final = read(output / "calls" / (cid + ".json")) if cid else None
                        if final:
                            request = final["request"]
                            expected_seed = (
                                plan["seed"]
                                + int(probe.runtime.digest(p)[:6], 16)
                                + repeat * 100
                                + 2
                            )
                            if (
                                cid != episode["episode_id"] + "-final"
                                or final["call_id"] != cid
                                or final["role"] != "final"
                                or final["condition"] != condition
                                or probe.runtime.digest(request) != final["request_digest"]
                                or request["seed"] != expected_seed
                                or request["adapter_enabled"]
                            ):
                                raise ValueError("native final identity/seed/base model mismatch")
                            contract = {k: request[k] for k in ("seed", "sampling", "model")}
                            if (p, repeat) in sampling_seen and sampling_seen[
                                p, repeat
                            ] != contract:
                                raise ValueError("matched final sampling differs")
                            sampling_seen[p, repeat] = contract
                        values[p, repeat, policy] = outcome(
                            final, cases[p]["answer"], episode["status"]
                        )
                rows = [values[p, r, policy] for p in cases for r in range(2)]
                expected = report["conditions"][condition]
                if any(abs(mean(r[m] for r in rows) - expected[m]) > 1e-12 for m in ("em", "f1")):
                    raise ValueError("native official regrade disagrees with prior report")
                if sum(r["valid"] for r in rows) != expected["valid_final"]:
                    raise ValueError("protocol validity disagrees with official report")
    if set(identities) != set(POLICIES):
        raise ValueError("missing required policy")
    partitions = {
        "all32": list(cases),
        **{kind: [p for p in cases if labels[p] == kind] for kind in ("bridge", "comparison")},
    }
    results = {
        kind: summarize(parents, values, draws=draws) for kind, parents in partitions.items()
    }
    for path in (
        Path(__file__),
        Path(score_hotpot.__file__),
        Path(audit_hotpot_dose.__file__),
        Path(analyze_helper.__file__),
        Path(probe.__file__),
    ):
        track(path)
    return dict(
        method=dict(
            metric="official HotpotQA answer EM/F1, strict answer JSON",
            repeats=2,
            bootstrap="Paired parent repeat means; singleton-parent clusters; "
            "unadjusted interpolated percentile95",
            draws=draws,
            seed=SEED,
            join="literal source ID plus exact question, no fuzzy matching or type inference",
        ),
        mirror=acquisition,
        labels=labels,
        strata=results,
        source_identities=identities,
        input_source_consumed_receipt_sha256=hashes,
        per_attempt=[dict(case_id=p, repeat=r, policy=c, **v) for (p, r, c), v in values.items()],
        cautions=[
            "Already-exposed32 explorer parents, not a fresh confirmatory holdout. "
            "Repeats are not independent examples; type strata are small.",
            "Official type is not the topology of the model's generated plan. "
            "Subgroup associations are not causal type effects.",
            "Direct uses different prompts and fewer calls; its advantage is an "
            "end-to-end policy contrast, not a controlled decomposition effect.",
            "Invalid returned protocols score zero; unavailable outcomes stay explicit. "
            "No model weights rehashed; prior official audit identities are retained.",
        ],
    )


def markdown(report):
    lines = [
        "# Exposed Hotpot explorer: official type audit",
        "",
        "Exact source-ID and question join to the pinned HF mirror; official Hotpot answer metrics.",
        "",
    ]
    for kind, result in report["strata"].items():
        lines += [
            f"## {kind}: {result['parents']} parents × 2 repeats",
            "",
            "| Policy | Correct/planned | EM | F1 | Valid | Unobserved |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in result["groups"].items():
            lines.append(
                f"| {name} | {row['correct']:g}/{row['planned']} | {row['em']:.1%} | "
                f"{row['f1']:.4f} | {row['valid']} | {row['unobserved']} |"
            )
        lines += [""]
        for name, pair in result["comparisons"].items():
            intervals = "; ".join(
                f"{m.upper()} {pair[m]['estimate'] * 100:+.2f}pp "
                f"[{pair[m]['ci95'][0] * 100:+.2f}, {pair[m]['ci95'][1] * 100:+.2f}]"
                for m in ("em", "f1")
            )
            lines.append(
                f"- {name}: {intervals}. Wins {pair['wins']['episodes']} "
                f"{pair['wins']['categories']}; losses {pair['losses']['episodes']} "
                f"{pair['losses']['categories']}."
            )
        lines += [""]
    lines += [
        f"Bootstrap: {report['method']['draws']} paired-parent draws, seed {SEED}.",
        "",
        *["- " + note for note in report["cautions"]],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "mirror", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    md = args.report.with_suffix(".md")
    if args.report == md or args.report.exists() or md.exists():
        parser.error("choose unused immutable report paths")
    result = analyze(args.root.resolve(), args.mirror.resolve(), draws=args.draws)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with md.open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result))
