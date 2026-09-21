"""Secondary fresh003 strata from completed, previously native-audited policy reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import analyze_helper
import probe

CASES_SHA = "6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153"
SEED = 2026092121
CONTRASTS = (
    ("planner_sft", "planner_rl"),
    ("direct_base_control", "direct_helper_sft"),
    ("direct_base", "planner_sft"),
    ("direct_base", "planner_rl"),
    ("direct_helper_sft", "planner_rl"),
    ("direct_base", "direct_base_control"),
)


def strata(cases, exposure):
    parents = [c["id"] for c in cases]
    details = exposure["parents_detail"]
    by_id = {r["parent_id"]: r for r in details}
    if (
        len(parents) != 64
        or len(set(parents)) != 64
        or len(details) != 64
        or set(by_id) != set(parents)
    ):
        raise ValueError("frozen exposure/case inventory differs")
    result = {"all64": parents}
    for name, hops in (("two_hop", 2), ("three_hop", 3)):
        result[name] = [c["id"] for c in cases if c["metadata"]["hops"] == hops]
    for name, match in (
        ("exact_train_document_overlap", True),
        ("no_exact_train_document_overlap", False),
    ):
        result[name] = [
            p for p in parents if bool(by_id[p]["exact_title_text"]["match_occurrences"]) == match
        ]
    if [len(v) for v in result.values()] != [64, 32, 32, 14, 50]:
        raise ValueError("not the predeclared fresh003 hop/exposure strata")
    return result


def restrict_clusters(clusters, parents):
    selected = set(parents)
    return [members for cluster in clusters if (members := [p for p in cluster if p in selected])]


def summarize_stratum(parents, values, policies, clusters, *, draws, seed):
    groups, comparisons = {}, {}
    denominator = len(parents) * 2
    for policy in policies:
        rows = [values[p, repeat, policy] for p in parents for repeat in range(2)]
        missing = sum(r["missing"] for r in rows)
        groups[policy] = dict(
            planned=denominator,
            correct=sum(r["em"] for r in rows),
            em=mean(r["em"] for r in rows),
            f1=mean(r["f1"] for r in rows),
            valid=sum(r["valid"] for r in rows),
            missing=missing,
            em_upper_bound=(sum(r["em"] for r in rows) + missing) / denominator,
            f1_upper_bound=(sum(r["f1"] for r in rows) + missing) / denominator,
            status_counts=dict(Counter(r["status"] for r in rows)),
        )
    for left, right in CONTRASTS:
        if left not in policies or right not in policies:
            continue
        pair = dict(
            left=left,
            right=right,
            observed_wins=Counter(),
            observed_losses=Counter(),
            unobserved_pairs=0,
        )
        for metric in ("em", "f1"):
            differences = {
                p: mean(values[p, r, right][metric] - values[p, r, left][metric] for r in range(2))
                for p in parents
            }
            pair[metric] = analyze_helper.clustered_interval(differences, clusters, draws, seed)
        for p in parents:
            for r in range(2):
                a, b = values[p, r, left], values[p, r, right]
                if a["missing"] or b["missing"]:
                    pair["unobserved_pairs"] += 1
                elif a["em"] != b["em"]:
                    direction = "observed_wins" if b["em"] > a["em"] else "observed_losses"
                    pair[direction][
                        "both_valid" if a["valid"] and b["valid"] else "protocol_involved"
                    ] += 1
        pair["incomplete"] = bool(pair["unobserved_pairs"])
        pair["interval_interpretation"] = (
            "Difference of missing-as-zero lower bounds, not a bound on treatment difference"
            if pair["incomplete"]
            else "Paired complete-panel metric difference"
        )
        comparisons[right + "_minus_" + left] = pair
    return dict(
        parent_ids=parents,
        parents=len(parents),
        component_clusters=clusters,
        groups=groups,
        contrasts=comparisons,
    )


def analyze(
    policy_report_path, adapted_report_path, exposure_path, cases_path, *, draws=20000, seed=SEED
):
    hashes, cache = {}, {}

    def read(path, audited=None):
        path = str(Path(path).resolve())
        if path not in cache:
            data = Path(path).read_bytes()
            hashes[path] = hashlib.sha256(data).hexdigest()
            cache[path] = json.loads(data)
        if audited is not None and audited.get(path) != hashes[path]:
            raise ValueError("receipt absent from or changed since native audit: " + path)
        return cache[path]

    policy_report, adapted_report, exposure = [
        read(p) for p in (policy_report_path, adapted_report_path, exposure_path)
    ]
    if (
        "MuSiQue" not in policy_report["method"]["metric"]
        or adapted_report["method"]["metric"] != "official_musique_alias_max_em_f1"
        or any(g.get("unlinked_call_ids") for g in policy_report["groups"].values())
    ):
        raise ValueError("expected complete official MuSiQue native audits without unlinked calls")
    raw = Path(cases_path).read_bytes()
    cases_hash = hashlib.sha256(raw).hexdigest()
    if (
        cases_hash != CASES_SHA
        or exposure["sources_sha256"]["fresh-dev-inputs-003/cases.jsonl"] != cases_hash
    ):
        raise ValueError("expected frozen fresh003 cases and corresponding exposure receipt")
    hashes[str(Path(cases_path).resolve())] = cases_hash
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    partitions = strata(cases, exposure["panels"]["freshdev64"])
    parents, by_id = partitions["all64"], {c["id"]: c for c in cases}
    full_clusters = analyze_helper.component_clusters(cases)
    if any(c.get("dataset", "musique") != "musique" or c["split"] != "development" for c in cases):
        raise ValueError("MuSiQue development only")
    if draws < 1:
        raise ValueError("positive bootstrap draw count required")
    sources = [
        (
            p["output"],
            {k: v for k, v in p.items() if k != "output"},
            policy_report["input_native_source_sha256"],
            False,
        )
        for p in policy_report["source_plans"]
    ]
    sources.append(
        (
            adapted_report["output"],
            adapted_report["source_plan"],
            adapted_report["input_source_receipt_sha256"],
            True,
        )
    )
    if adapted_report["unlinked_call_ids"] or adapted_report["unresolved_start_ids"]:
        raise ValueError("direct-adapted native audit has unresolved/unlinked receipts")
    if not adapted_report["all_direct_episodes_present"]:
        raise ValueError("direct-adapted audit is not a completed panel")
    first, values, groups, identities = sources[0][1], {}, {}, {}
    for output, plan, audited, adapted in sources:
        output = Path(output)
        # Owner boundary checked before reading outcomes. No unfinished evaluation is inspected.
        owners = list(output.glob("OWNER-*.json"))
        if not owners or {p.stem[6:] for p in owners} != {
            p.stem[9:] for p in output.glob("TERMINAL-*.json")
        }:
            raise ValueError("source not completed/released")
        for owner in owners:
            terminal = read(owner.with_name(owner.name.replace("OWNER-", "TERMINAL-")))
            if terminal.get("failure") or terminal.get("stopped"):
                raise ValueError("failed/capped source needs a separate incomplete audit")
        if read(output / "PLAN.json", audited) != plan:
            raise ValueError("source plan differs from native audit")
        for field in (
            "cases_sha256",
            "case_ids",
            "repeats",
            "seed",
            "model",
            "model_manifest_sha256",
        ):
            if plan[field] != first[field]:
                raise ValueError("matched source panel differs: " + field)
        if (
            plan["cases_sha256"] != cases_hash
            or plan["case_ids"] != parents
            or plan["repeats"] != 2
        ):
            raise ValueError("source not the full frozen64 x2 panel")
        if not adapted and plan.get("mode", "planner") == "planner":
            helper = plan.get("helper_contract", {})
            if (
                helper.get("mode") != "trained_helper"
                or helper.get("adapter_binding")
                != (adapted_report["source_plan"]["helper_adapter_binding"])
            ):
                raise ValueError("planner fixed helper differs from direct-adapted checkpoint")
        sampling = (
            plan["sampling"]
            if adapted
            else dict(
                max_new_tokens=plan["caps"]["final"],
                temperature=plan["temperature"],
                top_p=plan["top_p"],
                top_k=plan["top_k"],
            )
        )
        if any(
            sampling[k] != v
            for k, v in dict(max_new_tokens=128, temperature=0.5, top_p=1.0, top_k=0).items()
        ):
            raise ValueError("final cap/sampling contract differs")
        indexed = {}
        for path in sorted((output / "episodes").glob("*.json")):
            row = read(path, audited)
            key = row["case_id"], row["repeat"], row["condition"]
            if key in indexed:
                raise ValueError("duplicate episode")
            indexed[key] = row
        if set(indexed) != {
            (p, r, c) for p in parents for r in range(2) for c in plan["conditions"]
        }:
            raise ValueError("completed report must contain all planned episode identities")
        for condition in plan["conditions"]:
            label = (
                {"base_direct": "direct_base_control", "helper_sft_direct": "direct_helper_sft"}[
                    condition
                ]
                if adapted
                else "direct_base"
                if plan.get("mode") == "direct"
                else "planner_" + condition
            )
            if label in groups:
                raise ValueError("duplicate policy; never pool repeated same-seed acquisitions")
            for p in parents:
                for repeat in range(2):
                    row = indexed[p, repeat, condition]
                    ids = [c for c in row["call_ids"] if c.endswith(("-final", "-answer"))]
                    if len(ids) > 1:
                        raise ValueError("multiple final receipts")
                    final = read(output / "calls" / (ids[0] + ".json"), audited) if ids else None
                    if final and final["role"] not in ("final", "direct_answer"):
                        raise ValueError("unexpected answer role")
                    grade = probe.grade(
                        final["text"] if final and final["available"] else "", by_id[p]
                    )
                    status = (
                        ("scored" if grade["valid"] else "invalid_final")
                        if final and final["available"]
                        else row["status"]
                    )
                    missing = not (final and final["available"]) and status not in {
                        "invalid_plan",
                        "invalid_dependency",
                        "invalid_helper",
                    }
                    values[p, repeat, label] = dict(
                        em=float(grade["correct"]),
                        f1=grade["f1"],
                        valid=grade["valid"],
                        missing=missing,
                        status=status,
                    )
            source_group = (
                adapted_report["groups"][condition] if adapted else policy_report["groups"][label]
            )
            rows = [values[p, r, label] for p in parents for r in range(2)]
            for metric in ("em", "f1"):
                if abs(mean(r[metric] for r in rows) - source_group[metric]) > 1e-12:
                    raise ValueError("lightweight regrade differs from complete native audit")
            groups[label] = source_group
            identities[label] = dict(output=str(output), condition=condition, plan=plan)
    if not {
        "planner_sft",
        "planner_rl",
        "direct_base",
        "direct_base_control",
        "direct_helper_sft",
    } <= set(groups):
        raise ValueError("missing required completed fresh policies")
    results = {
        name: summarize_stratum(
            ids, values, list(groups), restrict_clusters(full_clusters, ids), draws=draws, seed=seed
        )
        for name, ids in partitions.items()
    }
    for path in (
        Path(__file__),
        Path(analyze_helper.__file__),
        Path(probe.__file__),
        probe.MUSIQUE / "metrics/answer.py",
    ):
        hashes[str(path.resolve())] = probe.campaign.sha(path)
    return dict(
        method=dict(
            metric="official_musique_alias_max_em_f1",
            repeats=2,
            draws=draws,
            seed=seed,
            primary="all64 remains primary: deliberately balanced 32 two-hop/32 three-hop, "
            "not the natural whole-development benchmark mixture; no reweighting. "
            "All other strata exploratory, unadjusted",
            clustering="Full-panel connected atomic components intersected with each stratum; "
            "parent-weighted repeat means and whole-cluster bootstrap",
        ),
        strata=results,
        hop_by_exposure={
            name: dict(Counter(by_id[p]["metadata"]["hops"] for p in partitions[name]))
            for name in ("exact_train_document_overlap", "no_exact_train_document_overlap")
        },
        full_panel_audited_groups=groups,
        source_identities=identities,
        source_reports=[str(Path(p).resolve()) for p in (policy_report_path, adapted_report_path)],
        input_receipt_sha256=hashes,
        provenance_note="Reuses completed native audit coverage/model/checkpoint checks; "
        "only consumed plans, episodes and final receipts rehashed/regraded. No weights rehashed.",
        cautions=[
            "No case reselection: hop and exposure strata overlap and are not independent factors.",
            "Exposure strata may differ in hop count and difficulty; their outcomes describe "
            "associations, not the causal effect of TRAIN document exposure.",
            "Exact exposure means any literal (title,text) document shared with TRAIN, "
            "including distractors; no exact overlap does not mean no semantic exposure.",
            "Different policies/prompts/costs are not a causal decomposition comparison. "
            "Within-planner RL contrast retains the same frozen trained helper.",
            "Higher gain in one stratum, or significance in only one stratum, is not "
            "a tested interaction or evidence that an effect is confined there.",
            "Missing observations are not protocol failures. Incomplete subgroup "
            "differences of lower bounds are not treatment-effect bounds.",
            "The new direct base control and original direct base are separate "
            "same-seed acquisitions, never pooled as extra independent repeats.",
        ],
    )


def markdown(report):
    lines = [
        "# Fresh003 secondary strata",
        "",
        report["method"]["primary"],
        "",
        "All outcomes retain their planned denominator. EM/F1 use official MuSiQue metrics.",
        "Exposure-stratum hop counts: " + json.dumps(report["hop_by_exposure"], sort_keys=True),
        "",
    ]
    for name, result in report["strata"].items():
        lines += [
            f"## {name}",
            "",
            f"{result['parents']} parents; {len(result['component_clusters'])} component clusters.",
            "",
            "| Policy | Correct/planned | EM | F1 | Missing |",
            "|---|---:|---:|---:|---:|",
        ]
        for policy, group in result["groups"].items():
            lines.append(
                f"| {policy} | {group['correct']:g}/{group['planned']} | "
                f"{group['em']:.1%} | {group['f1']:.1%} | {group['missing']} |"
            )
        lines += [""]
        for contrast, pair in result["contrasts"].items():
            metrics = "; ".join(
                f"{m.upper()} {100 * pair[m]['estimate']:+.1f}pp "
                f"[{100 * pair[m]['ci95'][0]:+.1f}, {100 * pair[m]['ci95'][1]:+.1f}]"
                for m in ("em", "f1")
            )
            lines.append(
                f"- {contrast}: {metrics}. Wins {dict(pair['observed_wins'])}; "
                f"losses {dict(pair['observed_losses'])}; "
                f"unobserved pairs {pair['unobserved_pairs']}."
            )
        lines += [""]
    lines += ["## Limits", "", *["- " + note for note in report["cautions"]], ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-report", "adapted-report", "exposure", "cases", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    md = args.report.with_suffix(".md")
    if args.report == md or args.report.exists() or md.exists():
        parser.error("choose unused immutable JSON and Markdown paths")
    result = analyze(
        args.policy_report, args.adapted_report, args.exposure, args.cases, draws=args.draws
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with md.open("x") as stream:
        stream.write(markdown(result))
    print(markdown(result))
