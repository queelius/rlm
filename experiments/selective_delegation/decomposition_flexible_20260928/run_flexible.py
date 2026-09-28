"""V2 admission screen changes query-order enforcement, not the native task."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import routing_flexible as routing

OLD = Path(__file__).resolve().parent.parent / "decomposition_20260928"
sys.path.insert(0, str(OLD))
import prepare_data as data  # noqa: E402

base = data.load(OLD / "run.py", "decomposition_flexible_base")
base.routing = routing


def correct_group_counts(report, plan):
    counts = Counter(job.get("condition", job["policy"]) for job in plan["jobs"])
    for policy, group in report["groups"].items():
        group["planned"] = counts[policy]
        group["missing_or_unknown"] = counts[policy] - group["observed"]
    return report


def build(args, fixture=False):
    plan, tasks, world, tokenizer, collector, auditor = base.build(args, fixture=fixture)
    if not getattr(collector, "flexible_summary_installed", False):
        old_summarize = collector.summarize
        collector.summarize = lambda output, plan: correct_group_counts(
            old_summarize(output, plan), plan
        )
        collector.flexible_summary_installed = True
    plan.update(
        schema="textcraft-decomposition-flexible-admission-20260928-v2",
        comparison="Same publiccp23, native task/stock/seed/caps and public candidate rule. "
        "Recipe queries may occur in any model-chosen order. Flat versus fixed/adaptive "
        "one-helper requirement; at most32 real responses.",
        stop_contract="Before a next request after32 responses or two rejected delegate "
        "instructions. Native query/craft errors do not trigger this instruction cutoff. "
        "Partial roots remain UNKNOWN, never zero.",
        relation_to_v1="Task now reused for exploratory repair after v1 rejected valid "
        "non-lexical prerequisite queries. This is not a fresh holdout or efficacy trial.",
    )
    for path in (Path(__file__), OLD / "routing.py"):
        plan["source_sha256"][str(path.resolve())] = data.sha(path)
    return plan, tasks, world, tokenizer, collector, auditor


def audit(output, require_terminal=True):
    result = base.audit(output, require_terminal=require_terminal)
    result["schema"] = "textcraft-decomposition-flexible-admission-audit-20260928-v2"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("flat", "fixed", "adaptive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=0.25)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit(args.output.resolve())
        data.save(args.output / "ADMISSION-AUDIT.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "artifacts_sha256"}))
        return
    plan, tasks, world, _, collector, _ = build(args)
    path = args.output / "PLAN.json"
    if path.exists():
        if data.read(path) != plan:
            raise ValueError("immutable v2 admission PLAN changed")
    else:
        data.save(path, plan)
    try:
        collector.run(args, prepared_run=(plan, tasks), adapter=plan["adapter"], world=world)
    except RuntimeError as exc:
        if "AdmissionStop:" not in str(exc):
            raise
        data.save(args.output / "ADMISSION-STOP.json", dict(reason=str(exc), root_score=None))
        print(json.dumps(dict(planned_admission_stop=str(exc), root_outcome="unknown")))


if __name__ == "__main__":
    main()
