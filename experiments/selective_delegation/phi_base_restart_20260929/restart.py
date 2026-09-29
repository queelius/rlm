"""Restart two preflight-only Phi controls with JSON-equivalent plan comparison."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PHI = HERE.parent / "phi_transfer_20260928"
ORIGINAL_PLAN_SHA = {
    42: "befe6509a4e8f88f0b943ffff8708192315f103c10c227dd7ae16d14d962dccc",
    50: "45effd3acf39f9c2d8487dea60abab7f631c7104fa723e9aa04822a5c6cdc98b",
}


def canonical_matching_plan(original, candidate):
    """Ignore only Python container distinctions that disappear in the saved JSON."""
    canonical = json.loads(json.dumps(candidate, allow_nan=False))
    if canonical != original:
        raise ValueError("original scientific plan differs after JSON serialization")
    return canonical


def original_module():
    sys.path.insert(0, str(PHI))
    spec = importlib.util.spec_from_file_location("frozen_phi_evaluation", PHI / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(world):
    module = original_module()
    root = module.acquire.ROOT
    old = root / f"textcraft-phi-base-w{world}-20260928-001"
    old_plan = old / "PLAN.json"
    if module.acquire.sha(old_plan) != ORIGINAL_PLAN_SHA[world]:
        raise ValueError("original base plan changed")
    if list(old.glob("OWNER-*.json")):
        raise ValueError("original attempt acquired a scientific owner; inspect before restart")
    failed = root / f"information-first-tail-20260928-002/queue/phi-base-w{world}.json"
    if module.acquire.read(failed).get("returncode") != 1:
        raise ValueError("expected preflight failure receipt required")
    output = root / f"textcraft-phi-base-w{world}-20260929-001"
    args = SimpleNamespace(teacher="base", assistance="raw", world=world, hours=0.25, output=output)
    candidate, tasks, native_world, _, collector, _ = module.build(args)
    plan = canonical_matching_plan(module.acquire.read(old_plan), candidate)
    plan["restart_provenance"] = dict(
        original_plan=str(old_plan),
        original_plan_sha256=ORIGINAL_PLAN_SHA[world],
        failed_job=str(failed),
        failed_job_sha256=module.acquire.sha(failed),
        reason="Native rewrite metadata uses tuples in memory and lists in JSON. "
        "The entire saved plan equals the serialized rebuilt plan; scientific fields unchanged.",
        scientific_changes="None: same base weights, tasks, seeds, world, prompts and caps.",
    )
    plan["source_sha256"][str(Path(__file__).resolve())] = module.acquire.sha(Path(__file__))
    path = output / "PLAN.json"
    if path.exists():
        if module.acquire.read(path) != plan:
            raise ValueError("restart plan changed")
    else:
        module.acquire.save(path, plan)
    return module, args, plan, tasks, native_world, collector


def main(args):
    module, scientific, plan, tasks, world, collector = prepare(args.world)
    if args.prepare_only:
        print(json.dumps(dict(output=str(scientific.output), episodes=len(plan["jobs"]))))
    elif args.audit:
        result = module.audit(scientific.output)
        module.acquire.save(scientific.output / "PHI-AUDIT.json", result)
        print(json.dumps({key: result[key] for key in ("observed", "successes", "unknown")}))
    else:
        collector.run(scientific, prepared_run=(plan, tasks), adapter=None, world=world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, choices=(42, 50), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    main(parser.parse_args())
