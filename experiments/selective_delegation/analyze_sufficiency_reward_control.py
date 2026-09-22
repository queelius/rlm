"""Independent four-arm fixed-panel readout; does not audit new TRAIN gradients."""

import argparse
import json
from pathlib import Path

import analyze_sufficiency_rl as shared
import eval_sufficiency_heldout as reader

CONDITIONS = ("warm_joint32", "rl_terminal", "matched_sft_terminal", "additive_rl_terminal")


def validate_condition_contract(plan):
    binding = plan["control_binding"]
    if (
        plan["conditions"] != list(CONDITIONS)
        or plan["planned_calls"] != 512
        or binding["matched_sft_control_for"] != "rl_terminal"
        or binding["rl_terminal_reward"] != "product"
        or binding["additive_rl_terminal_reward"] != "additive"
        or binding["equal_actual_steps_and_sample_cursors"] is not True
    ):
        raise ValueError("four-arm product/SFT/additive condition contract differs")


def audit_endpoint(identity, audit):
    root = Path(identity["training_output"])
    plan = audit.read(root / "PLAN.json", identity["training_plan_sha256"])
    summary = audit.read(root / "SUMMARY.json", identity["summary_sha256"])
    terminals = audit.terminal(root)
    if any(t.get("failure") or t.get("stopped") for t in terminals):
        raise ValueError("training endpoint failed/interrupted")
    terminal_paths = list(root.glob("TERMINAL-*.json"))
    if identity["terminal_sha256"] not in {audit.hashes[str(p.resolve())] for p in terminal_paths}:
        raise ValueError("terminal identity differs")
    boundaries = [
        audit.read(p) for p in sorted((root / "boundaries").glob("sample-*/BOUNDARY.json"))
    ]
    for boundary in boundaries:
        audit.checkpoint(boundary)
    last = boundaries[-1]
    terminal = max(terminals, key=lambda t: t["sample_cursor"])
    if (
        plan != identity["training_plan"]
        or boundaries != identity["boundaries"]
        or last["checkpoint"] != identity["path"]
        or summary["endpoint"] != identity["path"]
        or terminal["endpoint"] != identity["path"]
        or (identity["step"], identity["sample_cursor"]) != (8, 8)
        or (last["state"]["step"], last["state"]["sample_cursor"]) != (8, 8)
        or (summary["actual_optimizer_steps"], summary["committed_sampled_blocks"]) != (8, 8)
        or (terminal["step"], terminal["sample_cursor"]) != (8, 8)
        or terminal["endpoint_selection"] != "last committed boundary, never held score"
    ):
        raise ValueError("readout is not the declared eight-update terminal endpoint")
    return plan


def audit_readout_contract(plan, audit):
    validate_condition_contract(plan)
    roots = {Path(p).parent for p in plan["source_sha256"]}
    if len(roots) != 1:
        raise ValueError("readout source root differs")
    source = roots.pop()
    profile = plan["panel_profile"]
    frozen = audit.read(source / "REWARD-CONTROL-PROFILE.json", profile["profile_sha256"])
    audit.hash(source / "eval_sufficiency_reward_control.py", profile["entrypoint_sha256"])
    audit.hash(source / "eval_sufficiency_heldout.py", plan["runner_sha256"])
    if (
        frozen
        != {k: v for k, v in profile.items() if k not in ("profile_sha256", "entrypoint_sha256")}
        or frozen["cases_sha256"] != plan["cases_sha256"]
        or frozen["manifest_sha256"] != plan["manifest_sha256"]
        or frozen["readout_schema"] != plan["schema"]
        or frozen["component_cluster_count"] != 29
    ):
        raise ValueError("fixed050 reward-control profile differs")
    ids = plan["adapters"]
    rp, sp, ap = [audit_endpoint(ids[c], audit) for c in CONDITIONS[1:]]
    if any(p["warmstart"] != ids["warm_joint32"] for p in (rp, sp, ap)):
        raise ValueError("common joint32 warmstart differs")
    if (
        ids["warm_joint32"]["step"] != 32
        or ids["warm_joint32"]["arm"] != "joint"
        or rp["mode"] != "rl"
        or ap["mode"] != "rl"
        or sp["mode"] != "sft_control"
        or sp["rl_plan_sha256"] != ids["rl_terminal"]["training_plan_sha256"]
        or sp["matched_rl_boundaries"] != ids["rl_terminal"]["boundaries"]
        or sp["cases_sha256"] != rp["cases_sha256"]
        or sp["parent_blocks"] != rp["parent_blocks"]
    ):
        raise ValueError("extra SFT is not bound to original product-RL training")
    reader.validate_additive_control(ids["rl_terminal"], ids["additive_rl_terminal"])


def main(args):
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    output = args.output.resolve()
    if (output / "SKIPPED.json").exists():
        raise ValueError("readout explicitly skipped; no scientific four-arm comparison")
    audit = shared.Audit()
    audit.terminal(output)
    plan = audit.read(output / "PLAN.json")
    audit_readout_contract(plan, audit)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        shared.training.BASE, local_files_only=True, trust_remote_code=False
    )
    held = shared.held_report(output, args.cases.resolve(), audit, tokenizer, expected_calls=512)
    if held["method"]["component_count"] != 29:
        raise ValueError("fixed050 component count differs")
    report = {
        "schema": "paired-sufficiency-four-arm-analysis-v1",
        "training": {},
        "held": held,
        "scope": "Native held calls/grades and actual TRAIN endpoint identity only; "
        "new additive TRAIN rewards/gradients need their separate objective-aware audit",
        "primary_contrast": "additive_rl_terminal_minus_rl_terminal",
        "multiplicity": "Six exploratory pair contrasts, no multiplicity correction",
        "source_sha256": audit.hashes,
        "analyzer_sha256": {
            str(Path(m.__file__).resolve()): reader.sha(Path(m.__file__))
            for m in (shared, shared.paired, reader)
        },
    }
    report["analyzer_sha256"][str(Path(__file__).resolve())] = reader.sha(Path(__file__))
    shared.paired.baseline.native.save(args.report, report)
    with args.report.with_suffix(".md").open("x") as stream:
        stream.write(shared.markdown(report))
        stream.write(
            "\nPrimary contrast: additive minus product RL. Old extra-SFT remains "
            "matched to product RL, not relabeled additive-matched. Six exploratory "
            "contrasts; equal steps are not equal FLOPs or credited tokens.\n"
        )
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("output", "cases", "report"):
        parser.add_argument("--" + arg, type=Path, required=True)
    main(parser.parse_args())
