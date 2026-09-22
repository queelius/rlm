"""Independent two-arm pairing-mean versus diagonal fixed050 readout audit."""

import argparse
import json
from pathlib import Path

import analyze_sufficiency_rl as shared
import eval_sufficiency_heldout as reader

CONDITIONS = ("rl_terminal", "pairing_mean_terminal")


def validate_condition_contract(plan):
    binding = plan["control_binding"]
    if (
        plan["conditions"] != list(CONDITIONS)
        or plan["planned_calls"] != 256
        or binding["rl_terminal_reward"] != "product"
        or binding["pairing_mean_terminal_reward"] != "product"
        or binding["rl_terminal_estimator"] != "diagonal"
        or binding["pairing_mean_terminal_estimator"] != "pairing_mean"
        or binding["equal_actual_steps_and_sample_cursors"] is not True
    ):
        raise ValueError("two-arm product diagonal/pairing-mean contract differs")


def audit_endpoint(identity, audit):
    root = Path(identity["training_output"])
    plan = audit.read(root / "PLAN.json", identity["training_plan_sha256"])
    summary = audit.read(root / "SUMMARY.json", identity["summary_sha256"])
    terminals = audit.terminal(root)
    if any(t.get("failure") or t.get("stopped") for t in terminals):
        raise ValueError("training endpoint failed/interrupted")
    boundaries = [
        audit.read(p) for p in sorted((root / "boundaries").glob("sample-*/BOUNDARY.json"))
    ]
    for boundary in boundaries:
        audit.checkpoint(boundary)
    last, terminal = boundaries[-1], max(terminals, key=lambda t: t["sample_cursor"])
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
    frozen = audit.read(source / "PAIRING-MEAN-PROFILE.json", profile["profile_sha256"])
    audit.hash(source / "eval_sufficiency_pairing_mean.py", profile["entrypoint_sha256"])
    audit.hash(source / "eval_sufficiency_heldout.py", plan["runner_sha256"])
    if (
        frozen
        != {k: v for k, v in profile.items() if k not in ("profile_sha256", "entrypoint_sha256")}
        or frozen["cases_sha256"] != plan["cases_sha256"]
        or frozen["manifest_sha256"] != plan["manifest_sha256"]
        or frozen["readout_schema"] != plan["schema"]
        or frozen["component_cluster_count"] != 29
    ):
        raise ValueError("fixed050 pairing-mean profile differs")
    diagonal, pairing = [audit_endpoint(plan["adapters"][c], audit) for c in CONDITIONS]
    if any(
        p["warmstart"] != plan["adapters"]["rl_terminal"]["training_plan"]["warmstart"]
        for p in (diagonal, pairing)
    ):
        raise ValueError("common joint32 warmstart differs")
    reader.validate_pairing_mean_control(
        plan["adapters"]["rl_terminal"], plan["adapters"]["pairing_mean_terminal"]
    )


def main(args):
    if args.report.exists() or args.report.with_suffix(".md").exists():
        raise FileExistsError("immutable report exists")
    output = args.output.resolve()
    if (output / "SKIPPED.json").exists():
        raise ValueError("incomplete pairing-mean endpoint is not a scientific comparison")
    audit = shared.Audit()
    audit.terminal(output)
    plan = audit.read(output / "PLAN.json")
    audit_readout_contract(plan, audit)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        shared.training.BASE, local_files_only=True, trust_remote_code=False
    )
    held = shared.held_report(output, args.cases.resolve(), audit, tokenizer, expected_calls=256)
    if held["method"]["component_count"] != 29:
        raise ValueError("fixed050 component count differs")
    report = {
        "schema": "paired-sufficiency-pairing-mean-analysis-v1",
        "training": {},
        "held": held,
        "scope": "Two actual fixed-dose product-RL endpoints. Pairing-mean is a conditional "
        "Rao--Blackwellized credit estimator, not a new reward or held-selected checkpoint.",
        "primary_contrast": "pairing_mean_terminal_minus_rl_terminal",
        "multiplicity": "One predeclared primary contrast; panel is already exposed exploratory "
        "readout.",
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
            "\nPrimary contrast: pairing-mean minus diagonal product RL. This is an exploratory "
            "post-training readout of already exposed frozen050, not a fresh confirmation.\n"
        )
    print(json.dumps({"report": str(args.report)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("output", "cases", "report"):
        parser.add_argument("--" + arg, type=Path, required=True)
    main(parser.parse_args())
