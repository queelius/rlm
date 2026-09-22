"""Immutable behavior receipt for the TextCraft procedure-prompt control."""

import argparse
import hashlib
import json
from pathlib import Path

import audit_textcraft_query_transfer as query_audit

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
PROCEDURE_REPORT_SHA256 = "020c12fcb69ec57aa1a2889cad69182bc78acc90b64611af00daeb546bf400b3"
OLD_REPORT_SHA256 = "f84ea2aa2a86c1190fe4db0cb069242f0c1e88e35849e6a1a63eca819f8e9887"
CONFIGS = (
    (
        "old_original",
        "textcraft-trained-readout-001",
        "analysis-textcraft-trained-readout-001.json",
        "original",
    ),
    (
        "procedure_control",
        "textcraft-procedure-control-001",
        "analysis-textcraft-procedure-control-001.json",
        "procedure_control",
    ),
)
PAIRS = (("old_original", "procedure_control"),)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    old_report = ROOT / CONFIGS[0][2]
    procedure_report = ROOT / CONFIGS[1][2]
    if sha256(old_report) != OLD_REPORT_SHA256:
        raise ValueError("old report hash differs from fixed control binding")
    if sha256(procedure_report) != PROCEDURE_REPORT_SHA256:
        raise ValueError("procedure report hash differs from fixed control binding")
    result = query_audit.analyze(
        ROOT,
        configs=CONFIGS,
        matched_pairs=PAIRS,
        known_recipe_diagnostic=True,
    )
    receipt = {
        "schema": "textcraft_procedure_behavior_receipt_v1",
        "binding": {
            "configs": CONFIGS,
            "matched_pairs": PAIRS,
            "old_report_sha256": OLD_REPORT_SHA256,
            "procedure_report_sha256": PROCEDURE_REPORT_SHA256,
            "known_static_repeat_semantics": (
                "An item whose recipe appeared in strictly earlier public feedback; "
                "current dynamic inventory is not treated as a static recipe fact."
            ),
        },
        "audit": result,
        "driver_sha256": sha256(Path(__file__)),
    }
    with args.report.open("x") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
