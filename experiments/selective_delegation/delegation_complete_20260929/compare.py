"""All three fixed routing arms, keeping censored/unattempted roots unknown."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import complete as c  # noqa: E402


def paired(left, right):
    deltas = [
        right[k] - left[k] if left.get(k) is not None and right.get(k) is not None else None
        for k in (0, 1)
    ]
    wins, losses = sum(x == 1 for x in deltas), sum(x == -1 for x in deltas)
    unknown = deltas.count(None)
    return dict(
        planned=2,
        wins=wins,
        losses=losses,
        ties=deltas.count(0),
        unknown=unknown,
        difference=None if unknown else (wins - losses) / 2,
        difference_bounds=[(wins - losses - unknown) / 2, (wins - losses + unknown) / 2],
    )


def analyze(study):
    cells, outcomes, hashes = {}, {}, {}
    for mode in c.MODES:
        path = study / mode / "NATIVE-AUDIT.json"
        if not path.exists():
            cells[mode], outcomes[mode] = dict(available=False, unknown=2), {}
            continue
        value, plan = c.read(path), c.read(path.with_name("PLAN.json"))
        c.check_plan(plan)
        if value["fixture"] or plan["fixture"]:
            raise ValueError("scripted fixture cannot become scientific evidence")
        if value["routing"] != mode or value["calls_without_episode"] or value["unresolved_starts"]:
            raise ValueError("unbound routing evidence")
        for name, digest in value["receipt_sha256"].items():
            if c.sha(name) != digest:
                raise ValueError("native-audited receipt changed")
        outcomes[mode] = {row["repeat"]: row["native_score"] for row in value["rows"]}
        cells[mode] = {k: v for k, v in value.items() if k not in ("receipt_sha256", "rows")}
        cells[mode]["available"] = True
        cells[mode]["rows"] = value["rows"]
        hashes[str(path)] = c.sha(path)
    return dict(
        schema="textcraft-complete-goal-three-arm-comparison-20260929-v1",
        cells=cells,
        paired={
            f"{right}_minus_{left}": paired(outcomes[left], outcomes[right])
            for left, right in (("flat", "fixed"), ("flat", "adaptive"), ("fixed", "adaptive"))
        },
        evidence_sha256=hashes,
        scope="One adaptively exposed val494 root/world42; two paired sampling seeds, not "
        "independent task clusters. No confidence interval or robust-generalization claim. "
        "Local child success is distinct from root success; compare all role/global costs. "
        "Different prompts/routing yield different trajectories despite paired seeds. "
        "No trained recursion, host-spawn or additional arithmetic treatment.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=c.STUDY)
    args = parser.parse_args()
    report = analyze(args.study)
    c.persist(args.study / "COMPARISON.json", report)
    print(json.dumps(report["paired"], indent=2))
