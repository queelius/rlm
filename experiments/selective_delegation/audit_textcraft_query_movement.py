"""Read-only public-history query labels for the already fixed first-update audit."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
INPUT = ROOT / "analysis-textcraft-update-001.json"
EXPECTED = "e19bd51cc230e636899f0690c9523bdd07d146947338eb2685aeef5179b4f151"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    assert sha(INPUT) == EXPECTED
    report = json.loads(INPUT.read_text())
    lookup = {r["call_id"]: r for r in report["call_rows"] if r["action"] == "get_info"}
    data = Path(report["native_data"])
    rows, pins = [], {}
    for path in sorted((data / "nodes").glob("*.json")):
        node = json.loads(path.read_text())
        pins[str(path)] = sha(path)
        roots, initial = set(node["targets"]), set(node["initial_inventory"])
        introduced, queried = roots | initial, set()
        for cid, history in zip(node["call_ids"], node["public_history"], strict=True):
            action = history.get("action", {})
            if action.get("action") == "get_info":
                items = action["items"]
                if cid in lookup:
                    assert len(items) == 1
                    item = items[0]
                    category = (
                        "repeated_lookup"
                        if item in queried
                        else "first_root_lookup"
                        if item in roots
                        else "first_initial_inventory_lookup"
                        if item in initial
                        else "first_other_publicly_introduced_lookup"
                        if item in introduced
                        else "first_unintroduced_name_lookup"
                    )
                    row = lookup[cid]
                    rows.append(
                        dict(
                            call_id=cid,
                            task_id=row["task_id"],
                            sign=row["sign"],
                            category=category,
                            item=item,
                            previously_queried=item in queried,
                            root_target=item in roots,
                            public_name_before_query=item in introduced,
                            tokens=row["tokens"],
                            delta=row["sequence_logp_delta"],
                        )
                    )
                queried.update(items)
                feedback = history["feedback"]
                if isinstance(feedback, list):
                    for value in feedback:
                        if isinstance(value, dict):
                            if value.get("item"):
                                introduced.add(value["item"])
                            for recipe in value.get("recipes", []):
                                introduced.update(recipe.get("ingredients", {}))
    assert len(rows) == len(lookup) == 96
    groups = {}
    for sign, category in sorted({(r["sign"], r["category"]) for r in rows}):
        selected = [r for r in rows if (r["sign"], r["category"]) == (sign, category)]
        groups[sign + "/" + category] = dict(
            calls=len(selected),
            tokens=sum(r["tokens"] for r in selected),
            sequence_logp_delta_sum=sum(r["delta"] for r in selected),
            likelihood_increased=sum(r["delta"] > 0 for r in selected),
            tasks=len({r["task_id"] for r in selected}),
        )
    return dict(
        input_sha256=EXPECTED,
        source_sha256=sha(Path(__file__)),
        groups=groups,
        rows=rows,
        native_node_sha256=pins,
        method="Scan complete saved public history in call order. "
        "Names initially visible are public root targets and inventory; subsequent names "
        "are introduced by returned public recipe ingredients. First/repeated labels do not "
        "establish need, usefulness, recipe necessity or causal action value. "
        "No hidden world lookup.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["groups"], indent=2))
