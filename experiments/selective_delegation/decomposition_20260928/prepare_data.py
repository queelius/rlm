"""One fresh official VAL root for a bounded, outcome-blind admission probe."""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect_textcraft_worlds as worlds  # noqa: E402
import prepare_textcraft_sft as source  # noqa: E402
import routing  # noqa: E402

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-decomposition-admission-20260928-001"
SEED = 2026092805
VAL_SHA = "84a123ee46e29e65f4d7b95f4943aa56907fc3fd5992268faa749602579dfc0c"
WORLD_SHA = "f76ce3978c038be9624b3c7387aa30033970508c6afecb1f8f08315aa0693808"
FRESH = routing.LIBRARY / "fresh_train_20260928/prepare.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(output=OUTPUT):
    if output.exists():
        raise FileExistsError("immutable admission selection exists; no reselection")
    val = source.original.TASKS
    if sha(val) != VAL_SHA or sha(source.TRAIN) != source.TRAIN_SHA:
        raise ValueError("official source changed")
    original = {json.loads(line)["id"]: line for line in val.read_bytes().splitlines(keepends=True)}
    rows = [json.loads(line) for line in original.values()]
    files = {source.TRAIN}
    patterns = ["textcraft*/tasks.jsonl", "textcraft*/*/tasks.jsonl", "textcraft*/*/*/tasks.jsonl"]
    for pattern in patterns:
        files.update(p for p in ROOT.glob(pattern) if not p.is_relative_to(output))
    excluded_ids, excluded_roots, inventories = set(), set(), []
    for path in sorted(files):
        tasks = [json.loads(line) for line in path.read_text().splitlines()]
        ids = {t["id"] for t in tasks}
        roots = {k for t in tasks for k in t["misc"]["target_items"]}
        excluded_ids.update(ids)
        excluded_roots.update(roots)
        inventories.append(
            dict(path=str(path), sha256=sha(path), task_ids=sorted(ids), roots=sorted(roots))
        )
    plans = []
    for path in sorted(ROOT.glob("textcraft*/PLAN.json")):
        ids = {job["task_id"] for job in read(path).get("jobs", []) if "task_id" in job}
        if ids - excluded_ids:
            raise ValueError("existing plan references an unprotected task source: " + str(path))
        plans.append(dict(path=str(path), sha256=sha(path), task_ids=sorted(ids)))
    remaining = [
        task
        for task in rows
        if task["id"] not in excluded_ids and not set(task["misc"]["target_items"]) & excluded_roots
    ]
    candidates = [t for t in remaining if t["misc"]["max_depth"] == 4]
    if not candidates:
        raise ValueError("declared fresh depth4 stratum exhausted; no silent substitute")
    task = min(candidates, key=lambda t: hashlib.sha256(f"{SEED}:{t['id']}".encode()).hexdigest())
    output.mkdir(parents=True)
    with (output / "tasks.jsonl").open("xb") as stream:
        stream.write(original[task["id"]])
    save(
        output / "EXCLUSIONS.json",
        dict(
            boundary="Every existing TextCraft task inventory at one-to-three directory levels, "
            "including selected future evaluation and HOLDOUT roots, plus all official TRAIN "
            "roots. Unused official VAL rows are eligible. Shared ingredient ancestors remain.",
            inventory_globs=patterns,
            inventories=inventories,
            plans=plans,
            task_ids=sorted(excluded_ids),
            roots=sorted(excluded_roots),
        ),
    )
    selection = dict(
        schema="textcraft-decomposition-admission-selection-20260928-v1",
        selected_at=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        rule="ONE original official VAL row at declared depth4, first SHA256(seed:task_id) "
        "after root/ID exclusion. Frozen before native feasibility; no outcome replacement.",
        remaining_depth_counts=dict(Counter(t["misc"]["max_depth"] for t in remaining)),
        stratum_rows=len(candidates),
        stratum_roots=len({k for t in candidates for k in t["misc"]["target_items"]}),
        task_ids=[task["id"]],
        tasks_sha256=sha(output / "tasks.jsonl"),
        exclusions_sha256=sha(output / "EXCLUSIONS.json"),
        source_json_order="Exact official source bytes and nested key order retained.",
    )
    save(output / "SELECTION.json", selection)
    fresh = load(FRESH, "decomposition_fresh_cpu_helpers")
    world = routing.native.load_world()
    if worlds.digest(worlds.snapshot(world)) != WORLD_SHA:
        raise ValueError("native world42 identity changed")
    structure = fresh.structure(task, world, {})
    feasible, trace = fresh.feasibility(task, world)
    save(output / "native-public-feasibility.json", dict(audit=feasible, trace=trace))
    proxy = routing.make_bridge("adaptive")
    frame = proxy.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        routing.native.Budget(),
        1,
    )
    prefix = []
    while frame.routing()["decision"]["query"] is not None:
        action = frame.routing()["decision"]["query"]
        frame.budget.charge(0)  # CPU tool count only; no fabricated model token cost.
        prefix.append(dict(action=action, feedback=frame.apply(action)))
    manifest = dict(
        schema="textcraft-decomposition-admission-data-20260928-v1",
        task_ids=[task["id"]],
        split="official VAL; fresh goal root excluded from every current task inventory",
        tasks_sha256=sha(output / "tasks.jsonl"),
        selection_sha256=sha(output / "SELECTION.json"),
        exclusions_sha256=sha(output / "EXCLUSIONS.json"),
        source=dict(
            url="https://github.com/ApGa/platoon",
            commit=source.original.COMMIT,
            official_file=str(val),
            official_sha256=VAL_SHA,
            license="MIT",
            license_path=str(routing.native.CACHE / "LICENSE"),
            license_sha256=sha(routing.native.CACHE / "LICENSE"),
        ),
        world_seed=42,
        world_sha256=WORLD_SHA,
        trusted_source=routing.native.trusted_provenance(),
        structure=structure,
        all_native_feasible=feasible["feasible"],
        native_audit=feasible,
        native_trace_sha256=sha(output / "native-public-feasibility.json"),
        example=dict(public_prefix=prefix, first_public_decision=frame.routing()),
        source_sha256={str(p): sha(p) for p in (Path(__file__).resolve(), FRESH)},
        gpu_used=False,
        scientific_model_calls=0,
        limitation="Native CPU feasibility and structural diagnostic do not establish model "
        "success or context feasibility. No gold actions or dependency-depth fields enter routing.",
    )
    save(output / "MANIFEST.json", manifest)
    if not feasible["feasible"]:
        raise ValueError("selected task native feasibility failed; retained, no replacement")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output.resolve())
    print(json.dumps({k: result[k] for k in ("task_ids", "structure", "native_audit")}, indent=2))
