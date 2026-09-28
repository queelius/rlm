"""Freeze two fresh, root-disjoint official TRAIN groups without model outcomes.

Structural inspection and native feasibility use public get_info replies. Original
official JSONL lines are copied verbatim, including dictionary insertion order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

LIBRARY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIBRARY))

import inspect_textcraft_worlds as worlds  # noqa: E402
import prepare_textcraft_public as teacher  # noqa: E402
import prepare_textcraft_sft as source  # noqa: E402
import textcraft_bridge as bridge  # noqa: E402

SEED = 2026092803
QUOTAS = {3: 2, 4: 3, 5: 3}
GROUPS = ("train", "diagnostic")
ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
OUTPUT = ROOT / "textcraft-fresh-train-20260928-001"
VAL_SHA = "84a123ee46e29e65f4d7b95f4943aa56907fc3fd5992268faa749602579dfc0c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def roots(task: dict) -> set[str]:
    return set(task["misc"]["target_items"])


def official_rows() -> tuple[list[dict], dict[str, bytes]]:
    if sha(source.TRAIN) != source.TRAIN_SHA:
        raise ValueError("official TRAIN source changed")
    lines = source.TRAIN.read_bytes().splitlines(keepends=True)
    tasks = [json.loads(line) for line in lines if line.strip()]
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate official TRAIN identities")
    if any(not t["id"].startswith("textcraft_synth.train.") for t in tasks):
        raise ValueError("only official TRAIN source rows permitted")
    return tasks, {json.loads(line)["id"]: line for line in lines if line.strip()}


def exclusions(root: Path, output: Path) -> dict:
    """Small task inventories/plans only; never traverse checkpoint/source ancestry."""
    val = source.TRAIN.with_name("textcraft_synth_val.jsonl")
    if sha(val) != VAL_SHA:
        raise ValueError("official VAL source changed")
    files = {val}
    for pattern in (
        "textcraft*/tasks.jsonl",
        "textcraft*/*/tasks.jsonl",
        "textcraft*/*/*/tasks.jsonl",
    ):
        files.update(p for p in root.glob(pattern) if not p.is_relative_to(output))
    inventories, ids, excluded_roots = [], set(), set()
    for path in sorted(files):
        raw = path.read_bytes()
        tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
        current_ids = sorted({t["id"] for t in tasks})
        current_roots = sorted({item for task in tasks for item in roots(task)})
        ids.update(current_ids)
        excluded_roots.update(current_roots)
        inventories.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                rows=len(tasks),
                task_ids=current_ids,
                target_roots=current_roots,
            )
        )
    plans = []
    planned_ids = set()
    for path in sorted(root.glob("textcraft*/PLAN.json")):
        raw = path.read_bytes()
        plan = json.loads(raw)
        current = sorted({j["task_id"] for j in plan.get("jobs", []) if "task_id" in j})
        planned_ids.update(current)
        plans.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), task_ids=current))
    # Every currently selected plan identity must already be covered by a source
    # inventory. Fail closed if a new plan references an unaccounted task source.
    if planned_ids - ids:
        raise ValueError("plan task identity absent from protected inventories")
    return dict(
        boundary="All official VAL roots plus every existing TextCraft task inventory at "
        "one-to-three directory levels and all top-level TextCraft PLAN job identities. "
        "Includes SFT32, readiness TRAIN8, reserved breadth06/07, and reserve worlds47-49. "
        "Protects goal roots, not all reachable recipe ingredients; shared-world recipes remain.",
        inventory_globs=[
            "textcraft*/tasks.jsonl",
            "textcraft*/*/tasks.jsonl",
            "textcraft*/*/*/tasks.jsonl",
        ],
        inventories=inventories,
        plans=plans,
        task_ids=sorted(ids),
        target_roots=sorted(excluded_roots),
    )


def structure(task: dict, world, cache: dict) -> dict:
    """Read static dependency recipes through native public replies, without gold."""
    frame = bridge.Frame(
        world,
        dict(task["misc"]["initial_inventory"]),
        task["misc"]["target_items"],
        bridge.Budget(),
        0,
    )
    products, edges, depths = set(), [], {}

    def visit(item: str, active: set[str]) -> int:
        if item in active:
            raise ValueError("cycle in public recipe graph")
        if item in depths:
            return depths[item]
        if item not in cache:
            reply = frame.apply({"action": "get_info", "items": [item]})[0]
            cache[item] = dict(
                is_base=reply["is_base"],
                recipes=[
                    dict(ingredients=dict(r["ingredients"]), result_count=r["result_count"])
                    for r in reply["recipes"]
                ],
            )
        info = cache[item]
        if info["is_base"]:
            depths[item] = 0
        else:
            if len(info["recipes"]) != 1:
                raise ValueError("single public recipe per product required")
            products.add(item)
            ingredients = info["recipes"][0]["ingredients"]
            edges.extend((item, child) for child in ingredients)
            depths[item] = 1 + max(visit(child, active | {item}) for child in ingredients)
        return depths[item]

    longest = max(visit(item, set()) for item in roots(task))
    indegrees = Counter(child for _, child in edges)
    return dict(
        declared_depth=task["misc"]["max_depth"],
        dependency_chain=longest,
        reachable_products=len(products),
        ingredient_edges=len(edges),
        maximum_fanin=max(
            (len(cache[item]["recipes"][0]["ingredients"]) for item in products), default=0
        ),
        shared_ingredient_nodes=sum(count > 1 for count in indegrees.values()),
        initial_item_types=len(task["misc"]["initial_inventory"]),
        initial_units=sum(task["misc"]["initial_inventory"].values()),
        target_items=dict(task["misc"]["target_items"]),
        metadata_source="Native public get_info recipes; CPU preselection diagnostic only, "
        "never inserted into actor prompts or inventories; no gold trajectory inspected.",
    )


def select_groups(tasks, structures, excluded_ids, excluded_roots, quotas, seed):
    if len({t["id"] for t in tasks}) != len(tasks) or any(
        not t["id"].startswith("textcraft_synth.train.") for t in tasks
    ):
        raise ValueError("unique official TRAIN rows required")
    used_ids, used_roots = set(excluded_ids), set(excluded_roots)
    groups = {}
    for group in GROUPS:
        selected = []
        for depth, quota in sorted(quotas.items()):
            candidates = [
                t
                for t in tasks
                if t["id"] not in used_ids
                and not roots(t) & used_roots
                and 3 <= t["misc"]["max_depth"] <= 5
                and structures[t["id"]]["dependency_chain"] == depth
            ]
            ordered = sorted(
                candidates,
                key=lambda t: (hashlib.sha256(f"{seed}:{t['id']}".encode()).hexdigest(), t["id"]),
            )
            count = 0
            for task in ordered:
                if roots(task) & used_roots:
                    continue
                selected.append(task)
                used_ids.add(task["id"])
                used_roots.update(roots(task))
                count += 1
                if count == quota:
                    break
            if count != quota:
                raise ValueError(f"{group}: depth{depth} distinct-root quota unavailable")
        groups[group] = selected
    return groups


def feasibility(task: dict, world) -> tuple[dict, list]:
    """Public-only native solution; 512 tool actions, no model/context-cost claim."""
    initial = dict(task["misc"]["initial_inventory"])
    targets = dict(task["misc"]["target_items"])
    frame = bridge.Frame(world, dict(initial), targets, bridge.Budget(512, 131072), 0)
    observed, trace = {}, []
    error = None
    try:
        for _ in range(512):
            action = teacher.next_action(targets, initial, dict(frame.inventory), observed)
            # Native feasibility counts actions, not fabricated model output tokens.
            frame.budget.charge(0)
            reply = frame.apply(action)
            trace.append(dict(action=action, feedback=reply))
            if isinstance(reply, str) and reply.startswith("Error:"):
                raise ValueError(reply)
            if action["action"] == "get_info":
                for item in reply:
                    observed[item["item"]] = dict(
                        is_base=item["is_base"],
                        recipes=[
                            dict(ingredients=dict(r["ingredients"]), result_count=r["result_count"])
                            for r in item["recipes"]
                        ],
                    )
            if frame.finished:
                break
        if not frame.finished:
            error = "public planner native-action cap reached"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    score, details = frame.score()
    return dict(
        task_id=task["id"],
        finished=frame.finished,
        native_score=score,
        native_details=details,
        error=error,
        actions=len(trace),
        action_counts=dict(Counter(row["action"]["action"] for row in trace)),
        public_recipe_count=len(observed),
        feasible=frame.finished and score == 1 and error is None,
        scope="CPU public-recipe solver; initial inventory unchanged; no hidden recipe access, "
        "gold replay, model calls, policy success filtering or full-context qualification.",
    ), trace


def prepare(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError("immutable curriculum already exists")
    tasks, original_lines = official_rows()
    protected = exclusions(root, output)
    excluded_ids, excluded_roots = set(protected["task_ids"]), set(protected["target_roots"])
    candidates = [t for t in tasks if t["id"] not in excluded_ids and not roots(t) & excluded_roots]
    world, cache, structures = bridge.load_world(), {}, {}
    for task in candidates:
        if 3 <= task["misc"]["max_depth"] <= 5:
            structures[task["id"]] = structure(task, world, cache)
    groups = select_groups(candidates, structures, excluded_ids, excluded_roots, QUOTAS, SEED)
    world_sha = worlds.digest(worlds.snapshot(world))
    prior = json.loads((root / "textcraft-inputs-001/TOKEN-AUDIT.json").read_text())
    if prior["world_sha256"] != world_sha:
        raise ValueError("native world42 differs from frozen campaign")
    output.mkdir(parents=True)
    save(output / "EXCLUSIONS.json", protected)
    selection = dict(
        schema="textcraft-fresh-train-selection-20260928-v1",
        split="train",
        seed=SEED,
        rule="Group train then diagnostic; within actual dependency-depth strata, ascending "
        "SHA256(seed:official_id), skipping every used root. Declared tiers restricted3-5. "
        "No model or native-success outcome available at selection; no replacements.",
        per_group_actual_depth_quotas=QUOTAS,
        official_source_depth_counts=dict(Counter(t["misc"]["max_depth"] for t in tasks)),
        after_exclusion_declared_depth_counts=dict(
            Counter(t["misc"]["max_depth"] for t in candidates)
        ),
        available_actual_depth_counts=dict(
            Counter(s["dependency_chain"] for s in structures.values())
        ),
        available_distinct_roots_by_actual_depth={
            depth: len(
                {
                    item
                    for t in candidates
                    if t["id"] in structures and structures[t["id"]]["dependency_chain"] == depth
                    for item in roots(t)
                }
            )
            for depth in QUOTAS
        },
        groups={name: [t["id"] for t in rows] for name, rows in groups.items()},
        exclusions_sha256=sha(output / "EXCLUSIONS.json"),
        before_native_feasibility=True,
        model_outcomes_used=False,
        no_replacement=True,
        structural_metadata={
            t["id"]: structures[t["id"]] for rows in groups.values() for t in rows
        },
    )
    # Freeze both groups and the selection receipt BEFORE any feasibility result.
    for name, rows in groups.items():
        folder = output / name
        folder.mkdir()
        with (folder / "tasks.jsonl").open("xb") as stream:
            for task in rows:
                stream.write(original_lines[task["id"]])
    save(output / "SELECTION.json", selection)
    return qualify(root, output)


def qualify(root: Path, output: Path) -> dict:
    """Complete qualification of already frozen rows; never reselect or rewrite them."""
    if (output / "MANIFEST.json").exists():
        raise FileExistsError("completed immutable qualification exists")
    tasks, lines = official_rows()
    by_id = {t["id"]: t for t in tasks}
    selection = json.loads((output / "SELECTION.json").read_text())
    if selection["seed"] != SEED or selection["per_group_actual_depth_quotas"] != {
        str(k): v for k, v in QUOTAS.items()
    }:
        raise ValueError("frozen selection does not match declared design")
    if sha(output / "EXCLUSIONS.json") != selection["exclusions_sha256"]:
        raise ValueError("frozen exclusion receipt changed")
    protected = json.loads((output / "EXCLUSIONS.json").read_text())
    used_ids, used_roots = set(protected["task_ids"]), set(protected["target_roots"])
    groups = {}
    for name in GROUPS:
        selected_ids = selection["groups"][name]
        if len(selected_ids) != 8:
            raise ValueError("exact frozen eight TRAIN tasks required")
        groups[name] = [by_id[task_id] for task_id in selected_ids]
        if (output / name / "tasks.jsonl").read_bytes() != b"".join(
            lines[task_id] for task_id in selected_ids
        ):
            raise ValueError("frozen official source-line bytes changed")
        for task in groups[name]:
            if task["id"] in used_ids or roots(task) & used_roots:
                raise ValueError("frozen selection violates protected root/identity boundary")
            used_ids.add(task["id"])
            used_roots.update(roots(task))
    world = bridge.load_world()
    world_sha = worlds.digest(worlds.snapshot(world))
    prior = json.loads((root / "textcraft-inputs-001/TOKEN-AUDIT.json").read_text())
    if world_sha != prior["world_sha256"]:
        raise ValueError("qualification world42 differs from frozen campaign")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source.BASE, local_files_only=True, trust_remote_code=False
    )
    group_manifests = {}
    for name, rows in groups.items():
        audits = []
        for task in rows:
            receipt, trace = feasibility(task, world)
            prompt = bridge.initial_prompt(task, "flat")
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            receipt.update(
                initial_prompt_tokens=len(tokens),
                initial_prompt_plus_cap=len(tokens) + 256,
                initial_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                initial_context_fits=len(tokens) + 256 <= 8192,
            )
            trace_path = output / name / "native" / (task["id"] + ".json")
            save(trace_path, dict(receipt=receipt, trace=trace))
            receipt["trace_sha256"] = sha(trace_path)
            audits.append(receipt)
        group_manifests[name] = dict(
            role="RL optimization"
            if name == "train"
            else "TRAIN-only held-away diagnostic; "
            "not official VAL/HOLDOUT and not a confirmatory generalization claim",
            task_count=len(rows),
            task_ids=[t["id"] for t in rows],
            target_roots=sorted({item for t in rows for item in roots(t)}),
            tasks_path=str(output / name / "tasks.jsonl"),
            tasks_sha256=sha(output / name / "tasks.jsonl"),
            ready=all(a["feasible"] and a["initial_context_fits"] for a in audits),
            audits=audits,
        )
    manifest = dict(
        schema="textcraft-fresh-train-curriculum-20260928-v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        split="train",
        world_seed=42,
        world_sha256=world_sha,
        world_items_per_domain_tier=25,
        semantic_names=False,
        groups=group_manifests,
        ready=all(g["ready"] for g in group_manifests.values()),
        official_source=dict(
            repository_url="https://github.com/ApGa/platoon",
            commit=source.original.COMMIT,
            train_url="https://github.com/ApGa/platoon/blob/"
            + source.original.COMMIT
            + "/plugins/textcraft/platoon/textcraft/textcraft_synth_train.jsonl",
            train_path=str(source.TRAIN),
            train_sha256=source.TRAIN_SHA,
            val_path=str(source.TRAIN.with_name("textcraft_synth_val.jsonl")),
            val_sha256=VAL_SHA,
            license="MIT",
            license_sha256=sha(bridge.CACHE / "LICENSE"),
            acquisition="Reused existing pinned research cache; no new network acquisition.",
            verified_utc=datetime.now(timezone.utc).isoformat(),
        ),
        selection_sha256=sha(output / "SELECTION.json"),
        exclusions_sha256=sha(output / "EXCLUSIONS.json"),
        source_sha256={
            str(p): sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(bridge.__file__).resolve(),
                Path(teacher.__file__).resolve(),
                Path(source.__file__).resolve(),
                Path(source.original.__file__).resolve(),
                Path(worlds.__file__).resolve(),
            )
        },
        tokenizer=dict(
            path=str(source.BASE),
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            manifest_sha256=sha(source.BASE / "local-research-manifest.json"),
        ),
        environment=dict(
            python=sys.version,
            platform=platform.platform(),
            transformers=version("transformers"),
            tokenizers=version("tokenizers"),
            gpu_used=False,
            environment_modified=False,
        ),
        trusted_native_source=bridge.trusted_provenance(),
        failure_policy="All16 selected rows retained regardless of native/context failures; "
        "failed tasks must remain visible, never silently replace or success-filter.",
        limitations="Only original official TRAIN roots/rows in original world42. Shared "
        "ingredients/recipes can overlap old tasks. Structural metadata is CPU-only and not "
        "policy input. Native solvability and initial prompt fit do not guarantee solving "
        "within96 model calls,8192 total emitted tokens or8192 context. No model results used.",
    )
    for name, group_manifest in group_manifests.items():
        save(
            output / name / "MANIFEST.json",
            dict(
                schema="textcraft-fresh-train-group-20260928-v1",
                group=name,
                split="train",
                **group_manifest,
                world_seed=42,
                world_sha256=world_sha,
                all_native_feasible=all(a["feasible"] for a in group_manifest["audits"]),
                initial_context_qualified=all(
                    a["initial_context_fits"] for a in group_manifest["audits"]
                ),
                selection_path=str(output / "SELECTION.json"),
                selection_sha256=manifest["selection_sha256"],
                exclusions_path=str(output / "EXCLUSIONS.json"),
                exclusions_sha256=manifest["exclusions_sha256"],
                official_source=manifest["official_source"],
                trusted_native_source=manifest["trusted_native_source"],
                source_sha256=manifest["source_sha256"],
                row_encoding="Exact original official TRAIN JSONL line bytes, including key order.",
                limitations=manifest["limitations"],
            ),
        )
        group_manifest["manifest_sha256"] = sha(output / name / "MANIFEST.json")
    save(output / "MANIFEST.json", manifest)
    return manifest


def load_group(output: Path, group: str, expected_manifest_sha256: str) -> list[dict]:
    """Minimal fail-closed adapter seam; caller pins manifest once before GPU lock."""
    if group not in GROUPS or sha(output / "MANIFEST.json") != expected_manifest_sha256:
        raise ValueError("unknown group or manifest pin mismatch")
    manifest = json.loads((output / "MANIFEST.json").read_text())
    selected = manifest["groups"][group]
    task_path = output / group / "tasks.jsonl"
    if not selected["ready"] or sha(task_path) != selected["tasks_sha256"]:
        raise ValueError("unready group or tasks pin mismatch; no silent filtering")
    for name, field in (
        ("SELECTION.json", "selection_sha256"),
        ("EXCLUSIONS.json", "exclusions_sha256"),
    ):
        if sha(output / name) != manifest[field]:
            raise ValueError("selection/exclusion provenance changed")
    tasks, official_lines = official_rows()
    del tasks
    raw_lines = task_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw_lines]
    if len(rows) != 8 or [t["id"] for t in rows] != selected["task_ids"]:
        raise ValueError("exact frozen eight TRAIN identities required")
    protected = json.loads((output / "EXCLUSIONS.json").read_text())
    seen_ids, seen_roots = set(protected["task_ids"]), set(protected["target_roots"])
    other = manifest["groups"][GROUPS[1] if group == GROUPS[0] else GROUPS[0]]
    seen_ids.update(other["task_ids"])
    seen_roots.update(other["target_roots"])
    for task, line in zip(rows, raw_lines, strict=True):
        if task["id"] in seen_ids or roots(task) & seen_roots:
            raise ValueError("protected TRAIN/evaluation root or identity overlap")
        if official_lines.get(task["id"]) != line:
            raise ValueError("non-original official TRAIN row bytes")
        seen_ids.add(task["id"])
        seen_roots.update(roots(task))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--qualify-frozen", action="store_true")
    args = parser.parse_args()
    operation = qualify if args.qualify_frozen else prepare
    result = operation(args.root.resolve(), args.output.resolve())
    print(
        json.dumps(
            dict(
                ready=result["ready"],
                output=str(args.output),
                manifest_sha256=sha(args.output / "MANIFEST.json"),
                groups={
                    k: dict(tasks=v["task_count"], ready=v["ready"])
                    for k, v in result["groups"].items()
                },
            )
        )
    )
