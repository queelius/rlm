"""Bounded CPU metadata/graph audit; no collectors, model weights or full traces.

Run in the existing training environment for networkx, with CUDA_VISIBLE_DEVICES=.
Native graph access is an offline oracle diagnostic, never policy input.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path("/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921")
PACKAGE = Path(
    "/project/alex_phd/research-cache/repos/"
    "platoon-rao-d9c5857d3a0a056ebc9b047241a2a0c9515aafbe/plugins/textcraft/platoon/textcraft"
)
GENERATOR_SHA = "1c33e0a6f61759eb3a8eb33b35f0b88361155525f5f575885b53acbd011efb0e"
WORLDS = (42, 50, 51, 52)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def graph(world, root: str) -> nx.DiGraph:
    result = nx.DiGraph()

    def visit(item):
        if item in result:
            return
        recipes = world.get_recipes_for_item(item)
        assert len(recipes) == (0 if item in world.base_items else 1)
        count = recipes[0].result_count if recipes else 0
        role = "root" if item == root else "product" if recipes else "base"
        result.add_node(item, role=role, quantity=(role, count))
        if recipes:
            for ingredient, amount in recipes[0].ingredients.items():
                visit(ingredient)
                result.add_edge(item, ingredient, quantity=amount)

    visit(root)
    assert nx.is_directed_acyclic_graph(result)
    return result


def classify(graphs: list[nx.DiGraph], weighted: bool) -> list[int]:
    """Exact rooted DAG isomorphism, preserving shared nodes; no tree unfolding."""
    buckets, classes = defaultdict(list), []
    node_key = "quantity" if weighted else "role"
    node_match = nx.algorithms.isomorphism.categorical_node_match(node_key, None)
    edge_match = (
        nx.algorithms.isomorphism.categorical_edge_match("quantity", None) if weighted else None
    )
    for current in graphs:
        key = (len(current), current.number_of_edges())
        candidates = buckets[key]
        match = next(
            (
                number
                for number, reference in candidates
                if nx.is_isomorphic(current, reference, node_match, edge_match)
            ),
            None,
        )
        if match is None:
            match = sum(map(len, buckets.values()))
            candidates.append((match, current))
        classes.append(match)
    return classes


def main(output: Path) -> None:
    receipts = {}

    def read(path, lines=False, expected=None):
        raw = path.read_bytes()
        receipts[str(path)] = hashlib.sha256(raw).hexdigest()
        assert expected is None or receipts[str(path)] == expected, str(path)
        return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)

    generator_path = PACKAGE / "synth_recipe_generator.py"
    assert hashlib.sha256(generator_path.read_bytes()).hexdigest() == GENERATOR_SHA
    receipts[str(generator_path)] = GENERATOR_SHA
    spec = importlib.util.spec_from_file_location("audited_recipe_generator", generator_path)
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    generator.set_naming_mode(False)
    worlds, snapshots = {}, {}
    for seed in WORLDS:
        world = generator.SynthRecipeDatabase()
        world.generate_all_recipes(seed=seed, items_per_domain_tier=25)
        worlds[seed] = world
        snapshots[seed] = {
            item: [
                {"ingredients": r.ingredients, "result_count": r.result_count, "depth": r.depth}
                for r in recipes
            ]
            for item, recipes in sorted(world.recipes.items())
        }

    train = read(ROOT / "textcraft-quantity-matched-inputs-004/tasks.jsonl", lines=True)
    public = read(ROOT / "textcraft-public-discovery-prototype-001/tasks.jsonl", lines=True)
    assert [
        (t["id"], t["misc"]["target_items"], t["misc"]["initial_inventory"]) for t in train
    ] == [(t["id"], t["misc"]["target_items"], t["misc"]["initial_inventory"]) for t in public]
    cohorts = {"sft": (42, train)}
    for world in WORLDS:
        tasks = []
        for panel in range(6):
            directory = ROOT / f"textcraft-breadth-inputs-001/p{panel:02d}-w{world}"
            manifest = read(directory / "MANIFEST.json")
            assert digest(snapshots[world]) == manifest["world_sha256"]
            tasks.extend(read(directory / "tasks.jsonl", True, manifest["tasks_sha256"]))
        cohorts[f"val{world}"] = (world, tasks)
    fresh = read(ROOT / "textcraft-fresh-train-20260928-001/MANIFEST.json")
    exclusions = read(
        ROOT / "textcraft-fresh-train-20260928-001/EXCLUSIONS.json",
        expected=fresh["exclusions_sha256"],
    )
    for group in ("train", "diagnostic"):
        metadata = fresh["groups"][group]
        cohorts["fresh_" + group] = (
            42,
            read(Path(metadata["tasks_path"]), True, metadata["tasks_sha256"]),
        )
    records, graphs = [], []
    for cohort, (seed, tasks) in cohorts.items():
        for task in tasks:
            targets = task["misc"]["target_items"]
            assert len(targets) == 1
            root = next(iter(targets))
            current = graph(worlds[seed], root)
            products = sorted(n for n in current if n not in worlds[seed].base_items)
            rootless = {"@ROOT" if n == root else n: snapshots[seed][n] for n in products}
            records.append(
                dict(
                    cohort=cohort,
                    world=seed,
                    task_id=task["id"],
                    root=root,
                    targets=targets,
                    inventory=task["misc"]["initial_inventory"],
                    declared_depth=task["misc"]["max_depth"],
                    products=products,
                    exact_root_erased_closure=digest(rootless),
                )
            )
            graphs.append(current)
    for weighted, key in ((False, "topology"), (True, "weighted_topology")):
        for row, label in zip(records, classify(graphs, weighted), strict=True):
            row[key] = label
    groups = {name: [r for r in records if r["cohort"] == name] for name in cohorts}
    sft_products = set().union(*(set(r["products"]) for r in groups["sft"]))
    sft_roots = {r["root"] for r in groups["sft"]}
    supervised_queries = {}
    for teacher, directory in (
        ("discovery", "textcraft-public-discovery-prototype-001"),
        ("known_recipe_corrected", "textcraft-quantity-matched-inputs-004"),
    ):
        rows = read(ROOT / directory / "rows.jsonl", lines=True)
        actions = [json.loads(row["target"]) for row in rows]
        queried = {item for a in actions if a["action"] == "get_info" for item in a["items"]}
        assert queried == sft_products
        supervised_queries[teacher] = len(queried)
    summaries = {}
    for name, rows in groups.items():
        summaries[name] = dict(
            tasks=len(rows),
            roots=len({r["root"] for r in rows}),
            root_multiplicity=dict(Counter(Counter(r["root"] for r in rows).values())),
            goal_quantity=len({canonical(r["targets"]) for r in rows}),
            inventories=len({canonical(r["inventory"]) for r in rows}),
            topology_classes=len({r["topology"] for r in rows}),
            weighted_classes=len({r["weighted_topology"] for r in rows}),
            root_erased_named_closures=len({r["exact_root_erased_closure"] for r in rows}),
            roots_in_sft_roots=sum(r["root"] in sft_roots for r in rows),
            roots_in_sft_dependency_products=sum(r["root"] in sft_products for r in rows),
            any_sft_product_names_shared=sum(bool(set(r["products"]) & sft_products) for r in rows),
            any_exact_sft_recipes_shared=sum(
                any(
                    snapshots[r["world"]][item] == snapshots[42][item]
                    for item in set(r["products"]) & sft_products
                )
                for r in rows
            ),
            topology_seen_in_sft=sum(
                r["topology"] in {t["topology"] for t in groups["sft"]} for r in rows
            ),
            weighted_topology_seen_in_sft=sum(
                r["weighted_topology"] in {t["weighted_topology"] for t in groups["sft"]}
                for r in rows
            ),
            inventory_overlap_sft=sum(
                canonical(r["inventory"]) in {canonical(t["inventory"]) for t in groups["sft"]}
                for r in rows
            ),
        )
    val = groups["val42"]
    by_world = {w: {r["task_id"]: r for r in groups[f"val{w}"]} for w in WORLDS}
    cross = []
    for a, b in itertools.combinations(WORLDS, 2):
        pairs = [(by_world[a][r["task_id"]], by_world[b][r["task_id"]]) for r in val]
        cross.append(
            dict(
                worlds=[a, b],
                tasks=len(pairs),
                same_namespace=worlds[a].all_items == worlds[b].all_items,
                same_roots=sum(x["root"] == y["root"] for x, y in pairs),
                same_inventory=sum(x["inventory"] == y["inventory"] for x, y in pairs),
                same_root_recipe=sum(
                    snapshots[a][x["root"]] == snapshots[b][y["root"]] for x, y in pairs
                ),
                same_named_closure=sum(
                    x["exact_root_erased_closure"] == y["exact_root_erased_closure"]
                    for x, y in pairs
                ),
                same_topology=sum(x["topology"] == y["topology"] for x, y in pairs),
                same_weighted_topology=sum(
                    x["weighted_topology"] == y["weighted_topology"] for x, y in pairs
                ),
            )
        )
    # Root identity itself is bijective with all 48 task IDs. This broader sensitivity
    # instead groups roots sharing any craftable prerequisite within any fixed world.
    overlap = nx.Graph()
    overlap.add_nodes_from(r["task_id"] for r in val)
    for a, b in itertools.combinations(overlap.nodes, 2):
        if any(set(by_world[w][a]["products"]) & set(by_world[w][b]["products"]) for w in WORLDS):
            overlap.add_edge(a, b)
    components = sorted((sorted(c) for c in nx.connected_components(overlap)), key=lambda c: c[0])
    finding = read(Path(__file__).with_name("finding_textcraft_breadth_20260928.json"))
    details = read(
        Path(finding["evidence"]["details"]), expected=finding["evidence"]["details_sha256"]
    )
    coefficients = {
        "discovery_binder_minus_raw": {"discovery/binder": 1, "discovery/raw": -1},
        "discovery_minus_known_recipe_raw": {"discovery/raw": 1, "known_recipe_corrected/raw": -1},
        "discovery_minus_known_recipe_binder": {
            "discovery/binder": 1,
            "known_recipe_corrected/binder": -1,
        },
    }
    intervals = {}
    for name, weights in coefficients.items():
        totals = Counter()
        for row in details["episodes"]:
            totals[row["task_id"]] += weights.get(row["arm"], 0) * row["success"]
        values = [
            (sum(totals[t] for t in component), 16 * len(component)) for component in components
        ]
        samples, rng = [], random.Random(20260928)
        for _ in range(20000):
            selected = rng.choices(values, k=len(values))
            samples.append(sum(s for s, _ in selected) / sum(n for _, n in selected))
        samples.sort()
        intervals[name] = dict(
            original_task_cluster_ci95=finding["contrasts"][name]["ci95"],
            original_task_and_literal_root_clusters=48,
            shared_recipe_components=len(components),
            difference=sum(totals.values()) / 768,
            component_ratio_bootstrap_ci95=[samples[int(19999 * q)] for q in (0.025, 0.975)],
        )
        assert intervals[name]["difference"] == finding["contrasts"][name]["difference"]
    official = {}
    for split in ("train", "val"):
        tasks = read(PACKAGE / f"textcraft_synth_{split}.jsonl", lines=True)
        official[split] = dict(
            rows=len(tasks),
            roots=sorted({n for t in tasks for n in t["misc"]["target_items"]}),
        )
    fresh_a, fresh_b = groups["fresh_train"], groups["fresh_diagnostic"]
    products_a = set().union(*(set(r["products"]) for r in fresh_a))
    products_b = set().union(*(set(r["products"]) for r in fresh_b))
    report = dict(
        schema="textcraft-generalization-audit-20260928-v1",
        cohorts=summaries,
        official={
            k: {"rows": v["rows"], "unique_roots": len(v["roots"])} for k, v in official.items()
        },
        official_train_val_root_overlap=len(
            set(official["train"]["roots"]) & set(official["val"]["roots"])
        ),
        sft_supervised_query_products=supervised_queries,
        val_roots_queried_during_sft=sorted(r["root"] for r in val if r["root"] in sft_products),
        root_identity_map=[{k: r[k] for k in ("task_id", "root", "declared_depth")} for r in val],
        fresh_exclusion=dict(
            all_protected_root_strings=len(exclusions["target_roots"]),
            official_val_roots_covered=set(official["val"]["roots"])
            <= set(exclusions["target_roots"]),
            fresh_protected_root_overlap=sum(
                r["root"] in exclusions["target_roots"]
                for r in records
                if r["cohort"].startswith("fresh_")
            ),
            rule="Exact item strings, independent of requested quantity; "
            "not prerequisite or graph disjointness",
        ),
        fresh_cross_group=dict(
            shared_craftable_products=len(products_a & products_b),
            pairs_sharing_craftable_products=sum(
                bool(set(a["products"]) & set(b["products"]))
                for a, b in itertools.product(fresh_a, fresh_b)
            ),
            task_pairs=len(fresh_a) * len(fresh_b),
            a_roots_in_b_closure=sorted(r["root"] for r in fresh_a if r["root"] in products_b),
            b_roots_in_a_closure=sorted(r["root"] for r in fresh_b if r["root"] in products_a),
        ),
        cross_world=cross,
        shared_recipe_components=components,
        interval_sensitivity=intervals,
        namespace_counts={str(w): len(worlds[w].all_items) for w in WORLDS},
        receipts=receipts,
        networkx=nx.__version__,
        bootstrap_seed=20260928,
        bootstrap_draws=20000,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat="Hidden native dependency graph is offline audit only. Shape excludes names, "
        "goal quantity and inventory; weighted shape retains yields and ingredient quantities. "
        "Five-or-few-component intervals are unstable descriptive sensitivity, "
        "not calibrated confirmatory inference.",
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("records", "receipts", "shared_recipe_components")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
