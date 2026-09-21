"""Bounded CPU comparison of pinned seed42/43 recipe worlds, not a task panel."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import textcraft_bridge as bridge


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot(world):
    return {
        item: [
            {"ingredients": r.ingredients, "result_count": r.result_count, "depth": r.depth}
            for r in recipes
        ]
        for item, recipes in sorted(world.recipes.items())
    }


def graph_summary(world):
    @lru_cache(None)
    def depth(item):
        if world.is_base_item(item):
            return 0
        recipes = world.get_recipes_for_item(item)
        if len(recipes) != 1:
            raise ValueError("this inventory is not single-recipe")
        return 1 + max(depth(i) for i in recipes[0].ingredients)

    return {
        "items": len(world.all_items),
        "base_items": len(world.base_items),
        "recipes": len(world.recipes),
        "declared_depth_counts": dict(Counter(world.item_depths.values())),
        "actual_longest_path_counts": dict(Counter(depth(i) for i in world.all_items)),
        "ingredient_degree_counts": dict(
            Counter(len(r[0].ingredients) for r in world.recipes.values())
        ),
        "strictly_lower_declared_dependencies": all(
            world.item_depths[i] < r[0].depth
            for r in world.recipes.values()
            for i in r[0].ingredients
        ),
    }


def compare(root, output):
    if output.exists():
        raise FileExistsError("immutable comparison exists")
    original = bridge.load_world()
    generator = sys.modules["pinned_textcraft_synth_generator_d9c5857d"]
    changed = generator.SynthRecipeDatabase()
    changed.generate_all_recipes(seed=43, items_per_domain_tier=25)
    a, b = snapshot(original), snapshot(changed)
    old_audit_path = root / "textcraft-inputs-001/TOKEN-AUDIT.json"
    old_audit = json.loads(old_audit_path.read_text())
    if digest(a) != old_audit["world_sha256"]:
        raise ValueError("regenerated seed42 world differs from frozen pilot")
    common = sorted(set(a) & set(b))
    item = "m0_i2"
    examples = {}
    for seed, world in ((42, original), (43, changed)):
        recipe = world.get_recipes_for_item(item)[0]
        frame = bridge.Frame(
            world, dict(recipe.ingredients), {item: 1}, bridge.Budget(), max_depth=0
        )
        info = frame.apply({"action": "get_info", "items": [item]})
        feedback = frame.apply(
            {
                "action": "craft",
                "ingredients": recipe.ingredients,
                "target_item": item,
                "output_count": recipe.result_count,
            }
        )
        frame.apply({"action": "finish", "message": "done"})
        score, details = frame.score()
        if score != 1:
            raise ValueError("correct native one-batch contract failed")
        examples[str(seed)] = {
            "public_info": info,
            "correct_craft_feedback": feedback,
            "native_score": score,
            "native_details": details,
        }
    old_recipe = original.get_recipes_for_item(item)[0]
    wrong = bridge.Frame(
        changed, dict(old_recipe.ingredients), {item: 1}, bridge.Budget(), max_depth=0
    )
    wrong_feedback = wrong.apply(
        {
            "action": "craft",
            "ingredients": old_recipe.ingredients,
            "target_item": item,
            "output_count": old_recipe.result_count,
        }
    )
    report = {
        "schema": "textcraft-changed-world-feasibility-v1",
        "seeds": [42, 43],
        "items_per_domain_tier": 25,
        "semantic_names": False,
        "world_sha256": {"42": digest(a), "43": digest(b)},
        "same_item_namespace": original.all_items == changed.all_items,
        "same_base_items": original.base_items == changed.base_items,
        "same_declared_item_depths": original.item_depths == changed.item_depths,
        "common_recipe_items": len(common),
        "same_named_recipe_changes": sum(a[i] != b[i] for i in common),
        "same_named_ingredient_identity_changes": sum(
            set(a[i][0]["ingredients"]) != set(b[i][0]["ingredients"]) for i in common
        ),
        "same_named_result_count_changes": sum(
            a[i][0]["result_count"] != b[i][0]["result_count"] for i in common
        ),
        "world_summaries": {"42": graph_summary(original), "43": graph_summary(changed)},
        "same_item_example": {
            "item": item,
            "worlds": examples,
            "seed42_recipe_in_seed43_feedback": wrong_feedback,
        },
        "trusted_source": bridge.trusted_provenance(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                Path(bridge.__file__).resolve(),
                bridge.PACKAGE / "synth_tasks.py",
                old_audit_path,
            )
        },
        "scope": "Two complete in-memory recipe worlds and one native API fixture. "
        "No model calls, generated task dataset, task selection, or experiment panel.",
        "limits": "Matching item names and declared tiers does not match concrete dependency "
        "depth, ingredient branching, base inventory demand, optimal cost or prompt "
        "length. Existing seed42 tasks/gold must not be reused unchanged in seed43.",
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.root.resolve(), args.output.resolve())))
