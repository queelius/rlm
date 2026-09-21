def test_inventory_prunes_constructive_chain_without_changing_declared_depth():
    from audit_textcraft_depth import depth_row

    task = {
        "id": "x",
        "misc": {
            "max_depth": 4,
            "initial_inventory": {"raw": 1},
            "target_items": {"root": 1},
            "gold_trajectory": [
                {"ingredients": {"raw": 1}, "target": ["middle", 1], "result_count": 1},
                {"ingredients": {"middle": 1}, "target": ["root", 1], "result_count": 1},
            ],
        },
    }
    row = depth_row(task, "fixture")
    assert (row["declared_depth"], row["gold_crafts"], row["target_chain_depth"]) == (4, 2, 2)
    task["misc"]["initial_inventory"] = {"middle": 1}
    task["misc"]["gold_trajectory"] = task["misc"]["gold_trajectory"][1:]
    row = depth_row(task, "fixture")
    assert (row["declared_depth"], row["gold_crafts"], row["target_chain_depth"]) == (4, 1, 1)
