"""Teacher-forced strata use public structure, not names mentioned in schema text."""

import importlib.util
import json
import sys
from pathlib import Path


def test_query_visibility_uses_prior_returned_recipe_not_whole_prompt_substrings():
    path = Path(__file__).with_name("nll.py")
    assert path.exists(), "optional NLL implementation has not been written"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("dose_nll_fixture", path)
    nll = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nll)
    tasks = [dict(id="train0", misc=dict(target_items={"root": 1}, initial_inventory={"ore": 1}))]
    rows = [
        dict(
            task_id="train0",
            step=0,
            prompt="example includes hidden",
            target=json.dumps(dict(action="get_info", items=["hidden"])),
            feedback=[dict(item="hidden", recipes=[])],
        ),
        dict(
            task_id="train0",
            step=1,
            target=json.dumps(dict(action="get_info", items=["root"])),
            feedback=[dict(item="root", recipes=[dict(ingredients={"leaf": 1})])],
        ),
        dict(
            task_id="train0",
            step=2,
            target=json.dumps(dict(action="get_info", items=["leaf"])),
            feedback=[dict(item="leaf", recipes=[])],
        ),
    ]
    assert nll.categories(rows, tasks) == [
        ["all", "get_info", "get_info_unseen", "first_action"],
        ["all", "get_info", "get_info_visible"],
        ["all", "get_info", "get_info_visible"],
    ]
