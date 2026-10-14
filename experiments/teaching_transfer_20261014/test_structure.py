"""Guard against calling renamed graphs or duplicated branches fresh structure."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).with_name("structure.py")
    assert path.exists(), "structural novelty implementation missing"
    spec = importlib.util.spec_from_file_location("teaching_structure", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_renaming_and_quantity_changes_do_not_create_novel_structure():
    m = module()
    a = m.dependency_graph({"r": {"p": 2}, "p": {"b": 1}}, "r")
    b = m.dependency_graph({"x": {"y": 17}, "y": {"z": 4}}, "x")
    index = m.StructureIndex()
    assert index.add(a)
    assert not index.add(b)
    assert index.contains(b)


def test_shared_ingredient_is_not_unfolded_into_two_independent_leaves():
    m = module()
    shared = m.dependency_graph({"r": {"p": 1, "q": 1}, "p": {"b": 1}, "q": {"b": 1}}, "r")
    separate = m.dependency_graph({"r": {"p": 1, "q": 1}, "p": {"b": 1}, "q": {"c": 1}}, "r")
    index = m.StructureIndex()
    index.add(shared)
    assert not index.contains(separate)
    assert len(shared) == 4
    assert len(separate) == 5


def test_cycles_are_rejected_and_large_graphs_are_explicitly_out_of_scope():
    m = module()
    with pytest.raises(ValueError, match="cycle"):
        m.dependency_graph({"r": {"p": 1}, "p": {"r": 1}}, "r")
    assert m.dependency_graph({"r": {"p": 1}, "p": {"b": 1}}, "r", max_nodes=2) is None


def test_hash_collision_does_not_mean_isomorphism():
    m = module()
    a = m.dependency_graph({"r": {"a": 1}}, "r")
    b = m.dependency_graph({"r": {"a": 1, "b": 1}}, "r")
    index = m.StructureIndex(bucket=lambda graph: "deliberate collision")
    index.add(a)
    assert not index.contains(b)


def test_selection_is_input_order_independent_and_has_no_duplicate_structure():
    m = module()
    recipes = {"a": {"raw": 1}, "b": {"raw": 4}, "c": {"raw": 1, "raw2": 1}}
    candidates = [(name, 3, m.dependency_graph(recipes, name)) for name in recipes]
    a = m.select(candidates, m.StructureIndex(), {3: 2}, 123)
    b = m.select(list(reversed(candidates)), m.StructureIndex(), {3: 2}, 123)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert "c" in [x[0] for x in a]
    assert len(a) == 2
    with pytest.raises(ValueError, match="insufficient"):
        m.select(candidates, m.StructureIndex(), {3: 3}, 123)
