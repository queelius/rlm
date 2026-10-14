"""Name/quantity-invariant rooted DAG checks for prospective teaching transfer."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import networkx as nx


def dependency_graph(recipes: dict, root: str, max_nodes: int = 64):
    """Retain shared prerequisites; return None only for the explicit size cap."""
    graph = nx.DiGraph()
    active = set()

    def visit(item):
        if item in active:
            raise ValueError("dependency cycle")
        if item in graph:
            return
        graph.add_node(
            item, kind="root" if item == root else ("product" if item in recipes else "base")
        )
        if len(graph) > max_nodes:
            raise OverflowError
        active.add(item)
        for child in recipes.get(item, {}):
            visit(child)
            graph.add_edge(item, child)
        active.remove(item)

    try:
        visit(root)
    except OverflowError:
        return None
    return graph


def graph_bucket(graph):
    """Cheap isomorphism invariant, not a proof of graph equality."""
    degrees = sorted(
        (graph.nodes[n]["kind"], graph.in_degree(n), graph.out_degree(n)) for n in graph
    )
    return tuple(degrees)


class StructureIndex:
    def __init__(self, bucket=graph_bucket):
        self.bucket = bucket
        self.graphs = defaultdict(list)
        self.node_match = nx.algorithms.isomorphism.categorical_node_match("kind", None)

    def contains(self, graph) -> bool:
        return any(
            nx.is_isomorphic(graph, other, node_match=self.node_match)
            for other in self.graphs[self.bucket(graph)]
        )

    def add(self, graph) -> bool:
        if self.contains(graph):
            return False
        self.graphs[self.bucket(graph)].append(graph)
        return True


def select(candidates: list, excluded: StructureIndex, strata: dict, seed: int) -> list:
    """Select by frozen hash rank, without model outcomes or native feasibility."""
    selected, used = [], StructureIndex()
    for depth, count in sorted(strata.items()):
        eligible = sorted(
            (row for row in candidates if row[1] == depth and row[2] is not None),
            key=lambda row: hashlib.sha256(f"{seed}:{row[0]}".encode()).hexdigest(),
        )
        group = []
        for row in eligible:
            if not excluded.contains(row[2]) and used.add(row[2]):
                group.append(row)
                if len(group) == count:
                    break
        if len(group) != count:
            raise ValueError(f"insufficient novel structures in depth{depth}: {len(group)}/{count}")
        selected.extend(group)
    return selected
