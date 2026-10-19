"""Unified scripted expert: a family-agnostic first peek, then family-specific parse+action, then aggregate.

Turn 1 prints the head and tail of the input (the learner cannot know the task family from the
bootstrap alone). Turn 2 reads the printed tail to identify the family and the question, parses the
input and performs the action (extraction/lookup for store logs; classification/counting for
OOLONG). Turn 3 aggregates and submits. Variants change chunk size, prompt phrasing and filter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import variants
from teach import response_object

HERE = Path(__file__).resolve().parent
EVENT = HERE / "expert_cells"
OOLONG = HERE / "oolong_cells"
PEEK = (EVENT / "peek.py").read_text().rstrip()
NEEDS_LABELS = re.compile(r"label|classified", re.I)


def family_of(stdout: str) -> str:
    if "LOG (" in stdout and "QUESTION:" in stdout:
        return "eventlog"
    if "|| User:" in stdout or "In the above data" in stdout or "For the following question" in stdout:
        return "oolong"
    raise RuntimeError("unknown task family in peek output")


def question_of(family: str, stdout: str) -> str:
    if family == "eventlog":
        return stdout.split("QUESTION:", 1)[1].split("\n", 1)[0].strip()
    return [l for l in stdout.strip().split("\n") if l.strip()][-1].strip()


class UnifiedExpert:
    def __init__(self, variant: dict | None = None):
        self.variant = variant or variants.pick("", False)
        self.cells = {}

    def cell(self, cells_dir: Path, *names: str) -> str:
        key = (str(cells_dir), names)
        if key not in self.cells:
            self.cells[key] = "\n".join(variants.render(cells_dir, n, self.variant) for n in names)
        return self.cells[key]

    def complete(self, request, *, timeout, context):
        items = [it for it in request["input"] if it.get("role") == "user"]
        if len(items) == 1:
            return response_object(f"```python\n{PEEK}\n```")
        last = json.loads(items[-1]["content"])
        execution = last.get("execution") or {}
        if execution.get("exception") or last.get("submission", {}).get("status") == "rejected":
            raise RuntimeError("expert cell failed: " + json.dumps(execution)[:500])
        namespace = execution.get("namespace") or {}
        stdout = execution.get("stdout", "")
        if "events" in namespace:
            return response_object(f"```python\n{self.cell(EVENT, 'aggregate')}\n```")
        if "batch" in namespace:
            return response_object(f"```python\n{self.cell(OOLONG, 'aggregate')}\n```")
        if set(namespace) <= {"text"}:
            family = family_of(stdout)
            question = question_of(family, stdout)
            if family == "eventlog":
                action = "lookup" if question.startswith("On which day did") else "extract"
                return response_object(f"```python\n{self.cell(EVENT, 'inspect', action)}\n```")
            action = "classify" if NEEDS_LABELS.search(question) else "count"
            return response_object(f"```python\n{self.cell(OOLONG, 'inspect', action)}\n```")
        raise RuntimeError("expert cannot determine its stage: " + json.dumps(namespace)[:300])
