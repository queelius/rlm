"""Teaching variants: equivalent expert programs that differ in chunk size, prompt phrasing and filter."""

from __future__ import annotations

import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNKS = (6, 12, 24)
PROMPTS = ("a", "b")
FILTERS = {"person_item": "person in l and item in l", "person": "person in l"}


def pick(task_id: str, diversify: bool) -> dict:
    """Deterministic per-task variant; variant 0 reproduces the original expert."""
    if not diversify:
        return dict(chunk=12, prompt="a", filter="person_item")
    rng = random.Random("variant:" + task_id)
    return dict(chunk=rng.choice(CHUNKS), prompt=rng.choice(PROMPTS), filter=rng.choice(list(FILTERS)))


def render(cells_dir: Path, name: str, variant: dict) -> str:
    text = (cells_dir / f"{name}.py").read_text().rstrip()
    if "{{SYSTEM_BODY}}" in text:
        text = text.replace("{{SYSTEM_BODY}}", (cells_dir / f"extract_prompt_{variant['prompt']}.txt").read_text().strip())
    if "{{SYSTEM_EXPR}}" in text:
        text = text.replace("{{SYSTEM_EXPR}}", (cells_dir / f"classify_prompt_{variant['prompt']}.txt").read_text().strip())
    text = text.replace("{{CHUNK}}", str(variant["chunk"])).replace("{{FILTER}}", FILTERS[variant["filter"]])
    assert "{{" not in text, name
    return text
