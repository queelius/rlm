"""Skill library: signature of a task, retrieval, and the hint block the harness injects (L1)."""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
LIBRARY = {
    "eventlog": dict(path=SKILLS_DIR / "eventlog.py", families={"eventlog"}),
    "oolong": dict(path=SKILLS_DIR / "oolong.py", families={"oolong"}),
}


def signature(task) -> dict:
    row = getattr(task, "row", None)
    family = "oolong" if row is not None else "eventlog"
    length = row["context_len"] if row is not None else task.n_records
    return dict(family=family, length=length, question_type=getattr(task, "question_type", None))


def retrieve(sig: dict) -> list[str]:
    return [name for name, s in LIBRARY.items() if sig["family"] in s["families"]]


def note_of(path: Path) -> str:
    text = path.read_text()
    return text.split('"""', 2)[1].strip()


def hint_block(names: list[str]) -> str:
    parts = []
    for name in names:
        path = LIBRARY[name]["path"]
        parts.append(
            f"Library skill `{name}` (verified on inputs like this one). Define its functions with\n"
            f"    exec(open({str(path)!r}).read())\n"
            f"then call them; they use ask_batch/ask internally.\n{note_of(path)}"
        )
    return "\n\n".join(parts)
