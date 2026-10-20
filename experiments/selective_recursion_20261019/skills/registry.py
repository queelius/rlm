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


def exports_of(path: Path) -> tuple:
    """(name, description) pairs from the docstring lines of the form `name(...) -> ... # note`."""
    out = []
    for line in note_of(path).splitlines():
        line = line.strip()
        if "->" in line and line.split("(")[0].isidentifier():
            name = line.split("(")[0].strip()
            desc = line.split("#", 1)[1].strip() if "#" in line else line
            out.append((name, (line.split("#")[0].strip() + ". " + desc).strip()))
    return tuple(out)


def prelude_for(names: list[str]):
    """A PreludeSpec with the retrieved skills' source and exports (None when nothing applies)."""
    from rlm.specs import PreludeSpec

    if not names:
        return None
    source = "\n\n".join(LIBRARY[n]["path"].read_text() for n in names)
    exports = tuple(e for n in names for e in exports_of(LIBRARY[n]["path"]))
    return PreludeSpec(source=source, exports=exports)


def policy_note(names: list[str]) -> str:
    return "\n".join(
        f"Library skill `{n}` is already defined (see the harness prelude list above); "
        f"typical use:\n{[l for l in note_of(LIBRARY[n]['path']).splitlines() if l.startswith('Typical use')][0] if any(l.startswith('Typical use') for l in note_of(LIBRARY[n]['path']).splitlines()) else ''}"
        for n in names
    )


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
