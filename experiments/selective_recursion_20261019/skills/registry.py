"""Skill library: signature of a task, retrieval, and the hint block the harness injects (L1)."""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
LIBRARY = {
    "eventlog": dict(path=SKILLS_DIR / "eventlog.py", families={"eventlog"}),
    "oolong": dict(path=SKILLS_DIR / "oolong.py", families={"oolong"}),
}
# Variants selected by RLM_SKILLS_VARIANT (e.g. "honest"): a variant replaces the shipped skill of
# the same name for the run, so a running sweep with the default library is unaffected.
VARIANTS = {
    "honest": {"eventlog": dict(path=SKILLS_DIR / "eventlog_honest.py", families={"eventlog"})},
    # hand-written reference libraries for the new families (E21 to E23), selected per run
    "musique_ref": {"musique": dict(path=SKILLS_DIR / "reference" / "musique_filter_read.py", families={"musique"})},
    "musique_ref3": {"musique": dict(path=SKILLS_DIR / "reference" / "musique_filter_read_v3.py", families={"musique"})},
    "musique_rec": {"musique": dict(path=SKILLS_DIR / "reference" / "musique_recursive.py", families={"musique"})},
    "mrcr_ref": {"mrcr": dict(path=SKILLS_DIR / "reference" / "mrcr_handwritten.py", families={"mrcr"})},
    # libraries written by a proposer model (bootstrap_library.py), reviewed before use; kept out of
    # skills/learned/ so they never shadow a variant in a running job
    "mrcr_boot8b": {"mrcr": dict(path=SKILLS_DIR / "bootstrapped" / "mrcr_8b.py", families={"mrcr"})},
    # E25c: the docstring's typical-use line, added to the 8B library and removed from the reference
    "mrcr_boot8b_use": {"mrcr": dict(path=SKILLS_DIR / "bootstrapped" / "mrcr_8b_use.py", families={"mrcr"})},
    # E25d: documentation form, signature only, and signature plus the usage line without a description
    "mrcr_boot8b_sigonly": {"mrcr": dict(path=SKILLS_DIR / "bootstrapped" / "mrcr_8b_sigonly.py", families={"mrcr"})},
    "mrcr_boot8b_sigusage": {"mrcr": dict(path=SKILLS_DIR / "bootstrapped" / "mrcr_8b_sigusage.py", families={"mrcr"})},
    "mrcr_ref_nouse": {"mrcr": dict(path=SKILLS_DIR / "reference" / "mrcr_handwritten_nouse.py", families={"mrcr"})},
}
LEARNED_DIR = SKILLS_DIR / "learned"


def learned_skills() -> dict:
    """Skills written by the librarian (E08): `skills/learned/<family>_<name>.py` with a
    `# families: a,b` header line; listed after the shipped skills so their helpers are bound."""
    out = {}
    for path in sorted(LEARNED_DIR.glob("*.py")) if LEARNED_DIR.exists() else []:
        head = path.read_text().splitlines()[:15]
        fams = next((l.split(":", 1)[1] for l in head if l.startswith("# families:")), "")
        kind = next((l.split(":", 1)[1].strip() for l in head if l.startswith("# question_kind:")), None)
        pattern = next((l.split(":", 1)[1].strip() for l in head if l.startswith("# question_pattern:")), None)
        if fams.strip():
            out[path.stem] = dict(path=path, families={f.strip() for f in fams.split(",")},
                                  question_kind=kind, question_pattern=pattern)
    return out


def learned_question_kinds() -> list[tuple[str, str]]:
    """(kind, regex) pairs registered by learned skills, so the signature extractor knows new forms."""
    return [(s["question_kind"], s["question_pattern"]) for s in learned_skills().values()
            if s.get("question_kind") and s.get("question_pattern")]


def signature(task) -> dict:
    """Family, length and question type of a task, from whichever task object the harness uses."""
    if getattr(task, "family", None):  # tasks that name their family (MRCR and later families)
        return dict(family=task.family, length=getattr(task, "n_records", None),
                    question_type=getattr(task, "question_type", None))
    row = getattr(task, "row", None)
    qt = getattr(task, "question_type", None) or ""
    is_oolong = row is not None or getattr(task, "context_len", None) is not None or str(qt).startswith("TASK_TYPE")
    family = "oolong" if is_oolong else "eventlog"
    if row is not None:
        length = row["context_len"]
    else:
        length = getattr(task, "context_len", None) or getattr(task, "n_records", None)
    return dict(family=family, length=length, question_type=qt or None)


def all_skills() -> dict:
    import os

    variant = VARIANTS.get(os.environ.get("RLM_SKILLS_VARIANT", ""), {})
    return {**LIBRARY, **variant, **learned_skills()}


def retrieve(sig: dict) -> list[str]:
    return [name for name, s in all_skills().items() if sig["family"] in s["families"]]


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
    lib = all_skills()
    source = "\n\n".join(lib[n]["path"].read_text() for n in names)
    exports = tuple(e for n in names for e in exports_of(lib[n]["path"]))
    return PreludeSpec(source=source, exports=exports)


def policy_note(names: list[str]) -> str:
    lib = all_skills()
    parts = []
    for n in names:
        typical = [l for l in note_of(lib[n]["path"]).splitlines() if l.startswith("Typical use")]
        parts.append(f"Library skill `{n}` is already defined (see the harness prelude list above); "
                     f"typical use:\n{typical[0] if typical else ''}")
    return "\n".join(parts)


def hint_block(names: list[str]) -> str:
    parts = []
    for name in names:
        path = all_skills()[name]["path"]
        parts.append(
            f"Library skill `{name}` (verified on inputs like this one). Define its functions with\n"
            f"    exec(open({str(path)!r}).read())\n"
            f"then call them; they use ask_batch/ask internally.\n{note_of(path)}"
        )
    return "\n\n".join(parts)
