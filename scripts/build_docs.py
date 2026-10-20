"""Generate the site's derived pages before `mkdocs build` (run locally or in CI).

- docs/notebook/index.md: every dated note under docs/notebook, newest first, with its title.
- docs/experiments/: a copy of experiments/INDEX.md and each experiments/*/EXPERIMENT.md.
- docs/results/index.md: the ledger files under docs/results.
Generated files are ignored by git; the sources of truth stay where they are.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def title_of(path: Path) -> str:
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def notebook_index() -> None:
    notes = sorted(DOCS.glob("notebook/*.md"), reverse=True)
    notes = [n for n in notes if n.name != "index.md"]
    lines = ["# Lab notebook", "",
             "Dated checkpoint notes, newest first. Each note records what was run, what it showed, "
             "and what it changed; the claims they support are summarized in the [claims ledger](../RESULTS.md).",
             ""]
    for n in notes:
        date = re.match(r"(\d{4}-\d{2}-\d{2})", n.name)
        lines.append(f"- {date.group(1) if date else ''} [{title_of(n)}]({n.name})")
    extra = sorted(DOCS.glob("notebook/*.json"))
    if extra:
        lines += ["", "Machine-readable companions: " + ", ".join(f"`{e.name}`" for e in extra)]
    (DOCS / "notebook" / "index.md").write_text("\n".join(lines) + "\n")


def experiments_pages() -> None:
    out = DOCS / "experiments"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    index = (ROOT / "experiments" / "INDEX.md").read_text()
    index = index.replace("`docs/RESULTS.md`", "[the claims ledger](../RESULTS.md)")
    (out / "index.md").write_text(index)
    for exp in sorted((ROOT / "experiments").glob("*/EXPERIMENT.md")):
        body = exp.read_text().replace("docs/notebook/", "../notebook/").replace("`docs/RESULTS.md`", "[the claims ledger](../RESULTS.md)")
        (out / f"{exp.parent.name}.md").write_text(body)


def results_index() -> None:
    files = sorted((DOCS / "results").glob("*.md"))
    lines = ["# Full ledgers", "", "Complete claim lists behind the [summary ledger](../RESULTS.md):", ""]
    lines += [f"- [{title_of(f)}]({f.name})" for f in files if f.name != "index.md"]
    (DOCS / "results" / "index.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    notebook_index(); experiments_pages(); results_index()
    print("generated notebook/index.md, experiments/, results/index.md")
