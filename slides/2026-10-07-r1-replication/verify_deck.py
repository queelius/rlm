"""Check this five-slide PDF and render its pages for visual review."""

import json
import unicodedata
from pathlib import Path

import fitz

root = Path(__file__).resolve().parent
document = fitz.open(root / "research-update.pdf")
notes = json.loads((root / "speaker-notes.json").read_text())["slides"]
assert len(document) == len(notes) == 5
output = root / "rendered"
output.mkdir(exist_ok=True)
problems = []
for index, page in enumerate(document):
    compact = "".join(unicodedata.normalize("NFKC", page.get_text()).split())
    assert "".join(notes[index]["title"].split()) in compact
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                rect = fitz.Rect(span["bbox"])
                if not page.rect.contains(rect):
                    problems.append({"page": index + 1, "text": span["text"]})
    page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(output / f"slide-{index + 1:02d}.png")
assert not problems, problems
print("Five slides; titles match notes; all text lies within the page bounds.")
print("Rendered all pages. Visual inspection is still required.")
