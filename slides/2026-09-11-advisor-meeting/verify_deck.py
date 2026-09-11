"""Check PDF content/bounds and render every page for human visual inspection."""

import json
import sys
import unicodedata
from pathlib import Path

import fitz

EXPECTED = [
    "Can a small model solve harder problems",
    "An RLM lets a language model use Python",
    "The helper reads the text",
    "We taught the main model what to do next",
    "Training improved performance",
    "An answer can be right for the wrong record",
    "Matching tags kept answers accurate",
    "Matching tags helped three models",
    "Repeating the same tag beside an input",
    "Small helper improvements were not enough",
    "Reward training has not yet improved",
    "The next test is whether these gains",
    "The longer-term goal is to learn",
    "Which result should we turn into",
]


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "research-update.pdf")
    out = path.parent / "rendered"
    out.mkdir(exist_ok=True)
    doc = fitz.open(path)
    assert len(doc) == len(EXPECTED), (len(doc), len(EXPECTED))
    notes = json.loads((path.parent / "speaker-notes.json").read_text())["slides"]
    assert len(notes) == len(doc), "Notes and compiled PDF have different slide counts"
    problems = []
    page_summaries = []
    for index, page in enumerate(doc):
        text = " ".join(page.get_text().split())
        assert EXPECTED[index] in text, (index + 1, EXPECTED[index])
        assert unicodedata.normalize("NFKC", notes[index]["title"]) in unicodedata.normalize(
            "NFKC", text
        ), (index + 1, "Notes title does not match the compiled PDF")
        assert not list(page.annots() or []), (index + 1, "Audience PDF contains annotations")
        assert f"{index + 1}/{len(EXPECTED)}" in text, (index + 1, "missing page footer")
        spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    rect = fitz.Rect(span["bbox"])
                    if (
                        rect.x0 < -0.5
                        or rect.y0 < -0.5
                        or rect.x1 > page.rect.width + 0.5
                        or rect.y1 > page.rect.height + 0.5
                    ):
                        problems.append(
                            {"page": index + 1, "text": span["text"], "bbox": list(rect)}
                        )
                    spans.append(span)
        page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2)).save(out / f"slide-{index + 1:02d}.png")
        page_summaries.append(
            {
                "page": index + 1,
                "words": len(text.split()),
                "smallest_font_pt": min(s["size"] for s in spans),
            }
        )
    result = {
        "pages": len(doc),
        "out_of_page_text": problems,
        "page_summaries": page_summaries,
        "note": "Bounds checks do not replace inspecting overlap and readability "
        "in every rendered page.",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert not problems, problems


if __name__ == "__main__":
    main()
