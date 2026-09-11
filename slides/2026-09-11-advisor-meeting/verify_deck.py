"""Check PDF content/bounds and render every page for human visual inspection."""

import json
import sys
from pathlib import Path

import fitz

EXPECTED = [
    "Can a small model solve harder problems",
    "An RLM gives the model a workspace",
    "The helper reads the text",
    "We trained the model on examples of actions",
    "Training gains appeared again",
    "An answer can be right for the wrong record",
    "Matching row numbers on both sides",
    "Matching tags helped both tested models",
    "Better individual labels did not reliably",
    "This reward-training recipe",
    "The next experiments should test",
    "The longer-term goal is to learn",
    "Which result should we turn into",
]


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "research-update.pdf")
    out = path.parent / "rendered"
    out.mkdir(exist_ok=True)
    doc = fitz.open(path)
    assert len(doc) == len(EXPECTED), (len(doc), len(EXPECTED))
    problems = []
    page_summaries = []
    for index, page in enumerate(doc):
        text = " ".join(page.get_text().split())
        assert EXPECTED[index] in text, (index + 1, EXPECTED[index])
        assert f"{index + 1}/{len(EXPECTED)}" in text, (index + 1, "missing page footer")
        spans = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    rect = fitz.Rect(span["bbox"])
                    if (rect.x0 < -0.5 or rect.y0 < -0.5
                            or rect.x1 > page.rect.width + 0.5
                            or rect.y1 > page.rect.height + 0.5):
                        problems.append({"page": index + 1, "text": span["text"],
                                         "bbox": list(rect)})
                    spans.append(span)
        page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2)).save(out / f"slide-{index + 1:02d}.png")
        page_summaries.append({"page": index + 1, "words": len(text.split()),
                               "smallest_font_pt": min(s["size"] for s in spans)})
    result = {"pages": len(doc), "out_of_page_text": problems,
              "page_summaries": page_summaries,
              "note": "Bounds checks do not replace inspecting overlap and readability in every rendered page."}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert not problems, problems


if __name__ == "__main__":
    main()
