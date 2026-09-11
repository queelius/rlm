"""Check PDF content/bounds and render every page for human visual inspection."""

import json
import sys
import unicodedata
from pathlib import Path

import fitz

EXPECTED = [
    "Can better handoffs make an RLM more reliable?",
    "An RLM divides a large task into smaller steps.",
    "We used supervised fine-tuning (SFT) to teach the main model a routine.",
    "Both separately trained copies solved more of the 72 test tasks.",
    "We tested whether names help link each helper answer to the right statement.",
    "Matching names helped helpers judge many statements in one request.",
    "Smaller requests were a strong alternative to adding names to large requests.",
    "Next test: let the RLM program organize helper requests and match their answers.",
    "We have a promising handoff result and a focused next research question.",
    "Example: the helper identifies question types, and Python adds the relevant points.",
    "Using the same name on a statement and its answer helped more than using different names.",
    "Reward training did not improve the final-answer count in this trial.",
    "Training also helped with new combinations of familiar steps.",
    "Misleading names can draw an answer toward the wrong input.",
    "Both row numbers and arbitrary names helped when helpers judged many statements together.",
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
        text = unicodedata.normalize("NFKC", " ".join(page.get_text().split()))
        # PDF extraction can omit a space after a ligature even when it is visible.
        compact = "".join(text.split())
        assert "".join(EXPECTED[index].split()) in compact, (index + 1, EXPECTED[index])
        assert "".join(unicodedata.normalize("NFKC", notes[index]["title"]).split()) in compact, (
            index + 1,
            "Notes title does not match the compiled PDF",
        )
        assert not list(page.annots() or []), (index + 1, "Audience PDF contains annotations")
        footer = f"Main talk {index + 1}/9" if index < 9 else f"Backup {index - 8}/6"
        assert footer in text, (index + 1, "missing main/backup page footer")
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
