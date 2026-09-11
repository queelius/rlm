"""Check PDF content/bounds and render every page for human visual inspection."""

import json
import sys
import unicodedata
from pathlib import Path

import fitz

EXPECTED = [
    "Can better handoffs make an RLM more reliable?",
    "A long file can become many small questions whose answers Python can combine.",
    "Training taught a more reliable routine for asking helpers and calculating answers.",
    "We changed how helper answers are linked to the text they describe.",
    "Matching names improved accuracy when a helper answered many questions at once.",
    "Smaller calls were faster than large named calls in this local test.",
    "Proposed RLM change: manage record links and make helper group size an explicit choice.",
    "We have a promising handoff result and a focused next research question.",
    "Example: the helper reads the text, and Python does the counting.",
    "A control suggests that matching matters, not merely having names on the page.",
    "Reward training did not improve the final-answer count in this trial.",
    "Training also helped with new combinations of familiar steps.",
    "Misleading names can draw an answer toward the wrong input.",
    "Both row numbers and arbitrary names helped in the larger-batch tests.",
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
        footer = f"Main talk {index + 1}/8" if index < 8 else f"Backup {index - 7}/6"
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
