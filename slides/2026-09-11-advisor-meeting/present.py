"""Build pdfpc notes and open a laptop-friendly presenter view (Python standard library)."""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def make_metadata(source: str, notes: dict) -> dict:
    """Reject missing/reordered notes instead of attaching them to the wrong slide."""
    titles = [
        " ".join(title.replace(r"\\", " ").split())
        for title in re.findall(r"\\frametitle\{([^{}]+)\}", source)
    ]
    slides = notes["slides"]
    if not titles or len(titles) != len(slides):
        raise ValueError("Provide exactly one note for every frame title.")
    pages = []
    for index, (title, slide) in enumerate(zip(titles, slides, strict=True)):
        if title != slide["title"]:
            raise ValueError(f"Slide {index + 1}: notes title does not match {title!r}.")
        note = slide["notes"].strip()
        if not note or len(note.split()) > 100 or len(note) > 850:
            raise ValueError(
                f"Slide {index + 1}: keep notes nonempty and under 100 words/850 characters."
            )
        # pdfpc 4.6 plain text does not wrap itself. Preserve paragraph breaks.
        wrapped = "\n\n".join(
            textwrap.fill(paragraph, width=42) for paragraph in note.split("\n\n")
        )
        if len(wrapped.splitlines()) > 18:
            raise ValueError(f"Slide {index + 1}: move longer explanations to the speaker guide.")
        pages.append({"idx": index, "label": str(index + 1), "overlay": 0, "note": wrapped})
    return {"pdfpcFormat": 2, "disableMarkdown": True, "noteFontSize": 16, "pages": pages}


def launch_args(pdfpc: str, pdf: str, metadata: str, mode: str, duration: int) -> list[str]:
    """Two windows allow a single-screen caller to share only the audience window."""
    command = [
        pdfpc,
        f"--pdfpc-location={metadata}",
        f"--duration={duration}",
        "--note-format=plain",
    ]
    if mode == "laptop":
        command.append("--windowed=both")
    elif mode == "rehearse":
        command.extend(["--single-screen", "--windowed=none"])
    elif mode == "dual":
        command.append("--windowed=none")
    else:
        raise ValueError(f"Unknown presenter mode: {mode}")
    return command + [pdf]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["laptop", "rehearse", "dual"], default="laptop")
    parser.add_argument("--duration", type=int, default=10, help="Talk timer, in minutes")
    parser.add_argument("--pdfpc", default=os.environ.get("PDFPC", "pdfpc"))
    parser.add_argument(
        "--write-notes", action="store_true", help="Regenerate the portable .pdfpc file"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check the portable notes match their source"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the launch command without opening windows"
    )
    args = parser.parse_args()
    if args.duration < 0:
        parser.error("duration must be nonnegative")
    try:
        metadata = make_metadata(
            (ROOT / "research-update.tex").read_text(),
            json.loads((ROOT / "speaker-notes.json").read_text()),
        )
        output = ROOT / "research-update.pdfpc"
        text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        if args.write_notes:
            output.write_text(text)
            print(f"Wrote {len(metadata['pages'])} slide notes to {output.name}.")
            return 0
        if args.check:
            if json.loads(output.read_text()) != metadata:
                raise ValueError("Portable notes are stale; run make notes.")
            print(f"All {len(metadata['pages'])} notes match the current frame order.")
            return 0
        pdf = ROOT / "research-update.pdf"
        if not pdf.is_file():
            raise ValueError("Build research-update.pdf first with make or make tectonic.")
        if args.dry_run:
            preview = ROOT / ".presenter" / "SESSION" / "research-update.pdfpc"
            print(
                shlex.join(
                    launch_args(args.pdfpc, str(pdf), str(preview), args.mode, args.duration)
                )
            )
            return 0
        if shutil.which(args.pdfpc) is None:
            raise ValueError(
                "pdfpc is not installed. See PRESENTING.md for installation instructions."
            )
        runtime = ROOT / ".presenter"
        runtime.mkdir(exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix="session-", dir=runtime))
        working_notes = session / "research-update.pdfpc"
        working_notes.write_text(text)
        config = session / "config" / "pdfpc"
        config.mkdir(parents=True)
        shutil.copyfile(ROOT / "pdfpcrc", config / "pdfpcrc")
        environment = dict(os.environ, XDG_CONFIG_HOME=str(session / "config"))
        print(
            "Share only the audience slide window, never the whole desktop or presenter view.",
            flush=True,
        )
        print(f"Any notes edited in pdfpc are saved in {working_notes}", flush=True)
        if args.mode == "laptop":
            print(
                "Click the presenter window and press w to fill the screen so all notes fit.",
                flush=True,
            )
        return subprocess.call(
            launch_args(args.pdfpc, str(pdf), str(working_notes), args.mode, args.duration),
            env=environment,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Presenter setup: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
