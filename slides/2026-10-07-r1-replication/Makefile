# On your local desktop, from the repository root:
#   make -C slides/2026-10-07-r1-replication present
#   make -C slides/2026-10-07-r1-replication rehearse
# These launch the committed PDF and speaker notes; no LaTeX build is needed.
# Install pdfpc first (see README.md). Share only the audience window.
LATEXMK ?= latexmk
TECTONIC ?= tectonic
PDFPC ?= pdfpc
PYTHON ?= python3

.PHONY: all pdf tectonic notes notes-check present rehearse
all: pdf notes
pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error research-update.tex
tectonic:
	$(TECTONIC) --keep-logs research-update.tex
notes:
	$(PYTHON) -c 'import json, runpy; from pathlib import Path; root=Path("."); build=runpy.run_path("../2026-09-11-advisor-meeting/present.py")["make_metadata"]; notes=build((root/"research-update.tex").read_text(),json.loads((root/"speaker-notes.json").read_text())); (root/"research-update.pdfpc").write_text(json.dumps(notes,ensure_ascii=False,indent=2)+"\n"); print("Wrote",len(notes["pages"]),"notes.")'
notes-check:
	$(PYTHON) -c 'import json, runpy; from pathlib import Path; root=Path("."); build=runpy.run_path("../2026-09-11-advisor-meeting/present.py")["make_metadata"]; expected=build((root/"research-update.tex").read_text(),json.loads((root/"speaker-notes.json").read_text())); actual=json.loads((root/"research-update.pdfpc").read_text()); assert actual==expected; assert len(actual["pages"])==6; assert max(len(p["note"].splitlines()) for p in actual["pages"])<=14; print("Six matching notes, at most 14 lines, wrapped at 42 characters.")'
# Two movable windows: audience slides and private presenter notes.
present:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=6 --note-format=plain --windowed=both research-update.pdf
# Presenter view only, for reviewing the slides and notes on one screen.
rehearse:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=6 --note-format=plain --single-screen --windowed=none research-update.pdf
