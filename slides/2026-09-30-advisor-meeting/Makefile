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
	$(PYTHON) -c 'import json, runpy; from pathlib import Path; root=Path("."); build=runpy.run_path("../2026-09-11-advisor-meeting/present.py")["make_metadata"]; expected=build((root/"research-update.tex").read_text(),json.loads((root/"speaker-notes.json").read_text())); actual=json.loads((root/"research-update.pdfpc").read_text()); assert actual==expected; assert len(actual["pages"])==8; assert max(len(p["note"].splitlines()) for p in actual["pages"])<=14; print("Eight matching notes, at most 14 lines, wrapped at 42 characters.")'
present:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=10 --note-format=plain --windowed=both research-update.pdf
rehearse:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=10 --note-format=plain --single-screen --windowed=none research-update.pdf
