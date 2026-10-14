TECTONIC ?= tectonic
PYTHON ?= python3
PDFPC ?= pdfpc

.PHONY: all pdf notes present rehearse
all: pdf notes
pdf:
	$(TECTONIC) --keep-logs research-update.tex
notes:
	$(PYTHON) -c 'import json,runpy; from pathlib import Path; p=Path("."); f=runpy.run_path("../2026-09-11-advisor-meeting/present.py")["make_metadata"]; m=f((p/"research-update.tex").read_text(),json.loads((p/"speaker-notes.json").read_text())); (p/"research-update.pdfpc").write_text(json.dumps(m,ensure_ascii=False,indent=2)+"\n"); print(len(m["pages"]),"matching notes")'
present:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=9 --note-format=plain --windowed=both research-update.pdf
rehearse:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=9 --note-format=plain --single-screen --windowed=none research-update.pdf
