TECTONIC ?= /project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic
PYTHON ?= python3
PDFPC ?= pdfpc
.PHONY: all pdf notes present rehearse
all: pdf notes
pdf:
	$(TECTONIC) --keep-logs research-update.tex
notes:
	$(PYTHON) make_notes.py
present:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=10 --note-format=plain --windowed=both research-update.pdf
rehearse:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=10 --note-format=plain --single-screen --windowed=none research-update.pdf
