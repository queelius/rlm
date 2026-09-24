LATEXMK ?= latexmk
TECTONIC ?= tectonic
PDFPC ?= pdfpc

.PHONY: all pdf tectonic present rehearse
all: pdf
pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error research-update.tex
tectonic:
	$(TECTONIC) --keep-logs research-update.tex
present:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=10 --note-format=plain --windowed=both research-update.pdf
rehearse:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=10 --note-format=plain --single-screen --windowed=none research-update.pdf
