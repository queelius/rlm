LATEXMK ?= latexmk
TECTONIC ?= tectonic
PYTHON ?= python3
PDFPC ?= pdfpc
MAIN := research-update

.PHONY: all pdf tectonic figures notes present rehearse present-dual check clean
all: pdf

# The checked-in vector figures allow compilation without Python.
pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(MAKE) notes

tectonic:
	$(TECTONIC) --keep-logs $(MAIN).tex
	$(MAKE) notes

notes:
	$(PYTHON) present.py --write-notes

# Use the included PDF; presenting does not require a LaTeX installation.
present:
	$(PYTHON) present.py --pdfpc "$(PDFPC)" --mode laptop

rehearse:
	$(PYTHON) present.py --pdfpc "$(PDFPC)" --mode rehearse

present-dual:
	$(PYTHON) present.py --pdfpc "$(PDFPC)" --mode dual

figures:
	$(PYTHON) figures.py

check:
	$(PYTHON) -m unittest discover -p 'test_*.py'
	$(PYTHON) present.py --check
	$(PYTHON) verify_deck.py $(MAIN).pdf

# Only build intermediates from this deck; keep the PDF and source figures.
clean:
	$(LATEXMK) -c $(MAIN).tex
