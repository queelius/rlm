LATEXMK ?= latexmk
TECTONIC ?= tectonic
PYTHON ?= python
MAIN := research-update

.PHONY: all pdf tectonic figures check clean
all: pdf

# The checked-in vector figures allow compilation without Python.
pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

tectonic:
	$(TECTONIC) --keep-logs $(MAIN).tex

figures:
	$(PYTHON) figures.py

check:
	$(PYTHON) verify_deck.py $(MAIN).pdf

# Only build intermediates from this deck; keep the PDF and source figures.
clean:
	$(LATEXMK) -c $(MAIN).tex
