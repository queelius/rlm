LATEXMK ?= latexmk
TECTONIC ?= tectonic
MAIN := learning-guide

.PHONY: all pdf tectonic
all: pdf

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

tectonic:
	$(TECTONIC) --only-cached --keep-logs $(MAIN).tex
