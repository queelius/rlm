LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

BASENAME := rlm-preliminary-results
SOURCE := $(BASENAME).tex
PDF := $(BASENAME).pdf

.PHONY: all meeting clean distclean

all: $(PDF)

meeting:
	$(MAKE) -C 2026-09-11-advisor-meeting pdf LATEXMK="$(LATEXMK)"

$(PDF): $(SOURCE)
	$(LATEXMK) $(LATEXMK_FLAGS) $(SOURCE)

clean:
	$(LATEXMK) -c $(SOURCE)

distclean:
	$(LATEXMK) -C $(SOURCE)
