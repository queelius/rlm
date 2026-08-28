LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error

BASENAME := rlm-preliminary-results
SOURCE := $(BASENAME).tex
PDF := $(BASENAME).pdf

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(SOURCE)
	$(LATEXMK) $(LATEXMK_FLAGS) $(SOURCE)

clean:
	$(LATEXMK) -c $(SOURCE)

distclean:
	$(LATEXMK) -C $(SOURCE)
