---
title: Presentation tooling and provenance
retrieved_date: 2026-09-10
scope: CPU-only presentation build
---

# Compiler

The PDF was compiled with Tectonic 0.17.0, downloaded from the official release:

https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz

Archive SHA-256:
8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7

Tag: tectonic@0.17.0. Tag object:
7141d67aba85eb1cf38a769eac0b1bb2180c51d7.
Source commit: 8c0126a9653239a2e6e0a5274af9b8510f643030.

The archive hash matched the official GitHub release digest before extraction.
The archive contained one executable. The executable's version was checked
before compilation.

License: the project's LICENSE states MIT for Tectonic, with varied open-source
licenses for underlying components. See the
[license at the release](https://github.com/tectonic-typesetting/tectonic/blob/tectonic%400.17.0/LICENSE).
No compiler binary is included in this Git repository.

Local tool directory:
/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl

# Figures and PDF inspection

A separate environment was created at:
/project/alex_phd/envs/rlm-advisor-figures-20260911

Python 3.11.14; uv 0.9.7. Matplotlib produced vector PDF plots.
PyMuPDF rendered the compiled deck and checked text bounds. This environment
does not use CUDA and does not change the active training or serving environment.

Direct dependencies are in requirements-figures.txt; the complete resolved
package versions are in requirements-resolved.txt. Packages were acquired from
the configured Python package index through uv. Their own distributions retain
their licenses; no third-party package code is copied into this repository.

# Local verification commands

    /project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python figures.py
    /project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic --keep-logs research-update.tex
    /project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python verify_deck.py research-update.pdf

The input is the included Beamer source and the included portable numerical
evidence file, not an external slide template or generated bitmap.
