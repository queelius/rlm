# How to add to these docs

The site is built from `docs/` by MkDocs Material on every push to `main` that touches `docs/`,
`experiments/INDEX.md`, any `experiments/*/EXPERIMENT.md`, `mkdocs.yml` or the generator. Nothing
is duplicated: the pages are the repository's own Markdown files.

- **A new result**: add a dated note `docs/notebook/YYYY-MM-DD-slug.md` (it appears in the
  notebook index automatically), then add or update a row in `docs/RESULTS.md` with a status and
  the evidence, and update the status column of the plan of record.
- **A new experiment**: create `experiments/<id>/EXPERIMENT.md` and `manifest.json` (see the
  selective-recursion study for the shape) and add a line to `experiments/INDEX.md`; the generator
  copies both into the site.
- **A new idea or question**: add it to the ranked queue in the plan of record with a one-line
  decision rule, and log the revision.
- **A new external resource**: add it to `docs/links.md` under the right heading.
- **Model or dataset release**: record it in `docs/hf/RELEASE_LIST.md`.

Build locally with `pip install mkdocs-material` then `python scripts/build_docs.py && mkdocs serve`.
