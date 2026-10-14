# Deck verification and consolidation

Verified 14 October 2026, approximately 19:12 UTC. Scientific evidence cutoff
remains 18:10 UTC; later slide editing does not represent new experiments.

## Requirements review

- Five main slides: motivation, experiment, measured result, lessons, RLM agenda.
- One optional appendix for protocol and interpretation-relevant disclosures.
- Plain-language definitions of RL, RLM, SPO and the math task are on the slides.
- The audience sees the starting score, paper target, own result, gain,
  evaluation definition, full curve and fixed checkpoint rule without narration.
- Confidence rests on our own training and independent checking, not merely
  scoring the authors' weights.
- One-seed scope and adaptations remain visible; RLM transfer is a proposal.
- No failed-attempt or debugging narrative in the slides or speaker guide.
- Editable TeX, CSV, source provenance, audience PDF, short pdfpc notes and
  longer speaker guide are included. Existing research records remain unchanged.
- No training, evaluation or new GPU jobs were launched for slide production.

## Compression review

The outline was reduced from ten to seven to six main slides during design.
The final implementation reduces six to five by combining the result and its
essential credibility checks on one slide. Definitions stay with the experiment;
protocol details move to the optional appendix. The lessons and RLM comparisons
remain separate because they answer different questions.

A further merge would either combine the advisor's conclusion with a distinct
future research agenda or crowd the result with background. Five main slides is
the stopping point for this pass. Body text was not reduced to accomplish this.
Repeated explanations were shortened instead; the speaker guide provides depth.

## Checks performed

1. Compiled with Tectonic 0.17.0; exit 0. Final log contains no overfull-box or
   missing-character warnings.
2. Verified six PDF pages against six slide titles, normalizing typographic
   apostrophes and ligatures for text extraction. All text bounding boxes lie
   within their pages.
3. Rendered and visually inspected all six pages at twice PDF resolution.
   Corrected a crowded lessons slide and a result-slide collision. Disabled
   PGFPlots' automatic axis multiplier because tick labels already use thousands.
   Re-rendered and inspected the changed pages after correction.
4. Regenerated pdfpc metadata with the existing repository helper and compared
   it exactly with the shipped file. Notes are 11, 10, 11, 10, 10 and 11 lines,
   wrapped to at most 42 characters. All are within the helper's word limits.
5. Rendered and visually inspected each private note in a static 560 by 420
   point pane at 16-point type; all fit. Checked both presenter launch commands
   using `make -n present rehearse`. A live laptop/projector GUI was not available;
   window placement on the user's laptop remains a local check.
6. Checked all 70 scheduled points against their original independent receipt
   SHA-256, fresh-run identity, correct/total counts and native metric files.
   Checked all 71 CSV rows, including the matched zero-update baseline. Verified
   the ordinary versus final-hook update mapping and the 18.418309-point gain.

The existing eight-page renderer was reused in memory with a six-page count;
its source and the older deck were not changed. Rendered page and static note
previews remain locally in the ignored `rendered/` directory. Runtime model tests
were not rerun for this presentation-only change.

## Published artifact hashes

SHA-256 of the verified artifacts:

```text
c16a3dbfd427876798dd0b2e9f53c4bb48558fcb8ffb1e3d4cdaba46228a9d6a  research-update.pdf
bcb80656c4d985f1b52f3e10bc78697b8b22eda6ed7c732da6c1d37b2a670445  research-update.pdfpc
b60205bbf7ff2787463707488eba2fd41db34ab270eb82dba23a470c2304aa84  learning-curve.csv
```

Rebuilding can change PDF metadata and therefore its file hash. Recheck layout
and scientific content after any edit rather than treating a build as validation.
