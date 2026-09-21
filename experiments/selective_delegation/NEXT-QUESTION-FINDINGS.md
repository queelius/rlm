# Feedback changed the next question, not exact-answer success

We froze the first two helper steps for16 preselected four-hop questions. Fifteen
had usable prefixes. From each prefix, the supervised root generated exactly one
new question, either seeing the actual previous answers or seeing them hidden.
The helper answered that question, and the unchanged final model then answered
the original question with full document access. Each condition had two repeats.

Both conditions solved **6 of30 eligible attempts**. Feedback changed the next
question in19of22 paired attempts where both outputs were valid. This establishes
behavioral sensitivity, not useful adaptation: there were no paired exact-match
changes. Feedback reduced invalid root outputs from7 to4, but that alone did not
produce correct final answers. Eligible token F1 rose2.11 percentage points, with
a component-bootstrap interval of−5.56 to+9.69.

The unavailable prefix is retained as one selected parent with missing outcomes,
not a model error or an undisclosed exclusion. Two repeats of one prefix do not
create two independent questions. All158 new physical model calls returned,
with327,725 tokens and no unknown usage or runtime failures.

## What this means for the next architecture experiment

This was a narrow frozen-state screen, not a trained incremental controller,
full recursive RLM or learned stopping policy. The root was trained to emit
complete question lists, not this one-question interface. Only one new step was
allowed after a two-step prefix on four-hop tasks. A weak result therefore does
not settle whether a properly trained adaptive policy helps.

Do not expand this same prompt-only screen into a large sweep. First identify a
task and matched reactive baseline where newly acquired observations change the
useful next decision, and then train/test that interface explicitly. The current
short-context setup lets the final reader solve or override the helper chain.

Immutable report: `R/analysis-next-question-screen-001.json` and `.md`.
Inputs, seeds, caps and limitations: [INCREMENTAL-DESIGN.md](INCREMENTAL-DESIGN.md).
