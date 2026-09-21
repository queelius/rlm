# A new development readout for component-controlled training

Authoritative input: external study directory `fresh-dev-inputs-003`.
Cases SHA-256: `6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153`.
No model has yet evaluated these cases at this document's September 21 preparation
cutoff. They are development data for an adaptive study, not a claim of an
independent confirmatory benchmark or absence from model pretraining.

The panel contains 64 official MuSiQue development questions: 32 two-hop and 32
three-hop. Selection sorts eligible source IDs by SHA256(seed:ID), with fixed seed
2026092113, and takes the first 32 in each group. It does not select using answers,
step answers, aliases, model outputs, or whether a proposed method succeeds.

Exclusions cover every parent, normalized question, and atomic question component
in this study's original 384 selected cases and the earlier 512-question MuSiQue
breadth panel. All 512 breadth IDs were resolved to the official cached source,
yielding 894 atomic component IDs. The final panel has zero overlap on these axes.
Atomic components may recur within the new panel; use clustered uncertainty.
There were 517 eligible two-hop and 39 eligible three-hop candidates after the
external exclusions, so this is a constrained sample, not the full benchmark.

Earlier preparation directories remain intact for provenance. `fresh-dev-inputs-001`
and `-002` are superseded and must not be used: `-002` still shared 69 breadth
components across 43 selected parents. This was caught before any model evaluation.
The corrected `-003` keeps full source revision, archive hash, license, exclusion
counts, and preparation-source hashes in its immutable manifest. Reproduction code
is `prepare_fresh_dev_panel.py`.

A CPU audit with the actual tokenizer found maximum public prompt lengths of
423 tokens for the title-index root, 3,884 for a direct answer, and 3,855 for a
full-source helper asked the original question. These fit the 8,192-token runtime
context limit with substantial headroom. Actual generated subquestions and helper
traces can add tokens, so the runtime still checks each real request. No cases
were dropped or replaced by the tokenizer audit.

For the next incremental RL comparison, first choose a fixed helper contract
using the existing helper/reminder development comparison. Freeze the training
dose and checkpoint rule before evaluating this new panel. Collect supervised
and RL root models under exactly the same chosen helper, final model, evaluation
temperature, and seeds. Use `--split development`; do not rename the saved split.
The earlier helper readout used a different downstream seed base and is not a
substitute for this matched baseline.
