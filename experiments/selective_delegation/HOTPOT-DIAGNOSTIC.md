# HotpotQA explorer-sample transfer diagnostic

This is a frozen 32-case, unevaluated diagnostic from the official 100-row
HotpotQA camera-ready explorer sample. It is not the canonical HotpotQA dev
set, a clean unseen/OOD test, or evidence against model pretraining
contamination. The selected cases use ten public documents: two supplied
support documents and eight distractors, reordered deterministically without
exposing support roles.

The transfer panel is selected label-blind after excluding exact normalized
question overlap with the existing MuSiQue train panel. Title and exact
normalized title-plus-paragraph overlap are audited and retained rather than
silently filtered, because Wikipedia overlap alone is not a justified exclusion
criterion. Answer and support annotations remain host-only.

The existing evaluator currently uses MuSiQue alias-max EM/F1. HotpotQA requires
its official answer normalization (lowercase, remove ASCII punctuation and
articles, then collapse whitespace), with no aliases in this source. A future
execution must use a separately sealed Hotpot-aware regrade; this preparation
does not alter `probe.py` or `eval_planner.py`.

The exact preparer and focused test that created the immutable inputs are
archived externally at
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/source-hotpot-inputs-001`.
