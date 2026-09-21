# Annotation-compatible first-helper diagnostic

This is not general intermediate accuracy. It retains only frozen first
subquestions that, after whitespace collapse only, exactly equal one unique
dependency-free MuSiQue reference question. All three helper conditions must
have asked that same question. Generated questions without that literal match,
and all later questions with generated history, are deliberately unscored.

All 64 root/repeat groups have the three planned conditions, but only 21 groups
(14 of 32 parents) satisfy the pre-output-defined annotation-compatibility
rule. Thus each condition's planned denominator is 21, not 64. All 21 first
calls were available and strict-JSON-valid. Official MuSiQue normalized exact
match against the single annotated step answer was base helper 14/21, trained
helper 11/21, and format reminder 14/21. Trained versus base has 0 wins, 3
losses, 11 both-correct, and 7 both-wrong rows; the reminder is identical to
base on these rows.

Every eligible row has zero exact `(title, text)` overlap with helper-SFT train
under the document-exposure audit. Example opaque case IDs are
`e4ce3a42fc5f961f06187458`, `eee09c7fa29f1b287c518c57`, and
`36fa2956346ce8cff668d19c`; the sealed JSON contains every included ID and its
exact-document count. This does not establish fact disjointness or generalize
beyond the literal subset. Reference step answers may lack aliases, so this
single-answer score can undercount semantically acceptable outputs.

## The three trained-helper exact-match losses

The three loss rows reduce to two parent cases, not three independent content
failures. For `9d1bc0d5dd098b52dd34dc38` (repeat 0), the trained output adds
"alone" to the annotated Bible answer. The designated Protestantism paragraph
explicitly contains that qualifier: this is a reference-granularity/EM
disagreement (official F1 2/3), not a clear fact error. For
`eee09c7fa29f1b287c518c57` (both repeats), trained says Singapore where the
Labu paragraph explicitly identifies Brunei; this is a clear content error,
repeated under two seeds. Base and format-reminder return the annotated answer
in all three rows. These checks explain the narrow score caveat, but they do
not replace the frozen official score or justify a hand-scored aggregate.

`analysis-helper-annotation-compatible-interpretation-001.json` records the
three rows, their source receipt hashes, source paragraph identity, and this
predefined classification without copying source paragraphs or broadening the
metric.

Reproduce with `analyze_helper_annotation_compatible.py --report PATH`; it
exclusive-creates the report and records script, cases, PLAN, exposure-report,
episode, and consumed-call hashes. The sealed report is
`analysis-helper-annotation-compatible-001.json` in the external study store.
