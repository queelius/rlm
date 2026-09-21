# MuSiQue missing-evidence paired asset

The cached official archive is `musique_data_v1.0.zip`, SHA-256
`98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`
(MuSiQue repository revision `922ac98f19a201998dbdae6d7f2887a5258dbdeb`). Its
`musique_full_v1.0_train.jsonl` has 39,876 rows and its full dev file has 4,834.
Each has the usual id, paragraphs, decomposition, answer/aliases, and boolean
`answerable`. Train has19,938 distinct IDs, each appearing exactly twice;
dev has2,417 IDs, also exactly twice. Counts per answerability state are:

| Split | Two-hop | Three-hop | Four-hop | Total per state |
|---|---:|---:|---:|---:|
| Train |14,376|4,387|1,175|19,938|
| Development |1,252|760|405|2,417|

These counts were independently streamed from the ZIP on September21,12:53UTC.
Thus this is
not annotation deletion: matched variants retain the original question and answer
but remove required evidence, so an answer can remain globally true while being
unsupported by the supplied documents.

The small falsifiable control is paired answerable/missing-evidence variants of
the same official ID family: compare base/direct and trained-helper executions
for (a) answer EM on answerable members and (b) an explicit `need_more_evidence`
decision on their paired missing-evidence member. The primary failure is
over-answering: producing the factual answer despite its designated support being
absent. This cannot be evaluated by normal answer EM alone; use the official
sufficiency/answerability label as the abstention target and retain unavailable
calls separately from observed protocol failures. The official grouped answer
sufficiency metric takes the supported member's answer EM/F1 and sets it to zero
unless **both** sufficiency predictions in its ID pair are correct. It is not
ordinary binary-classification F1. The ordinary answer metric scores only
answerable members. Use grouped score plus separate supported-answer accuracy,
false-abstention and unsupported-over-answering rates. A useful signal would be high
paired sensitivity (answer on supported, abstain on unsupported) without merely
abstaining everywhere; it could then gate a *fairly controlled* context-expansion
policy. It is not evidence that the answer is false or that retrieval is needed
on arbitrary questions.

Selection must be label-blind with respect to model outputs, then grouped before
splitting: keep both variants of an official parent together; exclude parent IDs,
normalized questions, and atomic decomposition component IDs already represented
by `inputs-001`, `fresh-dev-inputs-003`, and breadth. Component exclusion must
apply across both variants, because their IDs/decompositions identify the same
reasoning chain. Do not mix a supported sibling into train and an unsupported
sibling into evaluation. Existing panels remain frozen; this is only a candidate
control after the current studies.

Sentinel-triggered retrieval already has prior art; it is not our novelty claim.
A sentinel
can be learned from an artificial prompt or missing-document token rather than
evidence sufficiency. The paired official variants are stronger because they hold
the question/answer family fixed while changing supplied support, but they still
do not establish robustness to naturally incomplete corpora, pretraining
contamination, or a deployable retrieval policy.

Uncompressed member SHA-256: train
`b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034`;
dev `8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a`.
Pinned repository `evaluate_v1.0.py` SHA-256
`f5fe66ae61dbea5172cba9d428d9924a5811f3457edc945fc1d81369c30e74b7`;
`metrics/group_answer_sufficiency.py`
`6fd72646055b736cfae474eba58f7d4823ee8f539c0746bb87dd5c8208d7abcc`.
The [official repository](https://github.com/StonyBrookNLP/musique) distributes
the data under CC BY4.0 and warns about single-hop seed-dataset overlap. Cached
archive retrieval predates this session; see DATA-AUDIT.md for acquisition limits.
No new panel, model output or GPU experiment was produced by this asset audit.
