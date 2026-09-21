# Plan-content diagnostic after the format and validation checks

Historical CPU proposal, 2026-09-21. **Superseded at execution:** both model and
reference arms received fresh helpers/finals with three new seeds (312 calls),
not the reused baseline/156 calls proposed below. Only the26two-hop panel ran.
See `plan_probe.py`, `LEDGER.md` and `FINDINGS.md` for actual execution and results.
The original proposal below is retained to explain the decision trail.

At the time this proposal was written, there was no implementation or GPU launch.
Inspected only the 32 TRAIN parents in `pilot-001/PLAN.json`, their saved shared
checkpoints, and host reference annotations. No validation/transfer labels examined.

## What the pilot supports

The completed pilot has finish EM 27/96=28.1%, decompose 33/96=34.4%, targeted
31/96=32.3%, and reconsider 27/96=28.1%. Selecting an action using the other two
continuation repeats yields 33.3%, below the best fixed arm's 34.4%. This does not
identify a useful router yet. It also does not establish that the underlying plans
are good: all alternatives inherit one model-generated provisional checkpoint.

Automatic structural counts: 32 parents, 26 reference two-hop and 6 reference
three-hop; every generated checkpoint has exactly two subquestions by schema.
Thus the six three-hop references have three annotated questions versus two
generated questions. This is a structural mismatch, not a count of semantic errors:
two generated questions could legitimately combine steps. I did not apply or invent
a semantic-error classifier over these 32 cases.

Concrete, manually inspected examples follow. These are illustrative cases, not
a representative error-rate estimate, and must not determine the diagnostic panel.

| Parent | Generated-plan issue | Reference question chain | Observed original decompose finals |
|---|---|---|---|
| `01432ceceb7dc7d7f0700f8f` | Asks when the Hudson Greater Eight line began and ceased production; loses the manufacturer's identity. | `Hudson Greater Eight >> manufacturer`; `What year was the end of #1 ?` | 1932 in all three repeats. Source says the product was produced in 1931/1932 but Hudson Motor Car Company existed through 1954. All three inspected format replays still say 1932. |
| `fed7837fab20d653678477c6` | Asks about Josta's sweetener and formulation changes, instead of its manufacturer's switch. | `Josta >> manufacturer`; `when did #1 change from sugar to corn syrup` | All three claim information is unavailable. The source connects Josta to PepsiCo and discusses sugar replacement in the 1980s. |
| `4c38d2b798f1e4b80340ca23` | Second step asks who sings Never Say Never, dropping the question's relational word "with". | `Mistletoe >> performer`; `who sings never say never with #1` | Justin Bieber in all three repeats, rather than the featured Jaden Smith. All three inspected format replays still say Justin Bieber. |
| `56b0aa0b4e8659b080cc01ed` | Finds Michal's spouse and the matching TV character but stops before asking who plays that character. | `Michal >> spouse`; `who played #1 on one day at a time` | Two Jan Uczkowski answers and one refusal. Source includes David Kane (Richard Masur). |
| `fbfd442d88ebea6b5bc6bdfb` | Adds a United States publisher restriction absent from the question and asks for a book "published in Baghdad" rather than resolving the title slot. | `In what city was Dana Hussain born in?`; `They Came to #1 >> publisher` | Two Dodd, Mead and Company answers and one rejection of the question. Source gives both the UK publisher Collins Crime Club and US publisher Dodd, Mead; regional ambiguity remains a reference caveat. |
| `c65ba3f904fdec877f3bcabf` (three-hop) | Generated steps stop at performer and birthplace, omitting the requested venue. | Performer; birthplace; Army Navy venue in that birthplace. | All three wrong, including London/Twickenham. Reference answer is Lincoln Financial Field. This belongs in a separately labeled three-hop diagnostic. |

These observations suggest testing plan content before paying for action-selector
training. They do not prove a causal planning failure: the full original question
and documents are available to the helper and final model, which could override
an incomplete plan; checkpoint anchoring, retrieval, entity attribution and answer
delivery can all still matter.

## Smallest matched question-only oracle

Primary panel: all 26 two-hop TRAIN pilot parents, not selected failures. Use all
three original continuation seeds. The existing span-format decompose outcomes
from `answer-format-001` supply the matched baseline once complete.

For each original parent/repeat:

1. Preserve exactly the same full public documents, original question, original
   shared checkpoint answer and evidence_question, model, temperature and seeds.
2. Only in the decompose HELPER input, replace the checkpoint's two generated
   `subquestions` with the two annotated `question` strings in reference order.
   Do not serialize reference `answer`, `id`, `paragraph_support_idx`, host gold,
   aliases, support flags, original parent ID, or other metadata.
3. Resolve reference `#1` syntax as an instruction to use the helper's OWN inferred
   first answer, never by substituting the annotated answer. A deterministic
   question-text normalization may replace `#1` with "the answer you infer for
   subquestion 1"; freeze this normalization before observing outcomes. Preserve
   `>>` relation text without manual per-case rewrites. No oracle source selection.
4. Execute one new helper at the original helper seed, temperature and 384-token
   cap. Execute one new final at the original final seed, temperature and 128-token
   cap, using exactly the shared `answer_style='span'` prompt.
5. The FINAL still receives the ORIGINAL checkpoint, not the reference questions;
   only the actual helper report changes there. This keeps the intervention at the
   helper-plan seam instead of directly handing the final model the oracle plan.

New work: 26 parents x 3 repeats x 2 calls = 156 calls, up to 39,936 requested
helper/final completion tokens; no checkpoint reruns. Baseline's helper reports
and span final calls are reused, with provenance and costs disclosed. Per deployed
policy, both arms contain the same one checkpoint, one helper and one final, but
realized prompt/output token counts can differ and must be reported. Exclude
incomplete/unavailable source pairs explicitly; malformed returned answers remain
failures. Fix a one-hour cap and verify the first actual response within 90 seconds.

This is an ANNOTATION-PRIVILEGED diagnostic: reference question strings themselves
can reveal useful entities or disambiguating wording, even when answers and supports
are omitted. It is neither deployable nor evidence of an RLM improvement. It tests
the combined value of privileged decomposition wording/structure at this seam,
not a clean separation of abstract planning skill from annotation information.

Three-hop panel is secondary: six parents x three repeats x two calls = 36 calls.
It changes two proposed questions to three. The current helper instruction says
"the two proposed subquestions", so a generalized numbered-plan instruction would
also be required. Report this separately as question-count AND instruction changes;
do not pool it into the matched two-question primary. Defer it if the two-hop
diagnostic already answers the immediate decision.

## Decision and interpretation

Queue the 26-parent diagnostic after the accepted format-only and 32-parent span
validation jobs if no clearer validation signal has already justified a different
next step. It is short and targets concrete errors that format-only prompting
cannot repair. Do not train a 32-parent router merely to keep an optimizer busy.

Report per-parent paired EM/F1 changes, wins/losses, seed consistency, plan/report
examples, and observed costs. Keep all 26 parents in the readout. A repeatable gain
would motivate learning plan content on a larger training panel, then testing
predicted plans with no annotations at inference. A null result would weaken this
particular plan-replacement intervention; it would not rule out planning because
the original mistaken answer/evidence_question remains and may anchor the model.
Inspect whether the helper actually follows the replacement questions before
attributing a null result to plan irrelevance. Reference ambiguity and unsupported
gold remain reportable, including the publisher example above.

Evidence root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Inputs: `inputs-001/cases.jsonl`; selection: `pilot-001/PLAN.json`; observations:
`pilot-001/checkpoints/<parent>.json`; original outcomes:
`pilot-001/episodes/<parent>-r<repeat>-decompose.json`; format controls:
`answer-format-001/pairs/<parent>-r<repeat>-decompose.json`; aggregate pilot:
`analysis-pilot-001.md`. Format outcomes quoted above were existing completed pairs
at inspection time, not a claim that the entire format job had finished.
