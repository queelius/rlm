# Train-input lexical support audit

Date: 2026-09-21. Scope: the frozen `inputs-001` train split only. This is a
read-only diagnostic; it does not alter official cases, scores, exclusions, or
the running pilot.

## Method

For every selected train parent, locate its raw MuSiQue train row by the
host-only `metadata.source_id`. For each `question_decomposition` step, test
whether the step answer occurs in that step's designated
`paragraph_support_idx` after the official MuSiQue answer normalization
(lowercase; remove ASCII punctuation and articles; collapse whitespace). The
search is a normalized whole-token span in `title + paragraph_text`.

## Result

| subset | parents | decomposition steps | answers not literally contained |
| --- | ---: | ---: | ---: |
| frozen train | 256 | 570 | 0 |
| first pilot prefix | 32 | 70 | 0 |

This checks a narrow provenance/property of the source rows: every annotated
step answer is lexically recoverable from its designated support text. It is
not a semantic entailment audit and does not prove that a paragraph supports
the complete relation asserted by the step.

## Army--Navy verification and caveat

The selective-delegation projection is faithful for raw source parent
`3hop1__233892_369053_59330`: the raw archive has the same question and final
gold `Lincoln Financial Field`, and the same three designated supporting
paragraphs retained in `inputs-001`. Thus the observed issue is not caused by
the projection.

Its middle decomposition step labels the `Music of Pennsylvania` paragraph as
support for `Solomon Burke -> place of birth -> Philadelphia`. That paragraph
contains the literal word `Philadelphia` but does not establish Solomon Burke's
birthplace. It consequently passes the lexical test while remaining a concrete
warning that literal answer containment is insufficient for relation-level
support validity. Do not exclude or rescore this official case post hoc; report
the caveat separately when interpreting evidence-grounding behavior.
