# Public document exposure audit

Helper SFT uses the complete public source for its 256 MuSiQue train parents.
The split/component rules therefore do not by themselves establish unseen
documents or facts. This audit is descriptive only: no frozen panel was filtered,
reselected, or rescored.

## Method

Exact overlap is the literal `(title, text)` pair (paragraph IDs deliberately
ignored); title overlap is literal title and includes the exact matches. The
train corpus contains 4,452 unique exact documents and 3,840 unique titles.
For MuSiQue and the Hotpot explorer, host-only `supporting_*_ids` identifies a
matched document's support/distractor role. It does not reveal whether a shared
distractor is relevant to a particular generated plan. Title-only matches are
not evidence of identical text or fact exposure.

| Frozen panel | Exact shared unique docs | Parents with exact match | Shared titles | Parents with title match |
| --- | ---: | ---: | ---: | ---: |
| Held validation (32) | 6 | 4/32 | 53 | 25/32 |
| Four-hop MuSiQue (64) | 18 | 18/64 | 80 | 58/64 |
| Hotpot explorer (32) | 0 | 0/32 | 2 | 2/32 |
| Fresh development (64) | 22 | 14/64 | 110 | 48/64 |

For exact-match *occurrences* (a document can recur in several parent contexts),
the panel-side role counts are held validation 2 support/4 distractor; four-hop
7/22; Hotpot 0/0; and fresh development 3/24. The corresponding same-document
roles in MuSiQue train are held validation 1 support/5 distractor; four-hop
7/22; Hotpot 0/0; and fresh development 3/24. These are exposure indicators,
not a claim that a support fact or answer was learned.

The particularly large title-only rates make the four-hop and fresh-dev results
unsuitable as claims of document-disjoint evaluation. Hotpot has no exact public
paragraph overlap in this audit, but remains only a small explorer diagnostic and
not a pretraining-clean transfer claim.

## Reproduction and provenance

Run:

```bash
/project/alex_phd/envs/prime-rl-5990b1b/bin/python \
  experiments/selective_delegation/audit_document_exposure.py
```

Inputs and selected-parent plans are immutable external artifacts. The script
prints their SHA-256 values along with the counts, and uses the actual 32 held
validation IDs, all 64 four-hop IDs, all 32 Hotpot IDs, and all 64 fresh-dev IDs.
Its `parents_detail` records expose only opaque parent IDs and exact/title counts
plus support/distractor match counts for secondary stratification; it reads no
answer values, model calls, or outputs. The sealed report is
`analysis-document-exposure-001.json` in the external study directory.

Current input hashes: `inputs-001/cases.jsonl`
`0aebb983cf5c91b38f8bbee93e6fbf1ebe86588fd59048f54c140ae17bf95ef8`,
`hotpot-inputs-001/cases.jsonl`
`f16bc99b6b786920a5fecc516d6e2ecfc7c566cf0c38377764e7a8469e424417`, and
`fresh-dev-inputs-003/cases.jsonl`
`6251b27db8acc4fcc195f614b661acc49e5dcdb60c9c61f01826d3c4caf86153`.
Panel selection hashes: held validation
`55d64f3d00c157838965037ef11dee3f5d41130d1aefdc810b541e09c22f61e6`,
four-hop `fe9ba098c71d2cb64932e8a44cd9605e366c9eec325a4039ea8a1251de94e748`,
and Hotpot `d0a9827e0f4296b79900483e998a053336f370de4e659c5135581a1330db633f`.
