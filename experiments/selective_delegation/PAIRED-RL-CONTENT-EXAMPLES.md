# Paired RL block-1 content examples

This is an outcome-selected illustrative reading, not a rate estimate.  I took
the completed `sufficiency-rl-001/batches/sample-0001` pairs where the positive
and negative answerability labels were both predicted correctly but positive
official EM was zero.  I selected at most one candidate per parent by minimum
`sha256("2026092207:{parent_id}:{candidate}")`, then selected the three parents
with the smallest `sha256("2026092207:{parent_id}")`.  This yielded 3 of 11
eligible parents (16 eligible candidate pairs).  No held-out data was read.

All three positive calls return strict JSON with `answerable: true`; their paired
negative calls return `answerable: false, answer: ""`. Thus these are official
answer mismatches despite correct answerability decisions, not JSON/protocol errors.
An exact-match disagreement is not automatically a false factual answer.
For each example I read all 20 supplied public documents before describing an
alternative-support issue.  Gold and aliases remain host-side in the input
artifact; no score or label was changed.

| Parent / candidate | Positive call; negative call | Question | Gold / model answer | Reading |
|---|---|---|---|---|
| `2hop__20214_841802` / 2 | `p12-k2-v0`; `p12-k2-v1` | Who is the sibling of the gospel singer Freddie Mercury cited as an inspiration? | `Carolyn Franklin` / `Patti LaBelle` | **Clear entity-binding content error.** `Queen (band)` says Mercury was inspired by gospel singer Aretha Franklin; `If You Want Me` identifies Carolyn as Aretha's sister. The Patti document describes a gospel-inspiring ballad, not a sibling relation or Mercury inspiration. Across the other 18 documents, no alternative support connects Patti to the requested relation. |
| `2hop__394196_759679` / 0 | `p04-k0-v0`; `p04-k0-v1` | Where was the screenwriter of Sukumar Ray educated? | `Visva-Bharati University` / `Ballygunge Government High School` | **Potential granularity/annotation ambiguity, not a clear false fact.** The `Satyajit Ray` passage states both that Ray studied at Ballygunge Government High School and later studied at Visva-Bharati University; `Sukumar Ray (film)` identifies him as the documentary maker. The model selected an earlier, factually supported school, while the official target selects the later university. Exact EM calls it wrong, but the wording does not itself rule out the earlier school. |
| `2hop__482608_80728` / 0 | `p07-k0-v0`; `p07-k0-v1` | where does the state containing Giridih Lok Sabha rank in population? | `14th` / `16th` | **Clear attribute-selection content error.** `Giridih (Lok Sabha constituency)` locates it in Jharkhand. The `Jharkhand` document states population rank `14th` and area rank `16th`; the model selected the adjacent wrong field. None of the other documents supplies an alternative population rank. |

## Sources and boundary

- Input cases: `R/sufficiency-rl-input-proposal-001/cases.jsonl`, positive case
  IDs respectively `8fe8f57e0cc336ce1d14c029`, `a5371400b5b5dbc32e39e841`, and
  `0eec82de4cb0993a749408b1` (case-file SHA-256
  `774ceda6bbac38b23fec6f823909e606889e804e7201fd37f338ffb81e9f7f81`).
- Native pair receipts: `R/sufficiency-rl-001/batches/sample-0001/pairs/`
  `p12-k2.json`, `p04-k0.json`, `p07-k0.json`; call receipts under the sibling
  `calls/` directory using the IDs in the table.

These examples receive full credit under a label-only reward but zero under our
paired exact-answer reward. The latter catches two clear content mistakes and
also rejects one plausibly supported answer. They do not establish how frequently each
mechanism occurs, whether a content reward would improve learning, or a causal
explanation of the RL update.
