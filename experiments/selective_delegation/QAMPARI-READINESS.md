# QAMPARI fixed-pool reading: proposed readiness screen

Status: CPU asset inspection, frozen panel preparation, and token audit complete; no GPU
experiment or new training approved. The main agent completed and hashed the archive before
inspection. Only the regular BM25 development member was extracted. The proposed collector
still needs implementation and a real GPU memory/throughput smoke check after acceptance.

## Question and smallest comparison

Does reading the same ranked retrieved evidence in four smaller batches improve multi-answer
coverage enough to offset extra calls and any loss of cross-batch reasoning? This is a
fixed-pool reading comparison, not live global retrieval, adaptive question decomposition,
or an APT-RAG replication. The official QAMPARI baseline already includes passage-independent
generation, so map-and-union itself is not an architecture novelty claim.

Prespecify 16 development parents as the first 16 IDs ordered by
`SHA256('2026092186:' + qid)`. Preserve the natural question-type distribution; do not select
by gold-answer count, retrieval coverage, difficulty, or model outcome. Keep all planned slots,
including any malformed/unavailable source slots, with explicit accounting. Keep train and
test out of this panel. Selection is proposed now, before any model outcomes.

For each parent and seeds 2026092187/2026092188:

1. **Direct-200:** one base-4B call sees the public question and first 200 retrieved `ctxs`
   in stored rank order. It returns the answer list directly, with a 1,024-token cap.
2. **Map-50:** four independent base-4B calls each see the same question and one consecutive
   50-chunk block from that exact 200-chunk pool. Each has a 256-token cap. The final output
   is the deterministic, first-occurrence, exact-string union of the four lists. No learned
   aggregator, planner, gold-aware merge, or extra final call is introduced.

This is 160 native calls: 16 parents × two repeats × (one direct + four map calls).
The maximum generated-token budget is equal at 1,024 per parent/repeat/arm; actual tokens,
prompt tokens, prefills, latency, and call counts must still be reported. Each arm reads every
pool chunk once, but map repeats the question/instruction and creates independent sampling
opportunities. It is a package comparison, not a pure representation or call-count effect.
Use temperature 0.5, top-p 1, top-k 0, no adapters, and fixed seeds recorded per call.
Use the repeat seed for direct and repeat seed plus block index for map; there is no
claim that different prompts constitute paired random-token draws.

The common prompt should request all distinct entities that answer the entire question using
the supplied passages, prohibit unsupported additions, and require exactly
`{"answers": ["entity", "entity"]}`. A valid empty list means no answer found. Do not reveal
the gold answer count, question type, knowledge-graph relations, aliases, proof fields, or
support annotations. For both arms expose only question and `{docid, title, text}` passages;
contiguous doc IDs encode retriever rank, not support. Map does not see answers from prior blocks.

The initial screen should have a one-hour GPU cap, immutable inputs, receipts, planned
denominators, and a real first-response check. An inference exception is unavailable evidence,
not a wrong answer, and should halt without implicit retry. Returned malformed JSON is an
observed protocol failure. For map, any malformed block makes the primary parent/arm result
protocol-invalid rather than silently unioning only successful blocks; missing blocks are
unavailable. Partial unions can be secondary diagnostics only, clearly labeled.

Do not add a 20-chunk arm yet. Ten small map calls change output-budget fragmentation and
invocation overhead further. If map-50 is informative, a later fixed-granularity comparison
can distinguish a smaller working context from simple opportunity to emit more candidates.

## Where this helps and where union is insufficient

Independent reading can recover distributed list members obscured by distractors or position.
But deterministic union cannot join a relation found only in one block with an entity found
only in another, or validate two intersection predicates split across blocks. A local reader
must either abstain or supply an answer it cannot establish locally. Retain composition and
intersection cases rather than quietly filtering them away; report type-stratified outcomes
descriptively if sample counts permit. A later evidence-carrying intermediate representation
would be a different architecture, requiring its own contrast against this simple baseline.

Full-pool retrieval may itself omit required evidence. Host-only alias occurrences can help
diagnose that, but lexical presence is not semantic support or an exact attainable-recall
ceiling. Do not use gold proofs to repair the public pool. A map recall increase with a larger
false-positive list is not automatically a useful improvement: report precision and F1 as well.

## Official assets and license boundaries

Official repository: <https://github.com/samsam3232/qampari>, pinned commit
`30bed9505a74d27ffad22d9bb1e80c4e99d9b534`, local path
`/project/alex_phd/research-cache/repos/qampari-30bed950`, inspected September 21, 2026.
The repository declares CC0-1.0. This is not a claim that copied Wikipedia passages are CC0;
their upstream text licensing and provenance remain separate. Vendored model code also has
separate notices (for example GC-DPR's README declares inherited CC-BY-NC 4.0).
No model-training or retriever code from this repository was executed.

Official archive: <https://aggreg-qa.s3.amazonaws.com/qampari_with_contexts.zip>, linked from
the [official website](https://samsam3232.github.io/qampari/). Acquisition metadata is under
`/project/alex_phd/research-cache/datasets/qampari-contexts-20260921/ACQUISITION.md`.
The completed length is 9,191,367,104 bytes, SHA-256
`ba19385c33fd9f0b4b37ac5ffc9918996f61787d56cd5dc43288395d27309cd6`, independently
recorded by the main agent in `DOWNLOAD.json`; the multipart ETag is not a checksum.
This preparation reused that completed hash receipt rather than rehashing 9.19 GB.
The ZIP directory contains three stored nested archives (FiD BM25, FiD DPR, and RAG).
A read-only, range-bounded seek window into the BM25 archive avoided copying its 2.96 GB
container. After validating member paths and sizes, only regular
`qampari_v2/qampari_fid_format/full_dev_data.jsonl.gz` was extracted. The similarly named
`full_dev_data_gold.jsonl.gz`, train, test, DPR, and RAG members were not extracted.

The [paper, inspected version 4](https://arxiv.org/html/2205.12665v4), describes development
and test splits of 1,000 questions each, human paraphrasing and answer validation, a
2021-08-01 Wikipedia dump, and passages averaging about 100 words. It explicitly includes
simple, composition, and intersection questions. These are dataset properties, not claims
that every released chunk is exactly 100 tokens or that our selected panel has balanced types.

## Schema and grading facts already established from inspected code

`models/retrievers/BM25/to_dpr.py::parse_example` copies query retrieval results to `ctxs`
with `id`, `title`, `text`, and `score`. In contrast, `positive_ctxs` are formed from annotated
answer proofs; `hard_negative_ctxs` use gold aliases to reject answer-containing candidates.
Both are prohibited in model prompts. Actual regular BM25 development `ctxs` have exactly
these four fields, and all 990 rows contain 200 passages in nonincreasing score order.
Public projection removes original passage IDs and scores; rank survives through order.

The packaged FiD row has `id`, `question`, `answers`, `ans_mappings`, `target`, `ctxs`, and
`positive_ctxs`, not the raw dataset's `qid`/`answer_list` interface. Native `id` is the qid
used by the frozen hash rule; it embeds question type and remains host-only. Inspected
`data_reparser_qampari.py` shows that `answers` is the union of aliases, while `ans_mappings`
preserves canonical answer → alias list. Reconstruct the official grader's `answer_list`
from the latter. In 132 rows, alias-union strings differ from the set of canonical names;
this is expected, not inconsistent ground truth. The reparser also excludes examples with
more than 100 canonical answers. The actual packaged inventory has 990 rows rather than
the README's 1,000; without the raw split comparison we do not assert that this filter
explains all ten missing rows. This panel targets the 990 available packaged-dev questions.

`models/evaluation/reader_metrics.py::compute_metrics_qampari` scores entity sets using
`answer_list` entries containing `answer_text` and `aliases`. A prediction can match any
normalized alias, but each canonical answer is credited at most once. The denominator for
precision is `len(set(raw_prediction_strings))`, **not** normalized-string deduplication.
Preserve this behavior: the proposed merge uses exact strings for both arms, without gold
canonicalization. Macro precision, recall, F1, recall ≥ 0.8, and F1 ≥ 0.5 should be retained.
This is entity-set F1, not the MuSiQue token-overlap metric.

The inspected evaluator was exercised on a tiny CPU fixture: one correct string scores
precision/recall/F1 1; two case variants of that same answer score 0.5/1/0.6667; an empty
prediction list raises `ZeroDivisionError`. A wrapper must declare empty-list P/R/F1 = 0
and both thresholds false, while using the unchanged official routine for nonempty lists.
This is an explicit boundary convention, not parsing repair or answer fallback. Gold aliases
stay in the evaluator. Missing outcomes and returned invalid responses must remain distinct.

Inspected file hashes:

| File | SHA-256 |
|---|---|
| `LICENSE` | `a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499` |
| `reader_metrics.py` | `d0d1e2beca6280b9e81be78bbdf2324f245a3dcc4233dc13fe92c8dfeac8ebaf` |
| `to_dpr.py` | `b5eed497d64fdec76ffe575b19b870ff32ea6c9cd07eafb4e03fc95a881fe926` |

## Context and runtime admission

The cached Qwen3-4B-Instruct-2507 config declares `max_position_embeddings = 262144`.
The earlier collectors' 8,192-token limit is a local experiment cap, not this model's native
declared context limit. Config SHA-256 is
`5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`.
Exact non-thinking chat-template tokenization was performed for all selected direct and map
prompts, including output reservations. Do not truncate or reselect long cases to make direct
artificially weak. A config limit alone does not establish memory feasibility or effective
long-context performance on the assigned GPU.

A rough BF16 KV-cache calculation from the local config is 147,456 bytes per token
(2 × 36 layers × 8 KV heads × 128 head dimension × 2 bytes), before other allocations.
The maximum direct sequence including generation has approximately 5.04 GiB of BF16 KV cache
under this calculation, excluding model weights, prefill activations, attention workspace,
and allocator overhead. Device memory, attention implementation, and observed first-call memory
must govern execution. The initial 160-call screen is plausibly a tens-of-minutes single-A100
job, not a measured runtime promise. No large environment or retriever index is needed.

## Actual frozen panel and token audit

`R/qampari-inputs-001/cases.jsonl` and `MANIFEST.json` now exist, with proposed status.
Here `R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
The native IDs were selected before any model outcomes with the prespecified seed and
without a type filter. The 16 selected questions include seven Wikidata simple, four table
simple, two Wikidata composition, two table composition, and one intersection question.
Natural canonical-answer counts range from 5 to 27; they were not used for selection.

| Exact chat-template quantity | Measured value |
|---|---:|
| Direct input tokens, minimum / median / maximum | 30,168 / 31,463 / 35,647 |
| Direct maximum including 1,024 output tokens | 36,671 |
| Map-50 input tokens, minimum / maximum | 7,410 / 9,366 |
| Map-50 maximum including 256 output tokens | 9,622 |
| All direct input tokens, two repeats | 1,022,968 |
| All map input tokens, two repeats | 1,031,920 |

Map repeats add 8,952 input tokens, about 0.88%, despite equal passage coverage. A proposed
**40,960-token runtime guard** accommodates every direct sequence without truncation;
this is an experiment-specific guard, not the model's native limit. The common instruction
and every per-parent token count are frozen in the manifest. No parent was reselected.
The existing `alfworld_probe.BaseClient` cannot be reused unchanged: its local 8k-context
and 128-output-token caps must not silently constrain this experiment.

Example of the actual public projection (text excerpt shortened for display only):

```json
{"question":"What companies are producing modern armament in Pakistan?",
 "documents":[{"docid":"d0","title":"China–Pakistan relations",
 "text":"In 1986, President Muhammad Zia-ul-Haq visited China to improve diplomatic relations, ..."}]}
```

The gzip and its 145,744,783-byte plaintext are preserved under
`/project/alex_phd/research-cache/datasets/qampari-contexts-20260921/extracted-bm25/`
`qampari_v2/qampari_fid_format/`; plaintext decompression was bounded at 200 MB.

| Prepared artifact | SHA-256 |
|---|---|
| Regular development gzip | `b84c1a039d0720042c9420699cdf4427f1d62dfdb7c6cf0a981079036cdcd991` |
| Regular development plaintext | `58cccb13ff6d64ce8c0660c4f5d99aeee617c38074ca0b9c1dd275a4b46ef861` |
| Prepared cases | `4dd4215394608958bd1e3a2096ae8f0c8a065eb726d78d1373e1557f6731e962` |
| Prepared manifest | `e0d77aeae10b94f92d027aac0d3f99b553a6b3067794404544e1979177f2e211` |

The small `prepare_qampari_panel.py` preparer records tokenizer/config/source/grader hashes
and accepts only a fresh output directory. Two focused CPU fixtures passed for strict public
projection and outcome-independent ID ordering. CPU tokenization used the existing training
environment Python, with offline loading and no model weights loaded. This is input readiness,
not measured model reliability or a passing GPU preflight.

This document was developed as a read-only feasibility spike using the brainstorming skill;
no collector implementation, model call, or GPU launch is part of this preparation.
