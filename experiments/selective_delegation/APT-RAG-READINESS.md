# APT-RAG readiness: distributed-evidence capability screen

**Acquisition update, September 21, 16:51 UTC:** The earlier quick-screen
recommendation below is historical. The full official QAMPARI contexts archive
has since been downloaded while independent GPU work ran; regular BM25 DEV
contexts are extracted and a 16-question fixed-pool comparison is CPU-prepared.
See [QAMPARI readiness](QAMPARI-READINESS.md). The 9.19 GB download is no longer
a blocker. That screen tests reading and splitting an already retrieved pool,
not adaptive retrieval over APT-RAG's global index; the distinction remains.

Date: 2026-09-21. This is an asset/readiness audit, not an experiment or a
claim that the present short supplied-context QA results test retrieval.

## What the primary work establishes

[APT-RAG](https://arxiv.org/abs/2609.04981v1) is directly relevant because it
tests adaptive expansion over a large retrieval corpus: an answerability
checker selects sibling-answer reuse, direct retrieval, or decomposition; leaf
retrieval uses a fixed top-20 dense retriever. Its reported datasets are
substantially different from the present supplied-document panels: MoNaCo has
1,315 questions, 43.3 gold documents on average, a 1,212,878-chunk corpus;
QAMPARI has 1,000 questions, 13.0 gold documents on average, and a
25,856,230-chunk corpus. The paper uses Qwen3-Embedding-0.6B and reports
150,000-token evidence batching for its 4B setting.

Thus, a useful follow-up question is not “can a planner beat a direct model on
short full context?” It is:

> With the same public retriever and an evidence budget, can an adaptive root
> planner improve answer/evidence coverage on genuinely dispersed evidence
> relative to a one-query retriever and a fixed-expansion control, at a
> defensible extra-token/call cost?

This is **not** a novelty claim for dynamic trees, answerability, evidence
reuse, or adaptive compute: APT-RAG already studies those mechanisms. Nor is a
generic supervised “insufficient evidence”/abstention objective new in light
of [Learning Evidence Sufficiency Boundaries for Selective Answering in
Grounded Multi-Hop QA](https://arxiv.org/abs/2609.01687v1). A distinct result
would need to isolate an observable planner decision (expand, reuse, or stop)
under a fixed retrieval interface and cost accounting.

## Official assets actually available

I cloned the official code only, at
`/project/alex_phd/research-cache/repos/apt-rag-7a0dd1c3`, commit
`7a0dd1c3c67db111e023440976348b94feba8967` (2026-09-07). It deliberately
excludes benchmark files, retrieved passages, outputs, and traces. Its
committed `monaco_300_ids.txt` and `qampari_300_ids.txt` each contain a
300-example evaluation subset, but no examples or per-question retrieval
results.

| Asset | Revision / license | What is public now | Readiness for <=30 min, one 4B GPU |
| --- | --- | --- | --- |
| MoNaCo benchmark | `allenai/MoNaCo_Benchmark` `b6382af0b328af154d22cdb38cefcafa909c23a0`; gated, ODC-By | question JSONL and execution traces require accepting the dataset gate | Not ready: benchmark access plus corpus/retrieval staging remain required. |
| APT MoNaCo corpus/index | `KJ-Min/APT-RAG-artifacts` `6ea45242e4ed271d80c3327b02f9ee0f35b45a8b`; CC BY-SA-4.0/Wikipedia | 451 MiB compressed 1M-passage corpus, 1.88 GB HNSW graph, 1.40 GB docstore (about 3.75 GB download) | Feasible storage-wise, but the repo's server also loads Qwen3-Embedding-0.6B and its scripts reserve up to 220 GB RAM for indexing. No ready fixed retrieved-context shard is released. |
| QAMPARI labels | official `qampari.zip`, 108,419,600 bytes, 2022-06-09; source repository `30bed9505a74d27ffad22d9bb1e80c4e99d9b534`, CC0 | public train/dev/test questions | Labels alone do not enable retrieval. |
| QAMPARI BM25/DPR contexts | official `qampari_with_contexts.zip`, 9,191,367,104 bytes, 2022-11-03 | public precomputed contexts, but one large archive rather than a selected shard | Not a bounded acquisition for a quick screen. |
| APT QAMPARI index | same APT artifact revision/license | 40.13 GB HNSW graph plus 21.47 GB docstore; the passage corpus is explicitly omitted | Not usable alone: the official code needs the corpus/docstore, and it is far outside this screen's retrieval budget. |

The artifact repository is public and ungated, but its MoNaCo corpus was
constructed by retaining trace-linked gold passages plus sampled non-gold
groups. Do not construct a per-question candidate set from its execution trace
and call it retrieval: that would expose gold provenance. The ordinary global
corpus/index is legitimate only when the planner sees retrieval results, not
the trace or gold-document flags.

## Smallest defensible comparison, conditional on an input seam

**Do not launch now.** The release contains no compact, public,
per-question candidate/retrieval-output artifact. The smallest valid screen is
therefore conditional on acquiring one without labels influencing selection:

1. Use the committed 300-ID list for one benchmark and choose 16 IDs with a
   recorded seed before answers or prior outputs are read. Prefer MoNaCo only
   after its gate is accepted and the exact source revision is receipted.
2. Create or obtain a *public, fixed retriever-output* cache for exactly those
   IDs: title/text/score/top-20 for each permitted query. It must be generated
   from the global corpus by the frozen Qwen3-Embedding-0.6B retriever, with no
   trace, gold-document field, or answer in the inference interface. A cached
   answer-specific gold bundle is disallowed.
3. On one 4B GPU, cap each case at root plus four retrieval actions and a
   predeclared evidence-token budget. Compare (a) one root-query RAG,
   (b) a fixed four-query expansion control with the same retrieval/generation
   caps, and (c) the adaptive planner. The direct control must receive the
   root retrieval result, not an artificially impoverished context. Report
   answer metric, retrieval recall only if gold stays host-side, calls/tokens,
   wall time, stop/expand choices, and all failed slots.

With a precomputed retrieval cache, this is roughly 16 roots and at most 80
4B generation requests; it is plausibly a <=30-minute *screen*, not a
replication of the paper's 150k-token regime. Without that cache, retrieval
embedding/index setup consumes the critical resource and the comparison is not
ready. A single GPU must not silently host both a large inference server and a
retrieval service without recording serialization/concurrency and elapsed time.

## Go / no-go

**Go** only when an immutable 16-ID manifest, a globally retrieved (not
gold-selected) fixed-output cache, source/revision/license receipts, and all
three controls exist. Positive signal: adaptive expansion improves answer or
retrieval coverage over both controls at a measured additional cost, with
per-case expansion decisions that are not merely always-expand.

**No-go / defer** if the only quick option is (i) traces or gold-doc contexts,
(ii) labels without retrievable passages, (iii) the 9.19 GB QAMPARI context
archive, or (iv) a partial index whose sampling changes the retrieval universe.
Those setups would not test the distributed-evidence capability and could make
an RLM win artificial. A null on 16 cases only retires this cheap screen, not
the broader question.

Sources checked: the [APT-RAG paper](https://arxiv.org/html/2609.04981v1),
[official code](https://github.com/hyudsl/APT-RAG) at the commit above,
[official APT artifacts](https://huggingface.co/datasets/KJ-Min/APT-RAG-artifacts)
at the revision above, [MoNaCo](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark),
and the [official QAMPARI site](https://samsam3232.github.io/qampari/).
