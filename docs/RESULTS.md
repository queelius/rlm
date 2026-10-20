# Claims ledger

Every empirical claim this research makes, with its status and the evidence behind it. Statuses:
**exploratory** (single run or small n), **replicated** (repeated with new seeds or inputs),
**confirmed** (replicated under a sealed or audited protocol), **retracted** (superseded; the
entry says by what), **inconclusive**. Each entry names what would overturn it. Experiment ids
refer to `experiments/INDEX.md`; notes are under `docs/research-checkpoints/` (to become
`docs/notebook/`). Numbers are quoted as recorded; nothing here is rounded up in status.

## Selective recursion: teaching and cost-aware RL of a 4B controller (E06)

Setup: Qwen3-4B-Instruct-2507 served by vLLM as both controller and helper; the RLM kernel shows
the controller only request metadata plus the latest cell/observation; synthetic event-log tasks
(8 to 512 records, five question types) and OOLONG-synth (1k to 32k tokens, held-out window split).

| Id | Claim | Status | Evidence | Caveats and what overturns it |
| --- | --- | --- | --- | --- |
| E06-1 | The untrained 4B controller delegates the whole request (`model_complete`) and fails above about 64 records; a programmed per-line extraction split is about 97 percent accurate at every length and no costlier than direct answering at 8 records. | replicated | 2026-10-19 note, baselines section; `sweep-001-t0` | Synthetic family; the hinted harness variant had no effect. |
| E06-2 | A controller taught by native replay of a scripted expert (SFT on message-cell pairs) executes the decomposition routine; it must see a family-agnostic first turn (peek) or it guesses the family blindly (SFT-002: 40/40 event-log, 0/11 OOLONG). | replicated | 2026-10-19 note, teaching lessons; SFT-002 vs SFT-004/005 | One model size and family. |
| E06-3 | Short question-specific composed programs are reproduced reliably; a 3k-character monolithic program is not (SFT-004: 5/72 on OOLONG with syntax drift; SFT-005: 36/72). | replicated | 2026-10-19 note; `oolong-sft4`, `oolong-sft5` | Teaching-set sizes of a few hundred trajectories. |
| E06-4 | On 293 held-out OOLONG rows, 30 updates of on-policy RL (REINFORCE, leave-one-out baseline) raise the taught controller from 0.567 to 0.631 to 0.652 across four runs (seeds 0 and 1 at cost 0.1, no cost, cost 0.3), against direct answering 0.208 and the scripted expert 0.672. | replicated | 2026-10-19 note, full-sample sections; `oolong-*-full` | Format-tolerant exact-match verifier, not the official scorer; one benchmark. |
| E06-5 | The RL gain is driven by the correctness signal: it is mostly a fix of helper-call budget failures at 32k tokens (a length never seen in training) plus small gains at 1k to 2k; the token-cost term at 0, 0.1 and 0.3 changes accuracy by at most 2 points and trims about 7 percent of tokens. | replicated | 2026-10-19 note, no-cost and 0.3-cost sections | Cost strengths tried were small relative to correctness reward. |
| E06-6 | On the event-log dev split the RL controllers match the warm start (85 to 88 of 96); two cost-0.1 seeds moved to smaller chunks (12 lines), the no-cost and cost-0.3 runs did not; the mechanism is unresolved. | replicated | 2026-10-19 note, second-seed and 0.3-cost sections | Non-monotone in the cost coefficient; two runs per condition at most. |
| E06-7 | At 1024 records (twice the longest training length) every RL adapter is 2 to 5 of 16 tasks below the warm start regardless of chunking (expert 13, warm start 14, RL 9/12/11/11). | exploratory | 2026-10-19 note, out-of-range tables; `dev1024-*` | 16 tasks; cannot separate a small robustness loss from noise. |
| E06-8 | Retracted: "the chunk drift caused the out-of-range loss". | retracted | superseded by E06-7 | The no-cost and 0.3-cost adapters kept 24-line chunks and still lost tasks. |

## Learned routing: when not to decompose (E07, in progress)

| Id | Claim | Status | Evidence | Caveats and what overturns it |
| --- | --- | --- | --- | --- |
| E07-1 | A warm start taught mostly decomposition assigns the direct route near-zero probability on short inputs (0 of 63 at temperature 1), so on-policy RL from it cannot discover the route; RL from it still suppresses a worse direct route on OOLONG 1k within eight steps. | replicated | 2026-10-19 note, routing warm start and control sections; `dev-sft6-t1`, `rl-route-lam01` | Teaching prior measured on event-log 8/32 records only. |
| E07-2 | A route-balanced warm start (SFT-007) keeps both routes usable (direct on 15 of 36 sampled 8-record tasks, 13/15 correct at half the tokens) without losing accuracy (OOLONG 34/72, event-log 85/96). | exploratory | 2026-10-19 note, balanced warm start table | One warm start. |
| E07-3 | From that warm start, RL with an absolute token cost (0.1) extinguishes the direct route everywhere within five updates, including 8-record tasks where it is equally accurate at half the tokens; the saving is worth about 0.03 reward against order-1 correctness differences elsewhere. | exploratory | 2026-10-19 note, step-10 and completed sections; `rl-route7-lam01`, `dev-ctrl-rlroute7-01-t1` | One seed; the 0.3-cost arm is queued. |
| E07-4 | A group-relative cost (penalty scaled by the task group's mean tokens, coefficient 0.5) changes the dynamics (a surge of direct routing at step 2) but not the outcome: in two seeds the route is extinct from step 3 or 4 to step 30, the held-out adapters never take it (0 of 32 sampled 8-record tasks each) and reach 42/72 and 43/72 on the OOLONG sample. | replicated | 2026-10-19 note, relative-cost sections; `rl-route7-rel05`, `rl-route7-rel05-s1`, `dev-ctrl-rlroute7-rel05*-t1` | Short-weighted, sequence-normalized and strong-KL arms in progress. |
| E07-5 | A per-token KL penalty to the warm start at the usual strength (0.1) is inert: the run reproduces the unregularized collapse step for step with mean per-token KL of 1 to 5 x 10^-4, because the route decision is a few tokens among hundreds. | exploratory | 2026-10-19 note, KL section; `rl-route7-rel05-kl` (stopped at step 5) | A coefficient-5 arm is queued. |
| E07-6 | The gradient that removes the direct route is not only its own negative advantage: summed advantages on direct rollouts were -1.97 and -1.40 in two runs but -0.12 in a third that collapsed just as fast; token-sum loss weighting (a correct 300-token decomposition cell carries ten times the gradient of a 30-token direct cell) and the rarity of short tasks in the draw are the remaining candidates. | exploratory | 2026-10-19 note, advantage-mass section; `advantage_mass.py` | Sequence-normalized and short-weighted arms test the two candidates. |

## Harness teaching and transfer on TextCraft (E04, E05; September 22 to October 19)

Full entries C79 to C96 in `docs/results/claims-2026-08-to-10.md`. The load-bearing ones:

| Id | Claim | Status | Evidence | Caveats and what overturns it |
| --- | --- | --- | --- | --- |
| C79 | Answer-preserving repair of teaching histories (recipe lookups reordered so names are visible before they are queried; actions, answer tokens and denominators unchanged) raises TextCraft successes over the weak original teacher, for example 1 to 14, 2 to 13 and 3 to 17 of 32. | replicated (second fit seed, added roots, further panels, new structures, two new worlds) | `claims-2026-08-to-10.md` C79; `docs/research-audit-2026-09-29/teaching-and-execution.md`; `experiments/teaching_*` | Every panel is exploratory; the comparison is against a weak teacher. |
| C80 | Repair never beats the stronger discovery teacher or the extra-training-dose control. | replicated | C80; `experiments/teaching_broader_20261019` | Same panels as C79. |
| C81 | The 69-update dose control flips between 5/16 and 0/16 across training seeds, so dose comparisons are seed-limited. | exploratory | C81 | Two seeds. |
| C84 | Code-filled ingredient arguments (execution assistance) raise success 323 to 381 of 768. | replicated | C84 | Assistance changes the action space, not the policy. |
| C87 | Four fixed RL updates from the SFT warm start give no sustained gain on the eight-goal diagnostic (4, 4, 5, 4, 4 of 16 raw; 5, 4, 4, 4 assisted). | replicated null | C87; `docs/RESEARCH_RESUME_2026-10-04.md` | One training run on an exposed diagnostic. |

## Codex-era findings, August 24 to September 29 (E01 to E03)

The September notebook itself (questions, analyses, promising-results notes as of 2026-09-14) is archived read-only at https://github.com/queelius/rlm-research; the September 29 audit in `docs/research-audit-2026-09-29/` supersedes it. Ninety-six entries across sixteen threads are in `docs/results/claims-2026-08-to-10.md`
(extracted 2026-10-20 from the checkpoint notes and the September 29 audit; statuses are the
notes' own, and none reaches "confirmed"). The ones that carry weight:

| Id | Claim | Status | Caveats and what overturns it |
| --- | --- | --- | --- |
| C62 | Singleton delegation (one helper per candidate) on the generated record-selection task: complete sets 4/12 to 10/12 on six frozen problems and 10/24 to 22/24 on twelve fresh ones (twelve gains, no losses). | replicated | About 3.7 times the input tokens; a fixed program, not learned routing. |
| C60 | Python-organized state on the same task: 1/24 to 9/24. | replicated | Same task family. |
| C50 | MRCR procedural SFT taught a Python search routine: 0 to 10 of 16 on two fresh panels, transferring to longer inputs. | replicated | Later RL on it mostly improved delivery; one gain did not replicate (C51, C53). |
| C68 | Code-executed arithmetic roughly doubles strict FinQA matches (51 to 108 and 52 to 102 of 508). | replicated | Prior exposure of the panels; semantic audit pending. |
| C40 | Eight RL updates on the AG News helper: 422 to 437 and 436 of 512 across two seeds on the earlier panel; on fresh official-test articles 427/429 against 426 after SFT. | replicated on the earlier panel, inconclusive against SFT on fresh data | Helper-only RL; attenuates on fresh examples. |
| C06 | An explicitly taught adaptive routine (Python for visible metadata, else a 16-record facet) reaches 144/144 conditional answers with a 67.8 percent helper-token reduction over two optimization seeds. | replicated | The routine was taught, not learned; synthetic. |
| C32, C39, C41, C51, C53, C76 | Every other RL claim (self-SFT compute saving, AG News one-update gain, varied-versus-repeated data, MRCR one-step gains, ALFWorld manager gain). | exploratory, null or retracted | See the full ledger for each. |

Reading across the ledger: taught programs transfer and replicate; every attempt to *learn*
routing or decomposition by RL before October 19 was exploratory, null or retracted, which is why
E06 and E07 were designed around a taught warm start plus a verifiable reward.
