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
| E07-4 | A group-relative cost (penalty scaled by the task group's mean tokens, coefficient 0.5) changes the dynamics (a surge of direct routing at step 2) but not the outcome: the route is extinct again by step 4. | exploratory | 2026-10-19 note, relative-cost sections; `rl-route7-rel05` | Run in progress at the time of writing; seed 1, short-weighted and KL-regularized arms queued. |

## Harness teaching and transfer on TextCraft (E04, E05)

Entries pending import from the checkpoint notes of 14 to 19 October (teaching-repair transfer,
recovery teaching, second fit, broader two-world comparison).

## Codex-era findings, September 2026 (E01 to E03)

Entries pending import from `rlm-research` and the September checkpoint notes (fixed
one-candidate delegation, record grouping, matched RL updates, breadth campaigns).
