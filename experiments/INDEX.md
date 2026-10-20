# Experiment index

One line per experiment. Status: exploratory, replicated, confirmed, retracted, closed.
Dated folder names are the current locations; `E##` ids are the stable names used by the
claims ledger in `docs/RESULTS.md`.

| Id | Folder | Question | Status | Verdict (one line) |
| --- | --- | --- | --- | --- |
| E01 | (rlm-research, Sep 2026) | Does fixed one-candidate delegation improve selection on generated tasks? | replicated | 4/12 to 10/12 and 10/24 to 22/24 at about 3.7x input tokens; not learned routing. |
| E02 | (rlm-research, Sep 2026) | Does grouping unchanged records help despite longer input? | exploratory | raw 1/24, grouped 4/24, resolved 8/24 on separate panels. |
| E03 | (rlm-research, Sep 2026) | Do two matched RL updates transfer on held problems? | inconclusive | 6/24 to 8/24 and 7/24; repeat gains on one problem. |
| E04 | teaching_transfer_20261014, recovery_teaching_20261014, teaching_second_fit_20261015 | Does teaching repair transfer across worlds (TextCraft)? | replicated | Repair beats original; not better than stronger controls. |
| E05 | teaching_broader_20261019 | Two-world, four-arm teaching-repair transfer on frozen roots. | closed | Repair > original replicates; not better than stronger controls. |
| E06 | selective_recursion_20261019 | Can a 4B controller learn the RLM decomposition routine and improve it with cost-aware RL? | replicated | SFT 0.567, RL 0.631 to 0.652 (4 runs), direct 0.208, expert 0.672 on 293 held-out OOLONG rows; gain is the correctness signal. |
| E07 | selective_recursion_20261019 (routing) | Can RL learn when not to decompose? | exploratory | Direct route extinguished under absolute and relative cost from a balanced warm start; KL and short-weighted arms running. |

Reproductions (moving to separate repositories): curl_replication_20261004, r1_replication_20261007,
spo_reproduction_20261008.
