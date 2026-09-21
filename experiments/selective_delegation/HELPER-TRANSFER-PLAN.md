# Does the helper improvement transfer?

Accepted September21 at12:20UTC, before collecting these new continuations.
The initial result is [here](HELPER-FINDINGS.md); no transfer outcome is yet claimed.

Reuse exactly the saved SFT48 root plans on64four-hop MuSiQue questions and
32HotpotQA explorer questions, two plans per question. Compare base helpers,
fixed helper-SFT36, and base helpers with the exact existing JSON reminder.
The final model remains the released base with full documents. No new root
plans, changed prompts, answer repair, checkpoint search, or new training.
References bind each condition's actual previous helper answers.

Both panels have previously been examined under other conditions. These are
exploratory transfer replications, not untouched confirmation. Retain every
planned outcome, including root/dependency failures. The main contrast is
trained minus base; the reminder tests whether the training benefit is explained
by this cheaper output-format intervention. Report protocol versus both-valid
changes and actual costs. MuSiQue uses its alias-aware official metric; Hotpot
uses its own official rules, including exact-only F1 for yes/no/noanswer.
Hotpot's website explorer sample is not the full benchmark development set.

Each panel has a one-A100-hour cap. Upper bounds from the actual saved plans
are1,524new helper/final calls for MuSiQue and627for Hotpot. Root acquisition is
historical cost, not new calls. The helper dose and all other choices remain
fixed even if the first transfer panel disappoints.

External study root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Decision and hashes: `HELPER-TRANSFER-DECISION-001.json`.
- Sealed source: `source-014/`, including dataset-specific scorer and analyzer.
- Supervisor: `launch_helper_transfer.py`, tool session30777; waits for the
  accepted full-pass RL readout's authenticated release, then runs both panels.
- Outputs: `helper-transfer-musique-001`, `helper-transfer-hotpot-001`.
- Automatic completed-receipt analyses: `analysis-helper-transfer-musique-001`
  and `analysis-helper-transfer-hotpot-001`, each JSON and Markdown.

The generalized collector was exercised against both real immutable source
inventories on CPU, with focused tests for32/64parents, rejected training/mismatched
splits, retained invalid plans, and official Hotpot grading. Existing sealed
training and development-evaluation sources remain unchanged.
