# Fourth completed tool comparison

Recipe set B (world48), second discovery-trained model (seed2026092291).
No extra training. The same16 planned attempts were compared with and without
automatic ingredient filling from previously observed recipes.

- Model fills ingredients:9/16 successful attempts.
- Code fills ingredients:13/16 successful attempts.
- All16 paired outcomes known; four improvements, zero regressions.
- Task-group95% bootstrap interval for the difference:6.25–50 percentage points.
- Model calls:364 to307. Assisted collection:307 returned calls, zero failed calls.
- All16 assisted episodes independently replayed; no unresolved starts or orphaned calls.

Native audited report:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-world48-seed2026092291-public-recipe-binder-followup-001.json`

SHA-256:`b89b87b679527654a598506bc1e719d4d98f7e120b0b9710770209fe4394a9be`.
Collection ended September24 at18:26UTC. This completes the two-model,
two-recipe-setting comparison. Its goals overlap the other comparisons; do not
pool64 paired attempts as64 independent tasks or claim cross-domain transfer.
