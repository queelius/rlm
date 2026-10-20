# Repository reorganization plan (decided 2026-10-20)

Decisions taken with the user on 2026-10-20:

1. **One monorepo** (`rlm`) for the runtime and the RLM research; no library/research split.
2. **Hugging Face under the user's personal account** for adapters and datasets; adapters
   Apache-2.0, data CC-BY-4.0 unless a source license says otherwise.
3. **Git history is not rewritten.** Large data checked in earlier stays in history; the working
   tree stops carrying it.
4. **Paper reproductions leave the monorepo**: SPO on GSM8K becomes its own repository (advisor
   facing, audited, checkpoint on Hugging Face); R1-style math RL and CURL move together to a
   reproductions repository. None of them imports the runtime.

## Target layout

```
rlm/
  src/rlm/ tests/            runtime, unchanged
  experiments/INDEX.md       one line per experiment: id, question, status, verdict
  experiments/E##-slug/      EXPERIMENT.md, manifest.json, scripts/, results/, analysis/, env/
  docs/RESULTS.md            claims ledger: claim, status, evidence, caveats, what overturns it
  docs/notebook/             dated checkpoint notes (today's docs/research-checkpoints)
  docs/operations/           RESEARCH_OPERATIONS.md, A100 quickstart
  paper/ slides/
```

Rules: nothing regenerable or downloadable in git (target under 5 MB of data); every experiment
records code commit, environment lock, base-model revision, data revisions, seeds and compute
cap; results are machine-readable before they are prose; experiment ids `E##-slug`, ISO dates
inside files, not in folder names.

## Sibling repositories and their fate

| Repository | What it is | Decision |
| --- | --- | --- |
| `rlm-bootstrap` | Intended orchestration/training sibling (Sep 2026); stalled before selecting a recipe. Its worktree venv is the de-facto training environment used by `experiments/selective_recursion_20261019`. | Fold intent into `experiments/`; export the training lock into the monorepo (done: `experiments/selective_recursion_20261019/env/`); mark the repo superseded. |
| `rlm-research` | Publication snapshot of the Codex-era notebook (Sep 2026, 4,784 files): questions, claims, analyses, sidecars. | Import its claims into `docs/RESULTS.md` with evidence links; keep the repo as a read-only archive, linked from the ledger. |
| `structured-decomposition-benchmark` | Pre-RLM study (Aug 2026) of one-level decomposition; own data and benchmark. | Keep as its own repository; cite as prior work; harvest `rlm-experiment-ideas` into the notebook. |
| `rlm-research-r4` (directory, not a repo) | Empty remnant (one `sidecars/` dir). The real r4 store is under `/project/alex_phd/runs/rlm-research-r4`. | Delete candidate; user's call. |
| `exp1` (45 GB, no remote), `forgot` | Unrelated to RLM; `exp1` is not on GitHub. | Out of scope here; flag that `exp1` has no remote backup. |
| `evolved-integrators`, `use-mention-gating` | Unrelated projects, on GitHub. | Out of scope. |

## Hugging Face plan and constraints

- Namespace: the user's personal account (`queelius`). No login exists on the cluster yet; the
  user runs `hf auth login` (interactive) before any upload.
- Models (LoRA adapters, about 130 MB each at rank 32, Apache-2.0 with the Qwen3 base noted):
  `rlm-qwen3-4b-sft-005` (taught warm start), `rlm-qwen3-4b-rl-mixed-001` (best RL controller),
  `rlm-qwen3-4b-sft-007` (route-balanced warm start), and the most informative routing arm. Each
  card states base model revision, teaching data, recipe, held-out numbers from `docs/RESULTS.md`,
  and the harness contract it expects.
- Datasets (CC-BY-4.0): the unified teaching sets (synthetic event logs plus our controller cells;
  OOLONG-derived rows carry only row ids, window ids and our cells, not the source text), the
  held-out split definition for OOLONG-synth (window ids), evaluation row tables and traces for
  the headline comparisons with OOLONG text stripped to ids.
- OOLONG-synth itself declared **no license** at the revision used (f0d59eaf, retrieved
  2026-10-19, `research-cache/datasets/oolong-synth-20261019/PROVENANCE.json`), so its text is
  not redistributed; the cards link to the source dataset and revision instead.
- Reproductions: the SPO checkpoint goes to its own model repo from the SPO repository.

## Migration order (additive first)

1. Write `docs/RESULTS.md` (claims ledger) and `experiments/INDEX.md`.
2. Create `E##-slug` folders alongside the dated ones with manifests; add a machine-readable
   path map `docs/PATH_MAP.json` (old path -> new path) before removing anything.
3. Move `docs/research-checkpoints` to `docs/notebook` (git mv, history follows).
4. Create the two reproduction repositories from the existing folders (SPO alone; R1 + CURL),
   with the CURL raw curves as a release asset, not tracked files; then remove those folders here.
   **Done locally 2026-10-20** (`/project/alex_phd/repos/spo-gsm8k-reproduction`, 110 commits;
   `/project/alex_phd/repos/rl-reproductions`, 76 commits; history filtered with git filter-repo;
   see `docs/PATH_MAP.json`). Pushed 2026-10-20 as private repositories `queelius/spo-gsm8k-reproduction` and `queelius/rl-reproductions` (release v0.1.0 with the CURL curves). Removal of the folders from this repo awaits the user's review of both.
5. Fast-forward `main` to the research branch; remove stale worktrees after confirming no live
   process uses them.
6. Hugging Face: model cards and dataset cards for the salient adapters and teaching/eval sets.
7. Cluster: write a manifest for `/project/alex_phd/runs`, split archive vs scratch, propose a
   sized deletion list for the user.
