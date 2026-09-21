# TextCraft changed-world control: feasible, not a renamed copy

CPU feasibility only, 21 September 2026. No new task panel, model call, training, or GPU job was created. The existing ALF analysis watcher remains active.

The pinned official generator supports a genuine changed-recipe control with seed 43, retaining generic naming and 25 items per domain/tier. Regenerated seed 42 exactly matches the frozen pilot's world hash. Both worlds contain the same 1,452 item names, 50 base items, 1,402 craftable products, and per-item declared depths. Nevertheless:

- 1,399 of 1,402 same-named recipes change.
- 1,387 products change their ingredient-name set.
- 326 products change output batch size.
- Ingredient-degree distributions differ (for example, three-ingredient products: 393 versus 415). This graph invariant rules out interpreting the new world as merely a bijective relabeling of the old dependency graph.
- Actual longest dependency-path distributions differ, despite equal declared tiers; direct-from-base craftable products number 144 versus 150. Every dependency still points to a strictly lower declared tier.

## Concrete same-item example and native contract check

| World | Product | Public recipe | Batch output | Declared tier |
|---|---|---|---:|---:|
| Seed 42 | m0_i2 | 1 m5_ore + 1 m2_ore | 2 | 2 |
| Seed 43 | m0_i2 | 2 m6_ore + 2 m8_i1 | 3 | 2 |

The same native `get_info` returns these different recipes. Each correct single-batch craft followed by explicit finish receives native success when supplied its own required ingredients. Applying the old recipe in world 43 returns a native error: output count 2 is not divisible by the new batch size 3. This is a tiny API fixture, not a matched task evaluation: the world-43 fixture deliberately supplies its intermediate ingredient initially.

Tool meaning is unchanged: public queries reveal recipes; craft consumes exact per-batch ingredients and emits a batch-multiple output quantity; the evaluator checks net target growth and nonempty explicit finish. `Frame` already accepts a world object. A future additive world-factory profile can supply seed 43 without changing those methods or editing the live seed-42 bridge. No model string is executed as code.

## What remains to qualify before any experiment

Equal nominal depth is not equal difficulty. The changed recipe can alter effective chain length, branching, ingredient demand, gold craft count, initial inventory size, and prompt length. Existing seed-42 initial inventories and gold trajectories cannot simply be replayed under seed 43.

The official `synth_tasks.py` offers `extract_base_materials_synth`, `solve_crafting_task_synth`, and `create_synth_datasets(seed=..., items_per_domain_tier=25)`. We inspected these but did not import the task framework or generate tasks. The full dataset API couples recipe generation, task sampling, and train/VAL root splitting to the same seed. Calling it with 43 therefore changes more than recipes and may place old TRAIN root names in the new VAL split. For a targeted transfer comparison, preserve a predeclared held-out root-name/quantity inventory and reconstruct feasible inventories and native-verified trajectories in each world, then report realized work rather than silently matching only annotation depth. This is a proposed design seam, not a frozen selection.

The task generator converts a base-item set to a list before sampling distractors. A future generation process should pin `PYTHONHASHSEED`, Python/runtime versions, and the resulting task bytes; a recipe seed alone is insufficient to guarantee every task-inventory detail. The in-memory recipe generation inspected here uses ordered candidate lists and deterministic seed initialization; we did not call its disk-export method.

Proceed only if the already-proposed competence training first helps. A gain surviving changed dependencies would support reusable query/quantity skills, not learned recursion by itself. Same names with changed recipes also create possible negative transfer; report that honestly instead of calling every decline general incompetence. No new benchmark framework or panel is justified by this feasibility result alone.

## Immutable provenance

Official repository: https://github.com/ApGa/platoon, cached revision `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`, MIT license.

- Generator SHA256: `1c33e0a6f61759eb3a8eb33b35f0b88361155525f5f575885b53acbd011efb0e`.
- Native environment SHA256: `c58ffad58e535d91def7291db148fd8f6433a8b3a915642f634a5b75d6d587d0`.
- Task-generator SHA256: `5cc082f13d6e6671aae2dfcf043f68a0d099fb3ec05d75e9a50885cb35d033c0`.
- Seed-42 canonical recipe hash: `f76ce3978c038be9624b3c7387aa30033970508c6afecb1f8f08315aa0693808`.
- Seed-43 canonical recipe hash: `9d71915420a2844d4fc94afc4fd4c9a5e19c6cc220e98a21c50dc738fdb003ff`.

External `R/analysis-textcraft-worlds-001.json` binds generator/revision/license, native method hashes, old pilot world audit, both complete-world hashes and the public API example. Its SHA256 is `17ef45a5c386873171afaac43fdf7a3543df2fe6b628e1bf52eb7e1298575594`; R is the active selective-delegation research store. Reproduction source is `E/inspect_textcraft_worlds.py`, SHA256 `28a67bee29cca0be17e4de02a3fefe0cd36fbb77e17d2f3e9e973e1b4f3fec3a`. The script completed on CPUs and Ruff passed.
