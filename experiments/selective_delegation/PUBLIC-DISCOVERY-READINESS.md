# Public discovery prototype: CPU feasible, training not accepted

The deterministic public-only teacher completed all 32 exact source047 TRAIN tasks in native CPU replay, with no dropped/replaced tasks or budget failures. Every first action queries the public root target. The teacher API accepts only target quantities, initial/current inventory and prior queried recipes; the driver never reads the gold trajectory or declared depth for action selection. Native scoring occurs only after each trace for qualification.

Five focused fixtures passed after observed RED→GREEN development: root/prerequisite discovery, shared demand with non-unit batch yield and existing stock, net-goal finish/cycle/unavailable-resource handling, ascending item-ID tie-breaking, and guarded real-native replay with target masking plus a retained capped failure. An additional replay independently reconstructed all 366 prompts and JSON/EOS loss masks, checked past-query provenance, and confirmed all 32 native successes. This verifies this prototype's construction, not that a student can learn the procedure or that the planner is generally optimal.

| Frozen data | Source047 privileged-order | Public-discovery prototype |
|---|---:|---:|
| Tasks / successful teacher traces | 32 / 32 | 32 / 32 |
| Query / craft / finish rows | 167 / 167 / 32 | 167 / 167 / 32 |
| Total rows | 366 | 366 |
| Supervised JSON+EOS tokens | 8,821 | 8,820 |
| Prompt tokens | 414,754 | 438,065 |
| Updates at effective batch16, one epoch | 23 | 23 |
| Maximum prompt plus256 generation allowance | 3,551 | 3,551 |

The equal row/update counts arose naturally; no traces were trimmed, duplicated or sampled to force a match. Target tokens differ by one, and public-discovery prompts total 23,311 more tokens. Thus the immutable manifest's prospective dose caveat should be read as “not experimentally forced to match,” not a claim that row/update counts actually differ. Any later comparison still changes the demonstration procedure and history contents together, not only one ordering variable at perfectly identical compute.

The teacher aggregates demand over queried recipe edges, credits current stock once, rounds to public recipe batch sizes, and crafts only when shared ingredient demand is satisfied. Public replies supply all dependency edges; native `crafting_depth`, `can_craft` and `in_inventory` metadata are not supplied to its recipe map. Prompts remain the unchanged public bridge prompts, including actual native replies. No future recipe is inserted before its query. The bounded implementation explicitly rejects cycles, insufficient uncraftable resources and multiple alternative recipes rather than introducing a search framework. The pinned synthetic world has one recipe per product, so no alternative-recipe failure occurred.

## Artifacts and reproduction

A separate all-query visibility audit sharpens the first-action comparison.
For each row, it parses only the saved prompt's final public JSON frame and
recursively collects dictionary keys and complete string values, including the
entire nested history. Exact-string comparison finds **135/167 source047 query
targets absent from that prior frame, across 30 tasks**, versus **0/167** for
source055. All 366 rows per procedure and identical 32-task inventories were
checked. For example, source047 train.988 step0 queries `c9_i1_19`; train.2040
steps0 and2 query `o0_i1_10` and `o1_i1` before those identifiers occur as a frame
string/key. Arbitrary item queries are allowed: this is teacher-supplied
identifier choice, not an illegal action, an input leak, or proof the model
cannot know the name. The audit does not substring-match prose or read current
target/feedback into prior visibility. Reproduction source is
`audit_textcraft_query_visibility.py`; immutable receipt is
`R/analysis-textcraft-query-visibility-001.json`, with per-query prompt hashes
and input/source hashes. No sealed source055 or prototype artifact changed.
The additive `analysis-textcraft-query-visibility-002.json` also searches the
whole prior prompt, including free-text feedback, with escaped item names and
`[A-Za-z0-9_]` identifier boundaries. It obtains the same 135 versus zero absent
query names. Both producing sources and receipt001 remain preserved externally.

Research-store root `R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Exact source/import snapshot: `R/source-055-textcraft-public-discovery/`; `SOURCE.json` SHA256 `80bf7cc468d029cbab97307146ae67c5c5c220bbfe6e725eb0566a35fbfafb59`.
- Teacher source SHA256 `712236968fd485a35dfea81ff87f6e461c342a6380efbe0840ebf19fdfabdcc6`.
- Prototype: `R/textcraft-public-discovery-prototype-001/`, including original task bytes, pre-replay selection receipt, all 32 traces, training-eligible rows, manifest and `PUBLIC-REPLAY-AUDIT.json`.
- Original task SHA256 `390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30`.
- Prototype manifest SHA256 `c69ef258a07f4c4f9b45b5bc044880f1cc9aa884e510fe590f3ddf77f19441da`.
- Rows SHA256 `dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a`.

Actual sealed test command: `/project/alex_phd/envs/prime-rl-5990b1b/bin/python -m pytest -q R/source-055-textcraft-public-discovery/test_prepare_textcraft_public.py` → five passed in 0.24s. Ruff passed for the two new implementation/test files.

The prototype was generated with the existing training-environment Python, `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, running `prepare_textcraft_public.py --root R --output R/textcraft-public-discovery-prototype-001`. To reproduce, use the sealed source and a new immutable output path. Only the cached tokenizer and trusted CPU environment are loaded. No trainer or accepted collector changed; no GPU job or training is authorized by this readiness result. Source052's separate completion watcher remains active.
