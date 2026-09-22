---
status: completed_input_audit
audited_utc: 2026-09-22T12:23:00Z
question: Were the fresh goal items absent from our SFT examples or only absent as root goals?
claim_strength: exact_identifier_exposure_in_two_local_training_sets
outcome_selection: none
---

# What is new about the 16 fresh crafting goals?

All 16 are new **root goals** under the frozen selection rule. However, three
of their final items previously appeared as intermediate products in both SFT
datasets. The other 13 item identifiers are absent from the prompts and targets
in both datasets. This is a check of our local fine-tuning examples, not a claim
about the base model's pretraining.

The same three items were explicitly queried, crafted, and then used as
ingredients in the training demonstrations:

| Fresh task | Goal item | Training task containing it | Old-teacher prompt rows | Revised-teacher prompt rows | Target rows in each dataset |
|---|---|---|---:|---:|---:|
| VAL151 | `m8_i2` | TRAIN1901 | 4 | 12 | 3 |
| VAL24 | `o9_i2` | TRAIN2489 | 12 | 31 | 3 |
| VAL462 | `o6_i3` | TRAIN2489 | 4 | 34 | 3 |

“Prompt rows” counts examples containing the exact identifier at least once,
not the number of occurrences or independent exposures. A previously returned
recipe can remain in the history of later examples. The unequal prompt counts
are another reason to describe the intervention as a **teaching package**, not
as a pure ordering intervention with identical exposure.

## How this affects the interpretation

Keep all 16 goals and both samples per goal in the primary comparison. Do not
replace that panel after seeing results. A separate 13-goal breakdown can check
whether a gain is confined to the three demonstrated products; label it an
additional diagnostic, not the originally specified primary test. This split
was derived from training inputs while the evaluation was running, before
opening the completed paired outcome report. No policy outcomes entered it.

Even those 13 goals can reuse familiar intermediate recipes. The earlier
prerequisite audit found sharing for 14 of the full 16 goals. “Not present as a
goal,” “not present anywhere in our SFT,” and “has no familiar prerequisites”
are different claims. None establishes transfer to an unrelated task domain.

A future stricter panel could exclude every demonstrated product, not just
training root goals, and separately vary the recipe world. That would be a new
prospective comparison; it does not change the running evaluation or accepted
RL/control sequence.

## Reproduction

Files under R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`:

- `textcraft-fresh-inputs-001/tasks.jsonl`, SHA256
  `da9f7498ffc136d24cc348523fe7be63586fc09bb2220ec05c35e86474382e8c`.
- `textcraft-train-inputs-001/rows.jsonl`, SHA256
  `caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9`.
- `textcraft-public-discovery-prototype-001/rows.jsonl`, SHA256
  `dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a`.

For each fresh task, read its single `misc.target_items` key. For each training
row, search the `prompt` and `target` strings separately with this exact-item
pattern, using Python's `re.escape(item)`:

```python
pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(item) + r"(?![A-Za-z0-9_])")
prompt_rows = sum(bool(pattern.search(row["prompt"])) for row in rows)
target_rows = sum(bool(pattern.search(row["target"])) for row in rows)
```

Identifier boundaries matter: the fresh item `c0_i3` must not be counted as seen
merely because a longer item such as `c0_i3_10` occurs. Parse matching target
JSON to inspect its query/craft/ingredient role. The selected training tasks'
gold product lists independently confirm the same three intermediate products;
gold data was used only for this CPU audit, never added to policy observations.
