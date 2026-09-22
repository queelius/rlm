# TextCraft action SFT CPU implementation

Status: CPU-ready implementation only; no source seal, GPU acceptance, or launch.

`train_textcraft_sft.py` consumes the immutable
`textcraft-train-inputs-001/rows.jsonl` and manifest. It independently verifies the
366-row hash, 32 eligible TRAIN-task contract, unique `(task_id, step)` rows, exact
public crafting prompt prefix, and strict `get_info`/`craft`/`finish` JSON targets.
It never consumes saved feedback, labels, input IDs, native scores, task IDs, or gold
future actions as prompt content. The tokenizer is rerun and supervises target JSON
plus the native EOS only; it never truncates a row.

The fixed recipe is fresh base 4B, LoRA rank 8/alpha 16/dropout 0, BF16 base and FP32
adapter, AdamW LR `1e-4`/weight decay `0`, clip `1`, microbatch `1`, effective batch
`16`, seed `2026092208`, one 366-row epoch: 23 updates with a final 14-row batch. The
terminal endpoint is checkpoint 23, not a validation-selected checkpoint. The plan
records input, model, recipe, source, and environment identities. Every update,
including checkpoint 0, atomically saves model, optimizer, RNG, and state through the
qualified `train_planner.save_checkpoint`; resume accepts only checksum-verified
committed snapshots and continues from the saved cursor.

Focused CPU checks passed against the intended training environment: three tests cover
public target JSON+EOS masking and 23-update schedule, rejection of private/non-action
rows, and checksum-verified cursor resume; Ruff and an actual `--prepare-only` run
against immutable input047 also passed. This profile is deliberately TextCraft-specific
and does not inherit ALFWorld's successful-game count, prompt schema, target cap, or
fixed 33-update schedule.
