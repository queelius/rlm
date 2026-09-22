# Prospective public-discovery SFT: same recipe, different demonstrations

CPU preparation only; main decides acceptance after the complete052 readout.
No GPU training or new readout is authorized by this plan. Use the writing-plans
and focused TDD workflow inline; autonomous approval covers this bounded preparation.

Question: does teaching observable prerequisite discovery transfer better than
teaching a privileged action order, under the same small training recipe?
This tests a demonstration policy and its resulting public histories together,
not a pure action-order intervention or a novel recursive-learning method.

## Frozen contrast

Use source055's verified `textcraft-public-discovery-prototype-001` without
selection, repair, rewrites or regenerated trajectories. It uses exactly047's
32 official TRAIN tasks; the public teacher queries the root before discovering
prerequisites and never reads future gold actions or declared crafting depth for
action selection. All32 native teacher traces succeed; all366 rows are retained.
The source055 public replay audit independently reconstructs prompts, provenance
and JSON/EOS masks. See `PUBLIC-DISCOVERY-READINESS.md` for the teacher limitations.

Compare against048's fresh-base privileged-order adapter at its fixed step23,
not against a continuation warmstart. Both use the same32 tasks,366 rows,23
updates and167 query /167 craft /32 finish targets. This equality arose naturally.
Public-discovery targets contain8,820 supervised tokens versus8,821; prompts
contain438,065 versus414,754 tokens (+23,311). Histories, token dose and wall/FLOP
cost are therefore not exactly matched, despite equal row/update counts.

Retain048 byte-for-byte: base Qwen3-4B-Instruct-2507, fresh rank8/alpha16/dropout0
LoRA, BF16 base/FP32 adapters, AdamW LR1e-4/weight decay0, clip1, microbatch1,
effective batch16, seed2026092208, one epoch, final batch14,23 updates. Supervise
only target JSON plus EOS, context8192/target256, no truncation. Save checkpoint0
and each real update with optimizer/RNG/state; fixed endpoint checkpoint0023,
never chosen on validation. Cumulative cap30minutes; incomplete23 is incomplete,
not permission to substitute a different endpoint or extend the dose.

Only a thin `train_textcraft_public.py` wrapper is new. It validates exact055
manifest/rows/tasks/public-replay/source hashes and byte-identical048 recipe
dependencies, writes an immutable TEACHER-CONTRACT receipt, then calls the
unchanged trainer. Existing trainer PLAN already binds prepared input manifest
and row hashes. Keep all host IDs, gold paths and scores out of model inputs;
the trainer retokenizes only the saved public prompt and action target, ignoring
host metadata and supplied mask/token arrays.

## Implementation and review scope

- [x] RED: actual055 inputs pass the proposed wrapper API; old047 input or changed
  teacher/audit identity rejects. Test complete32/366/root-first qualification.
- [x] GREEN: implement only validation/receipt/entrypoint around unchanged048.
  Preserve exception-to-owner failure reporting, resume and cumulative cap.
- [x] Verify ignored host metadata cannot enter tokenized prompt/targets; reuse
  the three048 fixtures for JSON/EOS alignment, schedule and committed resume.
- [x] Seal056 separately with exact048 trainer/probe/recipe dependencies; run
  focused sealed tests and an actual CPU `--prepare-only` on all366 frozen rows.
  Record token totals and input/source hashes. No model weights or GPU calls.

## Prospective readout and decision

After main acceptance and a complete step23, a separately accepted matched
readout may reuse052's identical tasks, public interface, seeds and global
budgets for base /privileged-order SFT /public-discovery SFT. It must retain the
whole panel and report native success, unknown/protocol/budget failures, calls
and tokens. Recursive deployment is instruction transfer: these flat teacher
traces do not themselves supervise child delegation.

Promote public-discovery teaching if fixed-endpoint success improves with useful
public query-to-craft execution, not merely more well-formed actions or longer
rollouts. If query compliance improves but task success does not, revise the
competence/interface hypothesis instead of adding epochs automatically. A small
exposed development panel is qualification, not broad generalization; same-world
recipe/intermediate overlap remains. No outcome selects TRAIN tasks or step23.

Source056/output proposal: `source-056-textcraft-public-discovery-sft` and
`textcraft-public-discovery-sft-001` in the September21 research store R.
No reader057 preparation is included in this task.

## CPU-ready receipt

Six sealed fixtures pass in0.21s, including exact055 source/input qualification,
old privileged-input rejection, no dropped tasks/unchecked histories, ignored
host fields, target JSON+EOS alignment,23-update schedule and committed resume.
Ruff and actual CLI `--help` pass. The full366-row `--prepare-only` entrypoint
also completed with CUDA hidden and Hugging Face offline, leaving only PLAN and
TEACHER-CONTRACT in the proposed output—no OWNER, model or checkpoint.

The exact launch argv is recorded, **not accepted**, in
`R/TEXTCRAFT-PUBLIC-DISCOVERY-SFT-DECISION-001.json`. `--resume` is needed to
validate/reuse the precreated immutable PLAN even though no checkpoint exists:

```sh
/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python \
 "$R/source-056-textcraft-public-discovery-sft/train_textcraft_public.py" \
 --prepared "$R/textcraft-public-discovery-prototype-001" \
 --output "$R/textcraft-public-discovery-sft-001" --hours .5 --resume
```

For CPU preparation only, add `--prepare-only` with `CUDA_VISIBLE_DEVICES=''`.
Do not launch before main acceptance. Source056 SOURCE.json SHA256:
`c7e8dd4fb4d401a35877f1e0edac5b2a62cca48c34eda5177145bcbf072c40d1`;
wrapper SHA256 `0bc333af23132f120702470f1612574efb5419c6ab45e5c92d2fc19e437ad3ea`;
prepared PLAN SHA256 `5a34080562a39dc92dee2a813078e730b883fb60944dad96b613598c31f9624c`.
Source048 and source055 remain unchanged. The source056 snapshot preserves the
prospective plan as it stood before the CPU qualification; this section records
the subsequently completed checks without rewriting that seal.
