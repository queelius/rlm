# Proposed official-sampling direct baseline qualification

CPU outline only: not implemented or GPU accepted. The capped-request doubling
proposal was retired after inspection found repetitive/type-drifting outputs,
not merely long useful answer lists. The repetition-penalty1.1 idea is also not
the chosen control: native HF repetition penalties affect previously seen prompt
tokens and may penalize copying evidence names.

Reissue **all 32 direct200 calls**, preserving all 16 parents and both seeds,
exact public prompts/evidence, frozen 4B checkpoint, tokenizer/chat template,
strict parser and official metric, 1,024 output cap and 40,960 context guard.
Change only the sampling package from `.5 / 1.0 / 0` to the cached official
recommendation **temperature .7 / top-p .8 / top-k20**. This does not isolate
temperature causally. It is not another independently selected panel.

The cached model README recommends .7/.8/20 and min-p0. It separately discusses
presence penalty and a much larger generic output allowance (16,384 tokens).
We deliberately retain the existing 1,024 cap and existing EOS behavior, so call
this **official sampling parameters**, not every official generation default or
an optimally tuned direct baseline. No presence/repetition penalty is added.

The inspected `generation_config.json` contains .7/.8/20, `do_sample=true`,
EOS IDs `[151645,151643]`, and pad ID151643. Source028 explicitly overrides EOS
with the tokenizer's single EOS ID; preserve that in this comparison. The loaded
CPU GenerationConfig has no configured forced BOS/EOS, minimum length,
repetition penalty, no-repeat n-gram, bad-word list, sequence bias or token
suppression. Installed HF code only constructs the corresponding processors
when those fields are active; no hidden forced-EOS-at-cap mechanism was found.
Save the resolved generation configuration and exact explicit call overrides in
new receipts rather than relying on implicit library defaults.

Use new immutable native receipts and a planned denominator of 32. Recompute
official macro precision/recall/F1 and parent-paired differences, protocol
failures, EOS/cap stops, raw repetition diagnostics and measured costs. The old
32 calls are reused comparison data, never counted as new physical calls.
Cap the job at 30 minutes; 5–10 minutes is a rough expectation, while the original
direct native time was 10.4 minutes and is the safer planning reference.

If the direct baseline improves meaningfully in validity and content, run the
entire map arm with the same sampling package before comparing architectures.
If it does not, end the QAMPARI fan-out branch rather than sweep temperatures,
penalties, output caps or chunk sizes. No router/recursive training is justified
by the present negative/uncertain screen.

Inspected immutable model directory:
`/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554`.
Generation-config SHA-256:
`835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48`.
README SHA-256:
`8e3dd0c3b5b11897cc71092ccfe517bb7a9783479baa3665aad73c8d1a2041cd`.
No weights were loaded for this inspection.
