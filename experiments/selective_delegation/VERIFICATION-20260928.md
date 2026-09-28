# September 28 research checkpoint verification

This is exploratory research, not a production release. Focused checks run while
the independent GPU collector remains active; no full repository suite was run.

- Parent reran the familiar RL tests plus crossed-readout gate tests:9passed.
- Fresh-RL and compact interface tests together:13passed.
- Teacher-dose continuation plus follow-on queue tests:7passed.
- Fixed-endpoint NLL and fresh TRAIN selection tests:4passed.
- Interface/dose receipt and predecessor-release tests:3passed.
- Matched teacher scheduling:4passed; response-journal tests:2passed.
  Counts overlap and must not be summed as distinct tests.
- Actual saved-request compact native fixtures passed in both recipe worlds.
  Teacher-order and compact demonstrations passed independent native replay.
- Original teacher checkpoints contain504 Adam states atstep23 and a saved CUDA
  RNG state. Continuations restore these rather than silently resetting them.
- Accepted fresh-campaign source/input pins were rechecked before dispatch.
  Prepared plans are not claimed as completed training.
- A separate agent reviewed the changed fresh-RL, compact and queue-handoff seams
  at10:54–10:59UTC. It found no blocking defect, reran16focused tests and replayed
  both compact saved-request fixtures with their deliberate errors preserved.

Broader targeted Ruff inspection found five nonblocking compact-package warnings:
three101/102-character lines in frozen`prepare.py`, import ordering in the CPU
fixture, and its synchronous loop-local model closure. Accepted frozen source was
not altered for style. Other newly checked packages/launchers passed Ruff. Do not
describe the entire compact directory as lint-clean. The queue setup test caught
and fixed an actual mismatch between a scalar old source hash and a new source-hash
map before launch. Failed preparation did not acquire the GPU.
The generated Matplotlib SVG retains its standard path-line whitespace; the
staged whitespace check passes with that generated asset excluded.

First live breadth response checks returned actual model output within3seconds,
with zero failed calls. The first completed new comparison is7/16versus7/16,
with model calls724→573. At10:55UTC world50raw completed16/16recorded attempts,
7successes,689returned calls and zero transport failures; its binder counterpart
is active. No new optimizer improvement is claimed yet.
