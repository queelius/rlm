# ALFWorld TRAIN demonstration feasibility

Official ALFWorld TextWorld code can supply a hand-coded expert action while
executing the real game. A bounded qualification collection selected one TRAIN
game for each of the six task families supported by that expert, using SHA256 of
`2026092197:path`; it did not inspect outcomes or reselect failures. The source
is official ALFWorld commit `aaba6870f86c5be6a08a491f32a50b906227bc3e`, with
the cached text release under `alfworld-text-0.4.2-20260921`.
Collector source SHA-256 is `6afe8eb2479940dcd5bd33f8d5e01e532f6a3d36ae39a90d01e286fb06c51dc5`;
the successful trace receipt SHA-256 is
`22d94ec5393971ed09482b595206af2dd1f5dbab196e12105356b5c889024804`.

The successful artifact is `alfworld-train-demonstration-feasibility-003`: five
of six games reached a native `won` state. Their executed target counts were 42,
5, 23, 21, and 20. The selected `pick_two_obj_and_place` game timed out after
200 executed targets; its 201st target slot is retained with host result
`unavailable`, rather than being dropped or replaced. The two earlier empty
artifacts (`-001`, shell interruption; `-002`, collector reporting bug) are
preserved with status receipts and used the same selection.

Each saved policy input contains only `feedback` and the current
`admissible_commands`. The hand-coded expert may use engine facts privately, but
its command and zero-based action index are host-only supervised targets; raw
facts, PDDL/task parameters, trajectory data, and expert plans are not in the
public input. Native execution verified every completed target was admissible
before it was stepped.

This qualifies a small public-observation-to-indexed-action SFT recipe, not an
evaluation protocol or a novelty claim. It is useful only if a minimal SFT
baseline can be compared against the same public bridge and fixed TRAIN/DEV
separation. The immediate falsifiable follow-up is whether such a prior improves
unseen closed-loop success over the public-action baseline within a fixed action
and token budget; otherwise more hierarchy prompts are not justified.
