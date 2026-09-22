# TextCraft state representation: a conditional harness question

TextCraft already supplies the root target, initial inventory, **current inventory**, remaining
global calls/tokens, depth, and the full public action/feedback history on every turn. Thus it is
not a test of reconstructing inventory from history. Its practical ambiguity is narrower: after a
successful public `get_info`, can the policy use a compact cache of observed recipes without being
distracted by obsolete actions, rejected JSON, repeated inventory feedback, or earlier failed
attempts? Removing history may also remove the only public record of a queried recipe.

## Minimal falsifiable comparison, only if 049/052 leave a state-use failure plausible

On the same eight already-exposed pilot tasks and two fixed seeds, run flat base policy under two
fresh conditions (16 episodes each):

1. **Raw history:** exact existing public prompt/history.
2. **Observed-recipe state:** retain the unchanged public goal, target/initial/current inventories,
depth and global budgets; replace history with (a) a deterministic map from item to the exact
public `get_info` reply **only after that reply has actually occurred**, and (b) latest public
feedback. Do not cache an unqueried recipe, gold route, host quantity, model rationale, task ID,
or outcome. Invalid/rejected actions affect only latest feedback and do not enter the cache.

Same model/checkpoint, decoding, parser, native world, 96-call/8,192-emitted-token episode cap and
256-token call cap apply. Report success, protocol/native errors, calls, emitted tokens and prompt
tokens; do not pad the compact arm to equal input length. Its claim is a representation/cost
trade-off, not equal-compute superiority. This is at most 32 new episodes, 3,072 possible native
calls and 262,144 emitted-token upper bounds; actual cost should be reported, not inferred. Fresh
collection is needed because 044 has failed/missing slots and differs in selection/runtime context;
the changed prompt is the intervention itself, not a reason a paired baseline would be invalid.

The go signal is fewer repeated/rejected actions or higher native success with lower prompt tokens,
without a new public-information channel. A null result retires this local cache hypothesis; a
gain would justify a fixed exposed-task replication before any fresh-panel claim. It would **not**
show generic memory, state reconstruction, recursion, or a full Python-RLM advantage.

## Relation to prior work

SKILL.state replaces append-only histories with validated mutable state plus the latest observation;
its reported benefit targets long-horizon prompt growth and noise, which TextCraft only partially
shares because current inventory is already public. [SKILL.state §3, §5.6, limits](https://arxiv.org/html/2608.26263v3)
Recuris likewise separates evidence-grounded working state from experiential skill memory and shows
that the useful component depends on domain; it is far broader than this fixed recipe cache.
[Recuris §2–3.3](https://arxiv.org/html/2608.24876v1)

At present, the observed pilot has final node histories of 17–88 public entries, so historical
noise is plausible but not demonstrated as causal. The next practical step is to finish the
existing suffix/SFT controls; do not implement this cache unless their error traces leave that
specific uncertainty live.
