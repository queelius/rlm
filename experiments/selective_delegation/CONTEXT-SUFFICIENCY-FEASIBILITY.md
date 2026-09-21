# TextCraft context-sufficiency feasibility

## What the interface establishes—and does not

In the pinned TextCraft implementation, a recursive child is created with only
its requested target string (and optional parent-provided `context`) while its
executor shares the parent's inventory by reference. It can inspect live
inventory and query recipes, but does not automatically receive the root's other
targets. This alone does **not** establish that the child frame is insufficient:
the parent can request the correct quantity, and a child that faithfully
completes that request can be locally correct.

The cached synthetic multi-target validation file has 1,000 rows: 681 with two
targets and 156 with three (837 multi-target roots); train has 9,315 such rows.
The single-target suites cannot test multi-target planning. This is evidence of
parent request/quantity information, not yet a child local-sufficiency failure.

Evidence inspected in pinned RAO clone
`d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`:

- `plugins/textcraft/platoon/textcraft/env.py:280-303` forms the child goal from
  delegated targets and supplies parent context only if passed; `:407-418` and
  `:480-491` share mutable inventory.
- `tasks.py:382-401` adjusts **direct** target-as-ingredient needs (not an
  arbitrary transitive recipe closure). Reward checks final inventory relative
  to initial inventory (`env.py:565-598`).
- SHA-256: `tasks.py` `392d087a...8e2abc`, `env.py`
  `c58ffad5...6d587d`, multi-target validation data
  `e906a34f...096fe8a`.

## Native structural witness and conclusion

A public structural scan found 12/1,000 multi-target validation roots with a
direct target-as-ingredient relation. For `textcraft_synth.val.35`, root targets
are `2×a2_i10_10` and `2×a3_i11_11`; the sole `a3_i11_11` recipe consumes one
`a2_i10_10` per craft and produces one result. The sole `a2_i10_10` recipe also
produces one result. A child correctly fulfilling a local request for two
`a2_i10_10` produces two; the parent then consumes both while crafting the two
`a3_i11_11`, leaving none. The root requires four `a2_i10_10` produced.
The saved native reference trajectory accordingly contains a craft action with
target `['a2_i10_10', 4]` followed by `['a3_i11_11', 2]`; it is a witness to
the root quantity calculation, not evidence that a two-item child action is bad.

That is a concrete global consequence of an underspecified *parent request*, not
a conflict between two locally correct child actions. The synthetic recipe
database has one recipe per output (1,402 recipe outputs; zero with multiple
recipes). With a shared current inventory and one output recipe, inspection did
not find a same-child-goal/current-inventory case where different root
obligations make different child actions locally necessary. Therefore the prior
local/parent/compact-constraint proposal is **retired** as a decomposition-
capability study.

## If revisited, the honest narrow study

Do not launch a GPU branch from this memo. A future diagnostic could select the
12 direct-dependency roots by public structure and test whether a *root policy*
requests sufficient child quantities. Any proposed training or representation
comparison must give both roots the same complete task and available recipe
information; removing the root's goal would simply cripple the baseline.
It would measure parent planning/communication, not child-frame sufficiency.

Do not construct a host recipe-closure projection unless those recipe facts have
already been queried in every arm, or charge identical fixed queries to every
arm and disclose this. Otherwise the host leaks unobserved recipe facts. Any
future prompt comparison must leave full public root state untruncated and report
prompt/generated tokens. A genuine context-sufficiency study needs alternative
child actions or a necessary state not already shared through the executor.

## Literature boundary

The inspected preprint, *Recursive Language Models Generalize Out of Domain*
([arXiv:2609.20831v1](https://arxiv.org/pdf/2609.20831)), is relevant because its
local-frame argument assumes the target is a function of that frame; its reported
controlled example uses matched recursive traces on nested modulo-10 expressions.
The PDF is stamped 23 July 2026 despite its `2609` identifier, so this memo makes
no claim it was published today. Its premise is not validated by the TextCraft
evidence above.
