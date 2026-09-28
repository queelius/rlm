# ALFWorld index versus command — September28,2026

Worth a bounded screen, not a novel symbol-binding method. Old index SFT used524 targets from30
successful TRAIN games and33 updates. On the exposed12-game panel, flat success fell4/24→2/24,
with zero invalid outputs. Object confusion, repeated completed actions and a numeric index that
alternates between take/put are plausible failure signatures, but do not identify an index cause.
See the existing ACTION-SFT-FINDINGS, UNSEEN-FINDINGS and FAILURE-MECHANISMS notes.

Important counterevidence: the earlier exact-command screen had 76 well-formed but inadmissible
commands, and 25/32 episodes stopped after three invalid attempts. The later combined indexed-list
and rejection-history interface had zero invalid calls, but flat wins stayed 1/16. Those old command
prompts used an unindexed list and omitted rejected attempts from subsequent public history, so the
comparison did not isolate labels. This study keeps the indexed list and rejection history in both
arms and adds command-target training. Command-base admissibility may nevertheless regress; record
that explicitly instead of interpreting longer trajectories or higher raw command scores as learning.
See [ALFWORLD-SCREEN-FINDINGS.md](../ALFWORLD-SCREEN-FINDINGS.md).

We compare native-success gains: `(command-trained − command-base) − (index-trained − index-base)`.
One new command SFT uses the same rows, native commands, ordering and optimizer updates. Four
own-interface readouts use the same newly selected official valid_unseen games and seeds. Neither
the old failed hierarchy nor the conditional permutation proposal is revived. Both interfaces keep
the numbered list and complete public history; only output instruction/schema/targets differ.

Command tokens supply lexical action/argument supervision absent from index-only targets, but also
increase target-token exposure and generation latency. The token-mean optimizer reweights examples
when command lengths differ. Therefore a positive interaction supports this representation/training
package, not a pure symbol-binding diagnosis. A command-base gain without a training interaction is
an inference-interface benefit, not improved learning. Zero/negative gain or syntax-only changes
retire this small recipe; no larger dose or manager training follows automatically.

## Primary-source scope, read September28,2026

- [Xue et al.,2406.01026v2](https://arxiv.org/html/2406.01026v2), June6,2024:
  read §§2.1–3.3 and§7. Their MCQA SFT compares symbol-only and symbol+content targets;
  adding content alone has inconsistent bias effects. They add symbol reweighting and negative
  symbol/content pairs. This directly precedes the learning/label hypothesis, but not this exact
  content-only native-action decoder and closed-loop task-success comparison.
- [Wong et al.,2601.03914v1](https://arxiv.org/html/2601.03914v1), January7,2026:
  read experimental setup, winner/binding interpretation and limitations. Qwen3-8B and Llama3B
  probes/permutations distinguish content winner information from the emitted symbol. Their
  scope is fixed four-choice tasks, not action-SFT or sequential environment learning, and the
  paper warns that direct-answer-text prompting may use different strategies.

No claim that index/content sensitivity is new, or that this small cross-environment extension is
publishable alone. Generalizable contribution would require reproducible gain interactions across
tasks/interfaces and stronger mechanism controls. The current value is deciding whether this
campaign's ALFWorld negative SFT result is partly an action-target choice.

Assets are already present. Prior24-slot flat readouts used443–759 native seconds; one command
training plus96 new episodes should be roughly1–2 useful GPU hours, with explicit longer-command
latency uncertainty and bounded caps. A native saved-response fixture must pass before handoff.
