# Choose the next training run from the diagnostics

September 21. Decision guide, **not acceptance of any new training run**.
Finish the queued TRAIN-fit, repeated-execution, direct-adapter and plan-only
comparisons. Small exploratory panels require examining paired changes,
protocol failures, repeat stability and cost together, not automatic promotion
from an individual bootstrap threshold.

Current root RL scores56/128 versus SFT53/128. All changed answers are valid,
and gains occur on three-hop questions, but the overall interval spans no gain.
The root is not demonstrably irrelevant. Conversely, a correct full-source
final does not demonstrate a correct decomposition.

## If training-question fit improves and rewards are reasonably stable

Prefer a bounded dose continuation: checkpoint16 versus fixed checkpoint24,
eight fresh on-policy updates, same learning rate and frozen helper/final.
Predeclare second-pass parent order and preserve optimizer state. Roughly
40–45minutes additional training at observed throughput, **plus** paired
evaluation. Select checkpoint24 regardless of its development score. This tests
training dose, not sample efficiency or a new RL algorithm.

An explicit continuation implementation is CPU-tested: it preserves checkpoint16
weights, Adam moments, RNG and global step in a new run directory. Actual GPU
restoration remains unexercised. Resetting Adam would be a different treatment.
Never edit the completed run's immutable PLAN or source to make it appear extended.

## If fit stays flat despite stable, informative candidate rewards

Prefer one update-size comparison: four updates at learning rate5e-5 from SFT48,
versus original full-pass checkpoint4 produced at2e-5. Same first four parent
blocks, sampling and downstream contract. Select the old checkpoint for dose
matching, not its held score. New training is roughly20–25minutes; readouts of
both checkpoints also cost calls and may dominate. Record likelihood movement
and TRAIN reward alongside fresh performance. TRAIN memorization alone is not
a generalization win.

## If repeated execution changes which candidates appear useful

Consider four updates with two executions per candidate and mean terminal
reward, versus ordinary four-candidate RLOO. Require evidence that repeated
estimates improve selection on the excluded seed, not merely that answers vary.
Repeating only finals addresses final noise, not helper noise.

Keep the first execution's original seed contract in both treatments, including
its already-common downstream seed across candidates. Removing old helper2/final
seed collisions changes another factor; use a matched baseline if doing that. Admission must support
fractional returns. Estimate35–40minutes training plus evaluation, disclosing
extra compute. Equal updates are not equal compute. An eventual efficiency
claim also needs a competing use of that budget, such as more candidate plans.

## If helper returns matter but helper answers remain the bottleneck

Freeze the root and change one answering component. A further supervised
helper dose needs justified TRAIN targets for actual generated questions;
annotated-step answers cannot simply label arbitrary generated questions.
Use direct-adapter and transfer controls to distinguish generic answering
improvement from a delegated-question benefit. Do not start by changing root
and helper together and then guess which caused a gain.

## If plan-only or direct preserves useful performance

Prefer a small original-question answer SFT comparison over more root RL:
TRAIN documents + original question → annotated short answer, one fixed epoch,
against direct base and helper-SFT used directly. This tests the component
doing useful work, not an RLM innovation. It may be a control or prerequisite
rather than the publication direction.

Retire this short-context root-RL configuration if a bounded update-size test
still yields neither TRAIN fit nor useful fresh changes, particularly if direct
or plan-only retains comparable quality at much lower cost. Change the
information-access or compositional task instead of repeating blind audits,
prompt sweeps or larger recursion trees. ADAPTIVE-TASK-CANDIDATES.md and
COMPOSITION-DIRECTION.md describe separate conditional alternatives.
