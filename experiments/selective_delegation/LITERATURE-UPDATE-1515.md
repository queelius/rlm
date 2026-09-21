# Literature check: sharpen the claim before expanding the system

September21,2026, about15:15UTC. Primary paper text inspected while accepted
GPU controls run. These are prior-art observations and proposed decisions, not
local experimental findings or a claim of an exhaustive novelty search.

## Relevant overlap

[GRASP, July11, v1](https://arxiv.org/html/2607.10463v1) already learns which
retrieval tool and context granularity to use. Its reward includes answer F1,
gold-document reading, complementary search and turn efficiency. Therefore
“learn what to read and how much context to use” is not, by itself, a novel claim.
Its intermediate gold-evidence rewards also differ from our terminal-answer-only
planner objective. We should distinguish added supervision from an architectural
improvement rather than copy the method and rename it.

[APEX-Searcher, inspected v2](https://arxiv.org/html/2603.13853v2) separates
RL planning from supervised execution. Crucially, its planner reward matches
generated subquestions to annotated decompositions using semantic similarity and
bipartite matching. Our terminal end-to-end answer reward is a different learning
signal, but splitting planner and executor training is established prior art.
The retrieved v2 title differs from the search index's later wording; claims here
are tied to the inspected version, not inferred from that search snippet.

[Diagnosing the Fact-Grounding Gap, September15, v1](https://arxiv.org/html/2609.17043v1)
distinguishes retrieving a designated passage from actually possessing the
relation needed to answer a subquestion. Its fact-presence labels use a validated
judge, not answer-string presence alone. This reinforces a local concern:
annotation-compatible step scoring is useful, but a wrong intermediate answer
does not automatically establish defective reasoning when the supplied passage
does not state the needed fact. Do not silently remove such cases after outcomes.

[RAVEL, September18, v1](https://arxiv.org/html/2609.21924v1) trains questions
through their downstream retrieval effect, with a frozen answerer and retriever,
in interactive person retrieval. It contrasts that objective with offline
question ordering and keeps private answerer information out of the questioner's
observation. It supports testing the value of questions through actual execution;
it does not establish that this approach transfers to our QA setting. Question
learning with downstream rewards and a partial-information interface is already
represented in current work.

## Implications for our next experiments

These are our inferences, not reported claims of those papers:

- A stronger contribution would identify **when the decomposition interface
  changes what can be learned or transferred**, beyond adding more calls or
  copying annotated plans. Current short-context questions admit strong direct
  reading, and our final model can compensate for bad helper outputs.
- Separate question quality, evidence access and execution. Our accepted
  plan-only control directly tests whether helper execution is earning its cost.
  The next-question screen tests response to feedback, but does not train a new
  feedback-conditioned policy or establish learned recursive stopping.
- Do not add several new rewards merely to obtain a positive number. Compare
  terminal reward against an explicitly justified intermediate signal only when
  the diagnosed failure makes their predictions different. Gold decomposition
  matching rewards imitation of a reference structure, not necessarily a useful
  alternative solution.
- A future partial-information experiment should compare equally informed
  reactive and hierarchical policies. Removing information only from the flat
  baseline would manufacture the desired result. Keep full-information direct
  answering as a labeled privileged reference if necessary.

The existing [adaptive task candidates](ADAPTIVE-TASK-CANDIDATES.md) and
[composition direction](COMPOSITION-DIRECTION.md) remain conditional alternatives.
This check does not launch a new training branch. Promote an alternative only
with a small, capped comparison capable of distinguishing its mechanism; record
actual information and compute differences before claiming an RLM improvement.
