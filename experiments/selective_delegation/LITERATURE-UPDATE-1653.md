---
status: research_inputs_not_replicated
retrieved_utc: 2026-09-21T16:53:00Z
decision: qualify_new_information_and_action_tasks_before_more_unchanged_root_rl
---

# Further primary research and what it changes

These papers constrain our claims and suggest controls. Their reported results
are not results we have reproduced.

## Planning-specific rewards already have substantial prior art

[DecomposeR](https://arxiv.org/html/2605.30824v1), May 29, separates planner and
answerer training over typed graphs. Its planner uses coverage, search quality
and structural rewards; the answerer is trained afterward. It also revises a
plan after seeing search results. Its long-form setting and judge-assisted
rewards differ from our short-answer experiment.

**Our decision:** neither planner-only RL nor typed decomposition graphs would
be a novel contribution by themselves. Before borrowing a structural reward,
test whether rewarding that structure improves answers rather than merely
producing more branches. Our negative cheap-control results make that
distinction important. No new training job is accepted from this paper alone.

## More training needs informative examples, not just more steps

[Adaptive Data Scheduling](https://arxiv.org/html/2606.22305v1), June 21,
combines semantic clusters with a changing band of examples near the policy's
success boundary. It motivates this through the lack of useful relative
advantages when every candidate succeeds or every candidate fails.

**Our decision:** our zero-advantage batch does not establish convergence.
However, the fresh checkpoint-21 result gives no reason to extend unchanged
root training immediately. If a new task produces mixed, valid outcomes,
compare a bounded fresh-data schedule with a fixed schedule before adding a
more elaborate curriculum. Keep skipped batches and their acquisition costs.

## A helper's answer is not automatically useful evidence

[Evidence Integration in Large Language Models](https://arxiv.org/html/2609.04290v1),
September 3, studies how receivers combine their own answer preferences with
outside candidates. It reports that error alignment and the receiving model
affect whether supplied evidence helps; checking an answer and using that
check are distinct behaviors.

**Our inference:** this is relevant to our correct final answers despite wrong
helper chains and to the different TRAIN and DEV helper effects. A small
future comparison could replace answer-only helper messages with source-linked
facts under matched input budgets. We would measure both final accuracy and
whether the purported supporting facts actually determine the answer. Generic
verification or evidence-grounding claims are not new.

## Re-reading selected evidence is another necessary control

[ReContext](https://arxiv.org/html/2607.02509v1), July 2, uses query-conditioned
internal relevance signals to select and replay evidence spans while retaining
the original context. Its attention signals propose spans; they are not
treated as faithful explanations.

**Our decision:** an improved long-context answer is not automatically evidence
for recursive task decomposition. A source-extraction or evidence-replay
baseline may explain the improvement more cheaply. For the pending QAMPARI
screen, first compare full-pool direct reading with fixed passage-group reading;
only then decide whether a more complex evidence interface is warranted.

## A potentially distinctive question, still only a hypothesis

When should a helper return an answer, and when should it return evidence that
another helper must combine with evidence from elsewhere? A simple union of
answer lists can work for collecting independently supported entities, but it
cannot generally solve a relationship split across two input groups. We can
look for that boundary without inventing new gold-aware routing rules.

If the pending fixed-split screen shows this contrast, the next small study
would compare answer-only merging, source-linked evidence merging, and a
budget-matched direct answerer. Learning when to request another local pass
would come after establishing that the alternative message actually helps.
This is a proposed experiment, not an established novelty claim or an accepted
training run. A method must beat a simple control before we invest in search
or a large adaptive tree.
