---
date: 2026-09-29
scope: Selective delegation, MuSiQue/Hotpot/QAMPARI QA, evidence sufficiency, ALFWorld, and actual helper uptake
requested_window: 2026-09-14/2026-09-29
source_coverage: September 14-15 completed breadth screen and September 21-29 selective-delegation campaign; no separate September 16-20 continuation located in the bounded inventory
excluded: TextCraft teacher repair, fresh TextCraft RL, new experiments, process operations, and queue changes
evidence_cutoff_utc: "2026-09-29T13:31:13Z"
status: retrospective_of_completed_exploratory_evidence
research_store: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
verification: Targeted reads of native JSON reports and small admission artifacts; no new model execution, full ancestry hashing, or regrading campaign
---

# Decomposition, transfer, and the difference between using a helper and benefiting from one

The accumulated evidence supports a useful negative conclusion: this campaign has not
established that learned question decomposition or a manager/helper architecture improves
held-out task success enough to justify its additional computation. It has established
several narrower facts. Role-specific training can improve execution and sometimes answer
content; supported-answer/refusal behavior is trainable; a short-term manager can help
particular household games; and the current crafting harness can cause a trained flat actor
to create a real child that acts on shared inventory. These are different achievements.
The strongest changes in belief came from direct, plan-only, extra-SFT, local-reasoning,
and fresh-panel controls, which repeatedly narrowed initially promising interpretations.

This is not a verdict on the full RLM proposition. The QA planner emits a JSON list of
linked questions from the original question and document titles. Isolated helpers receive
the documents and their current resolved subquestion; a final reader receives the full
original documents, plan, and actual helper answers. It is a learned planning pipeline,
not an arbitrary program or learned recursive call tree. The sufficiency models make one
direct answerability-and-answer prediction. The ALFWorld manager periodically writes a
short goal for an action-selecting worker. The September 28 crafting diagnostic introduces
one genuine child boundary, selected by deterministic public-state rules. Calling all four
systems “RLM” would erase the very distinctions needed to interpret the results.

The following are the most decision-relevant completed comparisons. Counts retain the
planned denominator; repeated seeds are not independent tasks. Links lead to native
reports, with narrative sources developed below.

| Tested question | Completed result | What it establishes |
| --- | --- | --- |
| Does trained planning beat direct MuSiQue answering? | Fresh 64 parents × 2: direct **54/128**, SFT planner **53/128**, RL16 planner **56/128**; RL−SFT +2.34 pp, component interval −1.67 to +6.25. | No resolved planning/RL advantage; direct uses 27.6–28.5% of planner tokens. [Native policy report][musique] |
| Do helper executions earn their cost on that panel? | SFT executed/plan-only **53/56**; RL executed/plan-only **56/53**, each /128. | Effects are uncertain; plan-only uses approximately one-third of executed-policy tokens. [Native ablation][planonly] |
| Can helper reports help conditional on fixed TRAIN plans? | Executed **198/320**, direct **148/320**, plan-only **99/320** logical slots. | A positive TRAIN-local helper signal remains after the direct control; only 16 parents, 15 components, and 80 actual direct calls. [Execution report][trainexecution], [direct control][traindirect] |
| Does the architecture transfer to larger Hotpot data? | Fresh 128 parents × 2: direct **152/256**, planner plus trained helper **151/256**; planner uses **3.44× tokens**. | No resolved benefit; the smaller helper-transfer pilot is not a general architecture result. [Native comparison][hotpot] |
| Does fixed fan-out improve QAMPARI? | Direct/map-union F1 **.1863/.1661** on 16 parents × 2; sampling-control direct **.1916**. | No established fan-out gain or baseline rescue; the branch was reasonably retired. [Native screen][qampari], [sampling control][qamsampling] |
| Does sufficiency RL outperform another supervised dose? | Two-hop warm/RL8/extra-SFT8 paired EM **10/18/19 of 64**; deeper panel **6/7/6**. | Easier-panel joint gains, no RL-specific advantage, no clear deeper joint gain. [Two-hop][suffheld], [deeper][suffdeep] |
| Do revised sufficiency rewards/estimators help? | Product/additive RL **10/4 of 64**; product diagonal/pairing-mean **10/6** on the exposed control panel. | More active credit is not itself better learning; neither follow-up supports expansion. [Reward control][suffreward], [estimator control][suffestimator] |
| Is the early ALFWorld manager gain robust? | Exposed flat/manager/local reason **1/6/6 of 16**; unseen games **4/4/6 of 24**. | The early gain is not unique to a manager and its point advantage does not replicate. [Closed-loop][alfclosed], [unseen][alfunseen] |
| Does ALFWorld action representation unlock learning? | New panel: index base/SFT **1/2 of 24**; command base/SFT **0/4 of 24**. | A narrow package signal, concentrated in two games in one scene, with substantial legal-action failure. [Native factorial][alfrepresentation] |
| Do actual crafting children improve root completion? | Flat/fixed/adaptive admission screens: **0/9/13 child calls**; all root scores **unknown**. | Actual child uptake, including one locally successful child; root efficacy remains unmeasured. [Fixed audit][fixedaudit], [adaptive audit][adaptiveaudit] |

The initial QA question was whether another reasoning step helps enough, and differently
enough across questions, to justify learning when to delegate. On the first 32 MuSiQue TRAIN
parents, proposed subquestions scored 34.4% versus 28.1% for finishing, at about 62% more
tokens. The parent-bootstrap interval for the 6.25-point difference included zero. An
other-repeat selector, which already had privileged outcome information for the same
question, did not beat a fixed strategy. More importantly, changing the final instruction
to request a short answer raised scores by 11–16 points and shrank the subquestion gap to
2.08 points. The large early effect was therefore sensitive to the answer contract, with
format and answer selection not separately isolated. This was a reason to qualify the
baseline before attributing gains to delegation, not a model-training result.
[Original evolving findings](../../experiments/selective_delegation/FINDINGS.md).

The next stages separated planning, execution, and final synthesis. A 26-parent diagnostic
crossed model/reference questions with bundled/isolated helpers. Isolated execution did
not improve aggregate exact match; even privileged reference questions did not establish
a reliable advantage. The final reader could recover from a wrong helper or overwrite a
correct helper. A revealing manufacturer/date example remained wrong after removing the
initial answer, so the initial hypothesis of simple answer anchoring was not supported by
its direct intervention. These controls mattered: they prevented a plausible trace story
from becoming a purported mechanism.
[Execution findings](../../experiments/selective_delegation/EXECUTION-FINDINGS.md),
[aggregation findings](../../experiments/selective_delegation/AGGREGATION-FINDINGS.md).

Supervised planner training then fitted 256 annotated plans over 48 updates, and real
planner RL updated the adapter from final-answer rewards. Both optimization and nonzero
likelihood movement are documented. Their presence is not the missing evidence. The first
four-update held readout improved only 18→19/64, and its three wins all involved protocol
recovery. On the earlier four-hop transfer panel, base direct scored 36/128 versus SFT and
RL planners at 24/128 each; the direct−SFT component interval was positive, +2.73 to +17.86
points, while direct used 28.9% as many tokens. That finding concerns different complete
policies on a short-document task, not the isolated effect of decomposition.
[Training audit](../../experiments/selective_delegation/RL-TRAINING-FINDINGS.md),
[four-hop policy analysis](../../experiments/selective_delegation/TRANSFER-FINDINGS.md).

Training the helper was more promising than simply adding root updates. Under fixed saved
SFT plans on 32 development parents, helper SFT improved 19→28/64; a JSON reminder reached
22/64. Both interventions removed all 11 helper-format failures. The trained helper had
additional both-valid gains, although its advantage over the reminder was imprecise and
did not establish consistently correct intermediate reasoning. On the small Hotpot
explorer panel, fixed-plan base/reminder/trained helpers scored **22/35/40 of 64**. The
trained-helper gain over base included 13 protocol-related wins and five both-valid wins.
This is genuine role-transfer evidence under fixed plans. Comparing that 40/64 with the
older direct 36/64 would be misleading: the acquisitions used different seeds, and the
pilot comprised 32 questions from the website's 100-question explorer sample.
[Helper findings](../../experiments/selective_delegation/HELPER-FINDINGS.md),
[native Hotpot helper report][helperhotpot].

The later fresh controls changed the practical conclusion. With the trained helper fixed,
MuSiQue direct, SFT-planned, and RL16-planned execution all lay between 53 and 56/128.
The RL−SFT changes were now both-valid, so this null was not merely persistent broken JSON.
Plan-only matched the executed systems within uncertainty at roughly one-third the tokens.
A helper-adapted direct model scored 49/128 versus base direct 54/128, an uncertain
decline: generic answering improvement was not demonstrated either, and selecting only
the weaker adapted direct baseline would exaggerate planning's value. Five additional
committed root updates produced 54/128 at checkpoint21, versus 56 at checkpoint16;
training stopped when the next batch lacked admission signal. There was no scientific
case for another unchanged dose. The larger Hotpot comparison, 152 versus 151/256,
likewise did not reproduce an architecture advantage. Its uncertain F1 preference for
planning (+1.70 points) accompanied 3.44× tokens and 4.18× calls.
[Fresh MuSiQue findings](../../experiments/selective_delegation/FRESH-CONTRACT-FINDINGS.md),
[direct-adapter control](../../experiments/selective_delegation/DIRECT-ADAPTED-FINDINGS.md),
[stopped-dose native report][stoppeddose], [fresh Hotpot report][hotpot].

The positive TRAIN execution result must nevertheless remain visible. With frozen
historical plans, actual helper reports beat both empty-report plan-only finals and a
direct answer, by 15.63 points versus direct with component interval +1.76 to +32.67.
All paired comparisons were valid JSON. This rules out the narrow explanation that
helpers merely recover damage done by presenting a plan. It does not establish a
deployable policy: five settings and four candidates share just 16 TRAIN parents, and
each physical direct answer is copied into four logical candidate slots. The proper
research question is why helpful conditional execution does not produce a robust fresh
policy advantage. More planner dose was an inadequate answer; reward can credit a final
reader that bypasses or repairs the purported plan. A next-question screen reinforced
the distinction: seeing previous answers changed 19/22 valid next questions but changed
no exact-match outcomes, with both arms at 6/30 eligible attempts.
[TRAIN direct findings](../../experiments/selective_delegation/TRAIN-DIRECT-CONTROL-FINDINGS.md),
[incremental feedback screen](../../experiments/selective_delegation/NEXT-QUESTION-FINDINGS.md).

QAMPARI tested another proposition: whether splitting a fixed retrieved pool into four
50-passage maps and taking an exact union is better than one 200-passage reader. It did
not test adaptive retrieval or recursive decomposition. Recall increased slightly while
precision fell; F1 decreased .0201 with interval −.0755 to +.0274. Invalid capped outputs
often contained repeated strings, not merely useful lists awaiting closure. Recommended
sampling changed direct F1 by only +.0054, reduced validity 22→21/32, and increased native
generation time. The apparent 27.8% aggregate map time saving was also sensitive to long
degenerate direct outputs; a both-valid/all-EOS subset reversed the time ordering, though
that post-treatment subset is not a causal correction. This branch supplied useful
failure analysis, not a case for chunk-size or decoder sweeps.
[Screen and timing caveat](../../experiments/selective_delegation/QAMPARI-FINDINGS.md),
[sampling decision](../../experiments/selective_delegation/QAMPARI-SAMPLING-FINDINGS.md).

The sufficiency work was a substantial shift in question. It asked whether the model can
answer when an official MuSiQue context is sufficient and abstain on the paired negative
variant. A correct pair requires the supported answer and the negative-label behavior;
there is no helper architecture in this comparison. Positive-only SFT increased supported
answering but answered every negative variant, producing zero paired successes. Joint SFT
reduced overanswering while also refusing many supported questions. The fresh replication
preserved this tradeoff: base/joint paired EM 6/11 of 64, supported EM 20/16, supported
refusals 25/41, negative overanswers 24/6. Thus “better sufficiency” could not be inferred
from negative-label compliance alone.
[Joint-supervision findings](../../experiments/selective_delegation/SUFFICIENCY-TRAINING-FINDINGS.md),
[fresh canonical replication](../../experiments/selective_delegation/SUFFICIENCY-CANONICAL-FINDINGS.md).

Eight RL updates did repair part of the joint warmstart's conservatism on the separate
two-hop readout: paired success rose 10→18/64, with a component interval of +5.00 to
+20.31 percentage points. But eight extra supervised updates reached 19/64. Both methods
answered more supported questions and also overanswered more negatives. On the new
three-/four-hop panel, paired success was only 6/7/6 for warm/RL/SFT, despite supported
accuracy improving from 9 to 14/15. This is stronger evidence about the bottleneck than
the easier-panel headline: learned response propensity changed, while joint reliability
did not clearly carry over. Panels differ beyond depth, and deeper compositions already
occurred in TRAIN; the result does not isolate a causal depth effect or test unseen-depth
extrapolation. Matching eight updates and response slots also did not match information,
credited tokens, or FLOPs.
[Native easier readout][suffheld], [native deeper readout][suffdeep].

The objective follow-ups made the retirement decision more concrete. Additive per-side
reward created more potential learning signal but reduced paired success from product
RL's 10/64 to 4/64, interval −18.97 to −1.72 points. It refused 52 supported attempts
versus product's 40 while reducing negative overanswers 8→2. The pairing-mean estimator
retained the product objective but also scored lower, 6 versus 10/64; four discordant
outcomes were all losses. That latter panel was already exposed, and both results use one
training seed. Neither proves RL broadly ineffective. Together with the matched-SFT
controls, they argue against treating reward density or estimator variance reduction as
the principal research opportunity in this branch.
[Reward-control interpretation](../../experiments/selective_delegation/PAIRED-REWARD-CONTROL-FINDINGS.md),
[pairing-mean interpretation](../../experiments/selective_delegation/PAIRED-PAIRING-MEAN-FINDINGS.md).

ALFWorld initially revealed an interface bottleneck: all 76 invalid replies in the
string-command screen were valid JSON containing commands outside the current admissible
list. Switching to numbered choices plus an available rejection-feedback path eliminated
invalid outputs in the next screen. Flat remained at 1/16, while the manager reached
6/16; all gains occurred on three placement games. The model was now executing legal
actions, but could still endlessly repeat ineffective actions or continue cleaning after
the subtask was accomplished. A local-reasoning control also reached 6/16, weakening a
hierarchy-specific interpretation. It used fewer total tokens but approximately 2.8× the
manager's native generation time: output length and serial generation matter.
[Original interface screen](../../experiments/selective_delegation/ALFWORLD-SCREEN-FINDINGS.md),
[closed-loop findings](../../experiments/selective_delegation/ALFWORLD-CLOSED-LOOP-FINDINGS.md),
[local reasoning](../../experiments/selective_delegation/ALFWORLD-LOCAL-REASON-FINDINGS.md).

The fresh twelve-game panel broadened to six task families but only four scenes. Flat and
manager both won 4/24; local reasoning won 6/24 with uncertain contrasts and 3.8× flat's
native generation time. Manager used 30% more tokens and 53% more native time without
more aggregate wins. Training the action actor then lowered both flat and manager from
4→2/24 on that exposed panel; each loss pair was two seeds of one game. That result does
not establish general harm from SFT, but provided no learned-hierarchy gain to scale.
[Unseen findings](../../experiments/selective_delegation/ALFWORLD-UNSEEN-FINDINGS.md),
[actor-training factorial](../../experiments/selective_delegation/ALFWORLD-ACTION-SFT-FINDINGS.md).

The September 28 representation experiment was a better targeted question: given the same
524 successful TRAIN rows, is it easier to learn the actual command than a changing
action-list index? Index base/SFT scored 1/2 of 24; exact-command base/SFT scored 0/4.
However, command targets had 32.1% more tokens and changed token-mean example weights,
so this was a representation/training package comparison. All four command-SFT wins
came from two games in scene10 at both seeds. It also made 73 inadmissible outputs versus
54 base, and 20/24 trained-command episodes ended after three invalid outputs. Its
shorter failed episodes cannot be advertised as an efficiency gain. Index SFT cut
immediate repeated commands 571→4 yet solved just 2/24: loop suppression alone did not
create task competence. The common remaining issue is state-dependent action choice and
progress after a subgoal, not merely legal output syntax.
[Native representation report][alfrepresentation],
[harness synthesis](../../experiments/selective_delegation/harness_synthesis_20260929/README.md).

Actual child execution arrived late and answers a narrower question than all of the
training results above. The earlier trained TextCraft actor had received 366 flat-action
targets and initially made no delegates. The newer flat/fixed/adaptive diagnostic
gave every arm the same public subgoal facts and used actual model-emitted delegate
actions. Fixed routes the first eligible missing prerequisite; adaptive applies declared
public branching, shared-stock, and remaining-budget rules. Neither learns its router,
uses hidden recipe depth, reserves stock, or creates grandchildren. Parent and child
share inventory and the global call/output limits, with separate entry snapshots for
net-production scoring.
[Admission contract](../../experiments/selective_delegation/decomposition_20260928/README.md).

Relaxed query ordering produced real helpers: fixed had nine child responses and its
child produced the requested two units; adaptive had thirteen and produced only two of
four requested units before finishing. Every arm stopped at the admission ceiling of
32 responses, so every root outcome is unknown. The saved child scores are native local
records; interrupted roots received receipt binding rather than complete native state
replay. This establishes functional uptake, not success benefit, and adaptive's lower
screen token count cannot establish an efficiency improvement. At this audit's cutoff,
the complete-goal study's three arm directories contained only `PLAN.json`; there was no
`COMPARISON.json` or scientific terminal outcome. CPU scripted/replayed fixture outcomes
are explicitly excluded from model evidence.
[Fixed native admission][fixedaudit], [adaptive native admission][adaptiveaudit],
[accepted complete-goal specification](../../experiments/selective_delegation/delegation_complete_20260929/README.md).

The most defensible account of the trajectory is therefore progressive localization of
the problem. Early QA experiments confounded answer style, plan quality, helper protocol,
and final-reader rescue. Freezing and controlling those pieces produced real local gains,
but stronger direct and fresh-panel comparisons removed the case for scaling the
architecture. Sufficiency experiments found a trainable answer/refusal tradeoff and then
showed its limited compositional reliability. ALFWorld moved from illegal commands to
legal but ineffective action sequences, and controlled away the early hierarchy story.
The crafting child screen finally established that helpers can execute within the native
shared-state harness, leaving their contribution to root completion as the next missing
measurement. These negative and narrowing results are scientific progress; they should
change which experiments are repeated.

The smallest next falsifiable comparison in this scope is already specified: run the
accepted complete-goal flat/fixed/adaptive comparison on the same exposed crafting root
at two fresh paired seeds, removing only the 32-response screen ceiling. Keep checkpoint,
binder, shared inventory, one-child limit, and original 96-call/8,192-output/8,192-context
budgets fixed. Report all six native root outcomes, local child outcomes, errors, and
root/child/global costs. Planned work is roughly 15–30 GPU minutes, with a 60-minute total
science cap. This audit did not launch it. One root cannot establish transfer, but it
can falsify the claim that the admitted child is useful even in its selected mechanism
case. Flat ties at lower cost or child-local success without a root gain would argue
against deeper recursion; a coherent root gain would justify a prospectively frozen
multi-root replication before any learned router or larger hierarchy.

Other branches need a new discriminating hypothesis before new dose or breadth. For QA,
newly acquired information should actually change a necessary decision, and a matched
single-agent computation control should remain present. For ALFWorld, state-dependent
legality and subgoal completion should be evaluated at fixed saved states before another
manager grid; demonstrated whole-task improvement must follow any local repair. For
sufficiency, any revival needs supported accuracy and false abstention reported jointly
with negative overanswering, plus a stronger comparison to extra supervision. These are
prospective decision criteria, not accepted new jobs or observed results.

Several limits cut across the whole record. Fresh questions are not necessarily fresh
documents: 14/64 fresh MuSiQue parents shared exact TRAIN paragraphs, and 63/64 variants
in the deeper sufficiency panel contained a TRAIN document. Official negative labels
are not a perfect semantic support oracle. Connected-component resampling is more honest
than treating seeds as independent, but cannot manufacture new training seeds, unseen
facts, or enough ALFWorld scenes. Most intervals are exploratory and unadjusted across an
adaptive sequence. Full-source final correctness does not validate helper reasoning;
native child correctness does not imply root benefit; and token counts, summed service
time, wall time, and optimization work are distinct costs. The linked reports preserve
those distinctions better than a single cumulative “RLM win rate” could.

This retrospective checked headline values against the native policy, plan-only,
TRAIN-direct, Hotpot, sufficiency, ALFWorld, and child-admission reports named below.
It relied on their existing receipt/native-replay audits instead of repeating an ancestry
walk or loading weights. Earlier narrative notes sometimes describe a then-pending run;
the later completed report governs the conclusion here. No missing full-goal helper
result has been inferred from preparation status.

[musique]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-fresh-contract-policy-001.json
[planonly]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/plan-only-001/ANALYSIS.json
[trainexecution]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-training-execution-credit-001.json
[traindirect]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-training-direct-001.json
[hotpot]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-hotpot-fresh-architecture-001.json
[helperhotpot]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-helper-transfer-hotpot-001.json
[stoppeddose]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-fresh-rl-stopped-dose-001.json
[qampari]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-qampari-001.json
[qamsampling]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-qampari-sampling-control-001.json
[suffheld]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-sufficiency-heldout-rl-001.json
[suffdeep]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-sufficiency-compositional-rl-001.json
[suffreward]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-sufficiency-reward-control-001.json
[suffestimator]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-sufficiency-pairing-mean-001.json
[alfclosed]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-alfworld-closed-loop-001.json
[alfunseen]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-alfworld-unseen-001.json
[alfrepresentation]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/alfworld-representation-20260928-001/COMPARISON.json
[fixedaudit]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-decomp-flexible-fixed-20260928-001/ADMISSION-AUDIT.json
[adaptiveaudit]: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-decomp-flexible-adaptive-20260928-001/ADMISSION-AUDIT.json

## September 14–20 gap: the breadth campaign completed, and arithmetic deserves attention

The September 14 launch note understates the evidence now available. The unattended
breadth campaign completed normally on **September 15 at 01:13 UTC**, after 18.43 hours;
its terminal records no failures, stop request, or deadline stop. It produced 31,276
native calls and 17,236 episode records, including 3,060 common-context exclusions.
A bounded reread of the small episode JSON files reproduced every summary cell's
episode count, found no nonexcluded unavailable outcome or unknown call usage, and
totaled 38,080,763 input-plus-output tokens. Each model's call count also matched its
start-receipt count. This is stronger completion evidence than the old launch checkpoint,
but is not an independent tokenizer, native-response, or scoring re-audit.
[Native summary][breadthsummary], [terminal][breadthterminal], [episode store][breadthmodels].

This was fixed inference without adapters: released Qwen3-4B-Instruct-2507 and Qwen3-8B,
not a controlled model-size intervention. The direct reader saw all documents; the
summary/facts final reader saw only actual helper reports. Thus this comparison really
does test an information bottleneck, unlike the later full-source QA final. “Facts” was
a request to preserve subjects, relations, dates, quantities, units, and source IDs,
not a verified lossless representation. Helpers used contiguous character-balanced
chunks, with long source records eligible for splitting. A 512-token allowance applied
to each helper and final, so neither total output budget nor computation was matched.
[Sealed runner][breadthrunner], [campaign contract][breadthreadme].

The main two-way panel reached all scheduled stages. Entries below are correct answers
under the campaign's exploratory supplied-target scoring; LongBench denominators exclude
the common length rejection, which is shown explicitly.

| Dataset | 4B direct / summary / facts | 8B direct / summary / facts | Scope |
| --- | --- | --- | --- |
| MuSiQue | 99 / 110 / 98 of 512 | 28 / 16 / 17 of 512 | Answerable development questions |
| FinQA | 55 / 64 / 66 of 512 | 52 / 78 / 65 of 512 | 508 numeric and four boolean targets |
| BoolQ | 408 / 420 / 388 of 512 | 431 / 417 / 412 of 512 | Validation questions |
| AG News counting | 51 / 29 / 40 of 238 | 54 / 23 / 38 of 238 | Disjoint constructed counting groups |
| LongBench v2 | 28 / 29 / 27 of 63 | 19 / 18 / 21 of 63 | **440/503 excluded** in every arm by the 24,000-token full-input bound |

These numbers do not support a generic advantage for evidence-preserving prose. It beats
summary on AG News while both lose to direct; its BoolQ result loses to direct in both
models. The small 4B summary gains on MuSiQue and BoolQ reverse in 8B. FinQA summary
improves descriptively in both models, more strongly in 8B, but is weaker than the simpler
arithmetic intervention below. LongBench's tiny admissible subset cannot speak to the
large-input setting most relevant to RLMs. These are aggregate exploratory counts; no
paired intervals or independent semantic adjudication accompanied the original summary.

The most substantial positive is the one-call **restricted arithmetic-expression arm**.
The model sees the full question, table, and text and must select the literal operands
and operators, returning `{"expression":"..."}`. The host then evaluates only bounded
`+ - * /` and parentheses; it does not select facts, insert gold operands, or supply a
reference program. This combines model-based source interpretation with deterministic
calculation, using no helpers or unrestricted generated code. Four boolean FinQA
questions are intentionally ineligible. On the matched
508 numeric cases, 4B direct scores **51/508** and arithmetic **108/508**: 71 paired wins,
14 losses, an 11.22-point gain. In 8B the figures are **52/508→102/508**, with 68 wins and
18 losses, a 9.84-point gain. Both arms make one call per question. Matched total tokens
are 723,567→730,383 for 4B and 726,672→732,452 for 8B, increases below 1%.
Arithmetic output validity is 473/508 and 495/508, respectively; failures remain counted.
These matched counts were recomputed from native episode IDs, model, case, seed, arm,
and split, rather than comparing arithmetic's 508 denominator with direct's 512.
[Native episodes][breadthmodels], [arithmetic and grading implementation][breadthrunner],
[numeric-only admission rule][breadthowner].

The effect is not confined to the pilot: in the final preselected 255 numeric cases,
4B direct/arithmetic scores **25/255 versus 53/255**, and 8B **26/255 versus 53/255**.
The later “four-way” stage repeats
the first 80 cases with new seeds; on its 79 numeric cases the same one-call arithmetic
arm scores 16 versus direct 5 in 4B, and 21 versus 7 in 8B. Chunk count has no effect on
the arithmetic/direct implementation, so this is a sampling repeat on exposed questions,
not a four-way arithmetic improvement or independent new-task replication. The actual
four-way helper checks did not reveal a broad rescue: AG News direct/summary/facts are
21/6/8 of 80 for 4B and 23/4/9 for 8B; MuSiQue 15/16/8 and 3/1/0. Only ten of the 80
revisited LongBench questions were admissible.

The 508 denominator is the entire numeric subset of 512 FinQA DEV rows shuffled with
seed 20260916. The final stage contains 256 new case IDs, of which 255 are numeric, after
the earlier 256 cases. These IDs are disjoint within the campaign, but the manifest
explicitly labels FinQA **exploratory prior-exposed**. Within the campaign alone, 140 of
the final 256 cases share a source filename/table with the first 256, and 99 exact full
document bundles occur in both blocks. Therefore “later preselected cases” must not be
upgraded to independent-document replication or untouched confirmation. Earlier-stage
success/failure gates also make the overall campaign adaptive.
[Frozen selection and exposure manifest][breadthmanifest].

This is an under-synthesized positive worth carrying into the main story: assigning exact
calculation to code can help much more than passing longer evidence notes, at nearly the
same token cost. It is not learned decomposition, unrestricted program synthesis, or a
new method claim; it is the established program-aided-arithmetic pattern. Success remains
only about 20–21%, and the campaign's numeric tolerance
is `1e-6` against supplied execution targets. The earlier independent FinQA16 audit had
already documented incorrect gold executions, percent/fraction conventions, and rounding
mismatches, so the breadth scores must not be presented as official FinQA accuracy or
as fully adjudicated semantic reasoning. Numeric direct-output validity was 506/508 in
4B and 508/508 in 8B, so the poor direct score is not primarily malformed output. Precision
does matter: a recorded direct `6.89` loses against target `6.88702`, while the expression
`451.1 / 65.5` passes. Five of 71 arithmetic wins in 4B and three of 68 in 8B have direct
answers exactly consistent with nonzero gold rounded to one through four decimal places.
This is a descriptive flag, not semantic adjudication. A post-hoc numerical sensitivity
using 1% relative tolerance, the same `1e-6` absolute tolerance, and unchanged validity
still favors arithmetic: direct/arithmetic **73/508 versus 150/508** in 4B and
**81/508 versus 132/508** in 8B.
No unit rescaling or target repair is applied. This sensitivity suggests the contrast is
not solely strict decimal precision; it does not replace the primary scorer or resolve
the documented source/target defects. The appropriate next step would be a bounded
independent audit of the paired arithmetic gains/losses and unit conventions, followed
by a prospectively frozen comparison only if that audit preserves the signal. This audit
did not prepare or launch such work.
[Earlier native FinQA caveats][finqacaveats].

The bounded inventory found one campaign admission and one normal terminal, no resume
receipt, and no separate September 16–20 breadth follow-on or completed semantic analysis
in the inspected campaign root and research analyses. The research handoff still retained
its September 14 launch-only account before moving to September 21 selective delegation.
That is a documentation/interpretation gap, not evidence that the GPU was idle throughout
the intervening dates. The later selective-delegation work reused the owned-service
launcher and excluded breadth cases in data-selection controls, but the inspected record
does not show that the FinQA breadth gain received its own follow-up. The absence claim
is restricted to this inventory; historical terminal timestamps alone cannot reconstruct
all allocation or research activity.

Reproducibility pins: native summary SHA-256
`4343ea01dd7b241280b7e836dfa86432fdeb65d3fb8448807e9cbe142f3ab959`;
`runner.py` SHA-256
`df1fcef621aa40603b0ff4849f1c7bcbaf79527630e1dfcdd9d8c19717fcc154`;
`campaign_v3.py` SHA-256
`dce17ae4c69ea7a7f8755b3051a1371c6c6bdbca2bddc1271f17f87534b3d904`.
Those three small files were hashed in this audit. The existing manifest binds
`cases-v2.jsonl` to
`356301e5a85a420ed1b740208031b3dbcb9cd5018fac57618bd392e83b144783`
and FinQA to official repository revision `0f16e2867befa6840783e58be38c9efb9229d742`;
large input ancestry was not rehashed. The newly derived paired counts match episode
records on `(model, case_id, seed, chunks)`, retaining invalid outcomes, and the displayed
sensitivity compares their stored finite numeric `parsed` values with `metadata.gold`.

[breadthsummary]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/outputs/campaign-001/SUMMARY.json
[breadthterminal]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/outputs/campaign-001/TERMINAL-dd6c23ad2927.json
[breadthmodels]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/outputs/campaign-001/models
[breadthrunner]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/runner.py
[breadthowner]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/campaign_v3.py
[breadthreadme]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/README.md
[breadthmanifest]: /project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/data/MANIFEST_V2.json
[finqacaveats]: /project/alex_phd/runs/rlm-research-r4/analyses/finqa-two-example-fresh16-independent-2026-09-13/outcome-002/FINDINGS.md
