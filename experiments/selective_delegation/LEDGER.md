# Research ledger — plan: experiments/selective_delegation/PLAN.md

2026-09-21 09:18 UTC: User approved autonomous overnight research. Existing GPU
allocation5879 has one idle A10040GB MIG, approximately66.7h left. Quota99% remaining
at09:15UTC. Both completed broad campaign and earlier follow-up12 are reviewed;
this experiment changes the information-access control and uses fresh parents.

Ruling: User's explicit uninterrupted exploratory-research instruction replaces
interactive plan/worktree approval and exhaustive production-suite gates. Worktree
research/selective-delegation-20260921 isolates source; focused scientific-contract
tests and independent audit remain. Cost if wrong: exploratory evidence may need
rerunning; no production runtime interface changes are being made.

Interfaces: Task1 emits immutable host cases; Task2 explicitly projects public
fields only. Task2 emits native calls and arm outcomes; Task3 never includes arm
outcomes or gold in controller observations. Task4 publishes source/docs, not data
or model weights. No interface conflicts identified.

Idle accounting: GPU idle at authorized execution resumption09:12UTC. CPU launch
preparation is currently on the critical path; elapsed prep is not GPU science.

09:25 UTC: Pilot owner3568037 launched with sealed source-001/probe.py and
inputs-001/cases.jsonl. Four focused collector tests and two data tests passed.
Inputs256/64/64 have disjoint atomic components between splits and exclude all
512 earlier breadth parents. Historical exposure outside that panel is not
exhaustively excluded; these are exploratory development data, not a claim of
unseen-pretraining examples. Service loading is in progress. Initial launch
preparation took about13minutes of idle reserved GPU time; this is an operational
cost, not training or inference time. Analysis and independent review run on CPU.

09:26 UTC: First native scientific return took2.4s. All18 receipts available at
first live inspection. Source-001 remains sealed; owner holds existing lock.

09:29 UTC review ruling: No observed gold leakage. Operational SUMMARY omitted
failed checkpoints from denominators; raw checkpoint receipts were preserved.
Corrected repository collector reporting with a failing-then-passing fixture;
live source is unchanged. Full analysis uses the planned inventory and includes
checkpoint failures. Physical cost counts unique calls; per-policy cost includes
checkpoint once per hypothetical attempt and averages continuation repeats.
Continuation repeats condition on one checkpoint, not independent full runs.
Within-panel atomic components can overlap;31 clusters among32 pilot parents.

09:32 UTC adaptive decision: Early training-panel inspection reveals verbose
correct-content answers rejected by EM, plus helpers sometimes overturning a
right initial answer. Add a final-answer-instruction-only ablation using identical
saved checkpoints and helper reports, without gold-dependent postprocessing.
This precedes controller training so we do not confuse answer formatting with
useful decomposition. Validation remains uncollected; transfer remains unopened.
Quota98% at09:28UTC. Detailed prior-art implications are in LITERATURE.md.

09:40 UTC: Pilot completed704calls/384episodes/32valid checkpoints, no transport
or final-format failures. Analysis-pilot-001 external report frozen. Decomposition
34.4% vsfinish28.1%, uncertain; other-repeat per-parentselection33.3% vsfixed34.4%.
This is not a strong routing-headroom signal. See FINDINGS.md. Followup supervisor
3575945 accepted final-only replay plus32newvalidationparents, with span-final
instructions frozen before replay results. Source-002 is sealed. No transfer
collection authorized until training decisions fixed. No advisor-deck change yet.

Review ruling: Followup supervisor permits a capped source; currentpilot was
explicitly checked complete384/384 so that concern does not alter this run.
Any future use on incomplete source must report its actual paired denominator,
not assume384. Independent review found no prompt leakage or source-reuse defect.

09:43 UTC: Final-only replay completed384/384 calls, no errors. Finish43.8%,
targeted46.9%,decompose45.8%; instruction effect11–16points dwarfs pilothelper
differences. CVselector50% vsfixed46.9%, conditionalCI0..8.3points. Validation32
span alreadyaccepted/running. Nextdiagnostic approved:26twohopTRAINparents,
modelplan vsreference-question plan, bothfreshhelper+final,3newseeds,312calls.
Only annotatedquestionsenteroraclehelper, neverannotatedanswers; finalretains
sameoriginalcheckpoint. This tests plancontentbottleneck, notdeployableperformance.
No automaticcontrollertraining onweak32-parentheadroom. Account97% at09:42UTC.

10:05 UTC: Validation completed704calls: finish36/96, reconsider30/96,
targeted39/96,decompose35/96. Reference-question diagnostic completed312calls:
model35/78,reference39/78; all4wins concentrated2parents. No optimizer run yet.
Inspection shows helpers sometimes ignore supplied plans and answer the original
composed question instead. New execution-probe-001 launched10:04 with sealed
source-004; first scientific return succeeded by10:04:53. It crosses plan source
and bundled/isolated execution on26two-hoptrainingparents,2repeats,520calls.
Each isolated helper receives only its current subquestion/fullsource; #1 binds
to an actual prediction. Finals retain original checkpoints. This is a package
intervention (visibility, granularity, helper answer format), not a pure recursion
or compute-controlled effect. Model plans have0/26literal#1 links, references26/26.
The ~2minute gap after prior owner completed was CPU launch preparation, not GPU
science. Question-planSFT inputs256parents and48update trainer are CPU-prepared;
paired base/SFT evaluator being prepared with frozen helpers. Quota96%@09:56.

10:16 UTC: Executionprobe completed208episodes/518calls, with2reference-isolated
helperJSONfailures. Modelbundled24/52,referencebundled28/52,modelisolated24/52,
referenceisolated26/52. Noaggregateisolationadvantage. Hudsonisolatedreference
helpercorrect1954 butfinalwrong1932 inbothrepeats; allotherexamplesretained.
SFTaccepteddespiteweakreferenceeffect: cheaptestoflearnableexecutablecontracts,
notassumptionofdownstreamgain. Owner launched10:10:22 immediatelyafterprobe;
48updatescomplete10:15:35,297.2optimizerseconds, threeepochs256parents. Source005
sealed,checkpoint0048primaryfixed. Adapterhashchanged; trainingloss2.250/.663/.468
perepoch. Noheldoutresultyet. Sameparentearlieranswerreplay started10:15:36 from
source006,412eligiblefinals; first57callsreturned0errors by10:16:27. Thenqueued
matchedbase/SFT isolatedvalidation32x2 undernew no-provisional-answer architecture.
Isolatedexecutor selectedfor explicitdependencycontract, notwinningpriorprobe.
Quota95%@10:08; source/docs8e50183pushedresearchbranch, notmain.

RL preparation decision: terminal-EM RLOO on fresh root plans, frozenhelpers,
common downstreamseeds acrossfourplans/parent;16trainparents fixedbeforeoutcomes.
PositiveSFTheldoutgain notrequired: protocolfitting mayprecederewardimprovement.
Requireatleast2mixed-valid-distinct-plan traininggroups for exploratoryupdates;
flatgroupbatchesarestoppedandreported. Fourupdatesmaximum,64freshrolloutseach,
LR2e-5. NoPPOmachineryforoneonpolicypass; noofflinebanditclaimmasqueradingasRL.
Agentimplementsboundedrunner; notlaunchedandnoRLclaimyet.

10:46 UTC: Actual planner RL is running from sealed source007. Three optimizer
updates are committed; update4 is collecting. Each uses64 fresh trajectories
(16 training parents ×4 candidate plans), frozen downstream models, terminal
exact-match rewards and leave-one-out advantages. Nonzero gradients and adapter
deltas are recorded; no held-out RL claim yet. Cache-versus-full-forward BF16
log probabilities differ slightly; full-forward train/eval replay matches exactly.
Retain this numerical limitation in interpretation, not an exact-policy claim.

Initial SFT validation completed: base16/64 versus SFT19/64, paired+4.7 points
with interval−4.7..+15.6. Protocol changes explain part of the difference.
Checkpoint-removal replay completed412 new finals with no aggregate gains.
The original anchoring hypothesis is not confirmed; stop multiplying that ablation.

Accepted serialized queue: direct and one-helper baselines on the first32
validation questions; SFT and last-committed-RL on validation questions32..63;
new full-source versus trace-only final calls on frozen RL batch1; then frozen
base/SFT/RL/direct comparisons on64 four-hop MuSiQue and32 Hotpot explorer cases.
Transfer checkpoint selection is fixed before reading transfer outcomes. The
Hotpot result must use official answer metrics, not the native MuSiQue summary.
Independent review reconstructed all64 frozen RL trajectories and found no
blocking defect in the aggregation comparison (58 complete helper traces,
six explicit source-failure zeros). GPU jobs remain serialized under the owner
lock. Quota92% at10:38 UTC; preserve10% shared reserve.

10:56 UTC: Held readout complete: SFT18/64, RL4 19/64; paired+1.56pp interval
−6.25..+10.94. All three wins involve protocol recovery, both losses have valid
finals. No convincing semantic-planning gain. The earlier 10:46 entry described
an interim three-update state; actual RL completion was10:44:29 (four updates,
1,053 calls,1,026.8 seconds). Direct and single-helper controls completed17/64
and14/64 respectively on initial validation32. Baseline and held readouts used
source008/009; actual summaries and terminal receipts establish their cutoffs.

Aggregation source010 is now loading. Transfer selection fixed to RL4 before
examining transfer outcomes. Accepted next CPU preparation:570 helper-SFT steps,
one fixed epoch/36 updates, fresh helper adapter, exact frozen SFT plans reused
for both evaluation arms, all new helpers/finals. This changes the trained role
while keeping the planner/final frozen. Helpers' gold-bound training histories
must not be mistaken for predicted-history training. No helper GPU run yet.
Quota91% at10:52; latest source checkpoint3d44d54 pushed to research branch.

10:58 UTC: Aggregation completed232 fresh finals, no errors. Full-source78/128
vs trace-only74/128; trace-only loses four and gains none. No observed final-repeat
EM disagreements in either condition across58 eligible candidate traces. Candidate
variation occurs in four parents with documents versus three without. This does
not support document removal or an immediate extra final-noise pilot. Keep the
full-source final; proceed with the accepted helper-only training comparison.
Frozen MuSiQue four-hop readout is already running. No transfer results have been
used to choose helper dose, inputs, architecture, or checkpoint (one epoch36updates).

12:14 UTC: Fixed transfer complete; direct answering beats SFT/RL on four-hop
(36vs24/128) and descriptively on Hotpot explorer (36vs23/24of64). These short
inputs fit one call. The new helper-only study completed36updates and192matched
episodes: base19/64, trained28/64, reminder22/64. Both interventions fix all11
base protocol failures, but trained vs reminder also has7both-valid wins/1loss.
Independent native routing audit passed613calls; full-source finals sometimes
rescue wrong chains, so this is not proof of faithful decomposition.

Selected frozen helperSFT36, original prompt without reminder. Accepted source013
full-pass rootRL fromSFT48, up to16fresh updates on256uniqueTRAINparents,3hours,
same RLOO objective/caps/optimizer and per-batch admission. Last committed
checkpoint fixed by stopping rule, never held score. Fresh00364parents reserved
for matched base/SFT/RL-with-same-helper plus direct controls. Actual GPU owner
started12:11:44, first24+calls verified by12:12, two models use17,160MiB and
helper has zero trainable parameters. Supervisor52519 queues independent readouts.
CPU support was focused-tested (31RL/eval tests); no production-wide suite.

Earlier one-epoch planner checkpoint Hotpot readout completed20/64, compared
with23/64 for three epochs, with more protocol failures at one epoch. This does
not support the simple extra-SFT-overtraining explanation; paired dose analysis
pending. KeepSFT48primary. Next CPU preparation targets exact frozen-root helper
transfer, with official Hotpot metrics. Quota85% at12:12; preserve10%. Latest
pushed8098d07 on research branch, notmain; raw runs/modelweights stay external.
