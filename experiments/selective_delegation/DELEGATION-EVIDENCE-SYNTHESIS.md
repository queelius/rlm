# Delegation evidence synthesis

The strongest current signal is conditional and TRAIN-local: with frozen historical
plans, executed helper reports score 198/320 versus 148/320 for a matched direct
control and 99/320 for plan-only. Parent/component-bootstrap contrasts are
+15.63 pp helper minus direct (95% CI +1.76 to +32.67) and +15.31 pp direct
minus plan-only (+1.25 to +34.00). The helper effect is therefore not solely a
recovery from plan-only damage, and all those paired contrasts were both-valid
JSON. It is nevertheless an observational reuse analysis over 16 TRAIN parents,
five execution settings, and four correlated candidates—not an adaptive-policy,
cost-matched, or held-out result.

The broader generalization evidence remains weak. On the fresh-question MuSiQue
panel, base direct was 54/128, planner SFT 53/128, and planner RL 56/128; the
RL-minus-SFT component interval included zero. Planner policies also used roughly
three to four times the direct tokens. Fresh Hotpot was likewise essentially tied
(planner 151/256, direct 152/256), though the F1 ordering differed. These results
do not support a general claim that question planning or helper execution improves
short full-context QA. They leave open a narrower claim: correct intermediate
evidence can materially help a fixed plan on in-distribution instances, while root
planning and/or execution does not yet transfer reliably. Fresh questions are
not necessarily fresh documents: 14 of the 64 MuSiQue parents share exact source
documents with the earlier training inventory.

The queued paired answerability SFT is a useful diagnostic rather than evidence
for delegation. Its joint arm learns the official answerability label and answer
from matched MuSiQue variants; its positive-only arm repeats the supported prompt
and target. Any advantage of joint over positive-only on the already-examined
32-parent DEV panel would support learning the dataset's answer/sufficiency
contract, provided supported-answer competence is retained. It would not establish
semantic grounding, general abstention, or a recursive interface result.

The single most belief-changing follow-up, conditional on a suggestive joint SFT
readout, is a fresh label-blind 32-parent paired MuSiQue panel: select complete
official DEV pairs by fixed seed after excluding IDs, normalized questions, and
atomic components from every current TRAIN, development, transfer, breadth, and
sufficiency panel. Run base, joint-SFT, and positive-only-SFT with the identical
full-context prompt and two fixed seeds (384 new calls:32 parents times two
variants times two seeds times three arms). Predeclare group
answer+sufficiency as primary; separately report supported-answer EM/F1, false
abstentions, negative overanswers, malformed JSON, and physical cost. A joint
gain over both controls with no supported-answer degradation would justify a
larger grounding/control study. A null result, or a gain explained by false
abstention, would retire this SFT direction before adding planning or RL machinery.
