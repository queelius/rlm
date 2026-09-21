# Helper transfer depends on dataset

Helper adaptation substantially improves this Hotpot panel, while MuSiQue
four-hop gains are mostly partial credit and protocol recovery. Neither panel
tests whether decomposition beats an adapted direct-answer model.

## MuSiQue four-hop

Completed MuSiQue four-hop panel: **all64 parents, two repeats,128 episodes per
arm**. The same saved SFT48 root plans, seeds, helper caps, and base final are
paired across base helpers, helper-SFT36, and base helpers with a JSON reminder.
There are no missing outcomes or native transport failures. This is helper-role
transfer under frozen plans, not planner learning or a decomposition-versus-direct
comparison.

| Helper | Correct /128 | F1 | Helper JSON failures | New calls | New native tokens |
|---|---:|---:|---:|---:|---:|
| Base | 25 | 23.53% | 28 | 421 | 1,262,073 |
| SFT36 | 27 | 32.64% | 0 | 480 | 1,436,800 |
| Base + reminder | 27 | 29.27% | 0 | 480 | 1,442,899 |

All three retain12 invalid-dependency outcomes from the fixed plans. Valid final
counts are88/116/116. New cost excludes historical roots; recovered execution
uses more calls rather than receiving free answers. SFT36 uses13.8% more native
tokens than base. Summed call service times are245.6/375.0/308.5 seconds; these
are not end-to-end wall times or an isolated adapter-latency benchmark.

## Why EM25→27 but F1+9.11 points?

SFT36 versus base has seven exact-answer wins and five losses. Five wins and all
five losses have valid finals in both arms: these cancel in net EM. Two further
wins involve baseline protocol failure. Thus the net two-answer EM improvement
is accounted for by protocol recovery—not a net both-valid exact-answer gain.

The **+9.112pp F1** difference decomposes over all128 paired slots into:

- **+6.548pp** from protocol-involved pairs:71.9% of the net F1 increase. Among
  the28 repaired helper-JSON failures, two become exact and13 gain nonzero partial
  F1 while remaining nonexact;13 still score zero.
- **+2.564pp** from both-valid pairs: +2.017pp among58 pairs where both answers
  remain nonexact, and +0.547pp net from the five exact wins/five exact losses.
  F1 losses need not be minus one when the losing answer retains token overlap.

Against the reminder, SFT36 has **six exact wins and six losses**, all both-valid,
so EM ties at27. Its **+3.372pp F1** advantage consists of +2.096pp among83
both-nonexact pairs (ten F1 increases, five decreases), plus +1.276pp from
asymmetric partial credit across the exact-answer swaps. Equal EM is not equal
answers; higher token overlap is not automatically faithful reasoning.

Exploratory connected-component bootstrap intervals,20,000 draws with seed
2026092113, average paired repeats within parent and resample24 component
clusters. SFT36−base: EM **+1.56pp [−3.12,+6.03]**, F1 **+9.11pp [+2.68,+16.37]**.
SFT36−reminder: F1 **+3.37pp [−0.55,+7.09]**. The latter does not establish a
robust content advantage over the cheap formatting control. These are unadjusted
exploratory comparisons, not128 independent parents.

## Three native examples, illustrative rather than a selected success estimate

1. **Partial recovery with a worse answer than the reminder:**
   `5351b7f215e75267da05e027`, repeat1, asks when an explorer reached a city linked
   through a record label. Base's third helper emits the bare date
   `August 3, 1769`, violating JSON; no final executes. SFT36 completes with
   `August 3`: EM=0/F1=0.8 against the gold date. Reminder completes with the full
   date: EM=1/F1=1.0. The intermediate bindings also differ, so this is not a pure
   formatting-only intervention, despite baseline's protocol failure.

2. **Both-valid exact gain:** `842e2dfff720c9f1bdb16cdb`, repeat0, asks for a
   county capital connected to Main Hall. Base and reminder trace
   `Appleton, Wisconsin → Pulaski, Wisconsin → Pulaski`; their final is wrong.
   SFT36 traces `Wisconsin → Brown County → Green Bay`, and the final exactly
   matches gold. The raw root plan is unchanged; earlier predictions change the
   resolved later questions. This is an observed final-answer gain, not proof
   that every intermediate answer or relation is correct.

3. **Both-valid exact loss:** `57c905ff3ca9ca9f7f74d2ff`, repeat1, asks about
   death-penalty abolition in a country related to the author of *The Book Thief*.
   All arms first return `Markus Zusak` and `Australia`. On the identical resolved
   third question, base/reminder return `1989`, but SFT36 returns `1961`;
   their respective finals retain those dates. Gold is1989, so SFT36 loses
   EM/F1. The saved plan also lacks an explicit question resolving the original
   “country near” relation; matching the final gold does not validate the chain.

These44 selected episode/call receipts were checked against the completed
analysis hashes. The quoted strings are saved model outputs, not independently
verified intermediate facts. Full-source finals can bypass, repair, or follow
helper reports; final scores cannot label intermediate accuracy or causal credit.

## Completed Hotpot transfer

All **32 parents, two repeats,64 planned episodes per arm** are present. The
owner terminated without failure before this inspection. Scoring uses official
Hotpot answer rules, including exact-only F1 for yes/no/noanswer—not MuSiQue F1.

| Helper | Correct /64 | F1 | Helper JSON failures | New calls | New native tokens |
|---|---:|---:|---:|---:|---:|
| Base | 22 | 43.86% | 16 | 178 | 295,891 |
| SFT36 | 40 | 69.11% | 0 | 209 | 339,870 |
| Base + reminder | 35 | 64.17% | 0 | 209 | 342,126 |

SFT36−base has **18 exact wins and no losses**:13 protocol-involved and five
both-valid. EM rises **28.12pp [15.62,42.19]**, F1 **25.25pp [12.05,39.58]**.
Reminder−base has13 wins and no losses, all protocol-involved. Both interventions
remove all16 helper-JSON failures, but SFT36 additionally beats the reminder by
**six wins to one loss, all both-valid**: EM **+7.81pp [0.00,15.62]**, F1
**+4.94pp [−0.87,11.50]**. This is an exploratory content-sensitive signal beyond
the reminder, not merely a larger count of valid outputs; it still does not
identify intermediate correctness or a causal decomposition mechanism.

Intervals use20,000 draws, seed2026092113, paired repeats averaged within parent.
All32 Hotpot parents lack component IDs, so bootstrap clusters are singleton
parents, **not verified atomic-component independence**. There are no missing
outcomes or transport failures. New costs exclude historical roots; summed
service times are92.0/139.4/113.6 seconds. All192 episode grades were independently
recomputed from native finals;788 episode/call hashes match the completed report.

The historical direct score36/64 uses different seeds and has no trained-direct
Hotpot arm. Comparing40/64 against36/64 would not establish decomposition's value.

## Research implication and provenance

Helper protocol robustness transfers to four-hop execution, but a fixed reminder
removes the same28 formatting failures. Role SFT changes content and downstream
bindings substantially; its incremental F1 advantage is promising but uncertain,
and strict EM shows no advantage over that control here. Retain exact outcomes,
partial overlap, and protocol accounting separately. The12 persistent dependency
failures and mixed native examples motivate inspecting plan/interface defects,
not claiming that helper SFT has solved decomposition. Nor do these results show
an advantage over an adapted model reading the original question directly.
Hotpot provides stronger evidence of helper-role transfer beyond the reminder
than MuSiQue four-hop does. The smallest discriminating control is paired base
versus helper-adapted direct answering, with identical observations and seeds;
generic reading/answer transfer remains an alternative to decomposition-specific
improvement. Do not pool the two panels into one mechanism claim.

Root and helper checkpoints are respectively `planner-sft-001/checkpoint-0048`
and `helper-sft-001/checkpoint-0036`. All paths below are under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- Complete result: `analysis-helper-transfer-musique-001.json` and `.md`.
  JSON SHA256: `4557bf3f15c7f3242127ee454177a6eb8714bd3cfe19bc1ced8f43ce977fdf26`.
- Native source: `helper-transfer-musique-001/episodes/` and `calls/`.
  PLAN SHA256: `f890cd66395e0c5a02b615819d65fba5d8a11c80b557531d82c7c25d4e87d1b0`.
  Example filenames start with the complete case ID, `-r<repeat>-<arm>`;
  all individual receipt hashes are in the result's `input_receipt_analysis_sha256`.
- Derived F1 contributions sum `trained.f1 − comparator.f1` over the indicated
  paired outcome category, then divide by128. No denominator or root was dropped.
- Hotpot complete result: `analysis-helper-transfer-hotpot-001.json` and `.md`.
  JSON SHA256: `fcc96638c274aa61651d6de2db846e25a1ba288825b0954e8ed1ebee8d2e92c8`.
  Native source: `helper-transfer-hotpot-001/episodes/` and `calls/`;
  PLAN SHA256: `40027e1333bd037fbbfa4986f7f2b4bab0192e447bab25833b55b49f6898e67d`.
  Terminal receipt: `helper-transfer-hotpot-001/TERMINAL-cdf40b48b061.json`.

Operationally, MuSiQue's1,381 calls sum to929 seconds of model service versus
approximately1,530 seconds wall time. Repeated full-receipt progress summaries
are a plausible overhead source, not a profiled attribution. Future collectors
should accumulate lightweight progress and summarize fully periodically or at
termination; no live/sealed source was changed. No GPU job was launched.
