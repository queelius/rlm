# Paired reward-control readout

The predeclared primary comparison disfavors the additive reward. On the fixed
32-parent paired MuSiQue panel (positive and missing-evidence variants, two
seeds), product-RL reached 10/64 joint-EM pairs (15.6%) and additive-RL 4/64
(6.25%): **−9.38 percentage points**, 29-component-cluster bootstrap 95% CI
[−18.97, −1.72]. This is a held readout of frozen terminal checkpoints, not a
claim that either objective learned general answerability.

## What was compared

All four adapters used the same full public documents, prompt/decoder contract,
64 paired attempts, and seeds. All 512 planned variants returned valid results;
there were no missing or protocol-invalid calls.

| Frozen endpoint | Joint EM | Positive EM | Both labels correct | Positive abstention | Negative over-answer |
|---|---:|---:|---:|---:|---:|
| Warm joint SFT32 | 6/64 (9.38%) | 8/64 | 12/64 | 48/64 | 5/64 |
| Product RL8 | 10/64 (15.6%) | 13/64 | 17/64 | 40/64 | 8/64 |
| Matched extra SFT8 | 10/64 (15.6%) | 15/64 | 20/64 | 34/64 | 11/64 |
| Additive RL8 | 4/64 (6.25%) | 6/64 | 10/64 | 52/64 | 2/64 |

The product-to-additive effect is an abstention tradeoff rather than a cost
change: additive produced 12 more positive abstentions and six fewer negative
over-answers. Its positive EM was −10.94pp [−20.69, −3.13] and negative
over-answer −9.38pp [−16.13, −3.23], both additive-minus-product paired
component-bootstrap intervals. So additive became more conservative, but in
this panel that conservatism lost answerable-pair reward.

Product RL versus warm joint SFT gained +6.25pp joint EM [0.00, +13.79] and
reduced positive abstention by 12.5pp [−20.97, −5.41], but increased negative
over-answer by 4.69pp [0.00, +10.34]. Matched extra SFT has the same +6.25pp
joint-EM point estimate versus warm and has 0.00pp [−6.25, +6.45] joint-EM
difference from product RL. It has higher positive EM (+3.13pp versus product)
and more negative over-answer (+4.69pp). Thus this readout does not identify a
unique product-RL benefit over the eight-update additional SFT control. Updates
are matched; training tokens and information are not identical.

Nominal-depth strata are descriptive only: the product/additive joint-EM gap is
−18.75pp on 16 two-hop parents (32 paired seeds), 0 on eight three-hop, and 0
on eight four-hop parents. The last two strata have near-zero success across
arms, so this is not evidence that the objective effect is two-hop-specific.

## Cost and limits

Each arm used 128 new native calls and about 362.7k known tokens; all four used
512 calls, 1,451,086 tokens, and 540.9 summed service seconds. Endpoint
training, source documents, and the warm checkpoint were reused, so these are
readout costs rather than full training-policy costs. Intervals resample 29
frozen atomic-component clusters (20,000 draws); 32 parents and their two
seeds are not independent observations. Six exploratory contrasts have no
multiplicity correction. The panel is a fixed held panel, but shared MuSiQue
documents/components and earlier training exposure limit any clean-unseen or
causal-generalization claim.

Provenance: authoritative report
`R/analysis-sufficiency-reward-control-001.json` SHA256
`85b12db8c0d7916b938895835bc863e26aaafc41c59ba18a435e7f4ba98a20f0`,
with sealed cases SHA256 `256e9a38…2f97d7c` and readout profile
`paired-sufficiency-reward-control32-terminal-v1`.

## Ranked next question

1. **Do not expand additive diagonal RL as specified.** Its predeclared primary
   held comparison is negative; a same-panel retry would add little information.
2. **Defer the product pairing-mean/Rao--Blackwell control.** The committed
   CPU audit found 42 nonzero diagonal-credit groups versus 52 raw-gradient
   pairing-mean groups, including ten restored index-mismatched groups. That
   establishes a pre-clipping variance-reduction rationale, not an outcome
   benefit after Adam. Because three joint panels have not identified a product
   RL advantage over matched SFT, another eight-block estimator run would test
   optimization mechanics without a decision-relevant capability signal.
3. **Retire this reward-optimization branch unless independent evidence revives
   it.** Only a future predeclared product-over-SFT capability signal would make
   the same-objective pairing-mean control informative; then keep the rollout
   budget, product reward, and held readout fixed. No further CPU calculation
   can turn the existing raw-gradient variance result into expected improvement.
