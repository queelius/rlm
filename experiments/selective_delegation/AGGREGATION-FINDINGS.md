# Full-source versus trace-only final aggregation

The completed frozen-trace probe does not support removing original documents
from the final answerer. This is an interface diagnostic on one RL batch, not a
planner-learning or generalization result.

## Result

Each condition has a planned denominator of 128: 64 frozen RL candidates times
two new final seeds. Six source invalid-helper candidates become 12 explicit
zero rows per condition; the 58 eligible candidates produce 116 returned final
calls per condition. There were no new transport failures, malformed finals, or
final-repeat disagreements.

| Condition | EM, all planned | F1, all planned | Returned eligible finals |
|---|---:|---:|---:|
| Full source | 78/128 (60.9%) | 79/128 (61.7%) | 116 |
| Trace only | 74/128 (57.8%) | 75/128 (58.6%) | 116 |

Among the 116 returned eligible pairs, trace-only had 0 wins and 4 losses.
Those four losses are two candidates repeated under both final seeds, from two
of the 15 parents with at least one eligible trace. The primary 20,000-draw
descriptive parent-cluster bootstrap includes all 16 planned parents and all
128 paired rows (including the six invalid-helper candidates as explicit
zeros): full-source minus trace-only EM is 3.1 percentage points, with a 0.0
to 7.8 percentage-point percentile interval. One `random.Random(20260921)` stream
generates the draws. Each draw samples 16 parent IDs with replacement, retains all
eight planned paired rows for every sampled parent, and computes their pooled
EM difference. It is not an independent-parent confidence interval.

For the separate returned-eligible conditional analysis, the same 20,000-draw
procedure samples the 15 parents with at least one eligible trace and retains
only their eligible paired rows (116 total); its interval is 0.0 to 8.6
percentage points. That conditional result does not replace the primary
planned-denominator analysis.

The public Abel Kirui country/emergency question lost its date under trace-only,
which instead returned the label “State of Emergency,” on both seeds. The public
Imperial Valley river-source question lost its source under trace-only, which
instead returned “Colorado River,” also on both seeds. These are examples of
original documents helping the final select a more specific answer; they do not
prove a universal causal mechanism.

## Cost and provenance

The probe made 232 new physical final calls: 362,240 prompt and 2,174 completion
tokens, with no unknown usage or unresolved starts. It reused each frozen
batch-one root plan and actual isolated helper trace; the source original final
was never the new full-source baseline. Consequently, 232 calls is new
acquisition cost, not end-to-end policy cost. Hypothetical deployment cost adds
the reused root/helpers once per policy trial and must not be summed across
conditions as physical work.

The zero final-repeat disagreement means a separate final-noise follow-up is not
currently decision-relevant. Keep the full-source final contract for the next
fixed readouts. This does not establish that plans receive faithful credit: the
candidate variation remains conditional on frozen helper traces, repeated
exposed TRAIN parents, and this one model/protocol.
