# Fresh Hotpot architecture readout

On the frozen 128-parent, two-repeat Hotpot panel, direct scored 152/256
(official EM .59375, F1 .69357) and the SFT-planner plus trained-helper execution
scored 151/256 (EM .58984, F1 .71053). The paired planner-minus-direct estimate
is -0.39 EM points (95% bootstrap [-6.64, +5.86]) and +1.70 F1 points
([-4.43, +7.91]); neither resolves a benefit.

All 256 planned attempts per arm remain in denominators. Direct has two invalid
finals; planner has two invalid dependencies/no final calls. Among outcome
changes, planner has 21 both-valid wins and 20 both-valid losses; two additional
planner losses involve protocol. This is not evidence that plans caused either
final outcome because planner finals retain full source access.

Direct used 256 calls and 436,983 tokens (1,706.96/attempt). Planner execution
used 1,071 calls and 1,501,958 tokens (5,867.02/attempt), including 256 roots,
561 helpers, and 254 finals. Summed service latency is not deployment wall time.
The uncertain F1 advantage comes at 3.44× the tokens and 4.18× the calls.
Without a resolved benefit, direct remains the practical baseline on this panel.

This is a fresh but still finite Hotpot development panel, not a generic transfer
or decomposition claim. Official cached Hotpot EM/F1 is authoritative; native
MuSiQue metrics in raw receipts are not comparable.

Sources: `analysis-hotpot-fresh-architecture-001.json`,
`analysis-hotpot-fresh-planner-001/REPORT.json`, and
`analysis-hotpot-fresh-direct-001/REPORT.json` under the external
September-21 selective-delegation sidecar.
