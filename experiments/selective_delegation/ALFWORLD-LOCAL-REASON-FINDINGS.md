# Local deliberation matches the manager on the exposed screen

The completed source032 control wins 6/16 episodes, matching source026 manager/worker's 6/16 and exceeding flat's 1/16. Local reasoning versus manager has one paired win and one loss; its game-bootstrap difference is 0, with 95% interval −18.75 to +18.75 percentage points. These results do not establish equivalence, but the observed manager advantage over bare flat is not unique to a persistent goal-manager architecture.

The comparison retains eight exposed seen-development games and both fixed seeds. All 553 new native calls returned, all 16 new episodes are observed, and no calls are unresolved or unlinked. The independent sealed audit reconstructed public prompts, strict decoding, actions, feedback, native won flags, seeds and charged token budgets. No model calls were made during analysis. Authoritative immutable reports are `analysis-alfworld-local-reason-001.json` and `.md` in the research store.

| Policy | Wins / 16 | Native calls | Total tokens | Native call seconds | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| Flat | 1 | 781 | 1,373,215 | 340.0 | 0 |
| Manager/worker | 6 | 697 | 1,252,404 | 356.5 | 0 |
| Local reason | 6 | 553 | 996,652 | 990.1 | 30 |

Local reasoning is not cheaper in measured service time: approximately 2.8 times the manager's native generation time despite fewer calls and fewer total tokens. It emits 23,992 completion tokens versus the manager's 7,434; long serial generation matters. Local episodes ended in six successes, six token-budget stops, three action-budget stops, and one three-invalid stop. Twenty-four invalid replies exceeded the 240-character reason contract; six were malformed JSON. Six calls hit their available output-token cap. There was no history trimming in any arm; these differences are not explained by context truncation.

Two grounded discordant examples illustrate both directions, not faithful internal reasoning:

- Game 0, “put some egg on microwave”: manager succeeds in both seeds (11 actions each), local succeeds only in seed 2026092179 (23 actions), and flat fails both. In local seed 2026092178, the agent searches several locations, takes an irrelevant spoon, then loops taking and replacing bread until its token budget expires after 46 actions. The matching manager finds the egg in the garbagecan and transfers it to the microwave. This is a real local-policy failure despite its aggregate tie.
- Game 7, “clean some cloth and put it in drawer”: local succeeds only in seed 2026092179 (21 actions); flat and manager fail both seeds. In the successful local trace, it discovers cloth in drawer 1, takes it, cleans it once at sinkbasin 1, and returns it to the drawer. The matching manager also discovers and cleans the cloth, but repeatedly cleans it through the 50-action limit instead of completing placement. This demonstrates a failure to progress after an accomplished subtask, not evidence that manager calls are inherently harmful. The other local seed also fails, at its token limit.

Small-sample sensitivity matters. Local-minus-flat game means are `[0.5,1,0,0.5,0,0,0,0.5]`; the bootstrap interval is +6.25 to +56.25 points, but exhaustive two-sided game sign-flipping gives p=0.125. Manager-minus-flat remains p=0.25, and local-minus-manager gives p=1.0. These are post-hoc sensitivity calculations over all 256 sign vectors under an exchangeable-sign null, not prespecified confirmatory tests. Episodes are not 16 independent games, and shared scenes further limit inference.

Decision: retain all three unchanged policies for the already-frozen unseen panel, rather than claiming hierarchy-specific improvement from the original two-arm screen. That panel tests new games and all six task families, but comprises only four scenes. The local-reason intervention jointly changes prompt/schema, generated deliberation and token allocation; equal success here does not isolate which component caused the change. This is exploratory scaffolding, not new hierarchical-learning methodology or evidence for learned recursive decomposition.
