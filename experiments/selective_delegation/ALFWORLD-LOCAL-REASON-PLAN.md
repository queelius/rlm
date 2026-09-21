# ALFWorld: one-call local deliberation control

Question: can brief per-action deliberation explain the manager/worker advantage
observed on source026's eight exposed seen-development games? This adds one arm
on exactly the same 16 game/seed slots, not a new confirmation panel.

The frozen base model receives the same public initial observation, current
feedback, executed-action/feedback history, and indexed native admissible commands.
It emits one strict JSON object, in this order:
`{"reason":"a short local reason","action_index":0}`.
The reason must be nonempty and at most 240 characters. The host executes the
indexed command exactly. The accepted reason is saved for audit but does not enter
future prompts. Rejected raw output and the parser error enter public history,
as in source026. There is no manager, persistent goal, fallback, or hidden state.

Both fields count toward 128 generated tokens per request and 2,048 per episode.
Other limits remain 50 actions, three consecutive invalid responses, 8,192-token
context, temperature 0.5, and the original per-action seed progression from
2026092178/2026092179. The source026 neutral trimming rule and goal-token reserve
are retained exactly. If the new instruction cannot fit with the retained history,
the collector records an unobserved context error rather than trimming differently.
All native requests, decoded outputs, decisions, observations, usage, and terminal
states are saved. Missing/inference failures are unobserved, not scientific losses.

Implementation uses a new collector and the unchanged source026 native model
client and environment bridge. The source and prepared input plan are sealed under
source032; the decision remains proposed until main acceptance. A bounded launcher
waits for authenticated release of `sufficiency-readout-positive_only-001`.
The maximum runtime is one hour. No GPU launch is part of CPU preparation.

Focused checks exercise the ordered parser, public projection, native receipt path,
reason exclusion from subsequent history, rejection feedback, and generated-token
accounting. The later comparison uses all 16 slots against completed source026
flat and manager/worker outcomes, with protocol and budget effects separated.
Reasoning-token allocation and response-format difficulty change together; this
does not isolate an internal reasoning mechanism or claim a novel hierarchy.
