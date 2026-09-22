---
status: completed_exploratory_serving_qualification
evidence_cutoff_utc: 2026-09-22T14:58:00Z
question: Can four concurrent requests reduce the cost of collecting future rollouts?
claim_scope: fixed_request_throughput_not_task_success
---

# Four concurrent requests were 2.75 times faster in a small serving test

On the same 32 saved TextCraft requests, serial serving took 38.59 seconds and
four-at-a-time serving took 14.02 seconds. Both conditions returned 32 valid
actions, with no native protocol failures and 931 total output tokens each.
The full run, including loading, compilation, five separate warmup requests and
shutdown, took 224.43 seconds. The first real reply arrived in 3.96 seconds.

This meets the predeclared qualification rule: at least 1.5 times the throughput
without fewer valid actions. It supports trying batched collection in a future
pilot. It does **not** measure completed-task success, establish a speedup for
whole interactive episodes, or demonstrate numerical equivalence with training.

The requests were selected by prompt length from historical saved requests,
without selecting successful outcomes. Both measured conditions used the same
requests and sampling settings, with prefix caching disabled. Serial ran first;
this is one fixed-order timing trial, not a randomized repeated benchmark.
The vLLM serving path uses BF16 LoRA arithmetic, whereas the existing training
and reference collection retain FP32 LoRA parameters. Only 13/32 responses in
each condition exactly matched the historical reference token sequence.
Matching validity or token totals must not be confused with matching policies.

## Reproducibility

Research store:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Output: `textcraft-batch-qualification-001`; source:
`source-textcraft-batch-qualification-002`; fixed inputs:
`textcraft-batch-qualification-inputs-003` (see the accepted receipt for exact paths).
The run was capped at fifteen minutes and 69 calls; it made exactly 69 calls.

`RESULT.json` SHA256:
`e062153534e9f6cb6c059f80573e5ce98f01d52f6aa77671d5445a9a945d850f`.
`TERMINAL.json` SHA256:
`761346a62348f383b5427e06b69bcf07616373daf7a768f492df49954454822e`.
The terminal records completed work and release. Main additionally checked that
the owner/server processes were gone and all three serving ports were free
before launching the next numerical probe.
