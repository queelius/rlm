# Cache-off serving qualification — CPU prepared, GPU not accepted

Question: can concurrency4 improve throughput by at least1.5× over concurrency1
for the **same frozen32 requests**, with native IDs/adapter/request contracts
preserved? This is a 15-minute engineering qualification, not task-success
evidence, a backend migration, or permission to interrupt062/063/fresh science.

## Fixed inputs and measurement

Use completed057 original-prompt calls only, fixed public056 checkpoint23.
Sort370 candidate calls by `(input_token_count, call_id)`, split at median rank,
then select16 per half by SHA256(`2026092241:call_id`). No response correctness,
action type, success or output length enters selection. Frozen ranges are
575–1643 and1853–5097 input tokens, covering7 task parents; all response caps256.
Historical reference32 calls used73,329 input/884 output tokens and70.84 summed
HF service seconds. These are acquisition costs, not new collection costs or a
fresh HF wall-throughput measurement.

One shared kernel warmup runs before either timed phase: one request then four
concurrent requests, from five hash-selected saved calls **excluded from32**.
Then measure32 serial requests followed by the identical32 at concurrency4.
**Prefix caching is disabled throughout**, both in the CPU-validated config and
an obligatory effective engine startup-log check. No second-arm free prompt
cache. Fixed phase order can retain residual system/kernel effects; this is a
small qualification, not a definitive throughput study. Total69 calls including
warmup, per-request timeout≤90s, whole owner≤900s including startup/release with
60s cleanup reserve and allocation-end margin600s. No retries; first transport
or native-contract failure cancels remaining work. Invalid JSON is separately
counted, never repaired or confused with transport failure.

Report each phase's wall seconds, requests/s, output tokens/s, prompt/output
counts, parse failures and exact HF-output-ID agreement/disagreement. Generated
lengths can differ, so report both request and token throughput. Admission to a
**later** fresh matched-backend pilot requires all64 measured native contracts
valid, ≥1.5× request throughput gain, and concurrency4 valid-action count at
least concurrency1 and nonzero. This is not automatic GPU acceptance or a
claim about end-to-end crafting success.

## Native contract and deliberately limited reuse

The successful old Prime/vLLM path provides reviewed launch/environment,
readiness, LoRA-loading and owned process-group stop helpers. Its allocation5801
wrapper is not reused: current allocation is5879, so its guard would reject it.
New qualification source uses those helpers under the current exclusive GPU
lock, current qualified driver580.126.20, separate experiment-owned caches,
loopback-only serving and immutable owner/start/call/terminal receipts. No
installation or unrelated environment/cache mutation.

Use installed **OpenAI CompletionRequest**, not an assertion that the historical
`/inference/v1/generate` route is interchangeable. `/v1/completions` receives the
exact saved token-list prompt, bypassing server chat rendering. Every response
must return `choices[0].prompt_token_ids` exactly equal to frozen native inputs
and `choices[0].token_ids` containing actual generated IDs; missing fields fail.
Outputs are decoded from those IDs, never reconstructed by re-tokenizing text.
Frozen tokenizer/template hashes and CPU re-encoding bind all selected prompts.
Actual `/v1/models` alias/root/parent must bind the selected adapter/base paths.

Sampling: T=.5,top-p1,top-k−1(vLLM's disabled filter, HF0),min-p0, no penalties,
native saved seed and cap, EOS151645, no prompt truncation or structured repair.
Serving base/LoRA explicitly BF16; stored LoRA is FP32, unlike HF's retained FP32
adapter. Engine seed0, max4 sequences, context8192, existing batch-invariant
kernel flag enabled. Numeric kernels, adapter casting, RNG implementation and
backend sampling remain differences even with exact input IDs/seeds. No
bitwise-equivalence promise; future comparisons must recollect both policy arms
under the same accepted backend. Cache-off results do not qualify cache-on use.

## Readiness and provenance

`R` is the selective-delegation-20260921 sidecar store.

- Source: `R/source-textcraft-batch-qualification-002`; runner SHA
  `3314384a4673fcfabb2ee1d159e6e483b76776327c1e36768f7409caaf7910cf`.
- Inputs: `R/textcraft-batch-qualification-inputs-003`; PLAN SHA
  `4b968a629e795eae7e3e323178960374de667d7a3a0f22889f269f0e165ee1d6`.
- Frozen measured REQUESTS SHA
  `ee79296eb345bfc953decfdc0c38bca6ec86f183d5bef399a27d40448d814cea`.
- Three sealed focused tests passed6.69s: outcome-independent length selection,
  strict missing/mismatched native-ID rejection, and actual057 input/tokenizer
  freeze with disjoint warmup. Ruff passes.
- Installed Prime InferenceConfig and vLLM CompletionRequest/ResponseChoice were
  inspected and CPU-validated: vLLM0.28.0/Torch2.13.0+cu130/Transformers5.6.2;
  exact schema fields exist. CPU import emits upstream Torch deprecation/no-CUDA
  warnings; **no native vLLM response has yet been observed**. That is the GPU
  qualification's first gate, not something a schema fixture proves.
- Preserve inputs001 (CPU failure: assumed standalone chat_template.jinja;
  this tokenizer embeds the template), inputs002 (superseded shared warmup
  profile), and source001. Measured32 REQUESTS bytes are identical in all three
  attempts; no reselection after results. Corrected source hashes the actual
  loaded template, and source002 uses five disjoint warmup calls.

Future **main-owned** invocation only, using the existing training Python:

```sh
python "$R/source-textcraft-batch-qualification-002/textcraft_batch_qualification.py" \
  --prepared "$R/textcraft-batch-qualification-inputs-003" \
  --output "$R/textcraft-batch-qualification-001"
```

No runtime output/owner is created by this preparation. No new GPU job accepted.
