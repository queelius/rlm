# Primer: what we are doing and why

## The idea of a recursive language model

A long input does not have to be read by the model that answers about it. In a Recursive Language
Model the input stays outside the context window as a variable; a *controller* model writes small
programs that peek at it, slice it, send pieces to *helper* model calls, combine the results with
ordinary code, and finally submit an answer. Zhang, Kraska and Khattab introduced the pattern
(arXiv:2512.24601) and showed that strong API models used this way handle inputs far beyond their
context windows. Our question is narrower and, we think, more fundamental for practice: **can a
small open model learn to be a good controller, and can it learn when recursion is not worth it?**

## Our kernel in one paragraph

The runtime (`src/rlm`) is a proxy for the OpenAI Responses API. A request arrives; the
controller sees only the request *metadata* (model name, input size, keys), never the content.
Each turn it writes exactly one Python cell, which runs in a persistent IPython process where the
request is bound as `request`. A small helper ABI is prebound: `model_complete` (send the whole
request to the public model), `ask` and `ask_batch` (send text prompts to helper calls, batched),
and `FINAL_TEXT` (submit the answer). The controller is shown the bootstrap plus only the *latest*
cell and its observation, not the whole history, so every cell has to carry what it needs. Run
limits bound turns, helper calls, tokens and wall time, and every run leaves a trace. Nothing is
retried or substituted: if the controller fails, the run fails.

## A worked trajectory (synthetic store log, 32 records, "who bought at least one X?")

Turn 1, the controller peeks (it cannot know the task family from metadata alone):

```python
text = request["input"]
print(len(text), "chars"); print(text[:500]); print("..."); print(text[-600:])
```

Turn 2, having seen a store log and a question, it splits the log into chunks and sends each chunk
to a helper with a fixed extraction prompt that returns one JSON object per line
(`ask_batch(chunks, system=SYSTEM)`), then parses the JSON into `events`.

Turn 3, it answers from `events` with plain Python, for example the set of people whose events
include a completed purchase of the item, and submits `FINAL_TEXT("ANSWER: Keiko, Olga")`.

That program is what we call the *decomposition routine*. The alternative at every turn is the
*direct route*: `model_complete(request)`, one call with the whole input, which is cheap and
accurate on short inputs and fails on long ones. Choosing between them by input length is what
"selective recursion" means here.

## The research program

1. **Teaching.** Can a 4B model learn the routine from a few hundred trajectories of a scripted
   expert run through the real kernel (native replay)? Yes, with two design facts that matter more
   than data volume: the first turn must be family-agnostic, and taught programs must be short and
   selected by the question.
2. **RL with a verifiable reward.** Does on-policy RL from that warm start improve it, and does a
   token cost term make it cheaper? Accuracy improves (0.567 to 0.63 to 0.65 on held-out
   OOLONG-synth), mostly by fixing helper budgets at lengths never seen in training; the cost term
   at the strengths tried barely matters.
3. **Selective recursion.** Can RL learn *when not to decompose*? Not so far: from a warm start
   that holds both routes, the direct route is extinguished within a few updates under an
   absolute cost, a group-relative cost, and a weak KL penalty. The literature names this routing
   collapse. We traced three contributing mechanisms (the negative gradient from lengths where the
   direct route fails, token-sum loss weighting, and a task-sampling artifact that starved short
   inputs for steps at a time) and are testing remedies on a corrected footing.

## Vocabulary

- **Controller**: the model that writes cells. **Helper**: a model call made from a cell.
- **Native replay**: generating teaching data by running an expert through the real kernel and
  recording the exact chat context and cell at each turn.
- **Warm start**: the SFT adapter RL starts from. Adapters are LoRA weights on
  Qwen3-4B-Instruct-2507, served by vLLM and hot-swapped between updates.
- **Route**: direct (one `model_complete` call) or decompose (chunked helper calls).
- **Routing collapse**: the policy settling on one route everywhere under policy gradient.
- **OOLONG-synth**: a long-context aggregation benchmark (Bertsch et al. 2025) we split by context
  window into teaching and held-out halves; **event-log tasks**: our synthetic family with an
  exact verifier.
- **Claims ledger**: `docs/RESULTS.md`, where each result carries a status and evidence.

## Reproducing the setup

`uv sync --extra dev` builds the runtime environment; `uv run pytest` checks it. Serving uses
vLLM 0.27.1 with LoRA enabled on one GPU; training runs on the other with the environment lock
in `experiments/selective_recursion_20261019/env/`. Each experiment folder has an `EXPERIMENT.md`
and a `manifest.json` with the code commit, base-model revision, data and adapter hashes. Models
and data are on [Hugging Face](hf/RELEASE_LIST.md).
