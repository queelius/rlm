---
title: What did the model learn from during SFT?
meeting_date: 2026-09-11
related_slides: [3, 4]
status: explanatory_example_with_recorded_source
---

# What did the model learn from during SFT?

Supervised fine-tuning (SFT) changes a model's learned parameters so that, when it sees a
situation, it becomes more likely to produce the demonstrated next action. Here we taught the
main model a routine: ask a helper to classify questions, calculate from the actual replies in
Python, and return the calculated answer. We did not train the helper in these runs.

## The short example on slide 3

**This is an illustration, not a literal training record.** We shortened the data and code to
make one training step understandable. Suppose the task is “Count questions asking for a
location,” and the helper has already classified three questions:

```python
types = ["location", "human being", "location"]
```

The model is shown that situation and trained to produce an action like this:

```python
count = 0
for kind in types:
    if kind == "location":
        count += 1
print(count)
```

Python prints **2**. The model is learning to generate the calculation, not merely to guess
the number two. In a full worked interaction, it also sees how to request the helper's
classifications and how to return the result after Python runs.

“Location” describes the expected answer type. “Where is Oslo?” asks for a location;
“Who wrote Hamlet?” asks for a person. The helper classifies these questions rather than
answering them. A wrong helper classification can still lead to a wrong final answer even
when the Python calculation is correct.

## How that connects to an actual training record

One recorded training task said:

> Consider only records owned by u0, u1. Count records whose category is 'location'.
> Return only Answer: N, replacing N with the exact nonnegative integer.

There were 16 records. The main model's demonstrated first action loaded them from a file,
asked the helper for their question types, and stored the returned types by record name.
The following is its actual next demonstrated calculation, with line breaks added for readability:

```python
matches = [
    record for record in items
    if record["user"] in ['u0', 'u1']
    and live_map[record["id"]] == 'location'
]
scalar = len(matches)
print(scalar)
```

In plain language: keep a record only if it belongs to one of the two requested users and the
helper called it a location question. Then count the selected records. Python printed **0**
in this recorded interaction, and the demonstrated final response was `Answer: 0`.

The slide removes the user restriction and record-name lookup, uses three example replies
instead of sixteen, and expands the selection into a simple loop. Its answer of two is
illustrative; it is not the answer in this actual record.

## Questions you might be asked

**Were these just question–answer pairs?** No. Each complete example showed three main-model
actions: obtain classifications, calculate from them, and return the result. Training rewarded
imitating those demonstrated main-model actions. The task text and helper replies supplied
context; they were not themselves the outputs the main model was trained to imitate.

**Where did the text come from?** The tasks used public TREC question texts. Users, record names,
and numeric weights were added for the calculation tasks. The real classification task has six
broad answer types. Our miniature examples show only the types needed to explain the routine.

**Who supplied the demonstrations?** The main-model actions were authored demonstrations.
They included real helper replies, even when the helper made a mistake. This was not evidence
that the main model discovered the routine unaided.

**How much training was there?** Each separately trained copy learned from 72 complete worked
interactions, containing 216 demonstrated main-model actions. Training changed a small adapter
attached to the main model. Both copies began with the same earlier-trained model and used
different example sets; one copy was not a continuation of the other.

**What did the later test establish?** On 72 separate test tasks, both trained copies had many
more confirmed correct answers with the requested calculations than the starting model.
The input sets were newly selected, but the task types were familiar. This supports learning
a useful routine, not a claim of general autonomous planning.

## Recorded source

This source is in the external research store, not bundled with the slide repository:

```text
/project/alex_phd/runs/rlm-research-r4/sidecars/root-question-sensitive-sft-v1/outputs/attempt-001/capture/7174335b73e979e63bec152f6098d26ab2d22573bee355daf2d470da1fcfbfbc/
```

- Task: `question-sensitive-sft-train-00:P1`.
- `TEACHER.json` SHA-256: `4b52bd0a8b3855fa3b02e53b59eb230cc6691db245fbb805004dbfadcb212827`.
- `EPISODE.json` SHA-256: `29670200adf65abbd88f235a1ee174f34b67cb1c90e0e2e15ad9cca9c3748d46`.
- Training report: `/project/alex_phd/runs/rlm-research-r4/analyses/root-question-sensitive-sft-live-2026-09-10/REPORT_MAIN.md`.

The slide-4 results come from the later shared evaluation described under S3 in
[evidence and methods](evidence-and-methods.md), not from this one training example.
