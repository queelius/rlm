---
license: mit
task_categories: [text-generation]
tags: [recursive-language-model, rlm, tool-use, distillation]
---

# rlm-onecell-library-calls (draft, not yet published)

436 (messages, completion) pairs: the controller turns of correct runs of the untrained
Qwen3-4B-Instruct-2507 controller calling the prebound routine library of
[queelius/rlm](https://github.com/queelius/rlm), on the teaching splits of two families (220
synthetic store-log tasks, 216 OOLONG-synth teaching windows). Each pair is one controller call:
the system prompt (harness policy and library contract), the metadata-only bootstrap envelope, and
the single Python cell the controller wrote.

No task input text is included: the controller never sees the input, so neither the store logs
nor OOLONG-synth content (which declares no license) appears in the data. Fields: `task_id`,
`turn`, `correct`, `messages`, `completion`, `source`.

Used to train `queelius/rlm-qwen3-0.6b-onecell-caller` (draft); subsets of 55 to 220 pairs and an
event-log-only subset are described in the repository's results ledger (S4).
