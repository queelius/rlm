# Idea registry: how it is run

Two Claude Code sessions share this directory, with one owner per file so they never edit the same file:

| File | Owner | Purpose |
| --- | --- | --- |
| `inbox.md` | the experiment session (append only) | raw ideas as they come up: origin, one-line question, a pointer to the evidence |
| `registry.md` | the registry session | curated entries with novelty verdicts, follow-ups and smallest tests |

## The registry session's loop

1. Read `inbox.md`; move each new item into `registry.md` as an `unchecked` entry and mark it `moved to I-0NN`
   in the inbox (the one edit the registry session makes to the inbox).
2. Check novelty for `unchecked` entries, batched (subagents are fine): arXiv, Semantic Scholar, OpenReview,
   ACL Anthology, and older classical work where relevant. Record the verdict (`checked: open`,
   `checked: partial`, `checked: prior art`), the 2 to 4 closest works with IDs, and the queries and date.
3. Where prior art exists, look for follow-ups that are still open and add them as new entries.
4. For `open` or `partial` entries with a cheap smallest test, write the test out (question, arms, metric,
   expected GPU shape and duration, what would promote or retire the idea) and add the entry to the backlog
   table in `../research-plans/2026-10-22-research-program-and-idea-backlog.md` with the note "proposed by
   the registry session". The experiment session decides when to run it; the registry session does not
   launch GPU jobs.
5. Commit only files under `docs/ideas/` (and the backlog row), pull with rebase before pushing, and use
   ordinary pushes.

## Ground rules

- No GPU jobs, no edits to experiment code, run directories, the job queue or the servers.
- Ideas outside RLMs are welcome; mark them `RLM? no`.
- "No prior art found" is a statement about a search, so it always carries its queries and date.
