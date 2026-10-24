"""Meaning-preserving shifts of generated families, for stress-testing frozen libraries (E59).

Conditions, all on split "stress" (never used for discovery, teaching or fresh scoring):
- control: the family as generated, 400 records.
- paraphrase: the same tasks (identical records and answers); only the question is reworded.
- nuisance: the same tasks; field order, separator and question label are redrawn, as if from another seed of the
  same family. The date style and answer format are kept.
- long: the family as generated with 2000 records (new tasks, 5 times E39's fresh length).
- paraphrase2, nuisance2 (E60): a second rewording and a second redraw, never used in any teaching set.

make_teach (E60) re-renders E39's own 30 teaching tasks (split "teach", 200 records, same order, so the same
records, slots and answers): 10 in the original wording, 10 with the first paraphrase, 10 with the first nuisance
redraw. The positions bootstrap seed 0 shows (27, 12, 24) get one of each, so the three shown examples are E39's
shown tasks, one per wording.

The task RNG is seeded by family, split, length and index, so control, paraphrase and nuisance tasks share records
and answers exactly; only the rendering or the question text differs.
"""

from __future__ import annotations

import random
import re

import family_gen as F

CONDITIONS = ("control", "paraphrase", "nuisance", "long", "paraphrase2", "nuisance2")

# one rewording per question form: (pattern on the generator's question, replacement); slots are kept verbatim
PARAPHRASE = {
    "count": (r"How many records have action (\w+) and item (\w+)\?",
              r"Count the entries whose item is \2 and whose action is \1."),
    "total": (r"How many units of (\w+) did (\w+) buy in total \(action bought only\)\?",
              r"What total quantity of \1 did \2 purchase? Only entries with action bought count."),
    "top_user": (r"Which user has the most records with action (\w+)\?",
                 r"Among entries whose action is \1, which user appears most often?"),
    "nth_item": (r"What was the item in (\w+)'s (\w+) record \(counting from the start of the log\)\?",
                 r"Taking only \1's entries in log order, which item is in the \2 one?"),
    "last_city": (r"In which city was (\w+)'s most recent record\?",
                  r"Which city is on the latest entry for \1?"),
    "first_after": (r"Which user made the first record at or after (.+)\?",
                    r"Who is the user on the earliest entry made no earlier than \1?"),
    "distinct": (r"How many different items did (\w+) have records with action (\w+) for\?",
                 r"For \1, how many distinct items appear among entries with action \2?"),
    "day_span": (r"How many whole days are there between the dates of (\w+)'s first and last records\?",
                 r"Counting whole days, how far apart are the dates of \1's earliest and latest entries?"),
    "semantic_total": (r"How many units of (\w+) items were bought in total \(action bought only\)\?",
                       r"Summing quantities over entries with action bought, how many units of \1 items were purchased?"),
}

# E60: a second rewording, test-only (written 2026-10-24 09:10 UTC, before any E60 run)
PARAPHRASE2 = {
    "count": (r"How many records have action (\w+) and item (\w+)\?",
              r"Give the number of events in which \2 occurs with the action \1."),
    "total": (r"How many units of (\w+) did (\w+) buy in total \(action bought only\)\?",
              r"Add up the quantities of \1 that \2 bought; what is the sum? Ignore other actions."),
    "top_user": (r"Which user has the most records with action (\w+)\?",
                 r"Name the person with the largest number of \1 events."),
    "nth_item": (r"What was the item in (\w+)'s (\w+) record \(counting from the start of the log\)\?",
                 r"Reading the log from the top and considering \1 alone, what item does their \2 event involve?"),
    "last_city": (r"In which city was (\w+)'s most recent record\?",
                  r"Where (which city) did \1's final logged event take place?"),
    "first_after": (r"Which user made the first record at or after (.+)\?",
                    r"At \1 or later, whose event comes first?"),
    "distinct": (r"How many different items did (\w+) have records with action (\w+) for\?",
                 r"Count the unique items \1 \2, without counting repeats."),
    "day_span": (r"How many whole days are there between the dates of (\w+)'s first and last records\?",
                 r"What is the number of calendar days from the date of \1's first event to the date of their last one?"),
    "semantic_total": (r"How many units of (\w+) items were bought in total \(action bought only\)\?",
                       r"Considering only purchases (action bought), what is the combined quantity of all \1 items?"),
}


class StressFamily(F.Family):
    def __init__(self, fid: str, condition: str):
        super().__init__(fid)
        if condition not in CONDITIONS:
            raise ValueError(condition)
        self.condition = condition
        if condition in ("nuisance", "nuisance2"):
            # nuisance2 must differ from both the original and nuisance 1 (label always; order or separator too)
            avoid = [(list(self.order), self.qlabel, self.sep)]
            if condition == "nuisance2":
                n1 = StressFamily(fid, "nuisance")
                avoid.append((list(n1.order), n1.qlabel, n1.sep))
            r = random.Random(f"{condition}:{fid}")
            while (list(self.order), self.qlabel, self.sep) in avoid or self.qlabel in [x[1] for x in avoid]:
                r.shuffle(self.order)
                self.qlabel = r.choice(("Question:", "QUESTION:", "Q:", "Task:"))
                self.sep = r.choice((" | ", " ; ", " :: "))

    def ask(self, r, recs):
        qa = super().ask(r, recs)
        if qa is None or self.condition not in ("paraphrase", "paraphrase2"):
            return qa
        pattern, repl = (PARAPHRASE if self.condition == "paraphrase" else PARAPHRASE2)[self.question]
        new, n = re.subn(f"^{pattern}$", repl, qa[0])
        if n != 1:
            raise ValueError(f"paraphrase pattern did not match: {qa[0]!r}")
        return new, qa[1]


SHOWN_SEED0 = {27: "control", 12: "paraphrase", 24: "nuisance"}  # random.Random(0).sample(range(30), 3)


def teach_condition(i: int) -> str:
    if i in SHOWN_SEED0:
        return SHOWN_SEED0[i]
    rest = [j for j in range(30) if j not in SHOWN_SEED0]
    return ("control", "paraphrase", "nuisance")[rest.index(i) % 3]


def make_teach(fid: str) -> list:
    """E60 teaching set: E39's 30 teaching tasks, 10 per wording, the seed-0 shown positions one per wording."""
    fams = {c: StressFamily(fid, c) for c in ("control", "paraphrase", "nuisance")}
    out = []
    for i in range(30):
        c = teach_condition(i)
        t = fams[c].task(i, "teach", 200)
        t.task_id = f"{t.task_id}-{c}"
        out.append(t)
    return out


def make(fid: str, condition: str, k: int = 30) -> list:
    fam = StressFamily(fid, condition)
    n = 2000 if condition == "long" else 400
    tasks = [fam.task(i, "stress", n) for i in range(k)]
    for t in tasks:  # keep task ids unique across conditions that share records
        t.task_id = f"{t.task_id}-{condition}"
    return tasks
