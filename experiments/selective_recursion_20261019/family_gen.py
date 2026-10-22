"""Procedural long-context task families for library discovery (E26).

A family fixes a record format, a question form and an answer format, plus nuisance details drawn from
its seed (field order, date style, question label); tasks within a family vary the records and the
question's slots. Answers are computed exactly from the records, so every family is verifiable. The
semantic question needs world knowledge about items (their category), so a program alone cannot answer
it and a library must call the helper.

Splits by construction: training families use TRAIN_FORMATS x TRAIN_QUESTIONS; held-out families use a
new format or a new question (HELDOUT_FORMATS, HELDOUT_QUESTIONS). Family ids look like
`gen-<format>-<question>-<answer>-s<seed>`.
"""

import datetime as _dt
import json
import random
import re
from types import SimpleNamespace

NAMES = ("alice bruno chen dalia emeka farah goran hana ivan jules kemal lena mateo nadia omar priya quinn rosa "
         "sami tara umar vera wen xavi yara zane amira boris celine dmitri elif felix gita hugo ines jonas kiri "
         "lucas mira nils olga pablo rhea stefan tomas uma viktor willa yusuf zoe").split()
CATEGORIES = {
    "fruit": "apple banana cherry mango pear plum peach grape lemon kiwi apricot papaya guava lychee fig".split(),
    "vegetable": "carrot potato onion spinach broccoli cabbage celery leek radish turnip kale lettuce beet parsnip garlic".split(),
    "tool": "hammer wrench screwdriver chisel pliers saw drill mallet crowbar trowel clamp sander awl spanner hacksaw".split(),
    "instrument": "violin cello flute trumpet oboe clarinet harp banjo ukulele tuba piccolo bassoon mandolin accordion xylophone".split(),
    "furniture": "sofa armchair bookcase wardrobe dresser nightstand ottoman bench stool desk cabinet futon recliner sideboard credenza".split(),
}
ITEM_CATEGORY = {it: c for c, its in CATEGORIES.items() for it in its}
ITEMS = sorted(ITEM_CATEGORY)
CITIES = ("Paris Lagos Lima Oslo Pune Quito Seoul Tunis Hanoi Accra Bogota Cairo Dakar Doha Kyiv Lyon Minsk Nairobi "
          "Osaka Porto Riga Sofia Tbilisi Turin Vienna Zagreb").split()
ACTIONS = ("bought", "returned", "viewed", "rated")
FIELDS = ("ts", "user", "action", "item", "qty", "city")

TRAIN_FORMATS = ("pipe", "kv", "jsonl", "csv", "table")
HELDOUT_FORMATS = ("prose", "xmlish", "chat")
TRAIN_QUESTIONS = ("count", "total", "top_user", "nth_item", "last_city", "first_after", "distinct")
HELDOUT_QUESTIONS = ("day_span", "semantic_total")
# E26e: training questions that need the helper's judgment (item categories); semantic_total stays held out
HELPER_TRAIN_QUESTIONS = ("semantic_count", "semantic_top_user")
ANSWER_FORMATS = ("plain", "answer_line", "prefix", "json")
ORD = {1: "1st", 2: "2nd", 3: "3rd"}


def ordinal(n: int) -> str:
    return ORD.get(n, f"{n}th")


def family_ids(split: str, seeds=(0,)) -> list[str]:
    if split == "train":
        combos = [(f, q) for f in TRAIN_FORMATS for q in TRAIN_QUESTIONS]
    else:
        combos = ([(f, q) for f in HELDOUT_FORMATS for q in TRAIN_QUESTIONS + HELDOUT_QUESTIONS]
                  + [(f, q) for f in TRAIN_FORMATS for q in HELDOUT_QUESTIONS])
    out = []
    for f, q in combos:
        for s in seeds:
            a = ANSWER_FORMATS[(len(f) + len(q) + s) % len(ANSWER_FORMATS)]  # deterministic spread over formats
            out.append(f"gen-{f}-{q}-{a}-s{s}")
    return out


def helper_training_family_ids(seeds=(0, 1)) -> list[str]:
    """Training families whose questions need the helper (E26e); formats are the training formats."""
    return [f"gen-{f}-{q}-{ANSWER_FORMATS[(len(f) + len(q) + s) % len(ANSWER_FORMATS)]}-s{s}"
            for f in TRAIN_FORMATS for q in HELPER_TRAIN_QUESTIONS for s in seeds]


def parse_family(fid: str):
    _, fmt, rest = fid.split("-", 2)
    q, a, s = rest.rsplit("-", 2)
    return fmt, q, a, int(s[1:])


class Family:
    def __init__(self, fid: str):
        self.fid = fid
        self.fmt, self.question, self.answer_fmt, self.seed = parse_family(fid)
        r = random.Random(f"family:{fid}")
        self.order = list(FIELDS)
        r.shuffle(self.order)
        self.date_style = r.choice(("iso", "dmy", "words"))
        self.qlabel = r.choice(("Question:", "QUESTION:", "Q:", "Task:"))
        self.sep = r.choice((" | ", " ; ", " :: "))

    # ---- rendering ----
    def fmt_ts(self, ts: _dt.datetime) -> str:
        if self.date_style == "iso":
            return ts.strftime("%Y-%m-%d %H:%M")
        if self.date_style == "dmy":
            return ts.strftime("%d/%m/%Y %H:%M")
        return ts.strftime("%B %d, %Y at %H:%M").replace(" 0", " ")

    def cell(self, rec, f):
        return self.fmt_ts(rec["ts"]) if f == "ts" else str(rec[f])

    def render(self, recs):
        o = self.order
        if self.fmt == "pipe":
            return "\n".join(self.sep.join(self.cell(r, f) for f in o) for r in recs)
        if self.fmt == "kv":
            return "\n".join(" ".join(f"{f}={self.cell(r, f)!r}" if f == "ts" else f"{f}={self.cell(r, f)}" for f in o) for r in recs)
        if self.fmt == "jsonl":
            return "\n".join(json.dumps({f: self.cell(r, f) for f in o}) for r in recs)
        if self.fmt == "csv":
            return ",".join(o) + "\n" + "\n".join(",".join(f'"{self.cell(r, f)}"' if f == "ts" else self.cell(r, f) for f in o) for r in recs)
        if self.fmt == "table":
            head = "| " + " | ".join(o) + " |\n|" + "---|" * len(o)
            return head + "\n" + "\n".join("| " + " | ".join(self.cell(r, f) for f in o) + " |" for r in recs)
        if self.fmt == "prose":
            return "\n".join(f"On {self.fmt_ts(r['ts'])}, {r['user'].capitalize()} {r['action']} {r['qty']} x {r['item']} in {r['city']}."
                             for r in recs)
        if self.fmt == "xmlish":
            return "\n".join("<event>" + "".join(f"<{f}>{self.cell(r, f)}</{f}>" for f in o) + "</event>" for r in recs)
        raise ValueError(self.fmt)

    # ---- records ----
    def records(self, r: random.Random, n: int):
        users = r.sample(NAMES, 12)
        items = r.sample(ITEMS, 30)
        t = _dt.datetime(2023, 1, 1) + _dt.timedelta(days=r.randrange(600))
        out = []
        for _ in range(n):
            t += _dt.timedelta(minutes=r.randrange(5, 400))
            out.append(dict(ts=t, user=r.choice(users), action=r.choice(ACTIONS), item=r.choice(items),
                            qty=r.randint(1, 9), city=r.choice(CITIES)))
        return out

    # ---- questions ----
    def ask(self, r: random.Random, recs):
        q = self.question
        users = sorted({x["user"] for x in recs})
        if q == "count":
            a, it = r.choice(ACTIONS), r.choice(recs)["item"]
            return f"How many records have action {a} and item {it}?", str(sum(x["action"] == a and x["item"] == it for x in recs))
        if q == "total":
            u, it = r.choice(recs)["user"], r.choice(recs)["item"]
            return (f"How many units of {it} did {u} buy in total (action bought only)?",
                    str(sum(x["qty"] for x in recs if x["user"] == u and x["item"] == it and x["action"] == "bought")))
        if q == "top_user":
            a = r.choice(ACTIONS)
            c = {u: sum(x["user"] == u and x["action"] == a for x in recs) for u in users}
            best = sorted(c.items(), key=lambda kv: -kv[1])
            if len(best) > 1 and best[0][1] == best[1][1]:
                return None
            return f"Which user has the most records with action {a}?", best[0][0]
        if q == "nth_item":
            u = r.choice(users)
            mine = [x for x in recs if x["user"] == u]
            n = r.randint(1, min(len(mine), 9))
            return f"What was the item in {u}'s {ordinal(n)} record (counting from the start of the log)?", mine[n - 1]["item"]
        if q == "last_city":
            u = r.choice(users)
            return f"In which city was {u}'s most recent record?", [x for x in recs if x["user"] == u][-1]["city"]
        if q == "first_after":
            k = r.randrange(len(recs) // 4, 3 * len(recs) // 4)
            ts = recs[k - 1]["ts"] + (recs[k]["ts"] - recs[k - 1]["ts"]) / 2
            ts = ts.replace(second=0, microsecond=0)
            if not recs[k - 1]["ts"] < ts <= recs[k]["ts"]:
                return None
            return f"Which user made the first record at or after {self.fmt_ts(ts)}?", recs[k]["user"]
        if q == "distinct":
            u, a = r.choice(users), r.choice(ACTIONS)
            return f"How many different items did {u} have records with action {a} for?", str(len({x['item'] for x in recs if x['user'] == u and x['action'] == a}))
        if q == "day_span":
            u = r.choice(users)
            mine = [x for x in recs if x["user"] == u]
            return (f"How many whole days are there between the dates of {u}'s first and last records?",
                    str((mine[-1]["ts"].date() - mine[0]["ts"].date()).days))
        if q == "semantic_count":
            cat = r.choice(sorted(CATEGORIES))
            return (f"How many records with action bought involve a {cat} item?",
                    str(sum(x["action"] == "bought" and ITEM_CATEGORY[x["item"]] == cat for x in recs)))
        if q == "semantic_top_user":
            cat = r.choice(sorted(CATEGORIES))
            c = {u: sum(x["qty"] for x in recs if x["user"] == u and x["action"] == "bought" and ITEM_CATEGORY[x["item"]] == cat)
                 for u in users}
            best = sorted(c.items(), key=lambda kv: -kv[1])
            if len(best) > 1 and best[0][1] == best[1][1]:
                return None
            return f"Which user bought the most units of {cat} items in total?", best[0][0]
        if q == "semantic_total":
            cat = r.choice(sorted(CATEGORIES))
            return (f"How many units of {cat} items were bought in total (action bought only)?",
                    str(sum(x["qty"] for x in recs if x["action"] == "bought" and ITEM_CATEGORY[x["item"]] == cat)))
        raise ValueError(q)

    def instruction(self, r: random.Random):
        if self.answer_fmt == "plain":
            return "Reply with the answer only.", ""
        if self.answer_fmt == "answer_line":
            return "Reply with one line of the form 'ANSWER: <answer>'.", "ANSWER: "
        if self.answer_fmt == "json":
            return 'Reply with a JSON object {"answer": "<answer>"} and nothing else.', None
        code = "".join(r.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(8))
        return f"Begin your reply with the code {code} immediately followed by the answer.", code

    def task(self, idx: int, split: str, n_records: int):
        r = random.Random(f"task:{self.fid}:{split}:{n_records}:{idx}")
        while True:
            recs = self.records(r, n_records)
            qa = self.ask(r, recs)
            if qa:
                break
        question, answer = qa
        instr, prefix = self.instruction(r)
        if self.fmt == "chat":
            msgs = [dict(role="system", content="You are a records clerk. Each user message reports one event.")]
            for x in recs:
                msgs.append(dict(role="user", content=f"Event: {x['user']} {x['action']} {x['qty']} x {x['item']} in {x['city']} on {self.fmt_ts(x['ts'])}."))
                msgs.append(dict(role="assistant", content="Noted."))
            msgs.append(dict(role="user", content=f"{self.qlabel} {question} {instr}"))
            prompt = msgs
        else:
            prompt = f"{self.render(recs)}\n\n{self.qlabel} {question} {instr}"
        gold = json.dumps({"answer": answer}) if self.answer_fmt == "json" else (prefix or "") + answer
        return SimpleNamespace(task_id=f"{self.fid}-{split}-{n_records}-{idx}", family=self.fid, prompt=prompt, answer=gold,
                               raw_answer=answer, answer_fmt=self.answer_fmt, n_records=n_records, question=question)


def make(fid: str, split: str, k: int, n_records: int = 200):
    fam = Family(fid)
    return [fam.task(i, split, n_records) for i in range(k)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().strip(".").lower())


def grade(output, task) -> float:
    out = (output or "").strip()
    if task.answer_fmt == "json":
        try:
            return float(_norm(json.loads(out)["answer"]) == _norm(task.raw_answer))
        except (ValueError, KeyError, TypeError):
            return 0.0
    if task.answer_fmt == "prefix":
        code = task.answer[: len(task.answer) - len(task.raw_answer)]
        return float(out.startswith(code) and _norm(out[len(code):]) == _norm(task.raw_answer))
    if task.answer_fmt == "answer_line":
        m = re.findall(r"ANSWER:\s*(.+)", out)
        return float(bool(m) and _norm(m[-1]) == _norm(task.raw_answer))
    return float(_norm(out) == _norm(task.raw_answer))


def describe(t) -> str:
    """Teaching-example rendering for the proposer: abbreviated input, question, gold answer."""
    if isinstance(t.prompt, list):
        shown = t.prompt[:7] + [dict(role="...", content=f"({len(t.prompt) - 9} more messages)")] + t.prompt[-2:]
        body = "\n".join(f"[{m['role']}] {m['content']}" for m in shown)
        return f"Input: a list of {len(t.prompt)} chat messages (request['input'] is this list of dicts with 'role' and 'content'):\n{body}\nExpected output: {t.answer!r}"
    lines = t.prompt.split("\n")
    shown = "\n".join(lines[:12] + [f"... ({len(lines) - 16} more lines) ..."] + lines[-4:])
    return f"Input: a string of {len(t.prompt)} characters (request['input']):\n{shown}\nExpected output: {t.answer!r}"
