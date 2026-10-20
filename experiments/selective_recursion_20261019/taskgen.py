"""Verifiable event-log aggregation tasks with controllable length and cross-record dependency.

Each task is a log of N natural-language activity records about people and products, followed
by a question whose exact answer is computed from the hidden structured events. Records use
varied templates, number words, negated distractors and unrelated filler so that per-record
understanding needs language, while the aggregation needs exact bookkeeping across records that
may be far apart. Length (N) and question type are the experimental knobs; the generator is
deterministic in (seed, N, question type) and records the template identities it used.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass

SCHEMA = "selective-recursion-eventlog-v1"

NAMES = [
    "Alice", "Bruno", "Carmen", "Dmitri", "Elena", "Farid", "Greta", "Hugo", "Ingrid", "Jonas",
    "Keiko", "Lars", "Maya", "Nikhil", "Olga", "Pedro", "Quinn", "Rosa", "Stefan", "Tariq",
    "Uma", "Viktor", "Wanda", "Xavier", "Yara", "Zoltan", "Amara", "Bastian", "Celine", "Diego",
    "Esther", "Felix", "Gloria", "Hamid", "Isabel", "Jamal", "Kirsten", "Leon", "Marta", "Noor",
]
ITEMS = [
    "kettle", "lamp", "backpack", "blender", "tent", "scarf", "printer", "guitar", "mirror",
    "toaster", "helmet", "hammock", "tripod", "umbrella", "wallet", "sweater", "thermos",
    "notebook", "speaker", "saucepan", "drill", "telescope", "keyboard", "suitcase", "raincoat",
    "compass", "lantern", "skateboard", "microphone", "vase",
]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
DAYS = [f"day {d}" for d in range(1, 61)]

# Event templates. {p}=person {q}=quantity text {i}=item (pluralized) {d}=day
BUY = [
    "On {d}, {p} bought {q} {i}.",
    "{p} purchased {q} {i} on {d}.",
    "{p} paid for {q} {i} at the counter on {d}.",
    "A receipt from {d} shows {p} picking up {q} {i}.",
    "{p} ordered {q} {i}; the order went through on {d}.",
    "On {d} the register logged a sale of {q} {i} to {p}.",
]
RETURN = [
    "On {d}, {p} returned {q} {i} for a refund.",
    "{p} brought back {q} {i} on {d} and was refunded.",
    "A refund was issued to {p} on {d} for {q} {i}.",
    "{p} sent {q} {i} back to the store on {d}.",
]
COMPLAIN = [
    "On {d}, {p} filed a complaint about the {s}.",
    "{p} called on {d} to complain that the {s} was defective.",
    "{p} left a one-star review of the {s} on {d}.",
    "Customer service logged a complaint from {p} about the {s} on {d}.",
]
RECOMMEND = [
    "On {d}, {p} recommended the {s} to a friend.",
    "{p} told the staff on {d} that the {s} was excellent.",
    "{p} posted a five-star review of the {s} on {d}.",
]
# Negated or hypothetical distractors mention a person and an item without any event happening.
DISTRACT = [
    "{p} looked at the {s} on {d} but did not buy it.",
    "{p} asked whether the {s} was in stock on {d}; it was not.",
    "On {d}, {p} almost returned the {s}, then kept it.",
    "{p} considered ordering {q} {i} on {d} but decided against it.",
    "A staff member suggested the {s} to {p} on {d}; {p} declined.",
    "{p} added {q} {i} to a wishlist on {d} without purchasing.",
]
FILLER = [
    "The store opened late on {d} because of a staff meeting.",
    "On {d}, the {s} display was moved closer to the entrance.",
    "Inventory of the {s} was recounted on {d}.",
    "A delivery of {i} arrived on {d}.",
    "The shop was closed for cleaning on {d}.",
]

QUESTION_TYPES = ("count_buyers", "both_items", "net_units", "bought_then_returned", "lookup_day")
AGGREGATE_TYPES = QUESTION_TYPES[:4]
# Held-out types for the library-growth experiment (E08): not covered by the taught routines or the
# shipped library, generated under the same schema so prior task ids are unchanged.
HELDOUT_TYPES = ("top_buyer", "purchase_span")


def plural(item: str, q: int) -> str:
    if q == 1:
        return item
    return item + ("es" if item.endswith("s") else "s")


@dataclass(frozen=True)
class Event:
    kind: str
    person: str
    item: str
    qty: int
    day: int


@dataclass
class Task:
    task_id: str
    seed: int
    n_records: int
    question_type: str
    prompt: str
    question: str
    answer: str
    answer_type: str
    support_records: int
    events: list
    template_ids: list


def _render(rng: random.Random, ev: Event, kind_templates: dict) -> tuple[str, str]:
    templates = kind_templates[ev.kind]
    t = rng.randrange(len(templates))
    qtext = NUMBER_WORDS[ev.qty] if rng.random() < 0.5 else str(ev.qty)
    text = templates[t].format(
        p=ev.person, q=qtext, i=plural(ev.item, ev.qty), s=ev.item, d=DAYS[ev.day]
    )
    return text, f"{ev.kind}:{t}"


def _answer(question_type: str, events: list[Event], a: str, b: str | None) -> tuple[str, str, int]:
    """Exact answer, answer type and number of records that bear on it."""
    buys = [e for e in events if e.kind == "buy"]
    rets = [e for e in events if e.kind == "return"]
    if question_type == "count_buyers":
        people = {e.person for e in buys if e.item == a}
        return str(len(people)), "integer", sum(e.item == a for e in buys)
    if question_type == "both_items":
        pa = {e.person for e in buys if e.item == a}
        pb = {e.person for e in buys if e.item == b}
        both = sorted(pa & pb)
        return (", ".join(both) if both else "none"), "name_list", sum(e.item in (a, b) for e in buys)
    if question_type == "net_units":
        net = sum(e.qty for e in buys if e.item == a) - sum(e.qty for e in rets if e.item == a)
        return str(net), "integer", sum(e.item == a for e in buys + rets)
    if question_type == "lookup_day":
        hits = [e for e in buys if e.item == a and e.person == b]
        return str(hits[0].day + 1), "integer", len(hits)
    if question_type == "bought_then_returned":
        first_buy: dict[str, int] = {}
        for e in buys:
            if e.item == a:
                first_buy[e.person] = min(first_buy.get(e.person, 99), e.day)
        people = sorted(
            {e.person for e in rets if e.item == a and e.person in first_buy and e.day > first_buy[e.person]}
        )
        return (", ".join(people) if people else "none"), "name_list", sum(e.item == a for e in buys + rets)
    if question_type == "top_buyer":
        totals: dict[str, int] = {}
        for e in buys:
            if e.item == a:
                totals[e.person] = totals.get(e.person, 0) + e.qty
        top = max(totals.values())
        winners = sorted(p for p, q in totals.items() if q == top)
        return winners[0], "name_list", sum(e.item == a for e in buys)
    if question_type == "purchase_span":
        days = [e.day for e in buys if e.item == a]
        return str(max(days) - min(days)), "integer", len(days)
    raise ValueError(question_type)


def _question(question_type: str, a: str, b: str | None) -> tuple[str, str]:
    if question_type == "count_buyers":
        return (f"How many distinct people bought at least one {a}?",
                "a single integer")
    if question_type == "both_items":
        return (f"Which people bought at least one {a} and also bought at least one {b}?",
                "the names in alphabetical order separated by commas, or the word none")
    if question_type == "net_units":
        return (f"How many units of {a} were bought in total, minus the units of {a} that were returned?",
                "a single integer")
    if question_type == "lookup_day":
        return (f"On which day did {b} buy a {a}? Answer with the day number only.",
                "a single integer")
    if question_type == "bought_then_returned":
        return (f"Which people bought a {a} and later (on a strictly later day) returned a {a}?",
                "the names in alphabetical order separated by commas, or the word none")
    if question_type == "top_buyer":
        return (f"Which single person bought the most units of {a} in total (returns do not count)?",
                "the name only")
    if question_type == "purchase_span":
        return (f"How many days passed between the earliest purchase of a {a} and the latest purchase of a {a}?",
                "a single integer")
    raise ValueError(question_type)


def make_task(seed: int, n_records: int, question_type: str, *, split: str = "train") -> Task:
    if question_type not in QUESTION_TYPES + HELDOUT_TYPES:
        raise ValueError(question_type)
    rng = random.Random(f"{SCHEMA}:{split}:{seed}:{n_records}:{question_type}")
    n_people = max(4, min(len(NAMES), n_records // 3))
    n_items = max(3, min(len(ITEMS), n_records // 4))
    people = rng.sample(NAMES, n_people)
    items = rng.sample(ITEMS, n_items)
    a = items[0]
    b = items[1] if question_type == "both_items" else None
    # Keep the question's support small but non-trivial regardless of N.
    support_target = max(3, min(24, round(n_records * 0.12)))
    events: list[Event] = []
    kinds = ["buy"] * 5 + ["return"] * 2 + ["complain"] * 2 + ["recommend"] * 1
    for _ in range(n_records):
        kind = rng.choice(kinds)
        relevant = rng.random() < (support_target / max(n_records, 1)) * 2.5
        item = (rng.choice([x for x in (a, b) if x]) if relevant else rng.choice(items[2:] or items))
        events.append(Event(kind, rng.choice(people), item, rng.randint(1, 6), rng.randrange(len(DAYS))))
    # Guarantee the target item appears in a few buys so answers are rarely degenerate.
    for k in range(min(3, n_records)):
        events[k] = Event("buy", rng.choice(people), a, rng.randint(1, 6), rng.randrange(len(DAYS)))
    if b:
        for k in range(3, min(6, n_records)):
            events[k] = Event("buy", events[k - 3].person if rng.random() < 0.6 else rng.choice(people),
                              b, rng.randint(1, 6), rng.randrange(len(DAYS)))
    if question_type == "lookup_day":
        b = rng.choice(people)
        # exactly one purchase of item a by person b; other buys of a are by other people
        for k, ev in enumerate(events):
            if ev.item == a and ev.person == b and ev.kind == "buy":
                events[k] = Event("buy", rng.choice([p for p in people if p != b]), a, ev.qty, ev.day)
        events[0] = Event("buy", b, a, rng.randint(1, 6), rng.randrange(len(DAYS)))
    if question_type == "top_buyer":
        # break ties for the top buyer of item a with one extra unit for the alphabetically first leader
        totals: dict[str, int] = {}
        for ev in events:
            if ev.kind == "buy" and ev.item == a:
                totals[ev.person] = totals.get(ev.person, 0) + ev.qty
        top = max(totals.values())
        leaders = sorted(p for p, q in totals.items() if q == top)
        if len(leaders) > 1:
            k = next(i for i, ev in enumerate(events) if ev.kind == "buy" and ev.item == a and ev.person == leaders[0])
            if events[k].qty < 6:
                events[k] = Event("buy", leaders[0], a, events[k].qty + 1, events[k].day)
            else:
                events.append(Event("buy", leaders[0], a, 1, rng.randrange(len(DAYS))))
    if question_type == "bought_then_returned":
        for k in range(6, min(9, n_records)):
            buyer = events[k - 6]
            events[k] = Event("return", buyer.person, a, 1, min(len(DAYS) - 1, buyer.day + rng.randint(1, 9)))
    rng.shuffle(events)
    # Interleave distractors and filler as extra lines: they mention entities but are not events.
    lines, template_ids = [], []
    kind_templates = {"buy": BUY, "return": RETURN, "complain": COMPLAIN, "recommend": RECOMMEND}
    for ev in events:
        text, tid = _render(rng, ev, kind_templates)
        lines.append(text)
        template_ids.append(tid)
        r = rng.random()
        if r < 0.18:
            t = rng.randrange(len(DISTRACT))
            q = rng.randint(1, 6)
            dist_item = rng.choice([a] + ([b] if b else []) + items[2:3])
            lines.append(DISTRACT[t].format(p=rng.choice(people), q=NUMBER_WORDS[q], i=plural(dist_item, q),
                                            s=dist_item, d=rng.choice(DAYS)))
            template_ids.append(f"distract:{t}")
        elif r < 0.26:
            t = rng.randrange(len(FILLER))
            lines.append(FILLER[t].format(s=rng.choice(items), i=plural(rng.choice(items), 2), d=rng.choice(DAYS)))
            template_ids.append(f"filler:{t}")
    answer, answer_type, support = _answer(question_type, events, a, b)
    question, fmt = _question(question_type, a, b)
    log = "\n".join(f"{k + 1}. {line}" for k, line in enumerate(lines))
    prompt = (
        "Below is a store activity log, one entry per line, followed by a question about it. "
        "Only lines that describe a completed purchase or a completed return count as purchases or "
        "returns (an order that went through is a completed purchase); lines about considering, "
        "declining, wishlists, complaints, reviews or stock do not.\n\n"
        f"LOG ({len(lines)} lines):\n{log}\n\n"
        f"QUESTION: {question}\n"
        f"You may reason briefly first. Finish with exactly one final line of the form "
        f"ANSWER: <answer>, where <answer> is {fmt}."
    )
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return Task(
        task_id=f"evlog-{split}-n{n_records}-{question_type}-s{seed}-{digest}",
        seed=seed, n_records=n_records, question_type=question_type, prompt=prompt,
        question=question, answer=answer, answer_type=answer_type, support_records=support,
        events=[asdict(e) for e in events], template_ids=template_ids,
    )


def final_answer_text(text: str) -> str:
    """Take the text after the last ANSWER: marker (first line only); else the whole text."""
    if text is None:
        return ""
    idx = text.rfind("ANSWER:")
    if idx < 0:
        return text
    return text[idx + len("ANSWER:"):].strip().split("\n", 1)[0]


def normalize(text: str, answer_type: str) -> str | None:
    s = final_answer_text(text).strip().strip("*").strip().strip(".").strip()
    if answer_type == "integer":
        tokens = [t for t in s.replace(",", " ").split() if t.lstrip("-").isdigit()]
        if len(tokens) == 1:
            return str(int(tokens[0]))
        return str(int(s)) if s.lstrip("-").isdigit() else None
    names = [t.strip().strip(".").capitalize() for t in s.replace(" and ", ",").split(",") if t.strip()]
    if not names or names == ["None"]:
        return "none"
    return ", ".join(sorted(set(names)))


def verify(task: Task | dict, output_text: str) -> bool:
    answer_type = task["answer_type"] if isinstance(task, dict) else task.answer_type
    answer = task["answer"] if isinstance(task, dict) else task.answer
    return normalize(output_text, answer_type) == normalize(answer, answer_type)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--qt", default="count_buyers")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    t = make_task(args.seed, args.n, args.qt)
    print(t.prompt[:3000]); print("...\nANSWER:", t.answer, "| support records:", t.support_records,
                                  "| prompt chars:", len(t.prompt))
