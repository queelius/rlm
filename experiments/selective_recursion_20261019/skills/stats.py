"""Outcome statistics over routines, keyed by the input signature (L3a-lite).

A routine is a way of answering: a library call pattern, the direct answer, or the controller's
own program. Every logged run contributes (signature, routine, correct, tokens). At run time the
harness computes the signature from the input text alone, retrieves the verified history for that
signature, and renders a short note with the recommended routine (highest accuracy minus a token
cost). Nothing here is learned into the controller's weights; the decision lives in the table.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

LIB_NAMES = {
    "eventlog": ("split_log", "extract_events", "answer_eventlog", "lookup_day"),
    "oolong": ("parse_oolong", "classify_rows", "answer_oolong"),
}
RECIPES = {
    "lib:extract_events+answer_eventlog": "lines, question = split_log(request['input']); "
    "events = extract_events(lines); FINAL_TEXT(answer_eventlog(events, question))",
    "lib:lookup_day": "lines, question = split_log(request['input']); FINAL_TEXT(lookup_day(lines, question))",
    "lib:classify_rows+answer_oolong": "rows, labels, question = parse_oolong(request['input']); "
    "classify_rows(rows, labels); FINAL_TEXT(answer_oolong(rows, labels, question))",
    "direct": "FINAL_RESPONSE(model_complete(request))",
    "own": "your own program (ask_batch over chunks, aggregate in Python)",
}
LABELS = {"direct": "direct answer", "own": "own program without the library"}


def question_kind(question: str) -> str:
    if re.search(r"did (\w+) buy an? (\w+)\?", question):
        return "lookup_day"
    if re.search(r"at least one (\w+) and also bought at least one (\w+)\?", question):
        return "both_items"
    if re.search(r"bought at least one (\w+)\?", question):
        return "count_buyers"
    if re.search(r"How many units of (\w+) were bought", question):
        return "net_units"
    if re.search(r"Which people bought a (\w+) and later", question):
        return "bought_then_returned"
    return "other"


def length_bucket(lines: int) -> int:
    """Power-of-two floor bucket of a line count (8..15 -> 8, 32..63 -> 32, ...)."""
    return 1 << max(int(lines).bit_length() - 1, 0)


def signature_from_input(text: str) -> dict | None:
    """Signature computed from the request input alone (event-log family); None when unknown."""
    m = re.search(r"LOG \((\d+) lines\)", text or "")
    if m and "QUESTION:" in text:
        question = text.split("QUESTION:", 1)[1].split("\n", 1)[0].strip()
        return dict(family="eventlog", length=length_bucket(int(m.group(1))), question_type=question_kind(question))
    return None


def routine_of(cells: list[str], family: str) -> str:
    joined = "\n".join(cells)
    used = [n for n in LIB_NAMES.get(family, ()) if re.search(rf"\b{n}\(", joined)]
    core = [n for n in used if n not in ("split_log", "parse_oolong")]
    if core:
        return "lib:" + "+".join(core)
    if "model_complete(request)" in joined:
        return "direct"
    if "ask_batch(" in joined or re.search(r"\bask\(", joined):
        return "own"
    return "other"


def _cells(trace_dir: str | None) -> list[str]:
    if not trace_dir:
        return []
    try:
        return [json.loads(l)["payload"]["code"] for l in open(Path(trace_dir) / "trace.jsonl")
                if '"controller.action"' in l]
    except OSError:
        return []


_TASK_ID = re.compile(r"evlog-(\w+)-n(\d+)-(\w+)-s(\d+)-")


def _row_signature(r: dict, family: str) -> dict:
    """The run-time signature of a logged row: regenerate the event-log task and read its input."""
    if family == "eventlog":
        m = _TASK_ID.match(r.get("task_id") or "")
        if m:
            import taskgen

            split, n, qt, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            sig = signature_from_input(taskgen.make_task(seed, n, qt, split=split).prompt)
            if sig:
                return sig
        return dict(family=family, length=r.get("n"), question_type=r.get("question_type"))
    return dict(family=family, length=r.get("context_len"), question_type=r.get("question_type"))


def collect(run_dir: Path, routine: str | None = None, source: str | None = None) -> list[dict]:
    """Records from one run directory; `routine` overrides classification (e.g. for the direct policy)."""
    out = []
    for line in open(Path(run_dir) / "rows.jsonl"):
        r = json.loads(line)
        family = "oolong" if "context_len" in r else "eventlog"
        if routine is not None:
            rt = routine
        elif r.get("policy") in ("direct", "fixed"):
            continue  # only routes the controller itself can take; bare policies are not its options
        elif r.get("harness", "default") == "default" and r.get("policy") == "rlm":
            rt = "own"
        else:
            rt = routine_of(_cells(r.get("trace_dir")), family)
        if rt == "other":
            continue
        sig = _row_signature(r, family)
        out.append(dict(family=family, length=sig["length"], question_type=sig["question_type"],
                        routine=rt, correct=bool(r.get("correct")), tokens=int(r.get("total_tokens") or 0),
                        source=source or Path(run_dir).name))
    return out


def summarize(records: list[dict]) -> dict:
    table: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: dict(n=0, correct=0, tokens=0)))
    for x in records:
        cell = table[f"{x['family']}|{x['length']}|{x['question_type']}"][x["routine"]]
        cell["n"] += 1
        cell["correct"] += int(x["correct"])
        cell["tokens"] += x["tokens"]
    return {k: dict(v) for k, v in table.items()}


def _key(sig: dict) -> str:
    return f"{sig['family']}|{sig['length']}|{sig['question_type']}"


def lookup(sig: dict, table: dict) -> tuple[str, dict] | tuple[None, None]:
    """Exact signature first; else the nearest length with the same family and question type."""
    k = _key(sig)
    if k in table:
        return k, table[k]
    cands = []
    for key in table:
        fam, length, qt = key.split("|")
        if fam == sig["family"] and qt == str(sig["question_type"]) and length not in ("None", ""):
            cands.append((abs(int(length) - int(sig["length"] or 0)), key))
    if cands:
        key = min(cands)[1]
        return key, table[key]
    return None, None


def score(cell: dict, lam: float) -> float:
    """Accuracy minus lam per 10k mean tokens; a routine that never succeeded is never preferred."""
    acc = cell["correct"] / cell["n"]
    if acc == 0:
        return -1.0
    return acc - lam * (cell["tokens"] / cell["n"]) / 10_000


def note_for(sig: dict, table: dict, lam: float = 0.05, min_n: int = 4) -> str:
    key, hist = lookup(sig, table)
    if not hist:
        return ""
    fam, length, qt = key.split("|")
    where = f"{'event-log' if fam == 'eventlog' else fam}, about {length} lines, {qt.replace('_', ' ')} question"
    exact = key == _key(sig)
    rows = sorted(hist.items(), key=lambda kv: -score(kv[1], lam))
    lines = [f"Verified history for inputs like this one ({where}{'' if exact else ', nearest length'}), "
             f"from logged runs; accuracy, then mean total tokens:"]
    best = None
    for rt, c in rows:
        label = LABELS.get(rt, rt.replace("lib:", "").replace("+", " + "))
        recipe = RECIPES.get(rt, "")
        lines.append(f"  - {label}: {c['correct']}/{c['n']} correct, ~{c['tokens'] / c['n'] / 1000:.1f}k tokens"
                     + (f"  [{recipe}]" if recipe else ""))
        if best is None and c["n"] >= min_n:
            best = rt
    if best is not None:
        lines.append(f"Recommended here: {LABELS.get(best, best.replace('lib:', '').replace('+', ' + '))} "
                     f"(best verified accuracy minus cost). Use it as the first cell unless the input clearly differs.")
    return "\n".join(lines)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())["table"]
