"""E68: does a question-canonicalization wrapper repair E59's template-bound libraries end to end?

The wrapper keeps the frozen library and changes only its input: it renders the question back into the family's
original template from parsed parameters, restores the family's original question label, and keeps the task's own
answer instruction (located by the generator's four instruction forms) and records. Parameters come from E61's
parses (qwen3-4b, qwen3-0.6b-nt; same stress tasks) or from the oracle (the task's original wording). Records are not
touched, so under nuisance 2 a changed field order or separator still reaches the library: the wrapper repairs the
question interface only. Writes one JSON line per (library, condition, parser).
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import e39_analysis as E  # noqa: E402
import e61_parse as P  # noqa: E402
import family_gen as F  # noqa: E402
import stress_gen as S  # noqa: E402

TEMPLATE = {
    "count": "How many records have action {action} and item {item}?",
    "total": "How many units of {item} did {user} buy in total (action bought only)?",
    "top_user": "Which user has the most records with action {action}?",
    "nth_item": "What was the item in {user}'s {position} record (counting from the start of the log)?",
    "last_city": "In which city was {user}'s most recent record?",
    "first_after": "Which user made the first record at or after {time}?",
    "distinct": "How many different items did {user} have records with action {action} for?",
    "day_span": "How many whole days are there between the dates of {user}'s first and last records?",
    "semantic_total": "How many units of {category} items were bought in total (action bought only)?",
}
INSTR = re.compile(r"(Reply with the answer only\.|Reply with one line of the form 'ANSWER: <answer>'\.|"
                   r"Reply with a JSON object \{\"answer\": \"<answer>\"\} and nothing else\.|"
                   r"Begin your reply with the code \S+ immediately followed by the answer\.)\s*$")


def render(q: str, slots: dict) -> str | None:
    s = dict(slots)
    if "position" in s:
        try:
            s["position"] = F.ordinal(int(s["position"]))
        except (TypeError, ValueError):
            return None
    try:
        return TEMPLATE[q].format(**s)
    except KeyError:
        return None


def canonicalize(t, fid: str, q: str, slots: dict | None):
    """A copy of task t whose final paragraph/message is '<original label> <canonical question> <its instruction>'."""
    final = t.prompt[-1]["content"] if isinstance(t.prompt, list) else t.prompt.rsplit("\n\n", 1)[-1]
    m = INSTR.search(final)
    canon = render(q, slots) if slots else None
    if m is None or canon is None:
        return t, False
    new_final = f"{F.Family(fid).qlabel} {canon} {m.group(1)}"
    u = copy.copy(t)
    if isinstance(t.prompt, list):
        u.prompt = t.prompt[:-1] + [dict(t.prompt[-1], content=new_final)]
    else:
        u.prompt = t.prompt.rsplit("\n\n", 1)[0] + "\n\n" + new_final
    return u, True


def raw_slots(row: dict) -> dict | None:
    """Unnormalized parameters from an E61 row (case kept for dates), position as an integer."""
    try:
        js = json.loads(re.search(r"\{.*\}", row["raw"], re.S).group(0))
    except (AttributeError, ValueError):
        return None
    keys = P.SCHEMA[row["question"]]
    out = {k: str(js.get(k, "")).strip() for k in keys}
    if "position" in out:
        out["position"] = P.norm("position", out["position"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="RUN_DIR:RULE (a or b), as in e59_stress.py")
    p.add_argument("--families", type=Path, required=True)
    p.add_argument("--parses", action="append", required=True, help="LABEL=E61 rows file (e.g. 4b=.../qwen3-4b.jsonl)")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--conditions", default="paraphrase2,nuisance2")
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--concurrency", type=int, default=6)
    a = p.parse_args()
    fams = a.families.read_text().strip().split(",")
    parses = {}
    for spec in a.parses:
        label, path = spec.split("=", 1)
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            parses[(label, r["family"], r["condition"], r["idx"])] = raw_slots(r)
    jobs = []
    for spec in a.run:
        run, rule = spec.rsplit(":", 1)
        run = Path(run)
        for fid in fams:
            if not (run / fid / "summary.json").exists():
                continue
            c = (E.pick_a if rule == "a" else E.pick_b)(run / fid)
            if c is not None:
                jobs += [(run, rule, fid, c, cond, parser) for cond in a.conditions.split(",")
                         for parser in ["oracle"] + [s.split("=", 1)[0] for s in a.parses]]
    print(f"{len(jobs)} scorings", flush=True)
    a.work.mkdir(parents=True, exist_ok=True)
    lock = Lock()

    def one(job):
        run, rule, fid, c, cond, parser = job
        q = F.parse_family(fid)[1]
        tasks, ok = [], 0
        for i, t in enumerate(S.make(fid, cond)):
            slots = P.oracle(fid, i, q) if parser == "oracle" else parses.get((parser, fid, cond, i))
            if parser == "oracle":  # oracle parameters with the original case, from the control task's question
                ct = S.make(fid, "control", i + 1)[i]
                mm = re.match(f"^{S.PARAPHRASE[q][0]}$", ct.question)
                slots = {k: mm.group(g) for k, g in zip(P.SCHEMA[q], P.GROUP_ORDER[q], strict=True)}
                if "position" in slots:
                    slots["position"] = P.norm("position", slots["position"])
            u, done = canonicalize(t, fid, q, slots)
            tasks.append(u)
            ok += done
        entry = "answer_" + re.sub(r"\W", "_", fid)
        v = B.verify(c["code"], tasks, lambda out, t: F.grade(out, t), a.work, f"{run.name}-{fid}-{cond}-{parser}",
                     a.helper_base_url, "qwen3-4b", 900, entry)
        row = dict(run=run.name, rule=rule, family=fid, condition=cond, parser=parser, canonicalized=ok, mean=v["mean"],
                   success=v["mean"] >= 0.9, errors=sum(bool(d.get("error")) for d in v["details"]))
        with lock, a.out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
