"""E69: can a small proposer edit a working library to parse its question with the helper?

For each family where a frozen library is correct (E39 arm A, boot seed 0), shows the proposer that library's code,
the same three examples it was bootstrapped from, and an instruction to replace its question parsing with one
helper call (question only, JSON parameters out) while keeping the rest. Draws K candidates; accepts the first that
runs on all 30 teaching inputs with varied outputs and reproduces the three shown examples (arm A's rule); scores the
accepted edit on the stress conditions (e59_stress.py's tasks). Writes candidates and one summary line per family.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import e39_analysis as E  # noqa: E402
import family_gen as F  # noqa: E402
import stress_gen as S  # noqa: E402

INSTRUCTION = (
    "\n\nThe library below is correct on inputs worded exactly like these examples, but in use the question may be "
    "worded differently (a paraphrase with the same meaning), its label may differ (for example 'Question:', "
    "'QUESTION:', 'Q:' or 'Task:'), and record fields may appear in a different order or with a different separator. "
    "Edit the library so that it no longer parses the question with patterns copied from the examples: find the "
    "question (the final paragraph or message), call ask() once, passing only the question text and the parameter "
    "keys your code needs (no records), asking for a JSON object of parameter values only; decode it with json.loads; "
    "then compute and format the answer exactly as the existing code does. Keep everything else that works. Return "
    "the complete edited module in one ```python block.\n\nCurrent library:\n```python\n{code}\n```")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True, help="frozen arm-A run whose libraries are edited")
    p.add_argument("--families", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--helper-base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--max-output-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=4)
    a = p.parse_args()
    fams = a.families.read_text().strip().split(",")
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=1800)
    a.out.mkdir(parents=True, exist_ok=True)

    def one(fid):
        d = a.out / fid
        if (d / "summary.json").exists():
            return
        src = E.pick_a(a.source_run / fid)
        if src is None:
            return
        d.mkdir(parents=True, exist_ok=True)
        api = B.family_api(fid)
        shown_ids = json.loads((a.source_run / fid / "summary.json").read_text())["shown_examples"]
        shown = [t for t in api["teach"] if t.task_id in shown_ids]
        entry = "answer_" + re.sub(r"\W", "_", fid)
        prompt = B.build_prompt(entry, [api["describe"](t) for t in shown]) + INSTRUCTION.format(code=src["code"])
        (d / "prompt.txt").write_text(prompt)
        accepted = None
        log = (d / "candidates.jsonl").open("a")
        for k in range(a.candidates):
            r = client.responses.create(model=a.model, input=prompt, temperature=0.7, max_output_tokens=a.max_output_tokens)
            code = B.extract_code(r.output_text)
            v = B.verify(code, api["teach"], api["grade"], d, f"k{k}", a.helper_base_url, "qwen3-4b", 600, entry)
            rec = dict(round=0, k=k, code=code, usage=dict(input=r.usage.input_tokens, output=r.usage.output_tokens), **v)
            log.write(json.dumps(rec, default=str) + "\n")
            log.flush()
            if accepted is None and E.valid(rec) and all(
                    dd["score"] >= 0.999 for dd in rec["details"] if dd["task_id"] in shown_ids):
                accepted = rec
                break
        row = dict(family=fid, accepted=accepted is not None, k=accepted["k"] if accepted else None,
                   uses_helper=bool(accepted and "ask(" in accepted["code"] and "json.loads" in accepted["code"]),
                   shown_examples=shown_ids)
        if accepted:
            for cond in S.CONDITIONS:
                v = B.verify(accepted["code"], S.make(fid, cond), lambda out, t: F.grade(out, t), d, f"stress-{cond}",
                             a.helper_base_url, "qwen3-4b", 900, entry)
                row[cond] = v["mean"]
        (d / "summary.json").write_text(json.dumps(row))
        print(json.dumps(row), flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, fams))


if __name__ == "__main__":
    main()
