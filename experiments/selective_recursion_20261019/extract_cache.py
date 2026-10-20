"""Cache model-extracted events for event-log tasks (E08 library growth).

The librarian verifies candidate routines against these extracted events and the gold answers of
the teaching split, so a routine is accepted on the same evidence a deployed controller would see
(model extraction noise included), not on the generator's hidden events.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taskgen  # noqa: E402



def extract_system() -> str:
    ns = {"ask_batch": None, "ask": None}
    src = open(Path(__file__).resolve().parent / "skills" / "eventlog.py").read()
    exec(compile(src, "eventlog", "exec"), ns)
    return ns["EXTRACT_SYSTEM"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="teach")
    p.add_argument("--seeds", default="16-23")
    p.add_argument("--ns", default="8,32,64,128,256,512")
    p.add_argument("--qts", default=",".join(taskgen.HELDOUT_TYPES))
    p.add_argument("--chunk", type=int, default=12)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="qwen3-4b")
    a = p.parse_args()
    from openai import OpenAI

    client = OpenAI(base_url=a.base_url, api_key="local")
    system = extract_system()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = a.seeds.split("-")
    tasks = [taskgen.make_task(s, n, qt, split=a.split) for qt in a.qts.split(",") for n in map(int, a.ns.split(","))
             for s in range(int(lo), int(hi) + 1)]
    tasks = [t for t in tasks if not (out / f"{t.task_id}.json").exists()]
    print(f"{len(tasks)} tasks to extract", flush=True)

    def one(task):
        log = task.prompt.split("LOG (", 1)[1].split("\n", 1)[1].split("\n\nQUESTION:", 1)[0]
        lines = log.split("\n")
        chunks = ["\n".join(lines[i:i + a.chunk]) for i in range(0, len(lines), a.chunk)]
        events, usage = [], [0, 0]
        for c in chunks:
            r = client.responses.create(model=a.model, instructions=system, input="Lines:\n" + c + "\n\nJSON array:",
                                        temperature=0, max_output_tokens=2048)
            usage[0] += r.usage.input_tokens
            usage[1] += r.usage.output_tokens
            m = re.search(r"\[.*\]", r.output_text or "", re.S)
            if m:
                try:
                    events += [e for e in json.loads(m.group(0)) if isinstance(e, dict) and e.get("kind") in ("buy", "return")]
                except json.JSONDecodeError:
                    pass
        rec = dict(task_id=task.task_id, split=a.split, seed=task.seed, n=task.n_records, question_type=task.question_type,
                   question=task.question, answer=task.answer, answer_type=task.answer_type, events=events,
                   gold_events=task.events, prompt_tokens=usage[0], completion_tokens=usage[1], chunks=len(chunks))
        (out / f"{task.task_id}.json").write_text(json.dumps(rec))
        return task.task_id

    with ThreadPoolExecutor(a.concurrency) as ex:
        for k, tid in enumerate(ex.map(one, tasks), 1):
            if k % 8 == 0:
                print(f"[{k}/{len(tasks)}] {tid}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
