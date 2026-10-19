"""Scripted expert controller run through the real kernel to produce teaching trajectories.

The expert is a deterministic ModelBackend that returns Python cells for the controller role. The
kernel executes them in the real IPython environment and calls the real helper model, so every
recorded controller request (bootstrap, observations) and target cell is a genuine trajectory of
the harness. Outputs: per-run rows (accuracy and cost of the expert inside the harness) and
`sft.jsonl` with one (messages, completion) pair per controller turn of verified-correct runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_train  # noqa: E402
import run_sweep  # noqa: E402
import taskgen  # noqa: E402

from rlm import RLM, OpenAIEndpoint, response_output_text  # noqa: E402

import variants  # noqa: E402

CELLS = HERE / "expert_cells"


def response_object(text: str) -> dict:
    return {
        "id": "resp_" + uuid.uuid4().hex[:12], "object": "response", "created_at": int(time.time()),
        "status": "completed", "model": "expert-controller",
        "output": [{"id": "msg_" + uuid.uuid4().hex[:12], "type": "message", "status": "completed",
                    "role": "assistant", "content": [{"type": "output_text", "annotations": [], "text": text}]}],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


class ExpertController:
    """Deterministic controller: inspect, then branch on the question printed in the observation."""

    def __init__(self, variant: dict | None = None):
        self.variant = variant or variants.pick("", False)
        self.cells = {n: variants.render(CELLS, n, self.variant) for n in ("inspect", "lookup", "extract", "aggregate")}

    def complete(self, request, *, timeout, context):
        # The kernel shows only the bootstrap plus the latest assistant/observation pair, so the
        # stage is read from the observation's namespace summary, not from the item count.
        items = [it for it in request["input"] if it.get("role") == "user"]
        if len(items) == 1:
            return response_object(f"```python\n{self.cells['inspect']}\n```")
        last = json.loads(items[-1]["content"])
        execution = last.get("execution") or {}
        if execution.get("exception") or last.get("submission", {}).get("status") == "rejected":
            raise RuntimeError("expert cell failed: " + json.dumps(execution)[:500])
        namespace = execution.get("namespace") or {}
        if "events" in namespace:
            return response_object(f"```python\n{self.cells['aggregate']}\n```")
        if "question" in namespace:
            stdout = execution.get("stdout", "")
            question = stdout.split("\n", 1)[1].strip() if "\n" in stdout else ""
            if question.startswith("On which day did"):
                return response_object(f"```python\n{self.cells['lookup']}\n```")
            return response_object(f"```python\n{self.cells['extract']}\n```")
        raise RuntimeError("expert cannot determine its stage: " + json.dumps(namespace)[:300])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ns", default="8,32,64,128,256,512")
    p.add_argument("--qts", default=",".join(taskgen.QUESTION_TYPES))
    p.add_argument("--seeds", default="0-3")
    p.add_argument("--split", default="teach")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--public-model", default="qwen3-4b")
    p.add_argument("--harness", default="default", choices=("default", "hinted"))
    p.add_argument("--diversify", action="store_true", help="sample chunk size, prompt phrasing and filter per task")
    p.add_argument("--unified", action="store_true", help="family-agnostic peek first (expert.UnifiedExpert)")
    p.add_argument("--route-direct-prob", type=float, default=0.0, help="teach the direct route on short inputs with this probability")
    p.add_argument("--max-subcalls", type=int, default=80)
    p.add_argument("--max-model-calls", type=int, default=96)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.seeds.split("-")
    tasks = [taskgen.make_task(s, int(n), qt, split=args.split) for n in args.ns.split(",")
             for qt in args.qts.split(",") for s in range(int(lo), int(hi) + 1)]
    ra = SimpleNamespace(base_url=args.base_url, api_key=args.api_key, public_model=args.public_model,
                         harness=args.harness, controller_max_output=2048, max_turns=6, max_model_calls=args.max_model_calls,
                         max_subcalls=args.max_subcalls, max_total_tokens=400_000, deadline=900.0, exec_timeout=540.0, temperature=0.0)
    trace_dir = args.out / "traces"
    rows_path, sft_path = args.out / "rows.jsonl", args.out / "sft.jsonl"
    lock = threading.Lock()

    def one(task):
        variant = variants.pick(task.task_id, args.diversify, short=task.n_records <= 32, route_direct_prob=args.route_direct_prob)
        row = dict(task_id=task.task_id, n=task.n_records, question_type=task.question_type, seed=task.seed,
                   answer=task.answer, policy="expert", variant=variant)
        rlm = run_sweep.make_rlm(ra, trace_dir, "expert-controller")
        rlm = RLM(rlm.backend, config=rlm.config, controller_backend=(__import__("expert").UnifiedExpert(variant) if args.unified else ExpertController(variant)))
        try:
            result = rlm.run(run_sweep.public_request(task, args.public_model, 0.0, 4096))
            row.update(output_text=response_output_text(result.response), total_tokens=result.usage.total_tokens,
                       prompt_tokens=result.usage.input_tokens, completion_tokens=result.usage.output_tokens,
                       calls=result.usage.calls, turns=result.turns, duration=result.duration_seconds,
                       trace_dir=str(result.trace_directory), error=None)
            row["correct"] = taskgen.verify(task, row["output_text"])
        except Exception as exc:  # noqa: BLE001
            row.update(correct=False, output_text=None, error=f"{type(exc).__name__}: {exc}"[:600], trace_dir=None)
        finally:
            rlm.close()
        pairs = []
        if row.get("trace_dir"):
            for k, t in enumerate(rl_train.controller_turns(row["trace_dir"])):
                pairs.append(dict(task_id=task.task_id, n=task.n_records, question_type=task.question_type,
                                  turn=k, correct=row["correct"], messages=t["messages"], completion=t["completion"],
                                  variant=variant))
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            with sft_path.open("a") as f:
                for pr in pairs:
                    f.write(json.dumps(pr) + "\n")
        return row

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, tasks))
    print(json.dumps(rl_train.summarize(rows), indent=1))
    errs = [r["error"] for r in rows if r.get("error")]
    print(len(errs), "errors;", errs[:3])


if __name__ == "__main__":
    main()
