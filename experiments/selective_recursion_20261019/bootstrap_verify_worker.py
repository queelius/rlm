"""Run one candidate library module on a list of teaching tasks in a separate process (E21).

Usage: bootstrap_verify_worker.py CANDIDATE.py TASKS.json OUT.json BASE_URL MODEL
Binds `ask` and `ask_batch` to the helper model, executes the candidate module, calls its entry
function on each task's request, and writes the raw outputs (and any exception) per task. The parent
enforces a wall-clock timeout on this process; scoring happens in the parent.
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


def main():
    cand, tasks_path, out_path, base_url, model = sys.argv[1:6]
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="local", timeout=600)
    pool = ThreadPoolExecutor(16)
    calls = {"n": 0}

    # KERNEL_SEMANTICS=1 mirrors the RLM kernel (src/rlm/executor.py): ask raises when the reply is not
    # "completed" or is empty (a reply cut at its token cap is a fault, not partial text), and ask_batch gives
    # None for such items. Without it (the default, kept for runs started earlier) partial text is returned.
    kernel = os.environ.get("KERNEL_SEMANTICS") == "1"

    def _one(p, system, options):
        options = options or {}
        calls["n"] += 1
        try:
            r = client.responses.create(model=model, input=p, instructions=system,
                                        temperature=options.get("temperature", 0),
                                        max_output_tokens=options.get("max_output_tokens", 1024))
        except Exception:  # noqa: BLE001
            if kernel:
                raise
            return ""
        text = r.output_text or ""
        if kernel and (r.status != "completed" or not text.strip()):
            return None
        return text

    def ask(prompt, system=None, options=None):
        out = _one(prompt, system, options)
        if out is None:
            raise RuntimeError("ask returned no usable text")
        return out

    def ask_batch(prompts, system=None, options=None):
        return SimpleNamespace(texts=list(pool.map(lambda p: _one(p, system, options), prompts)))

    meta = json.load(open(tasks_path))
    entry = meta["entry"]
    ns = {"ask": ask, "ask_batch": ask_batch}
    out = {"load_error": None, "results": []}
    try:
        exec(compile(open(cand).read(), "candidate", "exec"), ns)
        fn = ns[entry]
    except Exception:  # noqa: BLE001
        out["load_error"] = traceback.format_exc()[-1500:]
        json.dump(out, open(out_path, "w"))
        return
    for t in meta["tasks"]:
        t0 = time.time()
        n0 = calls["n"]
        try:
            ans = fn({"input": t["input"]})
            err = None
        except Exception:  # noqa: BLE001
            ans, err = None, traceback.format_exc()[-800:]
        out["results"].append(dict(task_id=t["task_id"], output=ans if isinstance(ans, str) else (None if ans is None else str(ans)),
                                   error=err, seconds=round(time.time() - t0, 1), helper_calls=calls["n"] - n0))
        json.dump(out, open(out_path, "w"))


if __name__ == "__main__":
    main()
