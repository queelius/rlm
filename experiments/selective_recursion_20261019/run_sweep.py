"""Paired policy sweep on event-log tasks: direct, RLM controller, programmed decomposition.

Every policy sees the same public request (same prompt, model, sampling). Costs are counted
from the serving endpoint's reported usage (prompt + completion tokens over every model call a
policy makes). Results are appended as JSONL rows; a summary table is printed at the end.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import taskgen  # noqa: E402

from rlm import (  # noqa: E402
    RLM,
    ControllerConfig,
    ExecutionConfig,
    OpenAIEndpoint,
    RLMConfig,
    RunLimits,
    TraceConfig,
    response_output_text,
)
from rlm.prompts import default_harness_spec  # noqa: E402

CHUNK_LINES = 12
EXTRACT_SYSTEM = (
    "You classify store log lines. For EVERY numbered input line output exactly one JSON object, in "
    "the same order, inside one JSON array. Object format: "
    '{"line": <line number>, "kind": "buy" | "return" | "none", "person": <name or null>, '
    '"item": <singular item noun or null>, "qty": <integer or null>, "day": <integer or null>}. '
    'Use "buy" only for a completed purchase (bought, purchased, paid for, picked up per a receipt, '
    "a sale logged by the register, or an order that went through). Use \"return\" only for a "
    "completed return or refund of items. Everything else (complaints, reviews, recommendations, "
    "wishlists, considering or declining a purchase, almost returning, stock questions, store "
    'notices, deliveries) is "none". Output only the JSON array.'
)
KIND_SYNONYMS = {"buy": "buy", "purchase": "buy", "bought": "buy", "sale": "buy", "purchased": "buy",
                 "order": "buy", "return": "return", "refund": "return", "returned": "return"}


def public_request(task, model: str, temperature: float, max_output_tokens: int) -> dict:
    return {
        "model": model,
        "input": task.prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }


HINTED_POLICY = """\
Use the least expensive reliable path. For a short input, delegating the whole request with
model_complete(request) is usually enough. For a long input made of many records, one model call
is unreliable: split the records into chunks in Python, use ask_batch to extract the needed facts
from each chunk as JSON, parse and aggregate them exactly in Python, and submit the computed answer.
"""


def hinted_harness_spec():
    from rlm import prompts as P
    from rlm.specs import HarnessSpec, PromptDigests, PromptSpec

    prompt = PromptSpec(name="hinted", version="1", protocol=P.DEFAULT_PROTOCOL, policy=HINTED_POLICY)
    return HarnessSpec(
        prompt=prompt,
        rendered_prompts=PromptDigests(
            nonrecursive=P._prompt_digest(P._render_prompt(prompt, allow_recursion=False)),
            recursive=P._prompt_digest(P._render_prompt(prompt, allow_recursion=True)),
        ),
        abi_version=P.ENVIRONMENT_ABI.version,
        abi_digest=P.ENVIRONMENT_ABI.digest(),
    )


def skills_harness_spec(task=None):
    """Per-task harness: the retrieved library skills prebound through the harness prelude (L1 proper).

    The default policy text is kept so a taught controller sees its usual prompt plus the prelude
    listing and a short usage note; the skill source never appears in the prompt."""
    from rlm import prompts as P
    from rlm.specs import HarnessSpec, PromptDigests, PromptSpec
    from skills import registry

    names = registry.retrieve(registry.signature(task)) if task is not None else []
    prelude = registry.prelude_for(names)
    note = registry.policy_note(names) if names else ""
    policy = P.DEFAULT_POLICY + ("\n" + note + "\n" if note else "")
    prompt = PromptSpec(name="skills", version="2", protocol=P.DEFAULT_PROTOCOL, policy=policy)
    return HarnessSpec(
        prompt=prompt,
        rendered_prompts=PromptDigests(
            nonrecursive=P._prompt_digest(P._render_prompt(prompt, allow_recursion=False, prelude=prelude)),
            recursive=P._prompt_digest(P._render_prompt(prompt, allow_recursion=True, prelude=prelude)),
        ),
        abi_version=P.ENVIRONMENT_ABI.version,
        abi_digest=P.ENVIRONMENT_ABI.digest(),
        prelude=prelude,
    )


def exemplars_harness_spec(task=None):
    """Per-task harness: the hinted policy plus the cells of one matching teaching trajectory."""
    from rlm import prompts as P
    from rlm.specs import HarnessSpec, PromptDigests, PromptSpec
    from skills import exemplars, registry

    block = exemplars.hint_block(registry.signature(task)) if task is not None else ""
    policy = HINTED_POLICY + ("\n" + block + "\n" if block else "")
    prompt = PromptSpec(name="exemplars", version="1", protocol=P.DEFAULT_PROTOCOL, policy=policy)
    return HarnessSpec(
        prompt=prompt,
        rendered_prompts=PromptDigests(
            nonrecursive=P._prompt_digest(P._render_prompt(prompt, allow_recursion=False)),
            recursive=P._prompt_digest(P._render_prompt(prompt, allow_recursion=True)),
        ),
        abi_version=P.ENVIRONMENT_ABI.version,
        abi_digest=P.ENVIRONMENT_ABI.digest(),
    )


HARNESSES = {"default": default_harness_spec, "hinted": hinted_harness_spec, "skills": skills_harness_spec,
             "exemplars": exemplars_harness_spec}


def make_rlm(args, trace_dir: Path, controller_model: str, task=None) -> RLM:
    spec = HARNESSES[args.harness]
    config = RLMConfig(
        harness=spec(task) if args.harness in ("skills", "exemplars") else spec(),
        limits=RunLimits(
            max_turns=args.max_turns,
            max_model_calls=args.max_model_calls,
            max_subcalls=args.max_subcalls,
            max_parallel_model_calls=8,
            max_depth=0,
            max_total_tokens=args.max_total_tokens,
            deadline_seconds=args.deadline,
            execution_timeout_seconds=getattr(args, "exec_timeout", 540.0),
        ),
        controller=ControllerConfig(
            model=controller_model,
            options={"temperature": args.temperature, "max_output_tokens": args.controller_max_output},
        ),
        execution=ExecutionConfig(working_directory=None),
        tracing=TraceConfig(enabled=True, directory=str(trace_dir), markdown=False),
    )
    return RLM(OpenAIEndpoint(base_url=args.base_url, api_key=args.api_key, timeout=args.deadline), config=config)


def run_direct(args, task, trace_dir):
    """One public-model call with the exact public request; usage is kept even when truncated."""
    start = time.monotonic()
    body = public_request(task, args.public_model, args.temperature, args.direct_max_output)
    r = http_client(args).post(
        f"{args.base_url}/responses",
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        json=body,
    )
    r.raise_for_status()
    data = r.json()
    u = data.get("usage") or {}
    return dict(
        output_text=response_output_text(data),
        prompt_tokens=int(u.get("input_tokens", 0)),
        completion_tokens=int(u.get("output_tokens", 0)),
        total_tokens=int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0)),
        calls=1,
        turns=1,
        duration=time.monotonic() - start,
        stop_reason=data.get("status"),
        incomplete=(data.get("incomplete_details") or {}).get("reason"),
    )


def run_rlm(args, task, trace_dir):
    rlm = make_rlm(args, trace_dir, args.controller_model, task=task)
    try:
        result = rlm.run(public_request(task, args.public_model, args.temperature, args.direct_max_output))
    finally:
        rlm.close()
    return dict(
        output_text=response_output_text(result.response),
        prompt_tokens=result.usage.input_tokens,
        completion_tokens=result.usage.output_tokens,
        total_tokens=result.usage.total_tokens,
        calls=result.usage.calls,
        unreported_calls=result.usage.unreported_calls,
        turns=result.turns,
        duration=result.duration_seconds,
        stop_reason=result.stop_reason,
        trace_dir=str(result.trace_directory),
        run_id=result.run_id,
        harness_fingerprint=result.harness_fingerprint,
    )


_http = threading.local()


def http_client(args) -> httpx.Client:
    client = getattr(_http, "client", None)
    if client is None:
        client = httpx.Client(timeout=args.deadline)
        _http.client = client
    return client


def helper_call(args, system: str, prompt: str, max_output_tokens: int) -> tuple[str, dict]:
    body = {
        "model": args.public_model,
        "instructions": system,
        "input": prompt,
        "temperature": args.temperature,
        "max_output_tokens": max_output_tokens,
    }
    r = http_client(args).post(
        f"{args.base_url}/responses",
        headers={"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"},
        json=body,
    )
    r.raise_for_status()
    data = r.json()
    return response_output_text(data), data.get("usage") or {}


def singular(item: str) -> str:
    item = item.strip().lower()
    if item in taskgen.ITEMS:
        return item
    if item.endswith("es") and item[:-2] in taskgen.ITEMS:
        return item[:-2]
    if item.endswith("s") and item[:-1] in taskgen.ITEMS:
        return item[:-1]
    return item


def parse_events(text: str) -> list[dict] | None:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        try:
            kind = KIND_SYNONYMS.get(str(it.get("kind", "")).lower().strip())
            if kind is None or it.get("person") is None or it.get("item") is None:
                continue
            out.append(
                dict(kind=kind, person=str(it.get("person", "")).strip(),
                     item=singular(str(it.get("item", ""))),
                     qty=int(it.get("qty", 1)), day=int(re.sub(r"\D", "", str(it.get("day", 0))) or 0))
            )
        except (TypeError, ValueError):
            continue
    return out


def aggregate(task, events: list[dict]) -> str:
    q = task.question
    # Recover the question's target items from the question text (same wording as the generator).
    items = re.findall(r"\b(" + "|".join(taskgen.ITEMS) + r")\b", q)
    a = items[0]
    b = items[1] if len(items) > 1 and task.question_type == "both_items" else None
    if task.question_type == "lookup_day":
        person = re.findall(r"\b(" + "|".join(taskgen.NAMES) + r")\b", q)[0]
        hits = sorted(e["day"] for e in events if e["kind"] == "buy" and e["item"] == a and e["person"] == person)
        return str(hits[0]) if hits else "unknown"
    buys = [e for e in events if e["kind"] == "buy"]
    rets = [e for e in events if e["kind"] == "return"]
    qt = task.question_type
    if qt == "count_buyers":
        return str(len({e["person"] for e in buys if e["item"] == a}))
    if qt == "both_items":
        both = sorted({e["person"] for e in buys if e["item"] == a} & {e["person"] for e in buys if e["item"] == b})
        return ", ".join(both) if both else "none"
    if qt == "net_units":
        return str(sum(e["qty"] for e in buys if e["item"] == a) - sum(e["qty"] for e in rets if e["item"] == a))
    if qt == "bought_then_returned":
        first = {}
        for e in buys:
            if e["item"] == a:
                first[e["person"]] = min(first.get(e["person"], 10**9), e["day"])
        people = sorted({e["person"] for e in rets if e["item"] == a and e["person"] in first and e["day"] > first[e["person"]]})
        return ", ".join(people) if people else "none"
    raise ValueError(qt)


def run_fixed(args, task, trace_dir):
    """Programmed split: chunk log lines, extract events per chunk with the helper, aggregate exactly."""
    start = time.monotonic()
    log = task.prompt.split("LOG (", 1)[1].split("\n", 1)[1].split("\n\nQUESTION:", 1)[0]
    lines = log.split("\n")
    chunks = ["\n".join(lines[i:i + CHUNK_LINES]) for i in range(0, len(lines), CHUNK_LINES)]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    parse_failures = 0
    events: list[dict] = []

    def one(chunk):
        return helper_call(args, EXTRACT_SYSTEM, "Lines:\n" + chunk + "\n\nJSON array:", 2048)

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for text, u in pool.map(one, chunks):
            usage["prompt_tokens"] += int(u.get("input_tokens", 0))
            usage["completion_tokens"] += int(u.get("output_tokens", 0))
            parsed = parse_events(text)
            if parsed is None:
                parse_failures += 1
            else:
                events.extend(parsed)
    answer = aggregate(task, events)
    return dict(
        output_text=answer,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["prompt_tokens"] + usage["completion_tokens"],
        turns=len(chunks),
        duration=time.monotonic() - start,
        stop_reason="completed",
        helper_calls=len(chunks),
        parse_failures=parse_failures,
        extracted_events=len(events),
    )


POLICIES = {"direct": run_direct, "rlm": run_rlm, "fixed": run_fixed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policies", default="direct,rlm,fixed")
    p.add_argument("--ns", default="8,32,128,512")
    p.add_argument("--qts", default=",".join(taskgen.QUESTION_TYPES))
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--split", default="train")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--public-model", default="qwen3-4b")
    p.add_argument("--controller-model", default="qwen3-4b")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--direct-max-output", type=int, default=4096)
    p.add_argument("--harness", default="default", choices=("default", "hinted", "skills", "exemplars"))
    p.add_argument("--controller-max-output", type=int, default=1024)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-model-calls", type=int, default=96)
    p.add_argument("--max-subcalls", type=int, default=80)
    p.add_argument("--max-total-tokens", type=int, default=400_000)
    p.add_argument("--deadline", type=float, default=900.0)
    p.add_argument("--exec-timeout", type=float, default=540.0, help="per-cell IPython execution cap")
    p.add_argument("--limit", type=int, default=0, help="debug: only first K jobs")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out / "traces"
    rows_path = args.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task_id"], r["policy"], r.get("harness")))
    lo, hi = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in args.seeds.split(",")]
    tasks = [taskgen.make_task(s, int(n), qt, split=args.split)
             for n in args.ns.split(",") for qt in args.qts.split(",") for s in seeds]
    jobs = [(t, pol) for t in tasks for pol in args.policies.split(",")
            if (t.task_id, pol, args.harness if pol == "rlm" else None) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    (args.out / "config.json").write_text(json.dumps(vars(args) | {"out": str(args.out)}, indent=2, default=str))
    print(f"{len(tasks)} tasks, {len(jobs)} jobs pending", flush=True)
    lock = threading.Lock()

    def work(job):
        task, policy = job
        started = time.time()
        row = dict(task_id=task.task_id, n=task.n_records, question_type=task.question_type, seed=task.seed,
                   policy=policy, harness=(args.harness if policy == "rlm" else None),
                   answer=task.answer, answer_type=task.answer_type,
                   support_records=task.support_records, started=started)
        try:
            row.update(POLICIES[policy](args, task, trace_dir))
            row["correct"] = taskgen.verify(task, row["output_text"])
            row["error"] = None
        except Exception as exc:  # noqa: BLE001 - keep every failure as a row
            row.update(correct=False, output_text=None, error=f"{type(exc).__name__}: {exc}"[:2000],
                       traceback=traceback.format_exc()[-3000:])
        row["wall"] = time.time() - started
        with lock:
            with rows_path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for k, row in enumerate(pool.map(work, jobs), 1):
            if k % 10 == 0 or k == len(jobs):
                print(f"[{k}/{len(jobs)}] last: {row['policy']} n={row['n']} {row['question_type']} "
                      f"correct={row['correct']} tok={row.get('total_tokens')} err={bool(row.get('error'))}", flush=True)
    summarize(rows_path)


def summarize(rows_path: Path):
    import collections
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    agg = collections.defaultdict(lambda: dict(n=0, c=0, tok=0, err=0))
    for r in rows:
        k = (r["policy"] + ("/" + r["harness"] if r.get("harness") else ""), r["n"], r["question_type"])
        a = agg[k]; a["n"] += 1; a["c"] += bool(r["correct"]); a["tok"] += int(r.get("total_tokens") or 0); a["err"] += bool(r.get("error"))
    print("\npolicy            n   question_type          acc    mean_tokens  errors")
    for k in sorted(agg):
        a = agg[k]
        print(f"{k[0]:<13} {k[1]:>5}  {k[2]:<22} {a['c']/a['n']:.2f}  {a['tok']/a['n']:>11.0f}  {a['err']}")


if __name__ == "__main__":
    main()
