"""On-policy RL (REINFORCE with leave-one-out baseline) for the RLM controller on event-log tasks.

Rollouts: the controller policy is a LoRA adapter served by vLLM (hot-swapped after every update);
helpers use the frozen base model. Reward = correctness minus a token-cost penalty. Training:
exact controller turns are rebuilt from the kernel traces (chat template verified to reproduce
vLLM's prompt token counts), and one gradient update per batch is applied to the adapter on the
training GPU. Everything a later analysis needs is logged as JSON per step and per rollout.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oolong_eval  # noqa: E402
import run_sweep  # noqa: E402
import taskgen  # noqa: E402

MODEL = os.environ.get("SR_MODEL", "/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554")
DEVICE = os.environ.get("SR_DEVICE", "cuda:0")


# ----------------------------------------------------------------------------- rollouts
def rollout_args(args, controller_model: str, temperature: float) -> SimpleNamespace:
    return SimpleNamespace(
        base_url=args.base_url, api_key=args.api_key, public_model=args.public_model,
        controller_model=controller_model, temperature=temperature, harness=args.harness,
        direct_max_output=4096, controller_max_output=args.controller_max_output,
        max_turns=args.max_turns, max_model_calls=args.max_model_calls, max_subcalls=args.max_subcalls,
        max_total_tokens=args.max_total_tokens, deadline=args.deadline, exec_timeout=args.exec_timeout,
    )


def controller_turns(trace_dir: str) -> list[dict]:
    """Exact (messages, completion_text, usage) per controller call, in order."""
    reqs, turns = {}, []
    with open(Path(trace_dir) / "trace.jsonl") as f:
        for line in f:
            ev = json.loads(line)
            p = ev.get("payload", {})
            ctx = p.get("context", {})
            if ctx.get("role") != "controller":
                continue
            if ev["type"] == "model.request":
                reqs[ctx["call_id"]] = p["request"]
            elif ev["type"] == "model.response":
                req = reqs.get(ctx["call_id"])
                if req is None:
                    continue
                msgs = [{"role": "system", "content": req["instructions"]}]
                for it in req["input"]:
                    if it.get("type") in (None, "message"):
                        c = it["content"]
                        if isinstance(c, list):
                            c = "".join(x.get("text", "") for x in c)
                        msgs.append({"role": it["role"], "content": c})
                resp = p["response"]
                text = "".join(c.get("text", "") for o in resp.get("output", []) if o.get("type") == "message"
                               for c in o.get("content", []))
                turns.append(dict(messages=msgs, completion=text, usage=resp.get("usage") or {},
                                  status=resp.get("status")))
    return turns


def correct_for(task, text) -> bool:
    row = getattr(task, "row", None)
    return oolong_eval.verify(row, text or "") if row is not None else taskgen.verify(task, text or "")


_OOLONG_CACHE: dict = {}


def oolong_tasks(split: str, max_len: int) -> list:
    key = (split, max_len)
    if key not in _OOLONG_CACHE:
        import teach_oolong
        _OOLONG_CACHE[key] = [teach_oolong.as_task(r) for r in teach_oolong.load_rows(split, max_len)]
    return _OOLONG_CACHE[key]


def step_tasks(args, step: int) -> list:
    rng = random.Random(f"{args.seed}:{step}")
    tasks = []
    if args.task_source in ("eventlog", "mixed"):
        cells = [(n, qt) for n in args.ns for qt in args.qts]
        count = args.tasks_per_step if args.task_source == "eventlog" else args.tasks_per_step // 2
        for i in range(count):
            n, qt = cells[(step * count + i) % len(cells)]
            tasks.append(taskgen.make_task(rng.randrange(10**6, 10**9), n, qt, split="rl"))
    if args.task_source in ("oolong", "mixed"):
        pool = oolong_tasks("train", args.oolong_max_len)
        by_len: dict = {}
        for t in pool:
            by_len.setdefault(t.n_records, []).append(t)
        lens = sorted(by_len)
        count = args.tasks_per_step if args.task_source == "oolong" else args.tasks_per_step - len(tasks)
        for i in range(count):
            tasks.append(rng.choice(by_len[lens[(step * count + i) % len(lens)]]))
    return tasks


def reward_of(row: dict, args) -> float:
    """Correctness minus a bounded token-cost penalty; failed runs pay the cap."""
    correct = 1.0 if row.get("correct") else 0.0
    tokens = float(row.get("total_tokens") or args.token_cap)
    return correct - args.lam * min(tokens, args.token_cap) / 10_000.0


def collect_rollouts(args, step: int, controller_model: str, out: Path) -> list[dict]:
    tasks = step_tasks(args, step)
    trace_dir = out / "traces" / f"step-{step:04d}"
    ra = rollout_args(args, controller_model, args.temperature)
    jobs = [(t, k) for t in tasks for k in range(args.group)]

    def one(job):
        task, k = job
        row = dict(task_id=task.task_id, n=task.n_records, question_type=task.question_type,
                   seed=task.seed, sample=k, answer=task.answer, step=step, controller=controller_model,
                   source="oolong" if getattr(task, "row", None) is not None else "eventlog")
        try:
            row.update(run_sweep.run_rlm(ra, task, trace_dir))
            row["correct"] = correct_for(task, row["output_text"])
            row["error"] = None
        except Exception as exc:  # noqa: BLE001
            row.update(correct=False, output_text=None, error=f"{type(exc).__name__}: {exc}"[:500],
                       total_tokens=None, trace_dir=None)
        row["reward"] = reward_of(row, args)
        return row

    with cf.ThreadPoolExecutor(max_workers=args.rollout_concurrency) as pool:
        rows = list(pool.map(one, jobs))
    # leave-one-out baseline within each task group
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    for group in by_task.values():
        rs = [r["reward"] for r in group]
        for r in group:
            others = [x for x in rs if x is not r["reward"]] if len(rs) > 1 else []
            n = len(rs)
            r["advantage"] = (r["reward"] - (sum(rs) - r["reward"]) / (n - 1)) if n > 1 else 0.0
            r["group_mean_reward"] = sum(rs) / n
    return rows


def completion_logprobs(model, prompt: list[int], comp: list[int], torch):
    """Per-token log-probabilities of the completion given the prompt, keeping only C+1 logits."""
    ids = torch.tensor([prompt + comp], device=DEVICE)
    out = model(input_ids=ids, logits_to_keep=len(comp) + 1)
    logits = out.logits[0, :-1, :].float()  # positions P-1 .. L-2 predict comp[0..C-1]
    target = torch.tensor(comp, device=DEVICE)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    del out, logits
    return logp


# ----------------------------------------------------------------------------- training
class Trainer:
    def __init__(self, args, out: Path):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": DEVICE}
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if args.init_adapter:
            self.model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        else:
            cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                             task_type="CAUSAL_LM")
            self.model = get_peft_model(model, cfg)
        self.model.enable_input_require_grads()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.n_params = sum(p.numel() for p in params)
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")

    def encode(self, turn: dict) -> tuple[list[int], list[int]]:
        ids = self.tok.apply_chat_template(turn["messages"], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        comp = self.tok(turn["completion"], add_special_tokens=False)["input_ids"] + [self.eos]
        return list(ids), comp

    def update(self, rows: list[dict]) -> dict:
        torch = self.torch
        seqs = []
        skipped_long = skipped_zero = 0
        for r in rows:
            if not r.get("trace_dir") or abs(r.get("advantage", 0.0)) < 1e-8:
                skipped_zero += 1
                continue
            for t in controller_turns(r["trace_dir"]):
                if not t["completion"]:
                    continue
                prompt, comp = self.encode(t)
                if len(prompt) + len(comp) > self.args.max_seq_len:
                    skipped_long += 1
                    continue
                seqs.append((prompt, comp, float(r["advantage"])))
        total_tokens = sum(len(c) for _, c, _ in seqs)
        if not seqs:
            return dict(sequences=0, skipped_long=skipped_long, skipped_zero_adv=skipped_zero, loss=None)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss_sum, logp_sum = 0.0, 0.0
        random.shuffle(seqs)
        for prompt, comp, adv in seqs:
            tok_logp = completion_logprobs(self.model, prompt, comp, torch)
            loss = -(adv * tok_logp.sum()) / total_tokens
            loss.backward()
            loss_sum += loss.item()
            logp_sum += tok_logp.sum().item()
            del tok_logp
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.args.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        (torch.cuda.empty_cache() if DEVICE.startswith("cuda") else None)
        return dict(sequences=len(seqs), completion_tokens=total_tokens, loss=loss_sum,
                    mean_token_logp=logp_sum / max(total_tokens, 1), grad_norm=float(grad_norm),
                    skipped_long=skipped_long, skipped_zero_adv=skipped_zero)

    def save(self, path: Path):
        self.model.save_pretrained(str(path))


# ----------------------------------------------------------------------------- serving control
def load_adapter(args, name: str, path: Path):
    r = httpx.post(f"{args.base_url}/load_lora_adapter", headers={"Authorization": f"Bearer {args.api_key}"},
                   json={"lora_name": name, "lora_path": str(path)}, timeout=300)
    if r.status_code >= 300:
        raise RuntimeError(f"load_lora_adapter {name}: {r.status_code} {r.text[:300]}")
    # verify the adapter answers
    r2 = httpx.post(f"{args.base_url}/responses", headers={"Authorization": f"Bearer {args.api_key}"},
                    json={"model": name, "input": "Reply with exactly READY.", "max_output_tokens": 8, "temperature": 0}, timeout=300)
    r2.raise_for_status()


def unload_adapter(args, name: str):
    httpx.post(f"{args.base_url}/unload_lora_adapter", headers={"Authorization": f"Bearer {args.api_key}"},
               json={"lora_name": name}, timeout=300)


# ----------------------------------------------------------------------------- evaluation
def dev_tasks(args) -> list:
    tasks = []
    if args.task_source in ("eventlog", "mixed"):
        tasks += [taskgen.make_task(s, n, qt, split="dev") for n in args.ns for qt in args.qts
                  for s in range(args.dev_seed, args.dev_seed + args.dev_per_cell)]
    if args.task_source in ("oolong", "mixed"):
        pool = oolong_tasks("test", args.oolong_max_len)
        by_len: dict = {}
        for t in pool:
            by_len.setdefault(t.n_records, []).append(t)
        rng = random.Random(args.dev_seed)
        for n in sorted(by_len):
            cell = sorted(by_len[n], key=lambda t: t.task_id)
            rng.shuffle(cell)
            tasks += cell[: args.oolong_dev_per_len]
    return tasks


def evaluate(args, controller_model: str, out: Path, step: int) -> dict:
    tasks = dev_tasks(args)
    ra = rollout_args(args, controller_model, 0.0)
    trace_dir = out / "traces" / f"dev-{step:04d}"

    def one(task):
        row = dict(task_id=task.task_id, n=task.n_records, question_type=task.question_type, answer=task.answer,
                   source="oolong" if getattr(task, "row", None) is not None else "eventlog")
        try:
            row.update(run_sweep.run_rlm(ra, task, trace_dir))
            row["correct"] = correct_for(task, row["output_text"])
            row["error"] = None
        except Exception as exc:  # noqa: BLE001
            row.update(correct=False, output_text=None, error=f"{type(exc).__name__}: {exc}"[:300], total_tokens=None)
        row["reward"] = reward_of(row, args)
        return row

    with cf.ThreadPoolExecutor(max_workers=args.rollout_concurrency) as pool:
        rows = list(pool.map(one, tasks))
    with (out / "dev" / f"step-{step:04d}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    by_n: dict[int, dict] = {}
    for r in rows:
        a = by_n.setdefault(r["n"], dict(n=0, correct=0, tokens=0, reward=0.0, errors=0, calls=0))
        a["n"] += 1; a["correct"] += bool(r.get("correct")); a["tokens"] += int(r.get("total_tokens") or 0)
        a["reward"] += float(r.get("reward") or 0.0); a["errors"] += bool(r.get("error")); a["calls"] += int(r.get("calls") or 0)
    table = {n: dict(acc=a["correct"] / a["n"], mean_tokens=a["tokens"] / a["n"], mean_reward=a["reward"] / a["n"],
                     mean_calls=a["calls"] / a["n"], errors=a["errors"], count=a["n"]) for n, a in sorted(by_n.items())}
    return dict(accuracy=sum(bool(r.get("correct")) for r in rows) / max(len(rows), 1),
                mean_reward=sum(float(r.get("reward") or 0.0) for r in rows) / max(len(rows), 1),
                mean_tokens=sum(int(r.get("total_tokens") or 0) for r in rows) / max(len(rows), 1), by_n=table)


# ----------------------------------------------------------------------------- main loop
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--tasks-per-step", type=int, default=16)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--ns", default="8,32,64,128,256,512")
    p.add_argument("--qts", default=",".join(taskgen.QUESTION_TYPES))
    p.add_argument("--task-source", default="eventlog", choices=("eventlog", "oolong", "mixed"))
    p.add_argument("--oolong-max-len", type=int, default=8192)
    p.add_argument("--oolong-dev-per-len", type=int, default=6)
    p.add_argument("--lam", type=float, default=0.1, help="reward penalty per 10k total tokens")
    p.add_argument("--token-cap", type=float, default=50_000, help="tokens charged to failed runs; also caps the penalty")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--harness", default="default", choices=("default", "hinted"))
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=12000)
    p.add_argument("--init-adapter", default=None)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--dev-seed", type=int, default=1000)
    p.add_argument("--dev-per-cell", type=int, default=2)
    p.add_argument("--rollout-concurrency", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api-key", default="local")
    p.add_argument("--public-model", default="qwen3-4b")
    p.add_argument("--controller-max-output", type=int, default=1024)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-model-calls", type=int, default=96)
    p.add_argument("--max-subcalls", type=int, default=80)
    p.add_argument("--max-total-tokens", type=int, default=400_000)
    p.add_argument("--deadline", type=float, default=900.0)
    p.add_argument("--exec-timeout", type=float, default=540.0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.ns = [int(x) for x in args.ns.split(",")]
    args.qts = args.qts.split(",")
    out = args.out
    for d in ("adapters", "dev", "rollouts", "traces"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log_path = out / "log.jsonl"

    start_step, controller = 1, args.public_model
    if args.resume and log_path.exists():
        last = json.loads(log_path.read_text().splitlines()[-1])
        start_step = last["step"] + 1
        args.init_adapter = str(out / "adapters" / f"step-{last['step']:04d}")
    trainer = Trainer(args, out)
    print(json.dumps(dict(event="trainer_ready", trainable_params=trainer.n_params)), flush=True)
    if args.init_adapter:
        controller = f"ctrl-{out.name}-init-{int(time.time())}"
        load_adapter(args, controller, Path(args.init_adapter))
    if start_step == 1:
        dev = evaluate(args, controller, out, 0)
        print(json.dumps(dict(event="dev", step=0, controller=controller, **dev)), flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(dict(step=0, controller=controller, dev=dev, time=time.time())) + "\n")

    for step in range(start_step, args.steps + 1):
        t0 = time.time()
        rows = collect_rollouts(args, step, controller, out)
        t1 = time.time()
        with (out / "rollouts" / f"step-{step:04d}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        train_stats = summarize(rows)
        upd = trainer.update(rows)
        t2 = time.time()
        adapter_dir = out / "adapters" / f"step-{step:04d}"
        trainer.save(adapter_dir)
        new_name = f"ctrl-{out.name}-s{step:04d}"  # unique per run so parallel/sequential runs never collide
        load_adapter(args, new_name, adapter_dir)
        if controller != args.public_model:
            unload_adapter(args, controller)
        controller = new_name
        record = dict(step=step, controller=controller, rollout_seconds=t1 - t0, train_seconds=t2 - t1,
                      train=train_stats, update=upd, time=time.time())
        if step % args.eval_every == 0:
            record["dev"] = evaluate(args, controller, out, step)
        with log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        print(json.dumps(dict(event="step", step=step, acc=train_stats["accuracy"], reward=train_stats["mean_reward"],
                              tokens=train_stats["mean_tokens"], seqs=upd.get("sequences"), loss=upd.get("loss"),
                              grad=upd.get("grad_norm"), rollout_s=round(t1 - t0), train_s=round(t2 - t1),
                              dev=record.get("dev", {}).get("accuracy"))), flush=True)
        # keep disk bounded: traces older than 3 steps are large; keep rollouts jsonl always
        old = out / "traces" / f"step-{step - 3:04d}"
        if old.exists() and not args.resume:
            shutil.rmtree(old, ignore_errors=True)


if __name__ == "__main__":
    main()
