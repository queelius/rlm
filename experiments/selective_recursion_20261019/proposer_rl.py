"""E26c: on-policy RL for the library proposer, reward = the verified teaching score (after SMITH, arXiv 2608.24571).

Each step samples training families (generated, never the held-out ones), builds the bootstrap prompt (contract plus
three solved examples), samples a group of libraries from the current LoRA adapter served by vLLM, and scores each
by running it on teaching tasks not shown in the prompt (bootstrap_verify_worker, in a subprocess). Advantage =
reward minus the mean of the rest of its group; groups with equal rewards are skipped. One policy-gradient update
per step (sequence-mean log-probability), then the adapter is saved and hot-swapped into vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bootstrap_library as B  # noqa: E402
import family_gen as G  # noqa: E402

MODEL = os.environ.get("SR_MODEL", "/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554")
DEVICE = os.environ.get("SR_DEVICE", "cuda:0")


def training_families(seeds=(0, 1, 2)) -> list[str]:
    fams = G.family_ids("train", seeds)
    fams += [f"gen-chat-{q}-{G.ANSWER_FORMATS[(len('chat') + len(q) + s) % 4]}-s{s}" for q in G.TRAIN_QUESTIONS for s in (1, 2)]
    return fams


_REAL = {}


def _real_family(fid: str):
    """A real family's teaching rows (bootstrap_library.family_api), cached; held-out rows are never loaded."""
    if fid not in _REAL:
        api = B.family_api(fid)
        _REAL[fid] = (api, [api["as_task"](r) for r in api["teach"]], list(api["teach"]))
    return _REAL[fid]


def prompt_for(fid: str, rng: random.Random, n_teach: int):
    if not fid.startswith("gen-"):  # a real family (e.g. mrcr): its own teaching rows
        api, tasks, rows = _real_family(fid)
        idx = rng.sample(range(len(rows)), 3)
        rest = [t for i, t in enumerate(tasks) if i not in idx]
        entry = "answer_" + fid
        return B.build_prompt(entry, [api["describe"](rows[i]) for i in idx]), entry, rng.sample(rest, min(n_teach, len(rest)))
    tasks = G.make(fid, "teach", 30)
    shown = rng.sample(tasks, 3)
    rest = [t for t in tasks if t not in shown]
    entry = "answer_" + fid.replace("-", "_")
    return B.build_prompt(entry, [G.describe(t) for t in shown]), entry, rng.sample(rest, n_teach)


def grader_for(fid: str):
    if not fid.startswith("gen-"):
        return _real_family(fid)[0]["grade"]
    return lambda out, t: G.grade(out, t)


def generate_ids(args, model_name: str, prompt_ids: list, n: int) -> list[tuple]:
    """I-019: sample from the exact prompt token ids through /v1/completions and keep the sampled token ids, their
    sampler log-probabilities and the finish reason, so the update trains on what was sampled (not re-tokenized text)."""
    def one(_):
        r = httpx.post(f"{args.base_url}/completions", headers={"Authorization": "Bearer local"}, timeout=1800,
                       json={"model": model_name, "prompt": list(prompt_ids), "temperature": args.temperature,
                             "max_tokens": args.max_output, "logprobs": 0, "return_tokens_as_token_ids": True})
        r.raise_for_status()
        c = r.json()["choices"][0]
        lp = c.get("logprobs") or {}
        ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens", [])]
        return c.get("text") or "", ids, lp.get("token_logprobs") or [], c.get("finish_reason")
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(one, range(n)))


def generate(args, model_name: str, prompt: str, n: int) -> list[str]:
    def one(_):
        r = httpx.post(f"{args.base_url}/responses", headers={"Authorization": "Bearer local"}, timeout=1800,
                       json={"model": model_name, "input": prompt, "temperature": args.temperature,
                             "max_output_tokens": args.max_output})
        r.raise_for_status()
        data = r.json()
        return "".join(c.get("text", "") for o in data.get("output", []) if o.get("type") == "message"
                       for c in o.get("content", []) if c.get("type") == "output_text")
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(one, range(n)))


def reward(args, text: str, tasks, entry: str, workdir: Path, tag: str, grade=None) -> float:
    code = B.extract_code(text)
    v = B.verify(code, tasks, grade or (lambda out, t: G.grade(out, t)), workdir, tag, args.base_url, "qwen3-4b", 120, entry)
    for suffix in (".py", ".out.json"):
        (workdir / f"{tag}{suffix}").unlink(missing_ok=True)
    return v["mean"]


class Policy:
    def __init__(self, args):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                                    attn_implementation="sdpa", device_map={"": DEVICE})
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        if args.init_adapter == "none":  # RL from the untrained model: a fresh LoRA, zero-initialized B
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = get_peft_model(base, cfg)
        else:
            self.model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        self.model.enable_input_require_grads()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.0)
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.args = args

    def prompt_ids(self, prompt: str) -> list:
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)

    def update(self, items: list[tuple]) -> dict:
        """items: (prompt, completion, advantage) where completion is text, or (sampled ids, finish reason) with
        --sample-ids; loss = -sum A * mean token log-prob, averaged over items."""
        from rl_train import completion_logprobs

        torch = self.torch
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        total, used = 0.0, 0
        for prompt, comp, adv in items:
            p_ids = self.prompt_ids(prompt)
            if isinstance(comp, tuple):  # sampled ids: train on exactly what was sampled
                ids, finish = comp
                c_ids = list(ids) + ([self.eos] if finish == "stop" else [])
            else:
                c_ids = self.tok(comp, add_special_tokens=False)["input_ids"] + [self.eos]
            if not c_ids:
                continue
            if len(p_ids) + len(c_ids) > self.args.max_seq_len:
                continue
            logp = completion_logprobs(self.model, list(p_ids), c_ids, torch)
            loss = -(adv * logp.mean()) / len(items)
            loss.backward()
            total += loss.item()
            used += 1
            del logp
        gn = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        return dict(loss=total, used=used, grad_norm=float(gn))

    def save(self, path: Path):
        self.model.save_pretrained(str(path))


def swap_adapter(args, name: str, path: Path, previous: str | None):
    h = {"Authorization": "Bearer local"}
    r = httpx.post(f"{args.base_url}/load_lora_adapter", headers=h, json={"lora_name": name, "lora_path": str(path)}, timeout=300)
    if r.status_code >= 300:
        raise RuntimeError(f"load {name}: {r.status_code} {r.text[:300]}")
    if previous and previous != args.start_name:
        httpx.post(f"{args.base_url}/unload_lora_adapter", headers=h, json={"lora_name": previous}, timeout=300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init-adapter", required=True, help='a LoRA directory, or "none" for a fresh LoRA on the base model')
    p.add_argument("--start-name", default="proposer-4b-8bcode-r2", help="the init adapter's name in vLLM (kept loaded)")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--families-per-step", type=int, default=8)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--n-teach", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-output", type=int, default=3072)
    p.add_argument("--max-seq-len", type=int, default=10000)
    p.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--families", default="", help="comma-separated families to train on (default: all generated training families); "
                   "a real family such as mrcr uses its own teaching rows")
    p.add_argument("--keep-solutions", action="store_true", help="save every sample whose reward reaches 0.9")
    p.add_argument("--with-helper-families", action="store_true", help="add the helper-dependent training families (E26e)")
    p.add_argument("--sample-ids", action="store_true",
                   help="I-019: sample through /v1/completions from the prompt token ids and train on the sampled ids")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / "verify"
    work.mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    fams = a.families.split(",") if a.families else training_families()
    if a.with_helper_families:
        fams += G.helper_training_family_ids()
    policy = Policy(a)
    current = a.start_name
    log = open(a.out / "steps.jsonl", "a")
    for step in range(1, a.steps + 1):
        t0 = time.time()
        picks = rng.sample(fams, a.families_per_step) if len(fams) >= a.families_per_step else [rng.choice(fams) for _ in range(a.families_per_step)]
        groups = []
        for fid in picks:
            prompt, entry, tasks = prompt_for(fid, random.Random(f"{a.seed}:{step}:{fid}:{len(groups)}"), a.n_teach)
            groups.append((fid, prompt, entry, tasks))
        if a.sample_ids:
            pids = [policy.prompt_ids(g[1]) for g in groups]  # tokenize outside the threads
            with ThreadPoolExecutor(len(groups)) as ex:
                raw = list(ex.map(lambda gi: generate_ids(a, current, pids[gi], a.group), range(len(groups))))
            samples = [[x[0] for x in g] for g in raw]
            sampled = [[(x[1], x[3]) for x in g] for g in raw]
        else:
            with ThreadPoolExecutor(len(groups)) as ex:
                samples = list(ex.map(lambda g: generate(a, current, g[1], a.group), groups))
            sampled = samples
        t_gen = time.time() - t0
        jobs = [(gi, k) for gi in range(len(groups)) for k in range(a.group)]
        with ThreadPoolExecutor(8) as ex:
            rewards = list(ex.map(lambda gk: reward(a, samples[gk[0]][gk[1]], groups[gk[0]][3], groups[gk[0]][2], work,
                                                    f"s{step}g{gk[0]}k{gk[1]}", grader_for(groups[gk[0]][0])), jobs))
        R = [[rewards[gi * a.group + k] for k in range(a.group)] for gi in range(len(groups))]
        if a.keep_solutions:
            for gi, rs in enumerate(R):
                for k, r in enumerate(rs):
                    if r >= 0.9:
                        (a.out / "solutions").mkdir(exist_ok=True)
                        (a.out / "solutions" / f"step{step}-g{gi}-k{k}-{groups[gi][0]}.py").write_text(B.extract_code(samples[gi][k]))
        items = []
        for gi, rs in enumerate(R):
            if max(rs) - min(rs) < 1e-9:
                continue
            for k, r in enumerate(rs):
                adv = r - (sum(rs) - r) / (len(rs) - 1)
                items.append((groups[gi][1], sampled[gi][k], adv))
        upd = policy.update(items) if items else dict(loss=0.0, used=0, grad_norm=0.0)
        ckpt = a.out / f"step-{step}"
        policy.save(ckpt)
        name = f"proposer-rl-{a.seed}-{step}"
        swap_adapter(a, name, ckpt, current if current != a.start_name else None)
        current = name
        flat = [r for rs in R for r in rs]
        rec = dict(step=step, families=picks, rewards=R, mean_reward=sum(flat) / len(flat),
                   solved=sum(r >= 0.9 for r in flat), groups_used=sum(1 for rs in R if max(rs) - min(rs) > 1e-9),
                   gen_seconds=round(t_gen), seconds=round(time.time() - t0), adapter=name, **upd)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"step {step}: mean reward {rec['mean_reward']:.3f}, solved {rec['solved']}/{len(flat)}, "
              f"groups used {rec['groups_used']}, {rec['seconds']}s", flush=True)
        if step > 2:
            import shutil
            shutil.rmtree(a.out / f"step-{step - 2}", ignore_errors=True) if (step - 2) % 10 else None
    (a.out / "FINAL").write_text(current + "\n")


if __name__ == "__main__":
    main()
