"""Supervised LoRA fine-tuning of the controller on (messages -> cell) pairs from teach.py traces."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = os.environ.get("SR_MODEL", "/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554")
DEVICE = os.environ.get("SR_DEVICE", "cuda:0")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--batch-tokens", type=int, default=16000, help="completion+prompt tokens per optimizer step")
    p.add_argument("--max-seq-len", type=int, default=12000)
    p.add_argument("--only-correct", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=0)
    args = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rl_train import completion_logprobs

    pairs = []
    for path in args.data:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if args.only_correct and not r.get("correct"):
                continue
            pairs.append(r)
    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    encoded = []
    for r in pairs:
        ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        comp = tok(r["completion"], add_special_tokens=False)["input_ids"] + [eos]
        if len(ids) + len(comp) <= args.max_seq_len:
            encoded.append((list(ids), comp))
    print(json.dumps(dict(pairs=len(pairs), encoded=len(encoded),
                          completion_tokens=sum(len(c) for _, c in encoded))), flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map={"": DEVICE})
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, args.epochs * sum(len(p_) + len(c) for p_, c in encoded) // args.batch_tokens)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / total_steps))
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train_log.jsonl").open("a")
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        random.Random(args.seed + epoch).shuffle(order)
        acc_tokens, acc_loss, acc_comp = 0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for idx in order:
            prompt, comp = encoded[idx]
            tok_logp = completion_logprobs(model, prompt, comp, torch)
            loss = -tok_logp.sum()
            loss.backward()
            acc_loss += loss.item(); acc_comp += len(comp); acc_tokens += len(prompt) + len(comp)
            del tok_logp
            if acc_tokens >= args.batch_tokens:
                for p_ in params:
                    if p_.grad is not None:
                        p_.grad.div_(acc_comp)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                rec = dict(step=step, epoch=epoch, loss=acc_loss / acc_comp, grad_norm=float(gn), lr=sched.get_last_lr()[0], seconds=time.time() - t0)
                log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
                acc_tokens, acc_loss, acc_comp = 0, 0.0, 0
        if acc_comp:
            for p_ in params:
                if p_.grad is not None:
                    p_.grad.div_(acc_comp)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        model.save_pretrained(str(args.out / f"epoch-{epoch + 1}"))
    model.save_pretrained(str(args.out / "final"))
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print("saved", args.out / "final")


if __name__ == "__main__":
    main()
