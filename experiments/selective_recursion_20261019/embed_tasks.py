"""E13: embed task inputs with the controller model's own hidden states (learned retrieval key).

For each task, the request text is tokenized; the first HEAD and last TAIL tokens (the task
description and the question) are run through Qwen3-4B, and the last hidden layer is mean-pooled.
The total token count is stored alongside, since length is invisible to a truncated encoding.
Writes an .npz with ids, family, split, vectors and token counts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import taskgen  # noqa: E402
import teach_oolong  # noqa: E402

MODEL = "/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554"


def tasks():
    for split, seeds in (("teach", range(16, 24)), ("dev", range(1000, 1004))):
        for n in (8, 32, 64, 128, 256, 512):
            for qt in taskgen.QUESTION_TYPES:
                for s in seeds:
                    t = taskgen.make_task(s, n, qt, split=split)
                    yield dict(id=t.task_id, family="eventlog", split=split, text=t.prompt)
    for split in ("train", "test"):
        for r in teach_oolong.load_rows(split, 32768):
            yield dict(id=f"oolong-{r['id']}", family="oolong", split=split, text=r["context"] + "\n" + r["question"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--head", type=int, default=768)
    p.add_argument("--tail", type=int, default=256)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16).cuda().eval()
    ids, fams, splits, vecs, ntoks = [], [], [], [], []
    items = list(tasks())
    print(len(items), "tasks", flush=True)
    for k, it in enumerate(items):
        t = tok(it["text"], add_special_tokens=False)["input_ids"]
        ntoks.append(len(t))
        x = t if len(t) <= a.head + a.tail else t[: a.head] + t[-a.tail:]
        with torch.no_grad():
            out = model(torch.tensor([x], device="cuda"), output_hidden_states=True)
        h = out.hidden_states[-1][0].float().mean(0).cpu().numpy()
        ids.append(it["id"]); fams.append(it["family"]); splits.append(it["split"]); vecs.append(h)
        if k % 100 == 0:
            print(k, flush=True)
    np.savez(a.out, ids=np.array(ids), family=np.array(fams), split=np.array(splits), vec=np.stack(vecs), ntok=np.array(ntoks))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
