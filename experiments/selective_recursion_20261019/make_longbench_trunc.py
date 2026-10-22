"""Exact middle truncation of LongBench-v2 contexts with the Qwen3 tokenizer (run with the vLLM environment).

Keeps the first and last HALF tokens of every context longer than 2*HALF tokens and writes {_id: context} to
/project/alex_phd/research-cache/derived/longbench-v2-2b48e494-trunc48k-qwen3.json (with a provenance file).
"""
import hashlib
import json
import sys
import time

sys.path.insert(0, ".")
from transformers import AutoTokenizer  # noqa: E402

import longbench_tasks as L  # noqa: E402

HALF = 24_000
MODEL = "/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554"
OUT = "/project/alex_phd/research-cache/derived/longbench-v2-2b48e494-trunc48k-qwen3.json"
tok = AutoTokenizer.from_pretrained(MODEL)
out, cut = {}, 0
for x in L._rows():
    ids = tok(x["context"], add_special_tokens=False).input_ids
    if len(ids) > 2 * HALF:
        out[x["_id"]] = tok.decode(ids[:HALF]) + "\n\n[... middle of the document omitted ...]\n\n" + tok.decode(ids[-HALF:])
        cut += 1
    else:
        out[x["_id"]] = x["context"]
blob = json.dumps(out)
open(OUT, "w").write(blob)
open(OUT + ".PROVENANCE.txt", "w").write(
    f"derived from {L.DATA} (zai-org/LongBench-v2, Apache-2.0)\ntokenizer: {MODEL}\nrule: contexts over {2 * HALF} "
    f"tokens keep the first and last {HALF} tokens\ntruncated: {cut} of {len(out)}\ncreated: "
    f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nsha256: {hashlib.sha256(blob.encode()).hexdigest()}\n")
print(f"{cut} of {len(out)} truncated -> {OUT}")
