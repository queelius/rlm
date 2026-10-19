"""Add the controller-side tokens a learned controller would pay to scripted-expert rows.

The expert's controller turns were produced by code, so the kernel's usage counted only helper
calls. This re-tokenizes each controller request (chat template) and completion (+EOS) from the
trace and writes `controller_tokens` and `total_tokens_with_controller` into rows.jsonl in place
(original totals kept as `helper_tokens`).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_train
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(rl_train.MODEL, local_files_only=True)
eos = 1
for path in sys.argv[1:]:
    rows_path = Path(path) / "rows.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()]
    done = 0
    for r in rows:
        if not r.get("trace_dir") or r.get("controller_tokens") is not None:
            continue
        ctrl = 0
        for t in rl_train.controller_turns(r["trace_dir"]):
            ids = tok.apply_chat_template(t["messages"], add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            ctrl += len(ids) + len(tok(t["completion"], add_special_tokens=False)["input_ids"]) + eos
        r["helper_tokens"] = r.get("total_tokens")
        r["controller_tokens"] = ctrl
        r["total_tokens_with_controller"] = (r.get("total_tokens") or 0) + ctrl
        r["total_tokens"] = r["total_tokens_with_controller"]
        done += 1
    rows_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    print(path, "updated", done, "rows")
