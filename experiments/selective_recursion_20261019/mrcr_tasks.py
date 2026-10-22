"""OpenAI MRCR (multi-round coreference) as an RLM task family (E21).

Source: huggingface.co/datasets/openai/mrcr (MIT), cached at REVISION below. Each row is a long
synthetic chat (a list of messages) in which a user asks for pieces of writing; the final user
message asks for the i-th instance of one repeated ask with a random string prepended. The answer
is that string plus the assistant reply to the requested ask. Score: the official grader (0 unless
the response starts with the string; otherwise difflib SequenceMatcher ratio after stripping it).

Split (fixed, seed 0): the two parquet files hold different length bins, so rows are split within
each bin: teaching rows come only from the three bins up to 32k tokens; held-out rows are disjoint
from them and stratified over all eight bins. Bins are recovered by length rank (100 rows per bin
per needle count).
"""

from __future__ import annotations

import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

ROOT2 = Path("/project/alex_phd/research-cache/datasets/openai-mrcr-f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
ROOT4 = Path("/project/alex_phd/research-cache/datasets/openai-mrcr-fourneedle-f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
BINS = [(4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536), (65536, 131072),
        (131072, 262144), (262144, 524288), (524288, 1048576)]


def grade(response: str, answer: str, prefix: str) -> float:
    """The official MRCR grader."""
    response = response or ""
    if not response.startswith(prefix):
        return 0.0
    return float(SequenceMatcher(None, response.removeprefix(prefix), answer.removeprefix(prefix)).ratio())


def approx_tokens(n_chars: int) -> int:
    return int(n_chars) // 4


def bin_of(n_chars: int) -> int | None:
    t = approx_tokens(n_chars)
    for lo, hi in BINS:
        if lo < t <= hi or (lo == 4096 and t <= hi):
            return hi
    return None


JSONL = Path("/project/alex_phd/research-cache/datasets/openai-mrcr-jsonl-f4c69fae")


def _rows(needles: int, part: int) -> list[dict]:
    """Rows of one source file with their official bin, from the JSONL export (see its PROVENANCE.json)."""
    f = JSONL / f"{needles}needle.jsonl"
    if f.exists():
        return [r for r in (json.loads(l) for l in open(f)) if r["part"] == part]
    return _rows_from_parquet(needles, part)


def _rows_from_parquet(needles: int, part: int) -> list[dict]:
    """Rows of one file with their official length bin. The dataset has exactly 100 rows per bin per
    needle count, so the bin is recovered by ranking both files' rows by length (n_chars) and cutting
    the ranking into eight groups of equal size."""
    import pyarrow.parquet as pq

    root = ROOT2 if needles == 2 else ROOT4
    every = []
    for prt in (0, 1):
        f = root / f"{needles}needle" / f"{needles}needle_{prt}.parquet"
        for k, r in enumerate(pq.read_table(f).to_pylist()):
            r["row_id"] = f"mrcr{needles}-{prt}-{k:03d}"
            r["needles"] = needles
            r["part"] = prt
            every.append(r)
    every.sort(key=lambda r: (r["n_chars"], r["row_id"]))
    per = len(every) // len(BINS)
    for i, r in enumerate(every):
        r["bin"] = BINS[min(i // per, len(BINS) - 1)][1]
    return [r for r in every if r["part"] == part]


def load(split: str, needles: int = 2, per_bin: int = 8, max_bin: int | None = None, seed: int = 0,
         teach_per_bin: int = 10) -> list[dict]:
    """Within each official bin the rows of both files are shuffled with a fixed seed. The first
    `teach_per_bin` rows of each bin up to 32k tokens form the teaching split; held-out rows are drawn,
    `per_bin` per bin, only from the remaining rows, so the two splits never share a row."""
    rows = _rows(needles, 0) + _rows(needles, 1)
    by_bin: dict[int, list[dict]] = {}
    for r in rows:
        by_bin.setdefault(r["bin"], []).append(r)
    picked = []
    for b in sorted(by_bin):
        rs = sorted(by_bin[b], key=lambda r: r["row_id"])
        random.Random(f"{seed}:{needles}:{b}").shuffle(rs)
        n_teach = teach_per_bin if b <= 32768 else 0
        if split == "teach":
            if b <= (max_bin or 32768):
                picked += rs[:n_teach]
        else:
            if b <= (max_bin or 1048576):
                picked += rs[n_teach:n_teach + per_bin]
    return picked


def as_task(r: dict) -> SimpleNamespace:
    msgs = json.loads(r["prompt"])
    return SimpleNamespace(task_id=r["row_id"], prompt=msgs, answer=r["answer"], prefix=r["random_string_to_prepend"],
                           family="mrcr", n_records=approx_tokens(r["n_chars"]), question_type=f"{r['needles']}needle",
                           seed=0, events=[], desired_msg_index=r["desired_msg_index"], bin=r["bin"])


def describe(r: dict, head: int = 160) -> str:
    """A compact rendering of one teaching example for the library proposer."""
    msgs = json.loads(r["prompt"])
    lines = [f"Input: request['input'] is a list of {len(msgs)} chat messages (dicts with 'role' and 'content')."]
    for i in (0, 1, 2, 3, 4):
        m = msgs[i]
        lines.append(f"  message {i} ({m['role']}, {len(m['content'])} chars): {m['content'][:head]!r}")
    lines.append("  ...")
    for i in (len(msgs) - 3, len(msgs) - 2, len(msgs) - 1):
        m = msgs[i]
        lines.append(f"  message {i} ({m['role']}, {len(m['content'])} chars): {m['content'][:head]!r}")
    lines.append(f"Gold answer ({len(r['answer'])} chars): {r['answer'][:head]!r} ...")
    d = r["desired_msg_index"]
    lines.append(f"(For reference: the gold answer equals the prepended string followed by the content of message {d + 1}, "
                 f"the assistant reply to message {d}: {msgs[d]['content'][:80]!r}.)")
    return "\n".join(lines)
