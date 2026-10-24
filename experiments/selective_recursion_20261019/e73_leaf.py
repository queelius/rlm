"""E73: how reliable is the neural leaf that semantic-total libraries depend on?

The generator knows every item's category (family_gen.ITEM_CATEGORY). This asks a helper model about every item in
two forms, (a) yes/no per (item, category), as the accepted libraries do, and (b) one choice question per item, then
computes the leaf-induced ceiling: the fraction of fresh semantic-total tasks whose exact answer survives the helper's
classifications (a library with perfect parsing and arithmetic can do no better). Temperature 0.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import family_gen as F  # noqa: E402

CATS = sorted(F.CATEGORIES)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3-4b")
    p.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    p.add_argument("--families", type=Path, required=True, help="families file; semantic_total ones are used")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=120)

    def ask(prompt: str) -> str:
        r = client.chat.completions.create(model=a.model, messages=[dict(role="user", content=prompt)], max_tokens=16,
                                           temperature=0)
        return (r.choices[0].message.content or "").strip().lower()

    jobs = [("yesno", it, c) for it in F.ITEMS for c in CATS] + [("choice", it, None) for it in F.ITEMS]

    def one(j):
        form, it, c = j
        if form == "yesno":
            return form, it, c, ask(f"Is the item '{it}' considered a {c} item? Answer with 'yes' or 'no'.").startswith("yes")
        ans = ask(f"Which category does the item '{it}' belong to? Options: {', '.join(CATS)}. Answer with one word.")
        return form, it, None, next((c for c in CATS if c in ans), None)

    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(one, jobs))
    yes = {(it, c): v for f, it, c, v in res if f == "yesno"}
    choice = {it: v for f, it, _, v in res if f == "choice"}
    yn_acc = sum(yes[(it, c)] == (F.ITEM_CATEGORY[it] == c) for it in F.ITEMS for c in CATS) / (len(F.ITEMS) * len(CATS))
    ch_acc = sum(choice[it] == F.ITEM_CATEGORY[it] for it in F.ITEMS) / len(F.ITEMS)
    wrong_yes = sorted((it, c) for (it, c), v in yes.items() if v != (F.ITEM_CATEGORY[it] == c))
    wrong_ch = sorted((it, choice[it], F.ITEM_CATEGORY[it]) for it in F.ITEMS if choice[it] != F.ITEM_CATEGORY[it])

    # leaf-induced ceiling on fresh semantic_total tasks: recompute each answer with the helper's classifications
    fams = [f for f in a.families.read_text().strip().split(",") if "-semantic_total-" in f]
    ceil = {"yesno": [], "choice": []}
    for fid in fams:
        fam = F.Family(fid)
        for i in range(30):
            r = random.Random(f"task:{fid}:fresh:400:{i}")
            while True:
                recs = fam.records(r, 400)
                qa = fam.ask(r, recs)
                if qa:
                    break
            cat = next(c for c in CATS if f" {c} items" in qa[0])
            gold = int(qa[1])
            bought = [x for x in recs if x["action"] == "bought"]
            pred_yn = sum(x["qty"] for x in bought if yes[(x["item"], cat)])
            pred_ch = sum(x["qty"] for x in bought if choice[x["item"]] == cat)
            ceil["yesno"].append(pred_yn == gold)
            ceil["choice"].append(pred_ch == gold)
    out = dict(model=a.model, yesno_pair_accuracy=yn_acc, choice_item_accuracy=ch_acc, wrong_yesno=wrong_yes,
               wrong_choice=wrong_ch, families=len(fams),
               ceiling_yesno=sum(ceil["yesno"]) / len(ceil["yesno"]), ceiling_choice=sum(ceil["choice"]) / len(ceil["choice"]))
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("wrong")}, indent=1))
    print("wrong yes/no pairs:", wrong_yes[:20])
    print("wrong choices:", wrong_ch)


if __name__ == "__main__":
    main()
