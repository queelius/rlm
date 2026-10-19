"""Summarize an RL run: per-step accuracy/reward/tokens by task source, update stats, dev evals."""
import json, sys, collections
from pathlib import Path
out = Path(sys.argv[1])
print(f"{'step':>4} {'acc':>5} {'reward':>7} {'tok':>7} {'ev_acc':>6} {'oo_acc':>6} {'seqs':>5} {'zeroadv':>7} {'loss':>8} {'grad':>6} {'roll_s':>6} {'train_s':>7} {'dev_acc':>7} {'dev_tok':>7}")
for line in (out / "log.jsonl").read_text().splitlines():
    d = json.loads(line)
    if d.get("step", 0) == 0:
        dev = d.get("dev") or {}
        print(f"{0:>4} {'':>5} {'':>7} {'':>7} {'':>6} {'':>6} {'':>5} {'':>7} {'':>8} {'':>6} {'':>6} {'':>7} {dev.get('accuracy', float('nan')):>7.2f} {dev.get('mean_tokens', 0):>7.0f}")
        continue
    t = d.get("train", {}); u = d.get("update", {}); dev = d.get("dev") or {}
    rows_path = out / "rollouts" / f"step-{d['step']:04d}.jsonl"
    by = collections.defaultdict(lambda: [0, 0])
    if rows_path.exists():
        for l in rows_path.read_text().splitlines():
            r = json.loads(l); a = by[r.get("source", "?")]; a[0] += 1; a[1] += bool(r.get("correct"))
    ev = by.get("eventlog", [0, 0]); oo = by.get("oolong", [0, 0])
    print(f"{d['step']:>4} {t.get('accuracy', 0):>5.2f} {t.get('mean_reward', 0):>7.3f} {t.get('mean_tokens', 0):>7.0f} "
          f"{(ev[1] / ev[0]) if ev[0] else float('nan'):>6.2f} {(oo[1] / oo[0]) if oo[0] else float('nan'):>6.2f} "
          f"{u.get('sequences', 0):>5} {u.get('skipped_zero_adv', 0):>7} {u.get('loss') or 0:>8.4f} {u.get('grad_norm') or 0:>6.3f} "
          f"{d.get('rollout_seconds', 0):>6.0f} {d.get('train_seconds', 0):>7.0f} {dev.get('accuracy', float('nan')):>7.2f} {dev.get('mean_tokens', 0):>7.0f}")
