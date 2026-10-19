import sys, json; sys.path.insert(0,'.')
import rl_train
from pathlib import Path
from types import SimpleNamespace
if __name__ == "__main__":
    args=SimpleNamespace(base_url="http://127.0.0.1:8001/v1", api_key="local", public_model="qwen3-4b", harness="default",
        controller_max_output=1024, max_turns=10, max_model_calls=48, max_subcalls=40, max_total_tokens=400000, deadline=600,
        temperature=1.0, lam=0.2, seed=0, ns=[8,128], qts=["count_buyers"], tasks_per_step=2, group=3, rollout_concurrency=6)
    out=Path('/project/alex_phd/runs/selective-recursion-20261019/rl-dryrun'); (out/'traces').mkdir(parents=True, exist_ok=True)
    rows=rl_train.collect_rollouts(args, 1, "qwen3-4b", out)
    for r in rows: print(r['task_id'][:40], 'n',r['n'],'k',r['sample'],'correct',r['correct'],'tok',r.get('total_tokens'),'calls',r.get('calls'),'reward',round(r['reward'],3),'adv',round(r['advantage'],3),'err',(r.get('error') or '')[:60])
    ok=[r for r in rows if r.get('trace_dir')]
    if ok:
        turns=rl_train.controller_turns(ok[0]['trace_dir'])
        print('turns in rollout 0:', len(turns), [len(t['completion']) for t in turns], [t['usage'].get('output_tokens') for t in turns])
        print(turns[0]['completion'][:300])
    print(json.dumps(rl_train.summarize(rows), indent=1)[:700])
