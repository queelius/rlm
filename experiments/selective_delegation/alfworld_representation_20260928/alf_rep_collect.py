"""Four fixed flat-policy cells; exact native episode loop and parent-only GPU dispatch."""

import argparse
import fcntl
import gc
import os
import signal
import time
import uuid
from collections import Counter
from pathlib import Path

import alf_rep as r
import alf_rep_data as data


def plan(mode, actor, output, fixture=False):
    panel = r.read(data.PANEL / "MANIFEST.json")
    if not panel["ready"] or panel["game_count"] != 12 or panel["seeds"] != list(r.SEEDS):
        raise ValueError("fixed12 fresh reset-qualified games required")
    if mode not in ("index", "command") or actor not in ("base", "trained"):
        raise ValueError("undeclared representation/actor cell")
    expected = r.STUDY / f"{mode}-{actor}"
    if output.resolve() != expected:
        raise ValueError("fixed cell output required")
    binding = r.checkpoint_binding(mode) if actor == "trained" and not fixture else None
    jobs = []
    for game in panel["games"]:
        if r.sha(game["game"]) != game["game_sha256"]:
            raise ValueError("selected native game changed")
        for seed in r.SEEDS:
            jobs.append(
                dict(
                    episode_id=f"game-{game['game_index']:02d}-seed-{seed}-flat",
                    game_index=game["game_index"],
                    game=game,
                    seed=seed,
                    policy="flat",
                )
            )
    result = dict(
        schema="alfworld-index-command-readout-20260928-v1",
        representation=mode,
        actor=actor,
        split="valid_unseen",
        panel=str(data.PANEL / "MANIFEST.json"),
        panel_sha256=r.sha(data.PANEL / "MANIFEST.json"),
        cases=jobs,
        planned_episodes=24,
        model=str(r.BASE),
        model_manifest_sha256=r.sha(r.BASE / "local-research-manifest.json"),
        checkpoint_binding=binding,
        action_limit=50,
        token_limit=2048,
        request_cap=128,
        context_limit=8192,
        history="complete; context overflow unknown, never trimmed",
        sampling=dict(temperature=0.5, top_p=1.0, top_k=0),
        budget_seconds=1440,
        alfworld_python=panel["alfworld_python"],
        data_root=panel["data_root"],
        source_sha256=r.source_pins(),
        fixture=fixture,
        comparison="Identical indexed affordance lists and full executed-action public histories; "
        "return instruction/strict response decoder and matched SFT targets differ. "
        "Native success and own-interface gains, not raw cross-interface superiority.",
    )
    r.persist(output / "PLAN.json", result)
    return result


def summarize(output, prepared):
    rows = [r.read(path) for path in (output / "episodes").glob("*.json")]
    calls = [r.read(path) for path in (output / "calls").glob("*.json")]
    observed = sum(row["observed"] for row in rows)
    return dict(
        representation=prepared["representation"],
        actor=prepared["actor"],
        planned=24,
        observed=observed,
        unknown=24 - observed,
        complete=observed == 24,
        won=sum(row["observed"] and row["won"] for row in rows),
        invalid_outputs=sum(row["invalid_outputs"] for row in rows),
        terminations=dict(Counter(row["termination"] for row in rows)),
        physical_cost=r.original.evaluation.cost(calls),
    )


def run(args):
    output = r.STUDY / f"{args.mode}-{args.actor}"
    if (
        args.actor == "trained"
        and args.mode == "command"
        and not (r.COMMAND_CHECKPOINT / "COMMIT.json").exists()
    ):
        r.persist(
            output / "CONDITIONAL-SKIP.json",
            dict(reason="No actual command checkpoint33; outcome unknown", GPU_loaded=False),
        )
        return
    prepared = plan(args.mode, args.actor, output)
    if args.prepare_only:
        return
    if list(output.glob("OWNER-*.json")):
        raise ValueError("existing attempt; no implicit retry")
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("parent must assign exactly one GPU")
    lease, started = int(os.environ["SLURM_JOB_END_TIME"]), time.time()
    deadline = min(started + prepared["budget_seconds"], lease - 600)
    if deadline <= started + 180:
        raise ValueError("insufficient allocation margin")
    lock = (r.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, failure, stopped, model, base = uuid.uuid4().hex[:12], None, False, None, None
    r.save(
        output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            create_time=psutil.Process().create_time(),
            source=str(Path(__file__)),
            source_sha256=prepared["source_sha256"],
            started=started,
            deadline=deadline,
        ),
    )

    def stop(*_):
        nonlocal stopped
        stopped = True

    def stopping():
        return stopped or time.time() >= deadline - 5 or (output / "STOP").exists()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(r.BASE, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            r.BASE,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = base
        controller = r.controller(args.mode)
        if args.actor == "trained":
            binding = prepared["checkpoint_binding"]
            model = PeftModel.from_pretrained(
                base, binding["path"], is_trainable=False, autocast_adapter_dtype=True
            )
            actor_module = r.module(Path(r.actor.__file__), "alf_rep_private_actor")
            actor_module.CHECKPOINT = Path(binding["path"])
            client = actor_module.ActionAdapterClient(
                model, tokenizer, output, deadline, binding["adapter_sha256"]
            )
        else:
            client = controller.BaseClient(model, tokenizer, output, deadline)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        r.save(
            output / "LOAD.json",
            dict(
                base_dtype=str(model.dtype),
                trainable_parameters=0,
                adapter_enabled=args.actor == "trained",
                gpu=torch.cuda.get_device_name(),
            ),
        )
        for job in prepared["cases"]:
            if stopping():
                stopped = True
                break
            env = r.actor.unseen.CheckedBridge(
                job["game"],
                prepared["alfworld_python"],
                prepared["data_root"],
                deadline,
                output / (job["episode_id"] + "-bridge.stderr"),
            )
            env.expected_public = job["game"]["public"]
            try:
                controller.play_episode(client, env, job, output, stopping)
            finally:
                env.close()
            r.probe.campaign.snapshot(output / "SUMMARY.json", summarize(output, prepared))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        model = base = None
        gc.collect()
        torch.cuda.empty_cache()
        summary = summarize(output, prepared)
        r.probe.campaign.snapshot(output / "SUMMARY.json", summary)
        r.save(
            output / f"TERMINAL-{owner}.json",
            dict(
                complete=summary["complete"] and failure is None,
                failure=failure,
                stopped=stopped,
                elapsed_seconds=time.time() - started,
                ended=time.time(),
            ),
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("index", "command"), required=True)
    parser.add_argument("--actor", choices=("base", "trained"), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
