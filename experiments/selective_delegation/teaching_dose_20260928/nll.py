"""Optional fixed-endpoint teacher-forced TRAIN NLL; separate from optimization."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import signal
import time
import uuid
from collections import defaultdict
from pathlib import Path

import dose_common as d
import readout


def categories(rows: list[dict], tasks: list[dict]) -> list[list[str]]:
    lookup = {task["id"]: task for task in tasks}
    grouped = defaultdict(list)
    result = [None] * len(rows)
    for index, row in enumerate(rows):
        grouped[row["task_id"]].append((index, row))
    for task_id, group in grouped.items():
        task = lookup[task_id]
        visible = set(task["misc"]["target_items"]) | set(task["misc"]["initial_inventory"])
        for position, (index, row) in enumerate(sorted(group, key=lambda pair: pair[1]["step"])):
            action = json.loads(row["target"])
            labels = ["all", action["action"]]
            if action["action"] == "get_info":
                public = set(action["items"]).issubset(visible)
                labels.append("get_info_visible" if public else "get_info_unseen")
                for info in row["feedback"]:
                    visible.add(info["item"])
                    for recipe in info["recipes"]:
                        visible.update(recipe["ingredients"])
            elif action["action"] == "craft":
                d.require(
                    str(row["feedback"]).startswith("Successfully crafted"),
                    "NLL teacher craft was not successful",
                )
                visible.add(action["target_item"])
            if position == 0:
                labels.append("first_action")
            result[index] = labels
    return result


def prepare(args):
    from transformers import AutoTokenizer

    _, shared, _ = readout.load_runtime("raw")
    binding = readout.endpoint(args.root, args.teacher, args.step, shared)
    name, digest = d.ORIGINALS[args.dataset]
    original_plan = d.ROOT / name / "PLAN.json"
    d.require(d.sha(original_plan) == digest, "NLL dataset plan changed")
    prior = d.read(original_plan)
    prepared = Path(prior["prepared"])
    d.require(d.sha(prepared / "rows.jsonl") == prior["rows_sha256"], "NLL rows changed")
    rows = [json.loads(line) for line in (prepared / "rows.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (prepared / "tasks.jsonl").read_text().splitlines()]
    labels = categories(rows, tasks)
    recipe = d.recipe()
    tokenizer = AutoTokenizer.from_pretrained(prior["model"], local_files_only=True)
    examples = recipe.tokenize_rows(rows, tokenizer)
    d.require(
        len(examples) == 366 and sum(len(e["target_ids"]) for e in examples) == 8820,
        "NLL dose changed",
    )
    counts = {
        key: sum(key in group for group in labels)
        for key in (
            "all",
            "get_info",
            "craft",
            "finish",
            "get_info_visible",
            "get_info_unseen",
            "first_action",
        )
    }
    plan = dict(
        schema="textcraft-teacher-fixed-nll-20260928-v1",
        teacher=args.teacher,
        cumulative_updates=args.step,
        dataset_teacher=args.dataset,
        split="train",
        adapter=binding,
        model=prior["model"],
        base_dtype="bfloat16",
        lora_dtype="float32",
        rows=366,
        supervised_tokens=8820,
        strata_rows=counts,
        rows_sha256=prior["rows_sha256"],
        tasks_sha256=d.sha(prepared / "tasks.jsonl"),
        budget_seconds=900,
        objective="Teacher-forced full target JSON+EOS NLL atT1; token weighted mean",
        visibility="Before action: goal/current or initial inventory and earlier "
        "returned get_info recipe ingredient names; no whole-prompt substring test",
        caveat="TRAIN fit only, not native success or name-only NLL. Cross-teacher "
        "dataset comparison changes conditioning histories, not just visibility.",
        source_sha256={
            str(p): d.sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(d.__file__).resolve(),
                Path(readout.__file__).resolve(),
                Path(recipe.__file__).resolve(),
                Path(recipe.target_loss.__code__.co_filename).resolve(),
            )
        },
    )
    output = args.root / f"nll-{args.teacher}-cp{args.step}-on-{args.dataset}"
    path = output / "PLAN.json"
    if path.exists():
        d.require(d.read(path) == plan, "immutable NLL plan differs")
    else:
        d.save(path, plan)
    return output, plan, rows, examples, labels


def run(args):
    output, plan, rows, examples, labels = prepare(args)
    if args.prepare_only:
        print(
            dict(output=str(output), GPU_loaded=False, strata_rows=plan["strata_rows"]), flush=True
        )
        return
    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    d.require(not list(output.glob("OWNER-*.json")), "existing NLL attempt; no silent retry")
    d.require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "parent must assign exactly one GPU",
    )
    started = time.time()
    deadline = min(started + 900, int(os.environ["SLURM_JOB_END_TIME"]) - 600)
    d.require(deadline > started + 180, "insufficient allocation margin")
    recipe = d.recipe()
    lock_path = recipe.probe.campaign.STORE / "sidecars/root-rlvr-campaign-v1/COORDINATOR.lock"
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = uuid.uuid4().hex[:12]
    d.save(
        output / f"OWNER-{owner}.json",
        dict(
            pid=os.getpid(),
            started=started,
            create_time=psutil.Process().create_time(),
            deadline=deadline,
            source_sha256=plan["source_sha256"],
        ),
    )
    stopped, failure, complete = False, None, False
    model = base = None

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        torch.set_num_threads(4)
        base = AutoModelForCausalLM.from_pretrained(
            plan["model"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(
            base, plan["adapter"]["path"], is_trainable=False, autocast_adapter_dtype=True
        )
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        d.require(
            all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n),
            "NLL LoRA arithmetic differs",
        )
        details, aggregate = [], defaultdict(lambda: dict(rows=0, tokens=0, nll_sum=0.0))
        with torch.no_grad():
            for row, example, groups in zip(rows, examples, labels, strict=True):
                if stopped or (output / "STOP").exists() or time.time() >= deadline - 10:
                    raise TimeoutError("bounded NLL stopped; incomplete measurement is unknown")
                ids = torch.tensor([example["input_ids"]], device="cuda:0")
                targets = torch.tensor([example["target_ids"]], device="cuda:0")
                nll = float(
                    recipe.target_loss(
                        model(input_ids=ids, logits_to_keep=targets.shape[1]).logits, targets
                    )
                )
                d.require(math.isfinite(nll), "nonfinite teacher-forced NLL")
                tokens = len(example["target_ids"])
                details.append(
                    dict(
                        task_id=row["task_id"],
                        step=row["step"],
                        strata=groups,
                        tokens=tokens,
                        nll_sum=nll,
                        nll_mean=nll / tokens,
                    )
                )
                for group in groups:
                    aggregate[group]["rows"] += 1
                    aggregate[group]["tokens"] += tokens
                    aggregate[group]["nll_sum"] += nll
        for record in aggregate.values():
            record["nll_mean"] = record["nll_sum"] / record["tokens"]
        d.save(
            output / "NLL.json",
            dict(
                plan_sha256=d.sha(output / "PLAN.json"),
                aggregate=dict(aggregate),
                rows=details,
                complete=True,
                teacher=args.teacher,
                dataset_teacher=args.dataset,
                cumulative_updates=args.step,
            ),
        )
        complete = True
        print(json.dumps(dict(aggregate)), flush=True)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        model = base = None
        gc.collect()
        torch.cuda.empty_cache()
        d.save(
            output / f"TERMINAL-{owner}.json",
            dict(
                complete=complete,
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
    parser.add_argument("--root", type=Path, default=d.OUTPUT)
    parser.add_argument("--teacher", choices=d.ORIGINALS, required=True)
    parser.add_argument("--step", type=int, choices=(23, 46, 69), required=True)
    parser.add_argument("--dataset", choices=d.ORIGINALS)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset or args.teacher
    run(args)
