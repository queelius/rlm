"""Report explicitly supplied three-arm cartpole runs; never select or pool partial cohorts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ARMS = ("curl", "no_curl", "shuffled_curl")
SEEDS = (123, 456, 789)
LABELS = (
    "CURL with image matching",
    "Same crops without image matching",
    "Shuffled image matching",
)
EXPECTED = list(range(10000, 10010))
SCHEDULE = [(step, step * 8) for step in range(0, 12501, 500)]


def read_run(arm: str, seed: int, path: Path | None) -> dict:
    run = dict(
        arm=arm,
        seed=seed,
        path=None,
        status="missing",
        reason="not supplied",
        config=None,
        curve=[],
        endpoint_return=None,
    )
    if path is None:
        return run
    path = path.resolve()
    if not (path / "config.json").exists() and (path / "run").is_dir():
        path = path / "run"
    run["path"] = str(path)
    if not path.exists():
        run["reason"] = "supplied path does not exist"
        return run
    try:
        config = json.loads((path / "config.json").read_text())
        run["config"] = config
        required = dict(
            arm=arm,
            seed=seed,
            domain_name="cartpole",
            task_name="swingup",
            action_repeat=8,
            num_train_steps=12500,
            num_eval_episodes=10,
            evaluation_seeds=EXPECTED,
        )
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError("explicit identity or prescribed scientific inputs differ")
        intervention = dict(
            kind="deranged_encoded_key_rows",
            labels="diagonal_unchanged",
            encoder_optimizer_steps="upstream_two",
        )
        if arm == "shuffled_curl" and any(
            config.get("contrastive_control", {}).get(k) != v for k, v in intervention.items()
        ):
            raise ValueError("shuffled arm differs from its declared intervention")
        records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        if any(not isinstance(row, dict) or "type" not in row for row in records):
            raise ValueError("invalid native record")
        run.update(status="incomplete", reason="missing complete fixed endpoint")
        failures = [r for r in records if r["type"] == "failure"]
        if failures:
            run.update(status="failure", reason=str(failures[-1]))
        for row in records:
            if row["type"] in ("eval_episode", "train_metric"):
                value = row.get("return") if row["type"] == "eval_episode" else row.get("value")
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("nonfinite or invalid native metric")
        batches = defaultdict(list)
        for row in records:
            if row["type"] == "eval_episode":
                batches[row.get("step"), row.get("env_steps")].append(row)
        for key, episodes in sorted(batches.items()):
            if key not in SCHEDULE:
                raise ValueError("evaluation outside the declared schedule")
            if len(episodes) < 10:
                continue
            if len(episodes) != 10 or sorted(r.get("eval_seed", -1) for r in episodes) != EXPECTED:
                raise ValueError("duplicate or wrong evaluation starts")
            run["curve"].append(
                dict(
                    step=key[0],
                    env_steps=key[1],
                    episodes=10,
                    mean_return=statistics.mean(r["return"] for r in episodes),
                )
            )
        starts = [r for r in records if r["type"] == "start"]
        ends = [r for r in records if r["type"] == "end"]
        fresh = (
            len(starts) == 1 and starts[0].get("step") == 0 and starts[0].get("config") == config
        )
        end = ends[-1] if ends else {}
        terminal = dict(
            reason="completed", step=12500, env_steps=100000, updates=11500, eval_env_steps=260000
        )
        receipt_path = path / "result.json"
        if not receipt_path.exists():
            receipt_path = path.parent / "result.json"
        receipt_ok = True
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            run["receipt"] = receipt
            receipt_ok = receipt.get("exit_code") == 0 and receipt.get("complete") is True
            if receipt.get("exit_code") != 0:
                run.update(status="failure", reason="unsuccessful owner receipt")
            elif not receipt_ok and not failures:
                run.update(
                    status="incomplete", reason=receipt.get("reason") or "incomplete owner receipt"
                )
        if (
            fresh
            and len(ends) == 1
            and records[-1] == end
            and not failures
            and receipt_ok
            and not any(r["type"] == "resume" for r in records)
            and all(end.get(k) == v for k, v in terminal.items())
            and len(run["curve"]) == 26
        ):
            run.update(
                status="completed",
                reason="completed",
                endpoint_return=run["curve"][-1]["mean_return"],
            )
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        run.update(status="invalid", reason=str(error), endpoint_return=None)
    return run


def science(config: dict) -> dict:
    ignored = {"arm", "seed", "device", "max_seconds", "checkpoint_seconds", "contrastive_control"}
    return {key: value for key, value in config.items() if key not in ignored}


def differences(scores: dict) -> dict:
    return dict(
        curl_minus_no_curl=scores["curl"] - scores["no_curl"],
        shuffled_minus_curl=scores["shuffled_curl"] - scores["curl"],
        shuffled_minus_no_curl=scores["shuffled_curl"] - scores["no_curl"],
    )


def summarize(inputs: list[tuple[str, int, Path]]) -> dict:
    supplied = {}
    for arm, seed, path in inputs:
        if arm not in ARMS or seed not in SEEDS:
            raise ValueError("only declared arms and training seeds123/456/789 are accepted")
        if (arm, seed) in supplied:
            raise ValueError(f"duplicate arm/seed: {arm}/{seed}; no attempt selected")
        supplied[arm, seed] = Path(path)
    runs = [read_run(arm, seed, supplied.get((arm, seed))) for seed in SEEDS for arm in ARMS]
    groups = []
    for seed in SEEDS:
        members = [r for r in runs if r["seed"] == seed]
        ready = all(r["status"] == "completed" for r in members)
        matched = ready and all(
            science(r["config"]) == science(members[0]["config"]) for r in members
        )
        scores = {r["arm"]: r["endpoint_return"] for r in members}
        groups.append(
            dict(
                seed=seed,
                status="completed" if matched else "config_mismatch" if ready else "partial",
                scores=scores,
                differences=differences(scores) if matched else None,
            )
        )
    complete = all(g["status"] == "completed" for g in groups)
    complete = complete and all(science(r["config"]) == science(runs[0]["config"]) for r in runs)
    arms = {}
    for arm in ARMS:
        members = [r for r in runs if r["arm"] == arm]
        values = [r["endpoint_return"] for r in members] if complete else []
        arms[arm] = dict(
            n_completed=sum(r["status"] == "completed" for r in members),
            mean_return=statistics.mean(values) if values else None,
            seed_sd=statistics.stdev(values) if values else None,
        )
    aggregate = (
        {
            key: statistics.mean(g["differences"][key] for g in groups)
            for key in groups[0]["differences"]
        }
        if complete
        else None
    )
    return dict(
        status="completed" if complete else "partial",
        target_env_steps=100000,
        planned_training_seeds=list(SEEDS),
        runs=runs,
        groups=groups,
        arms=arms,
        aggregate_differences=aggregate,
        interpretation="Exploratory three-arm comparison. Ten fixed evaluation starts "
        "are not training replicates. All nine matched runs must complete before pooling. "
        "Shuffled gradients are not mechanism proof.",
    )


def write_outputs(summary: dict, output: Path, plot: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Cartpole image-correspondence comparison",
        "",
        summary["interpretation"],
        "",
        "Fixed 100k training-step endpoints; means of ten starts 10000–10009. "
        "No best-point selection.",
        "",
        "| Seed | Condition | Status | Endpoint |",
        "| ---: | --- | --- | ---: |",
    ]
    for run in summary["runs"]:
        value = "missing" if run["endpoint_return"] is None else f"{run['endpoint_return']:.6f}"
        lines.append(
            f"| {run['seed']} | {LABELS[ARMS.index(run['arm'])]} | {run['status']} | {value} |"
        )
    lines += ["", "## Cohort averages", ""]
    if summary["status"] == "completed":
        lines += [
            f"{LABELS[ARMS.index(arm)]}: {stats['mean_return']:.6f}; "
            f"training-seed SD {stats['seed_sd']:.6f}."
            for arm, stats in summary["arms"].items()
        ]
        lines.append("Paired mean differences: " + json.dumps(summary["aggregate_differences"]))
    else:
        lines.append(
            "No pooled scores: the complete, scientifically matched nine-run cohort "
            "is not available."
        )
    lines += ["", "## Sources and accounting", ""]
    lines += [
        f"- {r['arm']} seed {r['seed']}: {r['reason']}; source `{r['path']}`."
        for r in summary["runs"]
    ]
    lines += [
        "",
        "Equal training interactions do not imply equal wall time or optimizer work. "
        "Original baselines and the later shuffled cohort remain separately identified.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if not plot:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True, sharey=True)
    for ax, seed in zip(axes, SEEDS, strict=True):
        for index, arm in enumerate(ARMS):
            run = next(r for r in summary["runs"] if (r["arm"], r["seed"]) == (arm, seed))
            ax.plot(
                [p["env_steps"] for p in run["curve"]],
                [p["mean_return"] for p in run["curve"]],
                color=("tab:blue", "tab:orange", "tab:green")[index],
                linestyle=("-", "--", ":")[index],
                label=LABELS[index]
                + (f" ({run['status']})" if run["status"] != "completed" else ""),
            )
        ax.set(title=f"Training seed {seed}", xlim=(0, 100000), ylim=(0, 1000))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean evaluation return (higher is better)")
    fig.supxlabel("Simulator steps used for training", y=0.12)
    note = (
        "All three training seeds complete; exploratory comparison"
        if summary["status"] == "completed"
        else "Partial cohort: incomplete runs remain visible; no pooled scores"
    )
    fig.suptitle(
        "Cartpole: three image-matching conditions; ten fixed starts per point\n" + note,
        fontsize=12,
    )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.20, 1, 0.90))
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"learning-curves.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=3, action="append", required=True, metavar=("ARM", "SEED", "PATH")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        result = summarize([(arm, int(seed), Path(path)) for arm, seed, path in args.run])
    except ValueError as error:
        parser.error(str(error))
    write_outputs(result, args.output, args.plot)
