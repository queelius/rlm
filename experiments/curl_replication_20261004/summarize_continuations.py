"""Separate 100k-to-500k CURL continuation accounting; never load model weights."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("curl", "no_curl")
RUNTIME = {"max_seconds", "checkpoint_seconds", "device"}


def read(path):
    return json.loads(path.read_text())


def records(path):
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                value = json.loads(line)
                require(isinstance(value, dict), "Native record must be an object")
            except ValueError:
                value = {"type": "unreadable_json", "line": number}
            result.append(value)
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scientific(config, ignored=RUNTIME):
    return {key: value for key, value in config.items() if key not in ignored}


def curve(rows, config, source):
    expected = config.get("evaluation_seeds", [])
    batches, result = defaultdict(list), []
    for row in rows:
        if row.get("type") == "eval_episode":
            batches[(row.get("step"), row.get("env_steps"))].append(row)
    for (step, env_steps), episodes in batches.items():
        values = [row.get("return") for row in episodes]
        if (
            len(expected) == len(set(expected)) == len(episodes) == 10
            and Counter(row.get("eval_seed") for row in episodes) == Counter(expected)
            and all(type(v) in (int, float) and math.isfinite(v) for v in values)
            and type(step) is int
            and type(env_steps) is int
        ):
            result.append(
                {
                    "step": step,
                    "env_steps": env_steps,
                    "mean_return": statistics.mean(values),
                    "episodes": 10,
                    "source": str(source.resolve()),
                }
            )
    return sorted(result, key=lambda point: (point["env_steps"], point["step"]))


def parent_curve(parent, config):
    checkpoint, config_path = Path(parent["checkpoint"]).resolve(), Path(parent["config_path"])
    require(config_path.resolve() == checkpoint.parent / "config.json", "Wrong parent config path")
    require(
        hashlib.sha256(config_path.read_bytes()).hexdigest() == parent["config_sha256"],
        "Parent config checksum changed",
    )
    previous = read(config_path)
    require(previous == parent["config"], "Parent config receipt differs")
    ignored = RUNTIME | {"num_train_steps"}
    require(
        scientific(previous, ignored) == scientific(config, ignored), "Scientific config changed"
    )
    require(
        previous.get("num_train_steps") == 12500
        and previous.get("init_steps") == 1000
        and previous.get("action_repeat") == 8
        and previous.get("domain_name") == "cartpole"
        and previous.get("task_name") == "swingup",
        "Wrong parent protocol",
    )
    rows = records(checkpoint.parent / "metrics.jsonl")
    require(not any(r.get("type") == "unreadable_json" for r in rows), "Parent JSON is incomplete")
    starts = [r for r in rows if r.get("type") in ("start", "resume", "failure")]
    require(len(starts) == 1 and starts[0]["type"] == "start", "Parent is not one fresh run")
    proof = {
        name: [r for r in rows if r.get("type") == name][-1]
        for name in ("end", "train_episode", "checkpoint")
    }
    end, episode, saved = (proof[name] for name in ("end", "train_episode", "checkpoint"))
    require(
        rows[-1] == end
        and end.get("reason") == "completed"
        and (end.get("step"), end.get("env_steps"), end.get("updates")) == (12500, 100000, 11500)
        and episode.get("step") == 12500
        and episode.get("truncated_by_budget") is False
        and saved.get("reason") == "completed"
        and saved.get("step") == 12500
        and saved.get("env_steps") == 100000
        and Path(saved.get("path", "")).resolve() == checkpoint
        and proof == parent["terminal_proof"],
        "Invalid parent terminal proof",
    )
    stat = checkpoint.stat()
    require(
        parent["checkpoint_identity"]
        == {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
        and saved.get("bytes") == stat.st_size,
        "Parent checkpoint identity changed",
    )
    points = curve(rows, previous, checkpoint.parent / "metrics.jsonl")
    require(
        sum(p["step"] == 12500 and p["env_steps"] == 100000 for p in points) == 1,
        "Parent lacks ten unique terminal evaluation episodes",
    )
    return points


def read_chain(job_path):
    entry = {
        "path": str(job_path.parent.resolve()),
        "status": "missing",
        "reason": "not started",
        "arm": None,
        "seed": None,
        "curve": [],
        "endpoint_return": None,
    }
    try:
        receipt = read(job_path)
        job, parent = receipt["job"], receipt["parent"]
        entry.update(arm=job["arm"], seed=job["seed"], provenance=receipt)
        require(entry["arm"] in ARMS and type(entry["seed"]) is int, "Invalid arm/seed")
        require(job["steps"] == 62500 and job["env_steps"] == 500000, "Wrong child budget")
        require(
            Path(job["resume"]).resolve() == Path(parent["checkpoint"]).resolve(),
            "Job points to a different parent",
        )
        output = job_path.parent / "run"
        result = (
            read(job_path.parent / "result.json")
            if (job_path.parent / "result.json").exists()
            else {}
        )
        entry["result"] = result
        if result.get("exit_code") not in (None, 0):
            entry.update(status="failure", reason="nonzero child exit")
        previous = parent_curve(parent, parent["config"])
        entry["parent_curve"] = previous
        entry["curve"] = [p for p in previous if p["env_steps"] <= 100000]
        config = read(output / "config.json")
        entry["config"] = config
        rows = records(output / "metrics.jsonl")
        failed = any(r.get("type") == "failure" for r in rows)
        if failed:
            entry.update(status="failure", reason="native failure record")
        points = curve(rows, config, output / "metrics.jsonl")
        entry["child_curve"] = points
        entry["curve"] += [p for p in points if 100000 < p["env_steps"] <= 500000]
        require(
            config.get("num_train_steps") == 62500
            and config.get("num_eval_episodes") == 10
            and config.get("arm") == job["arm"]
            and config.get("seed") == job["seed"],
            "Wrong child config",
        )
        ignored = RUNTIME | {"num_train_steps"}
        require(
            scientific(parent["config"], ignored) == scientific(config, ignored),
            "Scientific config changed",
        )
        resumes = [r for r in rows if r.get("type") in ("resume", "start")]
        require(
            len(resumes) == 1
            and resumes[0].get("type") == "resume"
            and resumes[0].get("step") == 12500
            and resumes[0].get("config") == config
            and Path(resumes[0].get("resume_path", "")).resolve()
            == Path(parent["checkpoint"]).resolve(),
            "Expected one resume from the authenticated 100k parent",
        )
        if entry["status"] == "failure":
            return entry
        ends = [r for r in rows if r.get("type") == "end"]
        endpoint = [p for p in points if p["step"] == 62500 and p["env_steps"] == 500000]
        entry.update(status="incomplete", reason="terminal endpoint or success receipt missing")
        if len(ends) == 1 and ends[0].get("reason") == "completed":
            require(
                not any(r.get("type") == "unreadable_json" for r in rows)
                and rows[-1] == ends[0]
                and (ends[0].get("step"), ends[0].get("env_steps"), ends[0].get("updates"))
                == (62500, 500000, 61500),
                "Wrong child terminal counters",
            )
            if (
                len(endpoint) == 1
                and result.get("complete") is True
                and type(result.get("exit_code")) is int
                and result["exit_code"] == 0
                and result.get("reason") is None
            ):
                entry.update(
                    status="completed",
                    reason="validated continuation endpoint",
                    endpoint_return=endpoint[0]["mean_return"],
                )
    except FileNotFoundError as error:
        entry["reason"] = str(error)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        if entry["status"] != "failure":
            entry["status"] = "invalid"
        entry["reason"] = str(error)
    return entry


def summarize(root: Path) -> dict:
    runs = [read_chain(path) for path in sorted(root.glob("*/job.json"))]
    counts = Counter((r["arm"], r["seed"]) for r in runs if r["arm"] in ARMS)
    for run in runs:
        if counts[run["arm"], run["seed"]] > 1:
            run["endpoint_return"] = None
            run["duplicate"] = True
            if run["status"] == "completed":
                run.update(
                    status="duplicate", reason="Repeated child arm/seed; no attempt selected"
                )
    scored = {(r["arm"], r["seed"]): r for r in runs if r["endpoint_return"] is not None}
    arms, pairs = {}, []
    for arm in ARMS:
        values = [r["endpoint_return"] for (a, _), r in scored.items() if a == arm]
        arms[arm] = {
            "n_seeds": len(values),
            "mean_return": statistics.mean(values) if values else None,
            "seed_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    for seed in sorted({seed for _, seed in scored}):
        if all((arm, seed) in scored for arm in ARMS):
            a, b = (scored[arm, seed] for arm in ARMS)
            if scientific(a["config"], RUNTIME | {"arm"}) == scientific(
                b["config"], RUNTIME | {"arm"}
            ):
                pairs.append(
                    {
                        "seed": seed,
                        "curl": a["endpoint_return"],
                        "no_curl": b["endpoint_return"],
                        "difference": a["endpoint_return"] - b["endpoint_return"],
                    }
                )
    return {
        "analysis_kind": "100k-to-500k continuation",
        "target_env_steps": 500000,
        "runs_root": str(root.resolve()),
        "runs": runs,
        "arms": arms,
        "pairs": pairs,
        "status_counts": dict(Counter(r["status"] for r in runs)),
    }


def write_outputs(summary, output: Path, plot=False):
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# CURL 500k continuation results",
        "",
        "Exploratory episode-boundary continuations: one parent-child chain is one training seed, "
        "not a new replicate. Simulator physics/frame history was not serialized. "
        "The old 100k cohort is separate. Fixed 500,000-step scores average ten evaluation "
        "episodes of the terminal policy; missing/incomplete/failure scores remain missing.",
        "",
        "| Arm | Completed chains | Mean return | Across-seed SD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm, stats in summary["arms"].items():
        values = [
            "missing" if stats[k] is None else f"{stats[k]:.2f}" for k in ("mean_return", "seed_sd")
        ]
        lines.append(f"| {arm} | {stats['n_seeds']}/3 | {' | '.join(values)} |")
    lines += ["", "## Matched endpoint pairs", ""]
    lines += [
        f"Seed {p['seed']}: CURL minus control = {p['difference']:+.2f}." for p in summary["pairs"]
    ] or ["No eligible matched pair."]
    lines += [
        "",
        "## All attempts",
        "",
        "Status counts: " + json.dumps(summary["status_counts"]),
        "",
    ]
    lines += [
        f"- {r['arm']} seed {r['seed']}: {r['status']}; {r['reason']}. `{r['path']}`"
        for r in summary["runs"]
    ] or ["No continuation job receipts discovered."]
    lines += [
        "",
        "JSON preserves parent receipts, config/checkpoint identities and curve source paths. "
        "Only the small parent config is rehashed; checkpoint weights are never read. "
        "Repeated child attempts cannot contribute endpoint scores. Pairs require matching "
        "scientific configurations. The control retains crops; "
        "it is not the paper's Pixel SAC baseline.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        seeds = sorted({r["seed"] for r in summary["runs"] if r["seed"] is not None})
        for run in summary["runs"]:
            points = run["curve"]
            if points:
                ax.plot(
                    [p["env_steps"] for p in points],
                    [p["mean_return"] for p in points],
                    color=plt.get_cmap("tab10")(seeds.index(run["seed"]) % 10),
                    linestyle="-" if run["arm"] == "curl" else "--",
                    label=f"{run['arm']} seed {run['seed']} ({run['status']})",
                )
        ax.set(
            xlabel="Training simulator steps",
            ylabel="Mean evaluation episode return",
            xlim=(0, 500000),
            ylim=(0, 1000),
            title="Exploratory 100k-to-500k continuations",
        )
        ax.axvline(100000, color="gray", linewidth=0.8, linestyle=":")
        ax.grid(alpha=0.2)
        if any(r["curve"] for r in summary["runs"]):
            ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"learning-curves.{suffix}", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true", help="Also write PDF/PNG using Matplotlib")
    args = parser.parse_args()
    write_outputs(summarize(args.runs), args.output, args.plot)
