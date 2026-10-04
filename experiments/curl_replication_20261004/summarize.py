"""Summarize native CURL records without checkpoint selection or pseudoreplication."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("curl", "no_curl")
TARGET = 100000


def read_attempt(path: Path) -> list[dict]:
    config = json.loads((path.parent / "config.json").read_text())
    if config.get("arm") not in ARMS or not isinstance(config.get("seed"), int):
        raise ValueError("config requires recognized arm and integer seed")
    segments, segment, parse_error = [], [], None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or "type" not in record:
                raise ValueError("record requires type")
        except (ValueError, TypeError) as error:
            parse_error = f"line {number}: {error}"
            break
        if record["type"] in ("start", "resume") and segment:
            segments.append(segment)
            segment = []
        segment.append(record)
    segments.append(segment)
    resumed = any(r["type"] == "resume" for s in segments for r in s)
    attempts = []
    for index, records in enumerate(segments):
        batches = defaultdict(list)
        for record in records:
            if record["type"] == "eval_episode":
                batches[(record.get("step"), record.get("env_steps"))].append(record)
        curve = []
        expected = config.get("evaluation_seeds", list(range(10000, 10010)))
        for (step, env_steps), episodes in sorted(batches.items()):
            values = [e.get("return") for e in episodes]
            if (
                len(episodes) == 10
                and len(set(expected)) == 10
                and sorted(e.get("eval_seed") for e in episodes) == sorted(expected)
                and all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
                and isinstance(step, int)
                and isinstance(env_steps, int)
            ):
                curve.append(
                    {
                        "step": step,
                        "env_steps": env_steps,
                        "mean_return": statistics.mean(values),
                        "episodes": 10,
                    }
                )
        end = next((r for r in reversed(records) if r["type"] == "end"), {})
        failure = next((r for r in reversed(records) if r["type"] == "failure"), None)
        endpoint = [p for p in curve if p["env_steps"] == TARGET and p["step"] == end.get("step")]
        eligible = (
            end.get("reason") == "completed"
            and end.get("env_steps") == TARGET
            and len(endpoint) == 1
            and not resumed
            and len(segments) == 1
        )
        status = "failure" if failure or parse_error else "completed" if eligible else "incomplete"
        attempts.append(
            {
                "path": str(path.parent.resolve()),
                "segment": index,
                "arm": config["arm"],
                "seed": config["seed"],
                "config": config,
                "status": status,
                "reason": parse_error or failure or end.get("reason", "running"),
                "resumed": resumed,
                "curve": curve,
                "endpoint_return": endpoint[0]["mean_return"] if status == "completed" else None,
            }
        )
    return attempts


def summarize(root: Path) -> dict:
    runs, excluded = [], []
    for path in sorted(root.rglob("metrics.jsonl")):
        if any("pilot" in part.lower() for part in path.relative_to(root).parts):
            excluded.append(str(path.parent))
            continue
        try:
            runs.extend(read_attempt(path))
        except (OSError, ValueError, TypeError, KeyError) as error:
            excluded.append({"path": str(path.parent), "error": str(error)})
    counts = Counter((r["arm"], r["seed"]) for r in runs)
    for run in runs:
        if counts[run["arm"], run["seed"]] > 1:
            run["endpoint_return"] = None
            if run["status"] != "failure":
                run.update(status="incomplete", reason="repeated arm/seed; no attempt selected")
    scored = {(r["arm"], r["seed"]): r for r in runs if r["endpoint_return"] is not None}
    arms = {}
    for arm in ARMS:
        values = [r["endpoint_return"] for (a, _), r in scored.items() if a == arm]
        arms[arm] = {
            "n_seeds": len(values),
            "planned_seeds": 3,
            "mean_return": statistics.mean(values) if values else None,
            "seed_sd": statistics.stdev(values) if len(values) > 1 else None,
        }
    pairs = []
    for seed in sorted({s for _, s in scored}):
        if all((a, seed) in scored for a in ARMS):
            a, b = (scored[arm, seed] for arm in ARMS)
            ignored = {"arm", "device", "max_seconds", "checkpoint_seconds"}
            if {k: v for k, v in a["config"].items() if k not in ignored} == {
                k: v for k, v in b["config"].items() if k not in ignored
            }:
                pairs.append(
                    {
                        "seed": seed,
                        "curl": a["endpoint_return"],
                        "no_curl": b["endpoint_return"],
                        "difference": a["endpoint_return"] - b["endpoint_return"],
                    }
                )
    return {
        "target_env_steps": TARGET,
        "runs_root": str(root.resolve()),
        "runs": runs,
        "arms": arms,
        "pairs": pairs,
        "excluded": excluded,
        "interpretation": "Exploratory; three planned training seeds per arm. "
        "Evaluation episodes are not training replicates.",
    }


def write_outputs(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# CURL replication results",
        "",
        summary["interpretation"],
        "",
        "Scores use the fixed 100,000 training-environment-step endpoint, averaging ten "
        "evaluation episodes of one unchanged policy. Missing results remain missing.",
        "",
        "| Arm | Completed seeds | Mean return | Across-seed SD |",
        "| --- | ---: | ---: | ---: |",
    ]

    def show(value):
        return "missing" if value is None else f"{value:.2f}"

    for arm, stats in summary["arms"].items():
        lines.append(
            f"| {arm} | {stats['n_seeds']}/3 | {show(stats['mean_return'])} | "
            f"{show(stats['seed_sd'])} |"
        )
    lines += ["", "## Matched pairs", ""]
    lines += [
        f"Seed {p['seed']}: CURL {p['curl']:.2f}, control {p['no_curl']:.2f}; "
        f"difference {p['difference']:+.2f}."
        for p in summary["pairs"]
    ] or ["No eligible matched pair."]
    lines += ["", "## Attempts", ""]
    lines += [
        f"- {r['arm']} seed {r['seed']}, segment {r['segment']}: {r['status']}; "
        f"{r['reason']}. Source: `{r['path']}`."
        for r in summary["runs"]
    ] or ["No native runs discovered."]
    lines += [
        "",
        "Pilot runs and invalid native inputs excluded: " + json.dumps(summary["excluded"]),
        "",
        "Repeated arm/seed attempts and resumed segments remain visible, but are excluded "
        "from endpoint aggregates. Pair differences require matching scientific configurations. "
        "The control retains crops; it is not the paper's Pixel SAC baseline. This comparison "
        "uses equal interactions, not equal wall time.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(8, 5))
    seeds = sorted({r["seed"] for r in summary["runs"]})
    colors = {seed: plt.get_cmap("tab10")(i % 10) for i, seed in enumerate(seeds)}
    for run in summary["runs"]:
        points = run["curve"]
        if points:
            ax.plot(
                [p["env_steps"] for p in points],
                [p["mean_return"] for p in points],
                color=colors[run["seed"]],
                linestyle="-" if run["arm"] == "curl" else "--",
                label=f"{run['arm']} seed {run['seed']} ({run['status']}, "
                f"segment {run['segment']})",
            )
    ax.set(
        xlabel="Training environment steps",
        ylabel="Mean evaluation episode return",
        ylim=(0, 1000),
        xlim=(0, TARGET),
        title="Exploratory CURL comparison; all recorded seed curves",
    )
    ax.grid(alpha=0.2)
    if any(r["curve"] for r in summary["runs"]):
        ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"learning-curves.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_outputs(summarize(args.runs), args.output)
