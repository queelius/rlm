"""Plot admitted completed 500k pairs from one immutable analysis snapshot."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("curl", "no_curl")
NAMES = {123: "First training pair", 456: "Second training pair", 789: "Third training pair"}


def select_pairs(summary: dict) -> list[dict]:
    pairs = sorted(summary["pairs"], key=lambda pair: pair["seed"])
    seeds = [p["seed"] for p in pairs]
    if (
        summary["target_env_steps"] != 500000
        or not 1 <= len(seeds) <= 3
        or len(set(seeds)) != len(seeds)
        or any(s not in NAMES for s in seeds)
    ):
        raise ValueError("Expected 1..3 unique admitted completed 500k training pairs")
    selected = []
    for pair in pairs:
        result = {"seed": pair["seed"]}
        for arm in ARMS:
            runs = [
                r
                for r in summary["runs"]
                if r["seed"] == pair["seed"] and r["arm"] == arm and r["status"] == "completed"
            ]
            if len(runs) != 1:
                raise ValueError("Expected one completed run per admitted arm/seed")
            run = runs[0]
            points = run["curve"]
            steps = [p["env_steps"] for p in points]
            if (
                not steps
                or steps != sorted(set(steps))
                or steps[0] != 0
                or steps[-1] != 500000
                or steps.count(100000) != 1
                or any(p["episodes"] != 10 for p in points)
                or any(not math.isfinite(p["mean_return"]) for p in points)
                or points[-1]["mean_return"] != run["endpoint_return"]
                or run["endpoint_return"] != pair[arm]
            ):
                raise ValueError("Joined curve or fixed endpoint disagrees with the admitted pair")
            result[arm] = run
        selected.append(result)
    return selected


def generate(summary_path: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    content = summary_path.read_bytes()
    selected = select_pairs(json.loads(content))
    fig, axes = plt.subplots(1, len(selected), figsize=(8, 5), sharey=True, squeeze=False)
    styles = {
        "curl": ("#1f77b4", "-", "CURL with image matching"),
        "no_curl": ("#d97706", "--", "Same crops without image matching"),
    }
    for ax, pair in zip(axes[0], selected, strict=True):
        for arm in ARMS:
            run = pair[arm]
            color, style, label = styles[arm]
            points = run["curve"]
            ax.plot(
                [p["env_steps"] for p in points],
                [p["mean_return"] for p in points],
                color=color,
                linestyle=style,
                linewidth=1.5,
                label=label,
            )
            endpoint = run["endpoint_return"]
            highest = endpoint > pair["no_curl" if arm == "curl" else "curl"]["endpoint_return"]
            ax.scatter(
                500000,
                endpoint,
                s=23,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
                clip_on=False,
            )
            ax.annotate(
                f"{endpoint:.0f}",
                (500000, endpoint),
                xytext=(340000, 930 if highest else 710),
                color=color,
                fontsize=11,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
            )
        ax.axvline(100000, color="gray", linestyle=":", linewidth=0.9)
        ax.set(
            xlim=(0, 500000),
            ylim=(0, 1000),
            xlabel="Retained training steps",
            title=f"{NAMES[pair['seed']]}\nOne training seed ({pair['seed']})",
        )
        ax.set_xticks((0, 100000, 300000, 500000))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k")
        )
        ax.tick_params(labelleft=True, labelsize=9)
        ax.grid(alpha=0.2)
    axes[0][0].set_ylabel("Mean evaluation return\n(higher is better)")
    fig.text(
        0.11, 0.955, "Each point averages ten fixed evaluation starts; no smoothing.", fontsize=10
    )
    fig.legend(
        *axes[0][0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.54, 0.145),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    for i, pair in enumerate(selected):
        costs = [pair[a]["physical_training_env_steps_for_completed_chain"] / 1000 for a in ARMS]
        fig.text(
            0.11,
            0.115 - i * 0.032,
            f"{NAMES[pair['seed']]}: physical training steps = CURL {costs[0]:.0f}k, "
            f"control {costs[1]:.0f}k; both retain 500k.",
            fontsize=9,
        )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.82, bottom=0.29, wspace=0.28)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"completed-500k-pairs.{suffix}", dpi=200)
    plt.close(fig)
    return {
        "data_sha256": hashlib.sha256(content).hexdigest(),
        "plotted_points": {p["seed"]: {a: len(p[a]["curve"]) for a in ARMS} for p in selected},
        "endpoints": {p["seed"]: {a: p[a]["endpoint_return"] for a in ARMS} for p in selected},
    }


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=directory / "extension-data" / "two-500k-pairs-summary.json"
    )
    parser.add_argument("--output", type=Path, default=directory / "figures")
    args = parser.parse_args()
    print(json.dumps(generate(args.summary, args.output), indent=2))
