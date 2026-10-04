"""Plot the immutable, validated seed-123 pair; never read native logs or weights."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("curl", "no_curl")


def select_pair(summary: dict) -> dict:
    pairs = [p for p in summary["pairs"] if p["seed"] == 123]
    if len(pairs) != 1 or summary["target_env_steps"] != 500000:
        raise ValueError("Expected one validated fixed 500k seed-123 pair")
    selected = {}
    for arm in ARMS:
        candidates = [
            r
            for r in summary["runs"]
            if r["arm"] == arm and r["seed"] == 123 and r["status"] == "completed"
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one completed seed-123 run for {arm}")
        run = candidates[0]
        points = run["curve"]
        steps = [p["env_steps"] for p in points]
        if (
            steps != sorted(set(steps))
            or steps[0] != 0
            or steps[-1] != 500000
            or steps.count(100000) != 1
            or any(p["episodes"] != 10 for p in points)
            or any(not math.isfinite(p["mean_return"]) for p in points)
        ):
            raise ValueError("Expected complete, unsmoothed joined evaluation curves")
        if (
            points[-1]["mean_return"] != run["endpoint_return"]
            or run["endpoint_return"] != pairs[0][arm]
        ):
            raise ValueError("Recorded endpoint disagrees with the validated pair")
        selected[arm] = run
    return selected


def generate(summary_path: Path, output: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    content = summary_path.read_bytes()
    selected = select_pair(json.loads(content))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {
        "curl": ("#1f77b4", "-", "CURL with image matching"),
        "no_curl": ("#d97706", "--", "Same crops without image matching"),
    }
    for arm, run in selected.items():
        color, style, label = styles[arm]
        points = run["curve"]
        ax.plot(
            [p["env_steps"] for p in points],
            [p["mean_return"] for p in points],
            color=color,
            linestyle=style,
            linewidth=1.7,
            label=label,
        )
        for point in [p for p in points if p["env_steps"] in (100000, 500000)]:
            x, y = point["env_steps"], point["mean_return"]
            ax.scatter(
                x, y, color=color, edgecolor="white", linewidth=0.7, s=28, zorder=4, clip_on=False
            )
            if x == 100000:
                ax.annotate(
                    f"{y:.0f}",
                    (x, y),
                    xytext=(-10, 12),
                    textcoords="offset points",
                    ha="right",
                    color=color,
                    fontsize=10,
                )
        endpoint = run["endpoint_return"]
        position = (395000, 705) if arm == "curl" else (395000, 968)
        ax.annotate(
            f"500k: {endpoint:.0f}",
            (500000, endpoint),
            xytext=position,
            color=color,
            fontsize=10,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )
    ax.axvline(100000, color="gray", linestyle=":", linewidth=1)
    ax.set(
        xlim=(0, 500000),
        ylim=(0, 1000),
        xlabel="Retained training steps",
        ylabel="Mean evaluation return\n(10 fixed starts)",
        title="The early lead reversed in this training pair",
    )
    ax.set_xticks(range(0, 500001, 100000))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{value / 1000:.0f}k")
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.27)
    curl_cost, control_cost = (
        selected[a]["physical_training_env_steps_for_completed_chain"] for a in ARMS
    )
    fig.text(
        0.11,
        0.10,
        "One training pair (seed 123); points are recorded means, with no smoothing.",
        fontsize=9,
    )
    fig.text(
        0.11,
        0.045,
        f"Recovery cost: CURL {curl_cost / 1000:.0f}k physical training steps; "
        f"control {control_cost / 1000:.0f}k. Both retain 500k.",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"first-500k-pair.{suffix}", dpi=200)
    plt.close(fig)
    return {
        "data_sha256": hashlib.sha256(content).hexdigest(),
        "plotted_points": {a: len(r["curve"]) for a, r in selected.items()},
        "endpoints": {a: r["endpoint_return"] for a, r in selected.items()},
    }


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=directory / "extension-data" / "first-500k-pair-summary.json",
    )
    parser.add_argument("--output", type=Path, default=directory / "figures")
    args = parser.parse_args()
    print(json.dumps(generate(args.summary, args.output), indent=2))
