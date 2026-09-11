"""Rebuild the small, vector-only meeting figures from the portable evidence file."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
NAVY, TEAL, GREY, GOLD = "#17324D", "#007F7B", "#718096", "#BA741B"


def validate(data):
    sft, rows, comp = (data[key] for key in ("sft", "row_matching", "composition"))
    assert all(0 <= n <= d for n, d in zip(sft["correct"], sft["planned"], strict=True))
    assert rows["planned_per_bar"] == (
        rows["context_clusters"] * rows["reference_conditions"] * rows["late_records_per_context"]
    )
    neither, out_only, in_only, both = rows["correct"]
    assert both - in_only - out_only + neither == rows["interaction_correct"]
    keys = data["stable_keys"]
    assert keys["planned_per_bar"] == (
        keys["context_clusters"] * keys["reference_conditions"] * keys["late_records_per_context"]
    )
    assert all(0 <= n <= keys["planned_per_bar"] for n in keys["correct"])
    cross = data["cross_model_keys"]
    assert cross["planned_per_bar"] == 16 * 3 * 32
    assert len(cross["models"]) == len(cross["correct"]) == 2
    assert all(len(row) == len(cross["labels"]) == 3 for row in cross["correct"])
    assert all(0 <= n <= cross["planned_per_bar"] for row in cross["correct"] for n in row)
    assert cross["correct"][0] == [keys["correct"][i] for i in (0, 1, 3)]
    assert comp["context_clusters"] == 4 and comp["episodes"] == 8
    assert all(0 <= x <= comp["label_slots"] for x in comp["correct_labels"])
    assert all(0 <= x <= comp["episodes"] for x in comp["exact_answers"])
    root = Path(data["research_store"])
    checked = 0
    unavailable_sources = []
    for item in data.values():
        if not isinstance(item, dict):
            continue
        for source in item.get("sources", []):
            path = root / source["path"]
            if path.exists():
                assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], path
                checked += 1
            else:
                unavailable_sources.append(source["path"])
    return checked, unavailable_sources


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B7C1CC")
    ax.tick_params(axis="both", length=0, pad=9)
    ax.grid(axis="y", color="#E6EBEF", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.pdf", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main():
    data = json.loads((HERE / "data/claims.json").read_text())
    checked, unavailable_sources = validate(data)
    (HERE / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "text.color": NAVY,
                         "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                         "pdf.fonttype": 42, "axes.titlesize": 17})

    s = data["sft"]
    fig, ax = plt.subplots(figsize=(9.5, 2.9), layout="constrained")
    positions = range(len(s["correct"]))
    bars = ax.bar(positions, s["correct"], width=0.5, color=[GREY, TEAL, TEAL])
    ax.set(ylim=(0, 72), yticks=[0, 18, 36, 54, 72],
           xticks=positions, xticklabels=s["labels"], ylabel="Verified successes")
    ax.bar_label(bars, labels=[f"{n} / {d}" for n, d in
                              zip(s["correct"], s["planned"], strict=True)],
                 padding=7, fontsize=19, color=NAVY)
    style(ax)
    save(fig, "training")

    r = data["row_matching"]
    fig, ax = plt.subplots(figsize=(9.5, 2.8), layout="constrained")
    values = [n / r["planned_per_bar"] * 100 for n in r["correct"]]
    bars = ax.bar(range(4), values, width=0.62, color=[GREY, GREY, GREY, TEAL])
    ax.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], xticks=range(4),
           xticklabels=r["labels"], ylabel="Correct labels")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=7, fontsize=18)
    style(ax)
    save(fig, "row-matching")

    k = data["stable_keys"]
    fig, ax = plt.subplots(figsize=(9.5, 2.8), layout="constrained")
    values = [n / k["planned_per_bar"] * 100 for n in k["correct"]]
    bars = ax.bar(range(4), values, width=0.62, color=[GREY, TEAL, TEAL, TEAL])
    ax.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], xticks=range(4),
           xticklabels=k["labels"], ylabel="Correct labels")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=7, fontsize=18)
    style(ax)
    save(fig, "stable-keys")

    cross = data["cross_model_keys"]
    fig, ax = plt.subplots(figsize=(9.5, 2.7), layout="constrained")
    for index, (model, counts) in enumerate(zip(cross["models"], cross["correct"], strict=True)):
        values = [100 * n / cross["planned_per_bar"] for n in counts]
        positions = [x + (index - 0.5) * 0.34 for x in range(3)]
        bars = ax.bar(positions, values, width=0.31, color=[NAVY, TEAL][index], label=model)
        ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=5, fontsize=16)
    ax.set(ylim=(0, 115), yticks=[0, 25, 50, 75, 100], xticks=range(3),
           xticklabels=cross["labels"], ylabel="Correct labels")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=15)
    style(ax)
    save(fig, "cross-model-keys")

    c = data["composition"]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.0), layout="constrained", width_ratios=[1, 1.3])
    values = [100 * n / c["label_slots"] for n in c["correct_labels"]]
    bars = axes[0].bar([0, 1], values, width=0.55, color=[GREY, TEAL])
    axes[0].set(ylim=(0, 112), yticks=[0, 25, 50, 75, 100], xticks=[0, 1],
                xticklabels=["Before", "After"], title="The helper's labels improved.")
    axes[0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[0].bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=6, fontsize=18)
    vals = c["exact_answers"] + [c["reference_labels_exact"]]
    bars = axes[1].bar(range(3), vals, width=0.58, color=[GREY, TEAL, GOLD])
    axes[1].set(ylim=(0, 9.2), yticks=[0, 2, 4, 6, 8], xticks=range(3),
                xticklabels=["Before", "After", "Reference\nlabels"],
                title="Exact final answers remained rare.")
    axes[1].bar_label(bars, labels=[f"{n} / 8" for n in vals], padding=7, fontsize=18)
    for ax in axes:
        style(ax)
    save(fig, "composition")
    print(json.dumps({"figures": 5, "local_source_hashes_verified": checked,
                      "source_files_unavailable_on_this_machine": unavailable_sources,
                      "note": "No error bars: exploratory clustered panels, not independent label trials."}))


if __name__ == "__main__":
    main()
