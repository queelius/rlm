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
    assert sft["possible_correct_bounds"] == [
        [n, n + missing] for n, missing in zip(sft["correct"], sft["unavailable"], strict=True)
    ]
    nested = data["nested_batch"]
    assert nested["planned_denominators"] == [16 * size for size in nested["sizes"]]
    assert nested["planned_calls"] == nested["available_calls"] == nested["valid_calls"] == 240
    assert all(len(row) == len(nested["sizes"]) for row in nested["correct"])
    assert all(
        0 <= n <= d
        for row in nested["correct"]
        for n, d in zip(row, nested["planned_denominators"], strict=True)
    )
    replication = data["nested_batch_mistral"]
    assert replication["sizes"] == nested["sizes"]
    assert replication["planned_denominators"] == nested["planned_denominators"]
    assert replication["planned_calls"] == replication["available_calls"] == 240
    invalid = replication["invalid_calls_by_format"]
    assert len(replication["correct"]) == len(invalid) == 3
    assert replication["valid_calls"] + sum(map(sum, invalid)) == 240
    for counts, failures in zip(replication["correct"], invalid, strict=True):
        for good, bad, size, total in zip(
            counts, failures, replication["sizes"], replication["planned_denominators"], strict=True
        ):
            assert 0 <= bad <= 16
            assert 0 <= good <= total - bad * size
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
    assert len(cross["models"]) == len(cross["correct"]) == 3
    assert all(len(row) == len(cross["labels"]) == 3 for row in cross["correct"])
    assert all(0 <= n <= cross["planned_per_bar"] for row in cross["correct"] for n in row)
    assert cross["correct"][0] == [keys["correct"][i] for i in (0, 1, 3)]
    assert cross["correct"][2] == [490, 857, 807]
    matching = data["literal_tag_matching"]
    assert matching["planned_per_bar"] == (
        matching["context_clusters"]
        * matching["reference_conditions"]
        * matching["tag_cells_per_bar"]
        * matching["late_records_per_context"]
    )
    assert matching["correct"] == [1000, 2420]
    assert (
        matching["planned_calls"] == matching["available_calls"] == matching["valid_calls"] == 192
    )
    for name in (
        "reward_training",
        "reward_training_lower_rate",
        "reward_training_longer",
        "compound_execution",
    ):
        reward = data[name]
        assert reward["possible_correct_bounds"] == [
            [n, n + missing]
            for n, missing in zip(reward["correct"], reward["unavailable"], strict=True)
        ]
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
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "pdf.fonttype": 42,
            "axes.titlesize": 17,
        }
    )

    s = data["sft"]
    fig, ax = plt.subplots(figsize=(9.5, 2.7), layout="constrained")
    positions = range(len(s["correct"]))
    bars = ax.bar(positions, s["correct"], width=0.5, color=[GREY, TEAL, TEAL])
    ax.set(
        ylim=(0, 72),
        yticks=[0, 18, 36, 54, 72],
        xticks=positions,
        xticklabels=s["labels"],
        ylabel="Correct answer\nand calculation",
    )
    ax.bar_label(
        bars,
        labels=[f"{n} / {d}" for n, d in zip(s["correct"], s["planned"], strict=True)],
        padding=7,
        fontsize=19,
        color=NAVY,
    )
    style(ax)
    ax.tick_params(labelsize=16)
    ax.set_ylabel("Correct answer\nand calculation", fontsize=16)
    save(fig, "training")

    n = data["nested_batch"]
    fig, ax = plt.subplots(figsize=(9.5, 2.95), layout="constrained")
    for counts, label, color, marker, dash, offset in zip(
        n["correct"],
        ["No matching tags", "Row numbers", "Arbitrary tags"],
        [GREY, TEAL, GOLD],
        ["o", "s", "^"],
        ["--", "-", "-"],
        [0, 7, -7],
        strict=True,
    ):
        values = [
            100 * good / total
            for good, total in zip(counts, n["planned_denominators"], strict=True)
        ]
        ax.plot(
            n["sizes"],
            values,
            color=color,
            marker=marker,
            linestyle=dash,
            linewidth=2.5,
            markersize=7,
        )
        ax.annotate(
            f"{label}: {values[-1]:.0f}%",
            xy=(64, values[-1]),
            xytext=(67, values[-1] + offset),
            textcoords="data",
            color=color,
            fontsize=14,
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8},
        )
    ax.set(
        xlim=(5, 91),
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
        xticks=n["sizes"],
        xlabel="Reading questions answered in one helper call",
        ylabel="Correct reading answers",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    style(ax)
    save(fig, "batch-size")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.2), layout="constrained", sharey=True)
    for ax, panel, title in zip(
        axes, [n, data["nested_batch_mistral"]], ["Qwen3-4B", "Mistral-7B"], strict=True
    ):
        for counts, label, color, marker in zip(
            panel["correct"],
            ["No matching tags", "Row numbers", "Arbitrary tags"],
            [GREY, TEAL, GOLD],
            ["o", "s", "^"],
            strict=True,
        ):
            values = [
                100 * good / total
                for good, total in zip(counts, panel["planned_denominators"], strict=True)
            ]
            ax.plot(
                panel["sizes"],
                values,
                color=color,
                marker=marker,
                label=label,
                linewidth=2.5,
                markersize=6,
            )
        ax.set(
            title=title,
            ylim=(0, 100),
            yticks=[0, 25, 50, 75, 100],
            xticks=panel["sizes"],
            xlabel="Reading questions in one call",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        style(ax)
    axes[0].set_ylabel("Correct reading answers")
    # A shared legend avoids six labels competing inside the small panels.
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
        columnspacing=1.8,
    )
    save(fig, "batch-size-models")

    # Main-talk view: exactly the unnamed and arbitrary-name conditions.
    # The complete three-condition view remains in the optional backup slides.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.1), layout="constrained", sharey=True)
    for ax, panel, title in zip(
        axes, [n, data["nested_batch_mistral"]], ["Qwen3-4B", "Mistral-7B"], strict=True
    ):
        for index, label, color, marker in (
            (0, "Without matching names", GREY, "o"),
            (2, "With matching names", TEAL, "s"),
        ):
            values = [
                100 * good / total
                for good, total in zip(
                    panel["correct"][index], panel["planned_denominators"], strict=True
                )
            ]
            ax.plot(
                panel["sizes"],
                values,
                label=label,
                color=color,
                marker=marker,
                linewidth=3,
                markersize=7,
            )
            ax.annotate(
                f"{values[-1]:.0f}%",
                (64, values[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=color,
                fontsize=19,
            )
        ax.set(
            title=title,
            xlim=(5, 76),
            ylim=(0, 100),
            yticks=[0, 50, 100],
            xticks=panel["sizes"],
            xlabel="Questions per helper call",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        style(ax)
        ax.tick_params(labelsize=17)
        ax.set_xlabel("Questions per helper call", fontsize=17)
        ax.set_title(title, fontsize=20)
    axes[0].set_ylabel("Correct answers", fontsize=17)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=17,
        columnspacing=2,
    )
    save(fig, "batch-size-simple")

    r = data["row_matching"]
    fig, ax = plt.subplots(figsize=(9.5, 2.8), layout="constrained")
    values = [n / r["planned_per_bar"] * 100 for n in r["correct"]]
    bars = ax.bar(range(4), values, width=0.62, color=[GREY, GREY, GREY, TEAL])
    ax.set(
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
        xticks=range(4),
        xticklabels=r["labels"],
        ylabel="Correct labels\n(last 32 answers)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=7, fontsize=18)
    style(ax)
    save(fig, "row-matching")

    k = data["stable_keys"]
    fig, ax = plt.subplots(figsize=(9.5, 2.8), layout="constrained")
    values = [n / k["planned_per_bar"] * 100 for n in k["correct"]]
    bars = ax.bar(range(4), values, width=0.62, color=[GREY, TEAL, TEAL, TEAL])
    ax.set(
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
        xticks=range(4),
        xticklabels=k["labels"],
        ylabel="Correct labels",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=7, fontsize=18)
    style(ax)
    save(fig, "stable-keys")

    cross = data["cross_model_keys"]
    fig, ax = plt.subplots(figsize=(9.5, 2.7), layout="constrained")
    for index, (model, counts) in enumerate(zip(cross["models"], cross["correct"], strict=True)):
        values = [100 * n / cross["planned_per_bar"] for n in counts]
        positions = [x + (index - 1) * 0.26 for x in range(3)]
        bars = ax.bar(positions, values, width=0.24, color=[NAVY, TEAL, GOLD][index], label=model)
        ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=5, fontsize=14)
    ax.set(
        ylim=(0, 115),
        yticks=[0, 25, 50, 75, 100],
        xticks=range(3),
        xticklabels=cross["labels"],
        ylabel="Correct labels\n(last 32 answers)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=15)
    style(ax)
    save(fig, "cross-model-keys")

    matching = data["literal_tag_matching"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6), layout="constrained")
    values = [100 * n / matching["planned_per_bar"] for n in matching["correct"]]
    bars = ax.bar(range(2), values, width=0.5, color=[GREY, TEAL])
    ax.set(
        ylim=(0, 100),
        yticks=[0, 25, 50, 75, 100],
        xticks=range(2),
        xticklabels=matching["labels"],
        ylabel="Correct reading labels\n(last 32 answers)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=7, fontsize=21)
    style(ax)
    save(fig, "literal-tag-matching")

    c = data["composition"]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 3.0), layout="constrained", width_ratios=[1, 1.3])
    values = [100 * n / c["label_slots"] for n in c["correct_labels"]]
    bars = axes[0].bar([0, 1], values, width=0.55, color=[GREY, TEAL])
    axes[0].set(
        ylim=(0, 112),
        yticks=[0, 25, 50, 75, 100],
        xticks=[0, 1],
        xticklabels=["Before helper\ntraining", "After helper\ntraining"],
        title="Correct helper labels",
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[0].bar_label(bars, labels=[f"{n:.1f}%" for n in values], padding=6, fontsize=18)
    vals = c["exact_answers"] + [c["reference_labels_exact"]]
    bars = axes[1].bar(range(3), vals, width=0.58, color=[GREY, TEAL, GOLD])
    axes[1].set(
        ylim=(0, 9.2),
        yticks=[0, 2, 4, 6, 8],
        xticks=range(3),
        xticklabels=["Before helper\ntraining", "After helper\ntraining", "Known-correct\nlabels"],
        title="Correct final answers",
    )
    axes[1].bar_label(bars, labels=[f"{n} / 8" for n in vals], padding=7, fontsize=18)
    for ax in axes:
        style(ax)
    save(fig, "composition")
    print(
        json.dumps(
            {
                "figures": 9,
                "local_source_hashes_verified": checked,
                "source_files_unavailable_on_this_machine": unavailable_sources,
                "note": "No error bars: exploratory clustered panels, "
                "not independent label trials.",
            }
        )
    )


if __name__ == "__main__":
    main()
