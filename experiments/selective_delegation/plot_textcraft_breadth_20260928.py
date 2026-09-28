"""Render the fixed breadth finding; requires matplotlib, no model environment."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot(report: Path, output: Path) -> None:
    result = json.loads(report.read_text())
    for suffix in (".png", ".svg", ".json"):
        if output.with_suffix(suffix).exists():
            raise FileExistsError(output.with_suffix(suffix))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    colors = {"discovery": "#176b91", "known_recipe_corrected": "#b56618"}
    labels = {"discovery": "Discovery", "known_recipe_corrected": "Known-recipe corrected"}
    modes = {"raw": ("--", "o"), "binder": ("-", "s")}
    for index, teacher in enumerate(colors):
        for offset, mode in ((-0.16, "raw"), (0.16, "binder")):
            row = result["arms"][f"{teacher}/{mode}"]
            axes[0].bar(
                index + offset,
                100 * row["success_rate"],
                width=0.28,
                color=colors[teacher],
                alpha=0.5 if mode == "raw" else 1,
            )
            axes[0].text(
                index + offset,
                100 * row["success_rate"] + 1.4,
                f"{row['successes']}/768",
                ha="center",
                fontsize=9,
            )
            depth = result["conditional"]["declared_depth"]
            values = [
                100 * depth[str(d)]["arms"][f"{teacher}/{mode}"]["success_rate"]
                for d in (2, 3, 4, 5)
            ]
            axes[1].plot(
                (2, 3, 4, 5),
                values,
                color=colors[teacher],
                linestyle=modes[mode][0],
                marker=modes[mode][1],
                label=f"{labels[teacher]}, {mode}",
                linewidth=2,
            )
    axes[0].set_xticks(
        (0, 1), ("Discovery\nraw    binder", "Known-recipe corrected\nraw    binder")
    )
    axes[0].set_ylim(0, 60)
    axes[0].set_ylabel("Native success (%)")
    axes[0].set_title("Teacher package dominates the gap", loc="left", fontsize=11)
    axes[1].set_xticks((2, 3, 4, 5))
    axes[1].set_ylim(0, 103)
    axes[1].set_xlabel("Declared task depth")
    axes[1].set_ylabel("Native success (%)")
    axes[1].set_title("Binding helps; deep tasks remain hard", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.subplots_adjust(bottom=0.25, top=0.9, wspace=0.30)
    figure.text(
        0.08,
        0.04,
        "Panels00–05 only · 48 task identities · four shared worlds · "
        "two training seeds · two repeats\n"
        "Same supervised action/label multisets; query names publicly available: discovery167/167, "
        "known-recipe32/167. Descriptive mechanism.",
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    with output.with_suffix(".json").open("x") as stream:
        json.dump(
            {
                "report": str(report),
                "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib": matplotlib.__version__,
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output stem for PNG/SVG/receipt"
    )
    args = parser.parse_args()
    plot(args.report, args.output)
