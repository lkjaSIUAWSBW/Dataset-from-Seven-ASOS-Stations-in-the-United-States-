"""Plot time coverage by index and save the publication-style figure as SVG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "dataset_time_coverage.csv"
OUTPUT_SVG = SCRIPT_DIR / "time_coverage_by_index.svg"


def main() -> None:
    data = pd.read_csv(INPUT_CSV)
    required_columns = {"index", "time coverage percent"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required column(s): {sorted(missing_columns)}")

    data = data.sort_values("index").copy()
    x = pd.to_numeric(data["index"], errors="raise").to_numpy()
    coverage = pd.to_numeric(data["time coverage percent"], errors="raise").to_numpy()

    
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",  
    })

    colors = [
        "#8FB5AA", "#C69AAA", "#A2A0C7", "#E6BA7B",
        "#9CB9C8", "#C8C783", "#9DB3C6", "#D3A087",
        "#AABF93", "#B9A5C9", "#D8C383", "#91B1A6",
    ]
    bar_colors = [colors[i % len(colors)] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(7.2, 4.7), dpi=300)
    ax.bar(
        x, coverage, width=0.66, color=bar_colors,
        edgecolor="#767676", linewidth=1.0, zorder=2,
    )
    ax.plot(
        x, coverage, color="#4A4A4A", linewidth=2.0, linestyle=(0, (3, 3)),
        marker="o", markersize=7.5, markerfacecolor="white",
        markeredgecolor="#4A4A4A", markeredgewidth=1.6, zorder=3,
    )

    for xi, value in zip(x, coverage):
        ax.annotate(
            f"{value:.2f}%", xy=(xi, value), xytext=(0, 6),
            textcoords="offset points", ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Index", fontsize=14, labelpad=8)
    ax.set_ylabel("Time coverage rate (%)", fontsize=14, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) if float(value).is_integer() else str(value) for value in x])
    ax.set_ylim(0, max(105, np.ceil((coverage.max() + 5) / 5) * 5))
    ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1, 20))
    ax.yaxis.grid(True, color="#D5D5D5", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.1, length=5)

    fig.tight_layout()
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG saved to: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
