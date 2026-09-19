"""Create violin plots for weather-variable distributions across seven datasets.

The script reads the original numeric observations from ``Processed dataset`` and
saves an SVG figure beside this script. To keep kernel-density estimation fast
and reproducible, each dataset-variable distribution is sampled to at most
20,000 values using a fixed random seed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "Processed dataset"
OUTPUT_SVG = SCRIPT_DIR / "weather_variable_violin_plots.svg"
MAX_POINTS_PER_GROUP = 20_000
RANDOM_SEED = 20260909

VARIABLES = [
    ("wind_speed_ms", "Wind speed (m/s)"),
    ("temperature_c", "Temperature (°C)"),
    ("pres1", "Pressure 1 (inches)"),
    ("pres2", "Pressure 2 (inches)"),
    ("pres3", "Pressure 3 (inches)"),
]


def read_values(file_path: Path, column: str, seed: int) -> np.ndarray:
    """Read valid numeric values for one variable and sample deterministically."""
    data = pd.read_csv(file_path, usecols=[column], dtype="string", keep_default_na=False)
    values = pd.to_numeric(data[column].replace("M", pd.NA), errors="coerce").dropna()
    if len(values) > MAX_POINTS_PER_GROUP:
        values = values.sample(MAX_POINTS_PER_GROUP, random_state=seed)
    return values.to_numpy(dtype=float)


def draw_violin_panel(ax: plt.Axes, values_by_dataset: list[np.ndarray], label: str, names: list[str]) -> None:
    positions = np.arange(1, len(names) + 1)
    valid = [(position, values) for position, values in zip(positions, values_by_dataset) if len(values)]
    if not valid:
        ax.text(0.5, 0.5, "No valid observations", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(label, fontsize=10)
        return

    violin = ax.violinplot(
        [values for _, values in valid], positions=[position for position, _ in valid],
        widths=0.76, showmeans=False, showmedians=True, showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#88AFC0")
        body.set_edgecolor("#4A4A4A")
        body.set_linewidth(0.7)
        body.set_alpha(0.82)
    violin["cmedians"].set_color("#252525")
    violin["cmedians"].set_linewidth(1.0)

    ax.set_ylabel(label, fontsize=10)
    ax.set_xlim(0.35, len(names) + 0.65)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="y", labelsize=8, width=0.8, length=3)
    ax.tick_params(axis="x", width=0.8, length=3)


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if len(files) != 7:
        raise ValueError(f"Expected 7 CSV datasets in {PROCESSED_DIR}, found {len(files)}")
    dataset_names = [file.stem for file in files]

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(nrows=len(VARIABLES), ncols=1, figsize=(7.2, 10.0), sharex=True)

    for variable_index, ((column, label), ax) in enumerate(zip(VARIABLES, axes)):
        values_by_dataset = [
            read_values(file, column, RANDOM_SEED + variable_index * 100 + file_index)
            for file_index, file in enumerate(files)
        ]
        draw_violin_panel(ax, values_by_dataset, label, dataset_names)

    axes[-1].set_xlabel("Dataset", fontsize=11, labelpad=8)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG saved to: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
