"""Plot 24-hour autocorrelation functions for seven wind-speed time series.

Dependencies: pandas, numpy, matplotlib.
The processed CSV files must contain the ``wind_speed_ms`` column at a regular
10-minute time step. Missing values are not interpolated: for each lag, the ACF
uses only pairs where both wind-speed values are valid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "Processed dataset"
OUTPUT_SVG = SCRIPT_DIR / "wind_speed_acf_24h.svg"
MAX_LAG_STEPS = 144  
TIME_STEP_MINUTES = 10


def pairwise_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Calculate Pearson ACF at each lag from valid same-time-grid pairs."""
    acf = np.full(max_lag + 1, np.nan, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        first = values[:-lag]
        second = values[lag:]
        valid = np.isfinite(first) & np.isfinite(second)
        if valid.sum() < 2:
            continue
        x = first[valid]
        y = second[valid]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denominator > 0:
            acf[lag] = np.sum(x_centered * y_centered) / denominator
    return acf


def load_wind_speed(file_path: Path) -> np.ndarray:
    data = pd.read_csv(file_path, usecols=["wind_speed_ms"], dtype="string", keep_default_na=False)
    return pd.to_numeric(data["wind_speed_ms"].replace("M", pd.NA), errors="coerce").to_numpy(dtype=float)


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if len(files) != 7:
        raise ValueError(f"Expected 7 CSV datasets in {PROCESSED_DIR}, found {len(files)}")

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()
    lags = np.arange(MAX_LAG_STEPS + 1)

    for ax, file_path in zip(axes, files):
        acf = pairwise_acf(load_wind_speed(file_path), MAX_LAG_STEPS)
        ax.plot(lags, acf, color="#3E7187", linewidth=1.3)
        ax.scatter(lags, acf, color="#3E7187", s=6, zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.75)
        ax.set_title(file_path.stem, fontsize=10, pad=5)
        ax.set_xlim(0, MAX_LAG_STEPS)
        ax.set_ylim(-0.2, 1.05)
        ax.set_xticks(np.arange(0, MAX_LAG_STEPS + 1, 24))
        ax.set_yticks(np.arange(-0.2, 1.01, 0.2))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(labelsize=8, width=0.7, length=3)

    axes[-1].set_visible(False)
    for ax in axes[::2]:
        ax.set_ylabel("Autocorrelation", fontsize=9)
    for ax in axes[-2:]:
        ax.set_xlabel("Lag (10-min steps)", fontsize=9)

    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG saved to: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
