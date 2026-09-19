"""Describe observed wind speeds and a two-parameter Weibull approximation.

Run: python analyze_wind_weibull.py
Requires: numpy, scipy, matplotlib

Reads the seven CSV files in ../Raw dataset. Writes one station-level CSV and
one seven-panel SVG beside this script. Zero wind speeds are reported as a
separate observed fraction; the Weibull model is fitted only to positive wind
speeds. The figure compares conditional empirical and fitted CDFs of positive
wind speeds. A good or bad fit is descriptive, not a test of sensor accuracy.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import weibull_min


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent.parent / "Raw dataset"
OUTPUT_CSV = SCRIPT_DIR / "wind_weibull_table.csv"
OUTPUT_SVG = SCRIPT_DIR / "wind_weibull_cdf.svg"
KNOT_TO_MS = 0.514444
MISSING_CODES = {"", "M", "NA", "N/A", "NULL", "NAN", "NONE"}


def read_wind(path: Path) -> tuple[np.ndarray, int, int, int]:
    positive: list[float] = []
    raw_rows = 0
    zero_rows = 0
    invalid_rows = 0

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sknt" not in reader.fieldnames:
            raise ValueError(f"{path.name}: missing sknt column")

        for row in reader:
            raw_rows += 1
            field = (row["sknt"] or "").strip()
            if field.upper() in MISSING_CODES:
                invalid_rows += 1
                continue
            try:
                knots = float(field)
            except ValueError:
                invalid_rows += 1
                continue
            if not math.isfinite(knots) or knots < 0:
                invalid_rows += 1
            elif knots == 0:
                zero_rows += 1
            else:
                positive.append(knots * KNOT_TO_MS)

    if raw_rows == 0 or not positive:
        raise ValueError(f"{path.name}: no positive wind speeds to fit")
    return np.asarray(positive, dtype=np.float64), raw_rows, zero_rows, invalid_rows


def fit_station(path: Path) -> tuple[dict[str, object], np.ndarray, float, float]:
    speed, raw_rows, zero_rows, invalid_rows = read_wind(path)
    speed.sort()
    shape, location, scale = weibull_min.fit(speed, floc=0)
    if location != 0 or not (math.isfinite(shape) and math.isfinite(scale)):
        raise ValueError(f"{path.name}: Weibull fit failed")

    # KS distance is a descriptive maximum CDF difference. No p-value is
    # reported because parameters were estimated from these same observations.
    fitted_at_observations = weibull_min.cdf(speed, shape, loc=0, scale=scale)
    n = speed.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    ks_distance = float(
        max(
            np.max(np.abs(fitted_at_observations - ranks / n)),
            np.max(np.abs(fitted_at_observations - (ranks - 1) / n)),
        )
    )
    observed_p90 = float(np.quantile(speed, 0.90))
    fitted_p90 = float(weibull_min.ppf(0.90, shape, loc=0, scale=scale))
    valid_rows = n + zero_rows

    result: dict[str, object] = {
        "station": path.stem,
        "raw_records": raw_rows,
        "valid_wind_records": valid_rows,
        "missing_or_invalid_wind_records": invalid_rows,
        "zero_wind_records": zero_rows,
        "zero_wind_pct_of_valid": round(100 * zero_rows / valid_rows, 2),
        "positive_wind_records_used_for_fit": n,
        "weibull_shape_k": round(float(shape), 4),
        "weibull_scale_lambda_ms": round(float(scale), 4),
        "cdf_max_absolute_difference": round(ks_distance, 4),
        "observed_positive_wind_p90_ms": round(observed_p90, 3),
        "weibull_positive_wind_p90_ms": round(fitted_p90, 3),
    }
    return result, speed, float(shape), float(scale)


def plot_results(fits: list[tuple[dict[str, object], np.ndarray, float, float]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",  # Preserve selectable text in the SVG.
    })
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharey=True)
    axes_flat = axes.ravel()

    for ax, (result, speed, shape, scale) in zip(axes_flat, fits):
        # A grid of display points keeps the SVG small while retaining the
        # exact empirical CDF at each displayed x value.
        xmax = max(float(np.quantile(speed, 0.995)), float(weibull_min.ppf(0.995, shape, scale=scale)))
        x = np.linspace(0, xmax, 600)
        empirical = np.searchsorted(speed, x, side="right") / speed.size
        fitted = weibull_min.cdf(x, shape, loc=0, scale=scale)
        ax.plot(x, empirical, color="#0072B2", lw=1.5, label="Observed")
        ax.plot(x, fitted, color="#D55E00", lw=1.4, ls="--", label="Weibull")
        ax.set_title(str(result["station"]), loc="left", pad=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#DDDDDD", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes_flat[-1].axis("off")
    fig.supxlabel("Positive wind speed (m/s)", y=0.02)
    fig.supylabel("Conditional cumulative probability", x=0.005)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.10), frameon=False)
    fig.tight_layout(rect=(0.025, 0.045, 1, 1), w_pad=1.2, h_pad=1.2)
    fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.name.casefold())
    if len(files) != 7:
        raise ValueError(f"Expected 7 CSV files in {RAW_DIR}; found {len(files)}")
    fits = [fit_station(path) for path in files]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fits[0][0]))
        writer.writeheader()
        writer.writerows(item[0] for item in fits)
    plot_results(fits)
    print(f"Saved {OUTPUT_CSV}")
    print(f"Saved {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
