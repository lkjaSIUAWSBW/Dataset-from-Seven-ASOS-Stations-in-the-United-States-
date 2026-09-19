"""Run with Python and pandas installed; output is saved beside this script.

Each variable uses its own available numeric observations. Missing values are
excluded; quality flags do not otherwise filter observations. Standard deviation
uses ddof=1 (sample SD). Q1 and Q3 use linear interpolation. Empty variables
have valid_observations=0 and blank statistics, not zero-valued statistics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "Processed dataset"
RESULT_PATH = SCRIPT_DIR / "basic_weather_statistics.csv"

VARIABLES = {
    "wind_speed_ms": "m/s",
    "temperature_c": "°C",
    "pres1": "inches",
    "pres2": "inches",
    "pres3": "inches",
}


def summarize_dataset(path: Path) -> list[dict[str, str | int | float]]:
    data = pd.read_csv(path, usecols=list(VARIABLES), dtype="string", keep_default_na=False)
    records: list[dict[str, str | int | float]] = []
    for variable, unit in VARIABLES.items():
        values = pd.to_numeric(data[variable].replace("M", pd.NA), errors="coerce").dropna()
        records.append({
            "dataset": path.name,
            "variable": variable,
            "unit": unit,
            "valid_observations": int(values.size),
            "mean": values.mean() if len(values) else float("nan"),
            "standard_deviation": values.std(ddof=1) if len(values) > 1 else float("nan"),
            "median": values.median() if len(values) else float("nan"),
            "first_quartile_q1": values.quantile(0.25, interpolation="linear") if len(values) else float("nan"),
            "third_quartile_q3": values.quantile(0.75, interpolation="linear") if len(values) else float("nan"),
        })
    return records


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV datasets found in {PROCESSED_DIR}")

    results = [record for path in files for record in summarize_dataset(path)]
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(results)} variable-level summaries to {RESULT_PATH}")


if __name__ == "__main__":
    main()
