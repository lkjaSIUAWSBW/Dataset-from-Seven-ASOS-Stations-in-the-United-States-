"""Create a seven-station table of joint core-variable availability.

Run: python analyze_joint_variable_availability.py

The script reads the seven CSV files in ../Raw dataset and writes one table,
joint_variable_availability.csv, beside this script. Percentages use the
number of raw rows at each station as their denominator. A variable is usable
when it is numeric and finite. Wind speed must also be nonnegative, and wind
direction must be within 0-360 degrees, inclusive. No arbitrary temperature
range is imposed. Missing pressure channels do not affect these results.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent.parent / "Raw dataset"
OUTPUT_CSV = SCRIPT_DIR / "joint_variable_availability.csv"
REQUIRED_COLUMNS = {"tmpf", "sknt", "drct"}
MISSING_CODES = {"", "M", "NA", "N/A", "NULL", "NAN", "NONE"}


def numeric_value(value: str | None) -> float | None:
    """Return a finite numeric value, or None for missing/invalid text."""
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned.upper() in MISSING_CODES:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def summarize(path: Path) -> dict[str, object]:
    counts = {
        "raw_records": 0,
        "wind_usable_n": 0,
        "temperature_usable_n": 0,
        "direction_usable_n": 0,
        "wind_temperature_usable_n": 0,
        "wind_direction_usable_n": 0,
        "wind_temperature_direction_usable_n": 0,
    }

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

        for row in reader:
            counts["raw_records"] += 1
            speed = numeric_value(row["sknt"])
            temperature = numeric_value(row["tmpf"])
            direction = numeric_value(row["drct"])

            wind_ok = speed is not None and speed >= 0
            temperature_ok = temperature is not None
            direction_ok = direction is not None and 0 <= direction <= 360

            counts["wind_usable_n"] += wind_ok
            counts["temperature_usable_n"] += temperature_ok
            counts["direction_usable_n"] += direction_ok
            counts["wind_temperature_usable_n"] += wind_ok and temperature_ok
            counts["wind_direction_usable_n"] += wind_ok and direction_ok
            counts["wind_temperature_direction_usable_n"] += (
                wind_ok and temperature_ok and direction_ok
            )

    denominator = counts["raw_records"]
    if denominator == 0:
        raise ValueError(f"{path.name}: no raw data rows")

    result: dict[str, object] = {"station": path.stem, "raw_records": denominator}
    for name, count in counts.items():
        if name == "raw_records":
            continue
        result[name] = count
        result[name.replace("_n", "_pct")] = round(100 * count / denominator, 2)
    return result


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.name.casefold())
    if len(files) != 7:
        raise ValueError(f"Expected 7 CSV files in {RAW_DIR}; found {len(files)}")

    results = [summarize(path) for path in files]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} station rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
