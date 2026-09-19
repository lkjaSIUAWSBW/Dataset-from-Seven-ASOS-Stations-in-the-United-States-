"""Calculate 10-minute time coverage for each processed ASOS dataset."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR.parent / "Processed dataset"
RESULT_PATH = SCRIPT_DIR / "dataset_time_coverage.csv"
TIME_FIELD = "valid(UTC)"
MISSING_TIMESTAMP_FIELD = "MissingTimestamp"
TIME_FORMAT = "%Y-%m-%d %H:%M"
INTERVAL_MINUTES = 10


def summarize_dataset(path: Path) -> dict[str, str | int | float]:
    earliest: datetime | None = None
    latest: datetime | None = None
    actual_records = 0
    missing_timestamps = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required_fields = {TIME_FIELD, MISSING_TIMESTAMP_FIELD}
        if not reader.fieldnames or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"{path.name} lacks required fields: {sorted(required_fields)}")
        for row in reader:
            timestamp = datetime.strptime(row[TIME_FIELD], TIME_FORMAT)
            earliest = timestamp if earliest is None or timestamp < earliest else earliest
            latest = timestamp if latest is None or timestamp > latest else latest
            if row[MISSING_TIMESTAMP_FIELD].strip() == "1":
                missing_timestamps += 1
            else:
                actual_records += 1

    if earliest is None or latest is None:
        raise ValueError(f"{path.name} contains no data rows")
    theoretical_points = int((latest - earliest).total_seconds() // (INTERVAL_MINUTES * 60)) + 1
    coverage_percent = actual_records / theoretical_points * 100
    return {
        "dataset": path.name,
        "start_time_utc": earliest.strftime(TIME_FORMAT),
        "end_time_utc": latest.strftime(TIME_FORMAT),
        "theoretical_time_points": theoretical_points,
        "actual_observation_records": actual_records,
        "missing_time_points": missing_timestamps,
        "time_coverage_percent": round(coverage_percent, 6),
    }


def main() -> None:
    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV datasets found in {PROCESSED_DIR}")
    summaries = [summarize_dataset(path) for path in files]
    fields = list(summaries[0])
    with RESULT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved {len(summaries)} dataset summaries to {RESULT_PATH}")


if __name__ == "__main__":
    main()
