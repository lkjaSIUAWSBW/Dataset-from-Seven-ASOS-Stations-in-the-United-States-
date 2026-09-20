"""Summarize temporal continuity of the seven raw ASOS station files.

Run: python analyze_time_continuity.py

The output is one station-level CSV table in this script's directory. All
stations use the same inclusive UTC study period. A usable window means that
EVERY 10-minute timestamp in the window has a raw record; meteorological
values are not checked here. Windows slide forward by one 10-minute step.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "Raw dataset"
OUTPUT_CSV = Path(__file__).resolve().parent / "time_continuity_table.csv"



START_UTC = datetime(2016, 1, 1, 0, 0)
END_UTC = datetime(2026, 7, 30, 23, 50)
STEP_MINUTES = 10
TIME_COLUMN = "valid(UTC)"
WINDOWS = {"24h": 24 * 6, "7d": 7 * 24 * 6, "30d": 30 * 24 * 6}


def count_windows(run_length: int, window_length: int) -> int:
    """Number of fully observed rolling windows within one continuous run."""
    return max(0, run_length - window_length + 1)


def summarize(path: Path, total_slots: int) -> dict[str, object]:
    present = bytearray(total_slots)
    row_count = 0
    duplicate_timestamps = 0
    first_observation: datetime | None = None
    last_observation: datetime | None = None

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or TIME_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path.name}: missing column {TIME_COLUMN!r}")

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            try:
                timestamp = datetime.strptime(row[TIME_COLUMN].strip(), "%Y-%m-%d %H:%M")
            except (AttributeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}, CSV line {line_number}: invalid UTC timestamp"
                ) from exc

            if timestamp < START_UTC or timestamp > END_UTC:
                raise ValueError(
                    f"{path.name}, CSV line {line_number}: {timestamp} is outside "
                    "the declared study period"
                )
            minutes_from_start = int((timestamp - START_UTC).total_seconds() // 60)
            if minutes_from_start % STEP_MINUTES:
                raise ValueError(
                    f"{path.name}, CSV line {line_number}: {timestamp} is off "
                    "the 10-minute time grid"
                )

            slot = minutes_from_start // STEP_MINUTES
            if present[slot]:
                duplicate_timestamps += 1
            else:
                present[slot] = 1
            if first_observation is None or timestamp < first_observation:
                first_observation = timestamp
            if last_observation is None or timestamp > last_observation:
                last_observation = timestamp

    if row_count == 0:
        raise ValueError(f"{path.name}: no data rows")

    observed_slots = sum(present)
    missing_slots = total_slots - observed_slots
    longest_gap = 0
    gap_length = 0
    longest_run = 0
    run_length = 0
    usable_windows = {label: 0 for label in WINDOWS}

    for value in present:
        if value:
            if gap_length > longest_gap:
                longest_gap = gap_length
            gap_length = 0
            run_length += 1
        else:
            if run_length:
                longest_run = max(longest_run, run_length)
                for label, length in WINDOWS.items():
                    usable_windows[label] += count_windows(run_length, length)
            run_length = 0
            gap_length += 1

    
    longest_gap = max(longest_gap, gap_length)
    longest_run = max(longest_run, run_length)
    if run_length:
        for label, length in WINDOWS.items():
            usable_windows[label] += count_windows(run_length, length)

    result: dict[str, object] = {
        "station": path.stem,
        "study_start_utc": START_UTC.strftime("%Y-%m-%d %H:%M"),
        "study_end_utc": END_UTC.strftime("%Y-%m-%d %H:%M"),
        "first_raw_record_utc": first_observation.strftime("%Y-%m-%d %H:%M"),
        "last_raw_record_utc": last_observation.strftime("%Y-%m-%d %H:%M"),
        "expected_10min_slots": total_slots,
        "raw_rows": row_count,
        "duplicate_timestamps": duplicate_timestamps,
        "observed_10min_slots": observed_slots,
        "missing_10min_slots": missing_slots,
        "temporal_coverage_pct": round(100 * observed_slots / total_slots, 2),
        "longest_gap_hours": round(longest_gap * STEP_MINUTES / 60, 2),
        "longest_continuous_run_days": round(
            longest_run * STEP_MINUTES / (60 * 24), 2
        ),
    }
    for label, length in WINDOWS.items():
        denominator = max(0, total_slots - length + 1)
        result[f"usable_{label}_windows"] = usable_windows[label]
        result[f"usable_{label}_windows_pct"] = round(
            100 * usable_windows[label] / denominator, 2
        ) if denominator else ""
    return result


def main() -> None:
    if END_UTC < START_UTC:
        raise ValueError("END_UTC must not precede START_UTC")
    span_minutes = int((END_UTC - START_UTC).total_seconds() // 60)
    if span_minutes % STEP_MINUTES:
        raise ValueError("Study period endpoints are not on one 10-minute grid")
    total_slots = span_minutes // STEP_MINUTES + 1

    files = sorted(RAW_DIR.glob("*.csv"), key=lambda p: p.name.casefold())
    if len(files) != 7:
        raise ValueError(f"Expected exactly 7 CSV files in {RAW_DIR}; found {len(files)}")

    results = [summarize(path, total_slots) for path in files]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} station rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
